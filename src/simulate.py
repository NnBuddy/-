"""Synthetic temperature-stream simulator.

Generates latent dynamics (Ornstein–Uhlenbeck or local-linear-trend), additive
observation noise, and missingness masks under MCAR / bursty / MAR / MNAR
mechanisms. Injectable jumps and ramps create known threshold-crossing times.

Assumptions
    Discrete time with unit steps. Process noise is Gaussian. Observation noise
    is independent of the latent process. MNAR depends on the contemporaneous
    latent state ``x(t)``; MAR depends only on the last *observed* value (not
    the current latent). Bursty dropout is a two-state Markov chain independent
    of ``x``. Forced outages overwrite the generated mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import expit

DynamicsName = Literal["ou", "local_linear_trend"]
NoiseFamily = Literal["gaussian", "student_t"]
MissingMechanism = Literal["mcar", "bursty", "mar", "mnar"]
NoiseTier = Literal["low", "medium", "high"]

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]

TARGET_MISSING_RATES: tuple[float, float, float, float] = (0.10, 0.30, 0.50, 0.70)

NOISE_TIER_SIGMA: dict[str, float] = {
    "low": 0.10,
    "medium": 0.30,
    "high": 1.00,
}

_G0_PAD = 30.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Jump:
    """Instantaneous additive shock to the latent level at ``time``.

    Attributes:
        time: Time index at which the jump is applied (after the stochastic
            update at that step).
        magnitude: Amount added to the latent level.
    """

    time: int
    magnitude: float


@dataclass(frozen=True)
class Ramp:
    """Deterministic per-step drift applied on ``[start, start + length)``.

    Attributes:
        start: First time index of the ramp (inclusive).
        length: Number of steps the drift is applied.
        slope: Amount added to the latent level at each step in the window.
    """

    start: int
    length: int
    slope: float


@dataclass
class DynamicsConfig:
    """Latent-process specification.

    Ornstein–Uhlenbeck uses the mean-reverting scalar recursion

        ``x(t+1) = mu + phi * (x(t) - mu) + w(t)``,
        ``phi = exp(-theta * dt)``,
        ``Var(w) = sigma^2 * (1 - phi^2) / (2 * theta)``  (``theta > 0``).

    Local-linear-trend uses the 2-D state ``(level, trend)``

        ``level(t+1) = level(t) + trend(t) + w_level(t)``,
        ``trend(t+1) = trend(t) + w_trend(t)``,

    with ``Q = diag(q_level, q_trend)`` (variances of the process noises).

    Attributes:
        name: ``"ou"`` or ``"local_linear_trend"``.
        mu: OU long-run mean (ignored by LLT except as a default start).
        theta: OU reversion speed. Must be ``> 0``.
        sigma: OU diffusion coefficient.
        dt: OU discretization step.
        x0: Initial latent level. If ``None``, OU starts at ``mu`` and LLT
            starts at ``mu``.
        q_level: LLT level-noise variance.
        q_trend: LLT trend-noise variance.
        trend0: LLT initial trend.
        jumps: Additive level shocks.
        ramps: Deterministic drifts used to manufacture crossings of ``tau``.
    """

    name: DynamicsName = "local_linear_trend"
    mu: float = 6.0
    theta: float = 0.05
    sigma: float = 0.15
    dt: float = 1.0
    x0: float | None = None
    q_level: float = 0.02
    q_trend: float = 0.002
    trend0: float = 0.01
    jumps: tuple[Jump, ...] = ()
    ramps: tuple[Ramp, ...] = ()


@dataclass
class NoiseConfig:
    """Observation-noise specification.

    Gaussian: ``e(t) ~ N(0, sigma(t)^2)``.
    Student-t: ``e(t) = sigma(t) * T_nu`` with ``T_nu`` standard Student-t
    (unscaled; variance is ``sigma(t)^2 * nu / (nu - 2)`` for ``nu > 2``).

    Occasional spikes replace ``e(t)`` with ``spike_scale * sigma(t) * Z``,
    ``Z ~ N(0, 1)``, independently with probability ``spike_probability``.

    If ``tier`` is set, it overrides ``sigma`` with a preset scale.

    Attributes:
        family: ``"gaussian"`` or ``"student_t"``.
        sigma: Scalar or length-``T`` array of scales.
        df: Student-t degrees of freedom.
        tier: Optional ``low`` / ``medium`` / ``high`` preset.
        spike_probability: Per-step probability of an outlier spike.
        spike_scale: Multiplier applied to ``sigma(t)`` for spikes.
    """

    family: NoiseFamily = "gaussian"
    sigma: float | Sequence[float] = 0.3
    df: float = 5.0
    tier: NoiseTier | None = None
    spike_probability: float = 0.0
    spike_scale: float = 8.0


@dataclass
class MissingConfig:
    """Missingness mechanism.

    MNAR: ``P(m(t)=1 | x(t)) = logistic(g0 + g1 * (x(t) - mu))``.
    ``g0`` is solved so the *mean* observation probability equals
    ``1 - target_rate``. ``g1 < 0`` yields dropout that increases with
    temperature.

    MAR: the same logistic form, but the covariate is the last observed
    ``y`` rather than ``x(t)``. ``g0`` is solved on the sequential mask.

    Bursty: two-state Markov chain with mean missing-run length
    ``mean_gap_length`` and stationary missing rate ``target_rate``.

    Attributes:
        mechanism: One of ``mcar``, ``bursty``, ``mar``, ``mnar``.
        target_rate: Desired ``P(m=0)``.
        mean_gap_length: Mean consecutive-missing length (bursty).
        g1: Logistic slope for MAR / MNAR.
        mu: Location subtracted inside the logistic. If ``None``, the
            generator uses the latent-process mean or the sample mean of
            ``x``.
        forced_outages: Hard missing windows ``(start, length)`` overlaid
            after the stochastic mask is drawn.
        always_observe_first: If True, ``m[0] = 1`` (needed for MAR).
        covariate_clip: Absolute clip applied to ``(last_obs - mu)`` in MAR.
            Bounds the logistic covariate so ``g0`` remains identifiable on
            drifting series. ``None`` disables clipping.
    """

    mechanism: MissingMechanism = "mcar"
    target_rate: float = 0.12
    mean_gap_length: float = 5.0
    g1: float = -1.5
    mu: float | None = None
    forced_outages: tuple[tuple[int, int], ...] = ()
    always_observe_first: bool = True
    covariate_clip: float | None = 5.0


@dataclass
class SimulationConfig:
    """Top-level simulator configuration."""

    T: int = 400
    tau: float = 8.0
    H: int = 15
    seed: int = 0
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)


@dataclass
class SimulationMetadata:
    """Parameters and diagnostics attached to a simulated stream."""

    T: int
    H: int
    seed: int
    dynamics: str
    noise_family: str
    missing_mechanism: str
    process_params: dict[str, Any]
    noise_params: dict[str, Any]
    missing_params: dict[str, Any]
    g0: float | None
    g1: float | None
    achieved_missing_rate: float
    mean_observation_probability: float | None
    trend: FloatArray | None = None


@dataclass
class SimulationResult:
    """Structured simulator output.

    Attributes:
        x_true: Latent temperature, shape ``(T,)``.
        y: Observations with ``NaN`` where ``m == 0``.
        m: Observation indicator (1 = observed, 0 = missing).
        tau: Excursion threshold.
        true_crossing_times: Indices of upcrossings of ``tau`` by ``x_true``.
        metadata: Generating parameters and achieved missing rate.
        y_full: Latent-plus-noise stream before masking (no NaNs).
    """

    x_true: FloatArray
    y: FloatArray
    m: IntArray
    tau: float
    true_crossing_times: IntArray
    metadata: SimulationMetadata
    y_full: FloatArray


# ---------------------------------------------------------------------------
# Latent generators
# ---------------------------------------------------------------------------


def _jump_map(jumps: Sequence[Jump]) -> dict[int, float]:
    acc: dict[int, float] = {}
    for jump in jumps:
        acc[jump.time] = acc.get(jump.time, 0.0) + jump.magnitude
    return acc


def _ramp_map(ramps: Sequence[Ramp]) -> dict[int, float]:
    acc: dict[int, float] = {}
    for ramp in ramps:
        if ramp.length < 0:
            raise ValueError("Ramp length must be non-negative.")
        for t in range(ramp.start, ramp.start + ramp.length):
            acc[t] = acc.get(t, 0.0) + ramp.slope
    return acc


def _apply_controls(level: float, t: int, jumps: Mapping[int, float], ramps: Mapping[int, float]) -> float:
    return level + jumps.get(t, 0.0) + ramps.get(t, 0.0)


def generate_ou(
    T: int,
    params: DynamicsConfig,
    rng: Generator,
) -> FloatArray:
    """Simulate a discrete-time Ornstein–Uhlenbeck path of length ``T``."""
    if T < 1:
        raise ValueError("T must be >= 1.")
    if params.theta <= 0.0:
        raise ValueError("OU reversion speed theta must be > 0.")
    if params.dt <= 0.0:
        raise ValueError("OU dt must be > 0.")

    phi = float(np.exp(-params.theta * params.dt))
    # Exact Gaussian transition variance of the OU SDE.
    q = (params.sigma**2) * (1.0 - phi**2) / (2.0 * params.theta)
    q = max(q, 0.0)
    std = float(np.sqrt(q))

    jumps = _jump_map(params.jumps)
    ramps = _ramp_map(params.ramps)

    x = np.empty(T, dtype=float)
    x0 = params.mu if params.x0 is None else params.x0
    x[0] = _apply_controls(x0, 0, jumps, ramps)
    for t in range(1, T):
        w = rng.normal(0.0, std)
        level = params.mu + phi * (x[t - 1] - params.mu) + w
        x[t] = _apply_controls(level, t, jumps, ramps)
    return x


def generate_local_linear_trend(
    T: int,
    params: DynamicsConfig,
    rng: Generator,
) -> tuple[FloatArray, FloatArray]:
    """Simulate a local-linear-trend latent level and trend.

    Returns:
        Tuple ``(level, trend)`` each of shape ``(T,)``. The observed latent
        temperature is the level component.
    """
    if T < 1:
        raise ValueError("T must be >= 1.")
    if params.q_level < 0.0 or params.q_trend < 0.0:
        raise ValueError("LLT process-noise variances must be >= 0.")

    jumps = _jump_map(params.jumps)
    ramps = _ramp_map(params.ramps)
    std_level = float(np.sqrt(params.q_level))
    std_trend = float(np.sqrt(params.q_trend))

    level = np.empty(T, dtype=float)
    trend = np.empty(T, dtype=float)
    x0 = params.mu if params.x0 is None else params.x0
    trend[0] = params.trend0
    level[0] = _apply_controls(x0, 0, jumps, ramps)

    for t in range(1, T):
        w_level = rng.normal(0.0, std_level)
        w_trend = rng.normal(0.0, std_trend)
        new_level = level[t - 1] + trend[t - 1] + w_level
        new_trend = trend[t - 1] + w_trend
        level[t] = _apply_controls(new_level, t, jumps, ramps)
        trend[t] = new_trend
    return level, trend


def generate_latent(
    T: int,
    params: DynamicsConfig,
    rng: Generator,
) -> tuple[FloatArray, FloatArray | None]:
    """Dispatch latent generation.

    Returns:
        ``(x_true, trend_or_none)``.
    """
    if params.name == "ou":
        return generate_ou(T, params, rng), None
    if params.name == "local_linear_trend":
        return generate_local_linear_trend(T, params, rng)
    raise ValueError(f"Unknown dynamics '{params.name}'.")


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def resolve_sigma(T: int, noise: NoiseConfig) -> FloatArray:
    """Build a length-``T`` observation-scale schedule."""
    if noise.tier is not None:
        if noise.tier not in NOISE_TIER_SIGMA:
            raise ValueError(f"Unknown noise tier '{noise.tier}'.")
        return np.full(T, NOISE_TIER_SIGMA[noise.tier], dtype=float)
    if np.isscalar(noise.sigma):
        return np.full(T, float(noise.sigma), dtype=float)  # type: ignore[arg-type]
    arr = np.asarray(noise.sigma, dtype=float)
    if arr.shape != (T,):
        raise ValueError(f"sigma array must have shape ({T},), got {arr.shape}.")
    if np.any(arr < 0.0):
        raise ValueError("Observation scales must be >= 0.")
    return arr


def generate_observation_noise(
    T: int,
    noise: NoiseConfig,
    rng: Generator,
) -> tuple[FloatArray, FloatArray]:
    """Draw observation noise and the scale schedule used.

    Returns:
        Tuple ``(e, sigma)``.
    """
    sigma = resolve_sigma(T, noise)
    if noise.family == "gaussian":
        e = rng.normal(0.0, 1.0, size=T) * sigma
    elif noise.family == "student_t":
        if noise.df <= 0.0:
            raise ValueError("Student-t degrees of freedom must be > 0.")
        e = rng.standard_t(noise.df, size=T) * sigma
    else:
        raise ValueError(f"Unknown noise family '{noise.family}'.")

    if noise.spike_probability < 0.0 or noise.spike_probability > 1.0:
        raise ValueError("spike_probability must be in [0, 1].")
    if noise.spike_probability > 0.0:
        spike = rng.random(T) < noise.spike_probability
        e = np.where(
            spike,
            noise.spike_scale * sigma * rng.normal(0.0, 1.0, size=T),
            e,
        )
    return e, sigma


# ---------------------------------------------------------------------------
# Missingness
# ---------------------------------------------------------------------------


def logistic(z: float | FloatArray) -> FloatArray:
    """Numerically stable logistic (expit)."""
    return np.asarray(expit(z), dtype=float)


def solve_mnar_g0(
    x: FloatArray,
    g1: float,
    mu: float,
    target_missing_rate: float,
) -> float:
    """Solve ``g0`` so ``E[logistic(g0 + g1 (x - mu))] = 1 - target_rate``.

    The expectation is the sample mean over the provided latent path, so the
    *mean observation probability* matches the target up to solver tolerance.
    Realized Bernoulli missing rates still have binomial sampling error.

    Args:
        x: Latent path.
        g1: Logistic slope. Negative values increase dropout as ``x`` rises.
        mu: Location subtracted from ``x``.
        target_missing_rate: Desired mean of ``1 - p_obs``.

    Returns:
        Intercept ``g0``.
    """
    rate = _clip_rate(target_missing_rate)
    z = g1 * (np.asarray(x, dtype=float) - mu)
    target_obs = 1.0 - rate

    def objective(g0: float) -> float:
        return float(np.mean(expit(g0 + z))) - target_obs

    lo, hi = _g0_bracket(z)
    return _safe_brentq(objective, lo, hi, fallback=_logit(target_obs) - float(np.mean(z)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return float(np.log(p / (1.0 - p)))


def _clip_rate(rate: float) -> float:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Missing rate must be in [0, 1], got {rate}.")
    return float(min(max(rate, 1e-8), 1.0 - 1e-8))


def _g0_bracket(z: FloatArray) -> tuple[float, float]:
    """Intercept range that can drive ``expit(g0 + z)`` to both 0 and 1."""
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    lo = -_G0_PAD - z_max
    hi = _G0_PAD - z_min
    if lo >= hi:
        return -_G0_PAD, _G0_PAD
    return lo, hi


def _safe_brentq(objective, lo: float, hi: float, fallback: float) -> float:
    f_lo = objective(lo)
    f_hi = objective(hi)
    if f_lo == 0.0:
        return float(lo)
    if f_hi == 0.0:
        return float(hi)
    if f_lo * f_hi < 0.0:
        return float(brentq(objective, lo, hi, xtol=1e-10))
    # Degenerate covariate (constant z) or numerically saturated logits.
    return float(fallback)


def generate_mcar(T: int, rate: float, rng: Generator) -> IntArray:
    """Independent Bernoulli observation mask (``1`` = observed)."""
    rate = _clip_rate(rate)
    return (rng.random(T) >= rate).astype(np.int8)


def bursty_transition_probs(target_rate: float, mean_gap_length: float) -> tuple[float, float]:
    """Stationary two-state chain matching missing rate and mean gap length.

    Returns:
        ``(p_drop, p_recover)`` where ``p_drop = P(O -> M)`` and
        ``p_recover = P(M -> O)``. Gap lengths (runs of missing) are
        geometric with mean ``1 / p_recover``.
    """
    if mean_gap_length < 1.0:
        raise ValueError("mean_gap_length must be >= 1 (geometric mean).")
    rate = _clip_rate(target_rate)
    p_recover = 1.0 / mean_gap_length
    p_recover = min(max(p_recover, 1e-8), 1.0)
    # Stationary P(M) = p_drop / (p_drop + p_recover) = rate
    # => p_drop = rate * p_recover / (1 - rate)
    p_drop = rate * p_recover / max(1.0 - rate, 1e-12)
    p_drop = min(max(p_drop, 0.0), 1.0)
    return float(p_drop), float(p_recover)


def generate_bursty(
    T: int,
    target_rate: float,
    mean_gap_length: float,
    rng: Generator,
) -> tuple[IntArray, float, float]:
    """Two-state Markov dropout. Starts from the stationary distribution."""
    p_drop, p_recover = bursty_transition_probs(target_rate, mean_gap_length)
    rate = _clip_rate(target_rate)
    m = np.empty(T, dtype=np.int8)
    # Stationary P(observed) = 1 - rate.
    state = 1 if rng.random() >= rate else 0
    for t in range(T):
        m[t] = state
        if state == 1:
            if rng.random() < p_drop:
                state = 0
        elif rng.random() < p_recover:
            state = 1
    return m, p_drop, p_recover


def _clipped_residual(value: float, mu: float, clip: float | None) -> float:
    residual = float(value) - mu
    if clip is None:
        return residual
    if clip < 0.0:
        raise ValueError("covariate_clip must be >= 0 or None.")
    return float(np.clip(residual, -clip, clip))


def _mar_mask_from_uniforms(
    y_full: FloatArray,
    g0: float,
    g1: float,
    mu: float,
    uniforms: FloatArray,
    always_observe_first: bool,
    covariate_clip: float | None,
) -> IntArray:
    """Sequential MAR mask with frozen uniforms (so ``g0`` can be calibrated)."""
    T = int(y_full.shape[0])
    m = np.zeros(T, dtype=np.int8)
    last_obs = float(y_full[0])
    y = np.asarray(y_full, dtype=float)
    u = np.asarray(uniforms, dtype=float)
    for t in range(T):
        if t == 0 and always_observe_first:
            m[t] = 1
            last_obs = float(y[t])
            continue
        p_obs = float(expit(g0 + g1 * _clipped_residual(last_obs, mu, covariate_clip)))
        observed = bool(u[t] < p_obs)
        m[t] = 1 if observed else 0
        if observed:
            last_obs = float(y[t])
    return m


def generate_mar(
    y_full: FloatArray,
    g1: float,
    mu: float,
    target_rate: float,
    rng: Generator,
    always_observe_first: bool = True,
    covariate_clip: float | None = 5.0,
) -> tuple[IntArray, float]:
    """MAR mask depending on the last observed value; ``g0`` hits ``target_rate``.

    Uniforms are drawn once, then ``g0`` is bisected so the realized sequential
    observation rate matches the target. Residuals are clipped so a drifting
    level cannot saturate the logistic and make the target unattainable.
    """
    rate = _clip_rate(target_rate)
    target_obs = 1.0 - rate
    y = np.asarray(y_full, dtype=float)
    uniforms = rng.random(y.shape[0])
    residual = y - mu
    if covariate_clip is not None:
        residual = np.clip(residual, -covariate_clip, covariate_clip)
    z = g1 * residual
    lo, hi = _g0_bracket(z)

    def obs_rate(g0: float) -> float:
        return float(
            _mar_mask_from_uniforms(
                y, g0, g1, mu, uniforms, always_observe_first, covariate_clip
            ).mean()
        )

    left, right = lo, hi
    for _ in range(48):
        mid = 0.5 * (left + right)
        if obs_rate(mid) < target_obs:
            left = mid
        else:
            right = mid
    g0 = 0.5 * (left + right)
    m = _mar_mask_from_uniforms(
        y, g0, g1, mu, uniforms, always_observe_first, covariate_clip
    )
    return m, g0


def generate_mnar(
    x: FloatArray,
    g1: float,
    mu: float,
    target_rate: float,
    rng: Generator,
) -> tuple[IntArray, float, FloatArray]:
    """MNAR mask with logistic dependence on current latent ``x``."""
    g0 = solve_mnar_g0(x, g1, mu, target_rate)
    p_obs = logistic(g0 + g1 * (x - mu))
    m = (rng.random(x.shape[0]) < p_obs).astype(np.int8)
    return m, g0, p_obs


def apply_forced_outages(m: IntArray, outages: Sequence[tuple[int, int]]) -> IntArray:
    """Force contiguous missing windows ``(start, length)``."""
    out = m.copy()
    T = out.shape[0]
    for start, length in outages:
        if length < 0:
            raise ValueError("Outage length must be >= 0.")
        lo = max(start, 0)
        hi = min(start + length, T)
        if lo < hi:
            out[lo:hi] = 0
    return out


def missing_run_lengths(m: np.ndarray) -> IntArray:
    """Lengths of consecutive missing runs (``m == 0``)."""
    mask = np.asarray(m) == 0
    if mask.size == 0 or not np.any(mask):
        return np.asarray([], dtype=int)
    # Pad so edge runs are captured.
    padded = np.concatenate([[False], mask, [False]])
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return (ends - starts).astype(int)


def upcrossing_times(x: FloatArray, tau: float) -> IntArray:
    """Indices ``t`` where ``x`` first exceeds ``tau`` (including ``t = 0``)."""
    above = np.asarray(x, dtype=float) > tau
    if above.size == 0:
        return np.asarray([], dtype=int)
    times: list[int] = []
    if above[0]:
        times.append(0)
    if above.size > 1:
        times.extend(np.where((~above[:-1]) & above[1:])[0] + 1)
    return np.asarray(times, dtype=int)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def simulate(config: SimulationConfig | None = None, **overrides: Any) -> SimulationResult:
    """Generate a synthetic temperature stream.

    Args:
        config: Full configuration. If ``None``, defaults are used.
        **overrides: Convenience field overrides for ``T``, ``tau``, ``H``,
            ``seed``, or nested dataclasses.

    Returns:
        :class:`SimulationResult` with latent path, masked observations, and
        true upcrossing times of ``tau``.
    """
    cfg = config if config is not None else SimulationConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown simulate() argument '{key}'.")
        setattr(cfg, key, value)

    if cfg.T < 1:
        raise ValueError("T must be >= 1.")

    rng = np.random.default_rng(cfg.seed)
    x, trend = generate_latent(cfg.T, cfg.dynamics, rng)
    e, sigma = generate_observation_noise(cfg.T, cfg.noise, rng)
    y_full = x + e

    miss = cfg.missing
    mu = miss.mu
    if mu is None:
        mu = float(cfg.dynamics.mu)

    g0: float | None = None
    g1: float | None = None
    mean_p: float | None = None
    extra_missing: dict[str, Any] = {
        "target_rate": miss.target_rate,
        "forced_outages": list(miss.forced_outages),
        "mu": mu,
    }

    if miss.mechanism == "mcar":
        m = generate_mcar(cfg.T, miss.target_rate, rng)
    elif miss.mechanism == "bursty":
        m, p_drop, p_recover = generate_bursty(
            cfg.T, miss.target_rate, miss.mean_gap_length, rng
        )
        extra_missing.update(
            {
                "mean_gap_length": miss.mean_gap_length,
                "p_drop": p_drop,
                "p_recover": p_recover,
            }
        )
    elif miss.mechanism == "mar":
        g1 = miss.g1
        m, g0 = generate_mar(
            y_full,
            g1=g1,
            mu=mu,
            target_rate=miss.target_rate,
            rng=rng,
            always_observe_first=miss.always_observe_first,
            covariate_clip=miss.covariate_clip,
        )
        extra_missing["always_observe_first"] = miss.always_observe_first
        extra_missing["covariate_clip"] = miss.covariate_clip
    elif miss.mechanism == "mnar":
        g1 = miss.g1
        m, g0, p_obs = generate_mnar(x, g1=g1, mu=mu, target_rate=miss.target_rate, rng=rng)
        mean_p = float(np.mean(p_obs))
        extra_missing["mean_observation_probability"] = mean_p
    else:
        raise ValueError(f"Unknown missingness mechanism '{miss.mechanism}'.")

    if miss.always_observe_first and miss.mechanism != "mar":
        m = m.copy()
        m[0] = 1

    m = apply_forced_outages(m, miss.forced_outages)
    y = np.where(m == 1, y_full, np.nan)
    crossings = upcrossing_times(x, cfg.tau)

    metadata = SimulationMetadata(
        T=cfg.T,
        H=cfg.H,
        seed=cfg.seed,
        dynamics=cfg.dynamics.name,
        noise_family=cfg.noise.family,
        missing_mechanism=miss.mechanism,
        process_params={
            "mu": cfg.dynamics.mu,
            "theta": cfg.dynamics.theta,
            "sigma": cfg.dynamics.sigma,
            "dt": cfg.dynamics.dt,
            "x0": cfg.dynamics.x0,
            "q_level": cfg.dynamics.q_level,
            "q_trend": cfg.dynamics.q_trend,
            "trend0": cfg.dynamics.trend0,
            "jumps": [(j.time, j.magnitude) for j in cfg.dynamics.jumps],
            "ramps": [(r.start, r.length, r.slope) for r in cfg.dynamics.ramps],
        },
        noise_params={
            "sigma_mean": float(np.mean(sigma)),
            "tier": cfg.noise.tier,
            "df": cfg.noise.df,
            "spike_probability": cfg.noise.spike_probability,
            "spike_scale": cfg.noise.spike_scale,
        },
        missing_params=extra_missing,
        g0=g0,
        g1=g1,
        achieved_missing_rate=float(1.0 - m.mean()),
        mean_observation_probability=mean_p,
        trend=trend,
    )
    return SimulationResult(
        x_true=x,
        y=y,
        m=m.astype(np.int8),
        tau=cfg.tau,
        true_crossing_times=crossings,
        metadata=metadata,
        y_full=y_full,
    )


def demo_scenario(T: int = 400, seed: int = 0, outage_start: int = 80) -> SimulationResult:
    """Baseline demo used throughout the project.

    Local-linear-trend with process-noise variances ``0.02`` (level) and
    ``0.002`` (trend), Gaussian observation noise ``σ = 0.3``, 12% MCAR
    dropout plus a 45-step outage, threshold ``τ = 8``, horizon ``H = 15``.

    The default initial level / trend are chosen so the latent path is
    expected to cross ``τ`` inside a length-400 window.
    """
    if outage_start < 0 or outage_start >= T:
        raise ValueError("outage_start must lie in [0, T).")
    length = min(45, T - outage_start)
    cfg = SimulationConfig(
        T=T,
        tau=8.0,
        H=15,
        seed=seed,
        dynamics=DynamicsConfig(
            name="local_linear_trend",
            mu=5.5,
            x0=5.5,
            q_level=0.02,
            q_trend=0.002,
            trend0=0.012,
        ),
        noise=NoiseConfig(family="gaussian", sigma=0.3),
        missing=MissingConfig(
            mechanism="mcar",
            target_rate=0.12,
            forced_outages=((outage_start, length),),
        ),
    )
    return simulate(cfg)


def to_dataframe(result: SimulationResult) -> pd.DataFrame:
    """Convert a simulation result to a time-indexed data frame."""
    return pd.DataFrame(
        {
            "t": np.arange(result.x_true.shape[0]),
            "x_true": result.x_true,
            "y": result.y,
            "m": result.m,
            "y_full": result.y_full,
        }
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_simulation(
    result: SimulationResult,
    ax: Axes | None = None,
    show_threshold: bool = True,
) -> Figure:
    """Plot latent path, noisy observations, and missingness.

    Two stacked panels: temperatures (top) and the observation mask (bottom).
    """
    if ax is None:
        fig, (ax_temp, ax_miss) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1]
        )
        draw_mask = True
    else:
        fig = ax.figure
        ax_temp = ax
        ax_miss = None
        draw_mask = False

    t = np.arange(result.x_true.shape[0])
    ax_temp.plot(t, result.x_true, color="black", lw=1.2, label=r"$x_{\mathrm{true}}$")
    observed = result.m == 1
    ax_temp.scatter(
        t[observed],
        result.y[observed],
        s=12,
        c="C0",
        alpha=0.7,
        label=r"$y$ (observed)",
        zorder=3,
    )
    if show_threshold:
        ax_temp.axhline(result.tau, color="C3", ls="--", lw=1.0, label=rf"$\tau={result.tau:g}$")
    for k, tc in enumerate(result.true_crossing_times):
        ax_temp.axvline(
            tc,
            color="C3",
            alpha=0.35,
            lw=0.8,
            label="upcrossing" if k == 0 else None,
        )
    ax_temp.set_ylabel("temperature")
    ax_temp.legend(loc="upper left", fontsize=8, frameon=False)
    ax_temp.set_title(
        f"{result.metadata.dynamics} | {result.metadata.missing_mechanism} | "
        f"missing={result.metadata.achieved_missing_rate:.1%} | H={result.metadata.H}"
    )

    if draw_mask and ax_miss is not None:
        ax_miss.fill_between(t, 0, 1 - result.m, step="mid", color="0.65", label="missing")
        ax_miss.set_ylim(-0.05, 1.15)
        ax_miss.set_ylabel("gap")
        ax_miss.set_xlabel("t")
        ax_miss.set_yticks([0, 1])
    else:
        ax_temp.set_xlabel("t")

    fig.tight_layout()
    return fig


def plot_missingness_diagnostics(
    result: SimulationResult,
    bins: int = 20,
) -> Figure:
    """Scatter / histogram diagnostics for missingness vs latent state."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = result.x_true
    m = result.m.astype(float)

    axes[0].scatter(x, m + 0.04 * np.random.default_rng(0).normal(size=x.size), s=8, alpha=0.35)
    axes[0].set_xlabel(r"$x_{\mathrm{true}}$")
    axes[0].set_ylabel(r"$m$ (jittered)")
    axes[0].set_title("Observation indicator vs latent")

    observed = m == 1
    axes[1].hist(x[observed], bins=bins, density=True, alpha=0.6, label="observed")
    axes[1].hist(x[~observed], bins=bins, density=True, alpha=0.6, label="missing")
    axes[1].set_xlabel(r"$x_{\mathrm{true}}$")
    axes[1].set_ylabel("density")
    axes[1].set_title("Latent distribution by missingness")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"{result.metadata.missing_mechanism}  g0={result.metadata.g0}  "
        f"g1={result.metadata.g1}  rate={result.metadata.achieved_missing_rate:.2%}"
    )
    fig.tight_layout()
    return fig
