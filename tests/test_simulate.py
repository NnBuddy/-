"""Tests for the synthetic temperature-stream simulator."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src.simulate import (
    NOISE_TIER_SIGMA,
    TARGET_MISSING_RATES,
    DynamicsConfig,
    Jump,
    MissingConfig,
    NoiseConfig,
    Ramp,
    SimulationConfig,
    bursty_transition_probs,
    demo_scenario,
    missing_run_lengths,
    plot_missingness_diagnostics,
    plot_simulation,
    simulate,
    solve_mnar_g0,
    to_dataframe,
    upcrossing_times,
)

SEED = 20261022


def _cfg(**kwargs: object) -> SimulationConfig:
    base = SimulationConfig(T=500, seed=SEED)
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


# ---------------------------------------------------------------------------
# Shapes, masking, reproducibility
# ---------------------------------------------------------------------------


def test_result_shapes_and_nan_mask() -> None:
    result = simulate(_cfg(T=250))
    assert result.x_true.shape == (250,)
    assert result.y.shape == (250,)
    assert result.m.shape == (250,)
    assert result.y_full.shape == (250,)
    np.testing.assert_array_equal(np.isnan(result.y), result.m == 0)
    np.testing.assert_allclose(result.y[result.m == 1], result.y_full[result.m == 1])
    assert result.tau == 8.0
    assert result.metadata.H == 15


def test_reproducibility_with_fixed_seed() -> None:
    a = simulate(_cfg(T=300, seed=11))
    b = simulate(_cfg(T=300, seed=11))
    np.testing.assert_allclose(a.x_true, b.x_true)
    np.testing.assert_allclose(a.y_full, b.y_full)
    np.testing.assert_array_equal(a.m, b.m)
    np.testing.assert_array_equal(a.true_crossing_times, b.true_crossing_times)


def test_to_dataframe_roundtrip_columns() -> None:
    result = simulate(_cfg(T=40))
    frame = to_dataframe(result)
    assert list(frame.columns) == ["t", "x_true", "y", "m", "y_full"]
    assert len(frame) == 40


# ---------------------------------------------------------------------------
# Latent dynamics
# ---------------------------------------------------------------------------


def test_llt_zero_noise_is_exact_line() -> None:
    dyn = DynamicsConfig(
        name="local_linear_trend",
        x0=1.0,
        trend0=0.25,
        q_level=0.0,
        q_trend=0.0,
        jumps=(),
        ramps=(),
    )
    result = simulate(
        SimulationConfig(
            T=21,
            seed=0,
            dynamics=dyn,
            noise=NoiseConfig(sigma=0.0),
            missing=MissingConfig(target_rate=1e-8, always_observe_first=True),
        )
    )
    expected = 1.0 + 0.25 * np.arange(21)
    np.testing.assert_allclose(result.x_true, expected, atol=1e-12)
    assert result.metadata.trend is not None
    np.testing.assert_allclose(result.metadata.trend, 0.25)


def test_ou_reverts_toward_mu() -> None:
    dyn = DynamicsConfig(
        name="ou",
        mu=3.0,
        theta=0.4,
        sigma=0.05,
        x0=10.0,
        jumps=(),
        ramps=(),
    )
    result = simulate(
        SimulationConfig(
            T=4000,
            seed=SEED,
            dynamics=dyn,
            noise=NoiseConfig(sigma=0.0),
            missing=MissingConfig(mechanism="mcar", target_rate=1e-8),
        )
    )
    # After burn-in the path should sit near the OU mean.
    assert abs(float(result.x_true[500:].mean()) - 3.0) < 0.15


def test_jump_creates_known_crossing() -> None:
    dyn = DynamicsConfig(
        name="local_linear_trend",
        x0=0.0,
        trend0=0.0,
        q_level=0.0,
        q_trend=0.0,
        jumps=(Jump(time=50, magnitude=10.0),),
    )
    result = simulate(
        SimulationConfig(
            T=80,
            tau=8.0,
            seed=0,
            dynamics=dyn,
            noise=NoiseConfig(sigma=0.0),
            missing=MissingConfig(target_rate=1e-8),
        )
    )
    np.testing.assert_array_equal(result.true_crossing_times, np.array([50]))
    assert result.x_true[49] <= 8.0
    assert result.x_true[50] > 8.0


def test_ramp_raises_level_by_slope_times_length() -> None:
    dyn = DynamicsConfig(
        name="ou",
        mu=0.0,
        theta=1e-6,  # nearly no reversion
        sigma=0.0,
        x0=0.0,
        ramps=(Ramp(start=10, length=20, slope=0.5),),
    )
    result = simulate(
        SimulationConfig(
            T=40,
            seed=0,
            dynamics=dyn,
            noise=NoiseConfig(sigma=0.0),
            missing=MissingConfig(target_rate=1e-8),
        )
    )
    # 20 steps of +0.5 after a near-unit-root OU.
    assert result.x_true[9] == pytest.approx(0.0, abs=1e-9)
    assert result.x_true[29] == pytest.approx(10.0, abs=1e-4)


def test_upcrossing_helper_matches_definition() -> None:
    x = np.array([0.0, 1.0, 9.0, 9.0, 7.0, 8.5])
    np.testing.assert_array_equal(upcrossing_times(x, 8.0), np.array([2, 5]))
    np.testing.assert_array_equal(upcrossing_times(np.array([9.0, 10.0]), 8.0), np.array([0]))


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def test_noise_tiers_scale_residuals() -> None:
    scales = []
    for tier in ("low", "medium", "high"):
        result = simulate(
            SimulationConfig(
                T=8000,
                seed=SEED,
                dynamics=DynamicsConfig(
                    name="local_linear_trend",
                    q_level=0.0,
                    q_trend=0.0,
                    trend0=0.0,
                    x0=0.0,
                ),
                noise=NoiseConfig(family="gaussian", tier=tier),  # type: ignore[arg-type]
                missing=MissingConfig(target_rate=1e-8, always_observe_first=False),
            )
        )
        scales.append(float(np.std(result.y_full - result.x_true)))
    assert scales[0] < scales[1] < scales[2]
    assert scales[0] == pytest.approx(NOISE_TIER_SIGMA["low"], rel=0.08)
    assert scales[2] == pytest.approx(NOISE_TIER_SIGMA["high"], rel=0.08)


def test_time_varying_gaussian_sigma() -> None:
    T = 200
    sigma = np.linspace(0.1, 2.0, T)
    result = simulate(
        SimulationConfig(
            T=T,
            seed=1,
            dynamics=DynamicsConfig(
                name="local_linear_trend",
                q_level=0.0,
                q_trend=0.0,
                trend0=0.0,
                x0=0.0,
            ),
            noise=NoiseConfig(family="gaussian", sigma=sigma, spike_probability=0.0),
            missing=MissingConfig(target_rate=1e-8),
        )
    )
    # Residual envelope should grow; compare early vs late RMSE.
    e = result.y_full - result.x_true
    early = float(np.sqrt(np.mean(e[:40] ** 2)))
    late = float(np.sqrt(np.mean(e[-40:] ** 2)))
    assert late > 2.0 * early


def test_student_t_has_heavier_tails_than_gaussian() -> None:
    def residuals(family: str, df: float = 3.0) -> np.ndarray:
        result = simulate(
            SimulationConfig(
                T=12000,
                seed=SEED,
                dynamics=DynamicsConfig(
                    name="local_linear_trend",
                    q_level=0.0,
                    q_trend=0.0,
                    trend0=0.0,
                    x0=0.0,
                ),
                noise=NoiseConfig(family=family, sigma=0.5, df=df),  # type: ignore[arg-type]
                missing=MissingConfig(target_rate=1e-8),
            )
        )
        return result.y_full - result.x_true

    g = residuals("gaussian")
    t = residuals("student_t", df=3.0)
    # Excess kurtosis / tail mass: fraction beyond 4 sigma_gauss.
    thresh = 4.0 * np.std(g)
    assert float(np.mean(np.abs(t) > thresh)) > float(np.mean(np.abs(g) > thresh))


def test_outlier_spikes_inflate_max_residual() -> None:
    clean = simulate(
        SimulationConfig(
            T=2000,
            seed=7,
            dynamics=DynamicsConfig(q_level=0.0, q_trend=0.0, trend0=0.0, x0=0.0),
            noise=NoiseConfig(family="gaussian", sigma=0.2, spike_probability=0.0),
            missing=MissingConfig(target_rate=1e-8),
        )
    )
    spiked = simulate(
        SimulationConfig(
            T=2000,
            seed=7,
            dynamics=DynamicsConfig(q_level=0.0, q_trend=0.0, trend0=0.0, x0=0.0),
            noise=NoiseConfig(
                family="gaussian",
                sigma=0.2,
                spike_probability=0.05,
                spike_scale=20.0,
            ),
            missing=MissingConfig(target_rate=1e-8),
        )
    )
    clean_max = float(np.max(np.abs(clean.y_full - clean.x_true)))
    spike_max = float(np.max(np.abs(spiked.y_full - spiked.x_true)))
    assert spike_max > 5.0 * clean_max


# ---------------------------------------------------------------------------
# Missingness rates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", TARGET_MISSING_RATES)
def test_mcar_achieved_missing_rates(rate: float) -> None:
    result = simulate(
        SimulationConfig(
            T=12000,
            seed=SEED,
            missing=MissingConfig(
                mechanism="mcar",
                target_rate=rate,
                always_observe_first=False,
            ),
        )
    )
    achieved = result.metadata.achieved_missing_rate
    assert achieved == pytest.approx(rate, abs=0.02)


@pytest.mark.parametrize("rate", TARGET_MISSING_RATES)
def test_mnar_mean_probability_matches_target(rate: float) -> None:
    result = simulate(
        SimulationConfig(
            T=4000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.08, sigma=0.4, x0=6.0),
            missing=MissingConfig(
                mechanism="mnar",
                target_rate=rate,
                g1=-1.5,
                always_observe_first=False,
            ),
        )
    )
    assert result.metadata.mean_observation_probability is not None
    assert result.metadata.mean_observation_probability == pytest.approx(1.0 - rate, abs=1e-6)
    assert result.metadata.achieved_missing_rate == pytest.approx(rate, abs=0.03)
    assert result.metadata.g0 is not None


@pytest.mark.parametrize("rate", TARGET_MISSING_RATES)
def test_mar_achieved_missing_rates(rate: float) -> None:
    result = simulate(
        SimulationConfig(
            T=8000,
            seed=SEED,
            missing=MissingConfig(
                mechanism="mar",
                target_rate=rate,
                g1=-1.0,
                always_observe_first=True,
            ),
        )
    )
    # First point is forced observed, so allow a slightly wider window.
    assert result.metadata.achieved_missing_rate == pytest.approx(rate, abs=0.025)


@pytest.mark.parametrize("rate", TARGET_MISSING_RATES)
def test_bursty_achieved_missing_rates(rate: float) -> None:
    result = simulate(
        SimulationConfig(
            T=15000,
            seed=SEED,
            missing=MissingConfig(
                mechanism="bursty",
                target_rate=rate,
                mean_gap_length=6.0,
                always_observe_first=False,
            ),
        )
    )
    assert result.metadata.achieved_missing_rate == pytest.approx(rate, abs=0.03)


def test_solve_mnar_g0_is_monotone_in_target_rate() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(6.0, 1.0, size=2000)
    intercepts = [solve_mnar_g0(x, g1=-1.0, mu=6.0, target_missing_rate=r) for r in TARGET_MISSING_RATES]
    # Higher missing rate => smaller observation probability => smaller g0.
    assert intercepts == sorted(intercepts, reverse=True)


# ---------------------------------------------------------------------------
# Burst lengths
# ---------------------------------------------------------------------------


def test_burst_length_distribution_matches_geometric_mean() -> None:
    mean_gap = 8.0
    result = simulate(
        SimulationConfig(
            T=40000,
            seed=SEED,
            missing=MissingConfig(
                mechanism="bursty",
                target_rate=0.30,
                mean_gap_length=mean_gap,
                always_observe_first=False,
            ),
        )
    )
    lengths = missing_run_lengths(result.m)
    assert lengths.size > 50
    assert float(lengths.mean()) == pytest.approx(mean_gap, rel=0.12)
    p_drop, p_recover = bursty_transition_probs(0.30, mean_gap)
    assert p_recover == pytest.approx(1.0 / mean_gap)
    # Empirical P(length >= 2) close to (1 - p_recover) under Geometric(p).
    emp = float(np.mean(lengths >= 2))
    assert emp == pytest.approx(1.0 - p_recover, abs=0.05)


def test_mcar_gaps_are_mostly_short() -> None:
    result = simulate(
        SimulationConfig(
            T=8000,
            seed=3,
            missing=MissingConfig(mechanism="mcar", target_rate=0.12, always_observe_first=False),
        )
    )
    lengths = missing_run_lengths(result.m)
    # MCAR with p=0.12 has geometric gap mean 1/(1-0.12)? Wait:
    # runs of missing: success probability of ending a miss run is 1-p = 0.88,
    # so mean miss-run length is 1 / 0.88 ≈ 1.14.
    assert float(lengths.mean()) == pytest.approx(1.0 / 0.88, rel=0.15)


# ---------------------------------------------------------------------------
# MNAR / MAR dependence
# ---------------------------------------------------------------------------


def test_mnar_negative_g1_correlates_missingness_with_high_x() -> None:
    result = simulate(
        SimulationConfig(
            T=8000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.05, sigma=0.6, x0=6.0),
            missing=MissingConfig(
                mechanism="mnar",
                target_rate=0.40,
                g1=-2.0,
                always_observe_first=False,
            ),
        )
    )
    corr = float(np.corrcoef(result.x_true, result.m.astype(float))[0, 1])
    # g1 < 0 => observe less when x is high => corr(m, x) < 0.
    assert corr < -0.15
    assert result.x_true[result.m == 0].mean() > result.x_true[result.m == 1].mean()


def test_mnar_positive_g1_flips_correlation() -> None:
    result = simulate(
        SimulationConfig(
            T=8000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.05, sigma=0.6, x0=6.0),
            missing=MissingConfig(
                mechanism="mnar",
                target_rate=0.40,
                g1=2.0,
                always_observe_first=False,
            ),
        )
    )
    corr = float(np.corrcoef(result.x_true, result.m.astype(float))[0, 1])
    assert corr > 0.15


def test_mcar_is_uncorrelated_with_latent() -> None:
    result = simulate(
        SimulationConfig(
            T=12000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.05, sigma=0.6),
            missing=MissingConfig(
                mechanism="mcar",
                target_rate=0.40,
                always_observe_first=False,
            ),
        )
    )
    corr = float(np.corrcoef(result.x_true, result.m.astype(float))[0, 1])
    assert abs(corr) < 0.05


def test_mar_depends_on_last_observation_not_current_x() -> None:
    # Strong MAR slope on last y. Current-x correlation should be weaker than
    # MNAR with the same |g1|, because the covariate is lagged and noisy.
    mnar = simulate(
        SimulationConfig(
            T=6000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.08, sigma=0.5),
            noise=NoiseConfig(sigma=0.3),
            missing=MissingConfig(mechanism="mnar", target_rate=0.35, g1=-2.5, always_observe_first=False),
        )
    )
    mar = simulate(
        SimulationConfig(
            T=6000,
            seed=SEED,
            dynamics=DynamicsConfig(name="ou", mu=6.0, theta=0.08, sigma=0.5),
            noise=NoiseConfig(sigma=0.3),
            missing=MissingConfig(mechanism="mar", target_rate=0.35, g1=-2.5, always_observe_first=True),
        )
    )
    corr_mnar = abs(float(np.corrcoef(mnar.x_true, mnar.m.astype(float))[0, 1]))
    corr_mar = abs(float(np.corrcoef(mar.x_true, mar.m.astype(float))[0, 1]))
    assert corr_mnar > corr_mar
    assert corr_mnar > 0.12


# ---------------------------------------------------------------------------
# Demo scenario
# ---------------------------------------------------------------------------


def test_demo_scenario_parameters_and_outage() -> None:
    result = demo_scenario(T=400, seed=0, outage_start=80)
    meta = result.metadata
    assert meta.dynamics == "local_linear_trend"
    assert meta.process_params["q_level"] == 0.02
    assert meta.process_params["q_trend"] == 0.002
    assert meta.noise_family == "gaussian"
    assert meta.noise_params["sigma_mean"] == pytest.approx(0.3)
    assert result.tau == 8.0
    assert meta.H == 15
    assert meta.missing_mechanism == "mcar"
    # Forced 45-step outage at t=80.
    np.testing.assert_array_equal(result.m[80:125], 0)
    assert int(np.sum(result.m[80:125] == 0)) == 45
    # Remaining missingness (outside the outage) close to 12%.
    outside = np.concatenate([result.m[:80], result.m[125:]])
    assert float(1.0 - outside.mean()) == pytest.approx(0.12, abs=0.06)


def test_forced_outage_clipped_to_horizon() -> None:
    result = demo_scenario(T=100, seed=1, outage_start=90)
    assert np.all(result.m[90:] == 0)
    assert result.m[90:].size == 10


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_utilities_return_figures(tmp_path) -> None:
    result = demo_scenario(T=120, seed=2)
    fig1 = plot_simulation(result)
    fig2 = plot_missingness_diagnostics(result, bins=10)
    path1 = tmp_path / "sim.png"
    path2 = tmp_path / "miss.png"
    fig1.savefig(path1)
    fig2.savefig(path2)
    assert path1.stat().st_size > 0
    assert path2.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close("all")
