"""Intel Berkeley Research Lab wireless sensor network loader.

The public dataset (28 Feb – 5 Apr 2004) is space-separated text with fields
``date time epoch moteid temperature humidity light voltage``. Epochs are
issued at a nominal 31-second cadence across ~54 motes.

Assumptions
    Recorded ``date``/``time`` strings are treated as UTC wall-clock (the
    files ship no timezone offset). Outlier rows are **never dropped**: low
    battery (``voltage < 2.3`` V) and extreme temperature spikes
    (``temperature > 100`` °C) are flagged with ``is_outlier`` and treated as
    invalid for the observation mask ``m`` (``m = 1`` means observed *and*
    valid). Missing epochs on the per-mote integer-epoch grid are inserted
    with ``m = 0`` so MNAR-like dropouts remain aligned with the original
    discrete-time index.

Ground-truth crossings (see :func:`define_intel_excursions`) are computed on
the cleaned valid series, not on outlier spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.simulate import missing_run_lengths

RAW_COLUMNS: tuple[str, ...] = (
    "date",
    "time",
    "epoch",
    "moteid",
    "temperature",
    "humidity",
    "light",
    "voltage",
)

EPOCH_SECONDS: float = 31.0
VOLTAGE_LOW_V: float = 2.3
TEMP_SPIKE_C: float = 100.0

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class IntelLabConfig:
    """Thresholds and cadence for Intel Lab preprocessing.

    Attributes:
        voltage_min: Readings with ``voltage < voltage_min`` are voltage
            anomalies (MNAR-like battery dropout).
        temp_max: Readings with ``temperature > temp_max`` are temperature
            spikes (sensor failure), retained but flagged.
        epoch_seconds: Nominal spacing used to stamp missing epochs.
        min_fields: Lines with fewer tokens than this are skipped as
            malformed (need at least date, time, epoch, moteid).
    """

    voltage_min: float = VOLTAGE_LOW_V
    temp_max: float = TEMP_SPIKE_C
    epoch_seconds: float = EPOCH_SECONDS
    min_fields: int = 4


class IntelExcursionSeries(NamedTuple):
    """Per-mote regularized series used for excursion labelling.

    Tuple order is ``(y, m, event_times, ...)`` so callers can unpack the
    three arrays requested by the project spec.

    Attributes:
        y: Valid observed temperature; ``NaN`` where ``m == 0``.
        m: Observation mask (1 = observed and valid).
        event_times: Grid indices where the *cleaned* series upcrosses
            ``tau`` (excursion starts).
        tau: Threshold (high quantile of the cleaned baseline).
        H: Forecast horizon carried for downstream detectors.
        mote_id: Mote identifier.
        timestamps: UTC timestamps on the regular epoch grid.
        is_outlier: Outlier flag aligned with the grid (not dropped).
        y_raw: Temperature including outlier spikes (``NaN`` only for
            epochs with no packet at all).
    """

    y: FloatArray
    m: IntArray
    event_times: IntArray
    tau: float
    H: int
    mote_id: int
    timestamps: NDArray[np.datetime64]
    is_outlier: BoolArray
    y_raw: FloatArray


class ChronologicalSplits(NamedTuple):
    """Strict temporal partitions with half-open unique-timestamp cuts."""

    train: pd.DataFrame
    val: pd.DataFrame
    calib: pd.DataFrame
    test: pd.DataFrame
    cutpoints: dict[str, Any]


def load_mote_locations(mote_locs_path: str | Path) -> pd.DataFrame:
    """Load mote coordinates (``moteid loc_x loc_y``), skipping bad lines.

    Args:
        mote_locs_path: Path to the space-separated locations file.

    Returns:
        Data frame with columns ``moteid``, ``loc_x``, ``loc_y``.
    """
    path = Path(mote_locs_path)
    rows: list[tuple[int, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                mote_id = int(float(parts[0]))
                loc_x = float(parts[1])
                loc_y = float(parts[2])
            except ValueError:
                continue
            rows.append((mote_id, loc_x, loc_y))
    return pd.DataFrame(rows, columns=["moteid", "loc_x", "loc_y"]).drop_duplicates(
        subset=["moteid"], keep="last"
    )


def _read_raw_lines(data_path: str | Path, min_fields: int) -> tuple[pd.DataFrame, int]:
    """Parse the space-separated dump, skipping truncated/malformed lines."""
    path = Path(data_path)
    n_malformed = 0
    records: list[list[str]] = []

    if path.suffix == ".gz" or path.name.endswith(".txt.gz"):
        import gzip

        handle = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        handle = path.open("r", encoding="utf-8", errors="replace")

    with handle:
        for line in handle:
            parts = line.split()
            if len(parts) < min_fields:
                if line.strip():
                    n_malformed += 1
                continue
            # Pad short-but-identifiable rows; ignore extra tokens.
            padded = (parts + [""] * len(RAW_COLUMNS))[: len(RAW_COLUMNS)]
            records.append(padded)

    df = pd.DataFrame(records, columns=list(RAW_COLUMNS))
    return df, n_malformed


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def _flag_outliers(
    df: pd.DataFrame, config: IntelLabConfig
) -> pd.DataFrame:
    """Add voltage/temperature anomaly flags. Does not drop rows."""
    out = df.copy()
    voltage = out["voltage"]
    temperature = out["temperature"]
    out["is_voltage_low"] = voltage.notna() & (voltage < config.voltage_min)
    out["is_temp_spike"] = temperature.notna() & (temperature > config.temp_max)
    out["is_outlier"] = out["is_voltage_low"] | out["is_temp_spike"]
    return out


def _collapse_duplicate_epochs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(moteid, epoch)``, keeping the last packet in time.

    Outlier packets are retained when they are the selected (last) reading.
    If any packet at the epoch was a voltage or temperature anomaly, the
    corresponding any-flags are stored in ``*_any`` columns; the primary
    ``is_outlier`` flag follows the selected row so the MNAR value in
    ``temperature`` is the one that would have been observed.
    """
    if df.empty:
        return df
    ordered = df.sort_values(["moteid", "epoch", "timestamp"], kind="mergesort")
    any_flags = (
        ordered.groupby(["moteid", "epoch"], sort=False)
        .agg(
            is_voltage_low_any=("is_voltage_low", "max"),
            is_temp_spike_any=("is_temp_spike", "max"),
            n_raw_readings=("epoch", "size"),
        )
        .reset_index()
    )
    last = ordered.drop_duplicates(subset=["moteid", "epoch"], keep="last")
    return last.merge(any_flags, on=["moteid", "epoch"], how="left")


def _fill_missing_timestamps(
    frame: pd.DataFrame, epoch_seconds: float
) -> pd.Series:
    """Stamp missing epochs from a reference ``(epoch, timestamp)`` pair."""
    ts = frame["timestamp"].copy()
    observed = frame["timestamp"].notna()
    if not bool(observed.any()):
        return ts
    ref = frame.loc[observed].iloc[0]
    offset = (frame["epoch"] - int(ref["epoch"])) * epoch_seconds
    missing = ~observed
    ts.loc[missing] = ref["timestamp"] + pd.to_timedelta(offset.loc[missing], unit="s")
    return ts


def _regularize_mote(
    mote_df: pd.DataFrame, config: IntelLabConfig
) -> pd.DataFrame:
    """Reindex one mote onto a contiguous integer-epoch grid."""
    if mote_df.empty:
        return mote_df
    mote_id = int(mote_df["moteid"].iloc[0])
    e_min = int(mote_df["epoch"].min())
    e_max = int(mote_df["epoch"].max())
    grid = pd.DataFrame({"epoch": np.arange(e_min, e_max + 1, dtype=np.int64)})
    out = grid.merge(mote_df, on="epoch", how="left")
    out["moteid"] = mote_id
    for col in ("is_voltage_low", "is_temp_spike", "is_outlier"):
        out[col] = out[col].fillna(False).astype(bool)
    for col in ("is_voltage_low_any", "is_temp_spike_any"):
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
    if "n_raw_readings" in out.columns:
        out["n_raw_readings"] = out["n_raw_readings"].fillna(0).astype(np.int32)
    else:
        out["n_raw_readings"] = (out["temperature"].notna() | out["voltage"].notna()).astype(
            np.int32
        )
    out["timestamp"] = _fill_missing_timestamps(out, config.epoch_seconds)
    valid_temp = out["temperature"].notna() & ~out["is_outlier"]
    out["m"] = valid_temp.astype(np.int8)
    return out


def load_intel_data(
    data_path: str | Path,
    mote_locs_path: str | Path | None = None,
    config: IntelLabConfig | None = None,
    regularize: bool = True,
) -> pd.DataFrame:
    """Load and preprocess the Intel Berkeley Lab dump.

    Parses space-separated text (plain or ``.gz``), builds a UTC timestamp,
    flags voltage/temperature outliers without dropping them, and optionally
    regularizes each mote onto a contiguous epoch grid with ``m = 0`` on
    missing epochs.

    Args:
        data_path: Path to ``data.txt`` or ``data.txt.gz``.
        mote_locs_path: Optional mote coordinate file.
        config: Thresholds and cadence. Defaults to the public-dataset rules.
        regularize: If True (default), insert missing epochs per mote.

    Returns:
        Preprocessed frame with at least ``timestamp``, ``epoch``, ``moteid``,
        ``temperature``, ``m``, and ``is_outlier``. Provenance is stored in
        ``DataFrame.attrs``.
    """
    cfg = config if config is not None else IntelLabConfig()
    raw, n_malformed = _read_raw_lines(data_path, min_fields=cfg.min_fields)

    raw["epoch"] = _to_numeric(raw["epoch"])
    raw["moteid"] = _to_numeric(raw["moteid"])
    for col in ("temperature", "humidity", "light", "voltage"):
        raw[col] = _to_numeric(raw[col])

    identifiable = raw["epoch"].notna() & raw["moteid"].notna()
    n_unidentifiable = int((~identifiable).sum())
    raw = raw.loc[identifiable].copy()
    raw["epoch"] = raw["epoch"].round().astype(np.int64)
    raw["moteid"] = raw["moteid"].round().astype(np.int64)

    stamp = raw["date"].astype(str) + " " + raw["time"].astype(str)
    raw["timestamp"] = pd.to_datetime(stamp, utc=True, errors="coerce")

    flagged = _flag_outliers(raw, cfg)
    collapsed = _collapse_duplicate_epochs(flagged)

    if regularize and not collapsed.empty:
        pieces = [
            _regularize_mote(group, cfg) for _, group in collapsed.groupby("moteid", sort=True)
        ]
        out = pd.concat(pieces, ignore_index=True)
    else:
        out = collapsed.copy()
        if not out.empty:
            valid_temp = out["temperature"].notna() & ~out["is_outlier"]
            out["m"] = valid_temp.astype(np.int8)

    if mote_locs_path is not None:
        locs = load_mote_locations(mote_locs_path)
        out = out.merge(locs, on="moteid", how="left")

    out = out.sort_values(["moteid", "epoch"], kind="mergesort").reset_index(drop=True)
    out.attrs["n_malformed_lines"] = n_malformed + n_unidentifiable
    out.attrs["config"] = {
        "voltage_min": cfg.voltage_min,
        "temp_max": cfg.temp_max,
        "epoch_seconds": cfg.epoch_seconds,
        "regularize": regularize,
    }
    out.attrs["source"] = str(data_path)
    return out


def get_mote_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-mote coverage, missingness, and anomaly counts.

    Args:
        df: Output of :func:`load_intel_data` (regularized grid).

    Returns:
        One row per mote with total epochs, missing rate, mean/median gap
        length, and voltage/temperature anomaly counts.
    """
    required = {"moteid", "epoch", "m"}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Data frame missing columns {sorted(missing_cols)}.")

    rows: list[dict[str, Any]] = []
    for mote_id, group in df.groupby("moteid", sort=True):
        ordered = group.sort_values("epoch")
        m = ordered["m"].to_numpy()
        gaps = missing_run_lengths(m)
        n_voltage = (
            int(ordered["is_voltage_low"].sum()) if "is_voltage_low" in ordered.columns else 0
        )
        n_temp = int(ordered["is_temp_spike"].sum()) if "is_temp_spike" in ordered.columns else 0
        n_outlier = int(ordered["is_outlier"].sum()) if "is_outlier" in ordered.columns else 0
        ts = ordered["timestamp"] if "timestamp" in ordered.columns else None
        rows.append(
            {
                "moteid": int(mote_id),
                "total_epochs": int(len(ordered)),
                "n_observed": int(m.sum()),
                "missing_rate": float(1.0 - m.mean()) if m.size else np.nan,
                "mean_gap_length": float(gaps.mean()) if gaps.size else 0.0,
                "median_gap_length": float(np.median(gaps)) if gaps.size else 0.0,
                "n_gaps": int(gaps.size),
                "n_voltage_anomalies": n_voltage,
                "n_temp_anomalies": n_temp,
                "n_outliers": n_outlier,
                "first_timestamp": ts.min() if ts is not None and len(ts) else pd.NaT,
                "last_timestamp": ts.max() if ts is not None and len(ts) else pd.NaT,
            }
        )
    return pd.DataFrame(rows)


def _excursion_starts(cleaned: FloatArray, tau: float) -> IntArray:
    """Excursion starts on a cleaned series that may contain ``NaN`` gaps.

    A grid index ``t`` is an excursion start when ``cleaned[t]`` is finite
    and strictly greater than ``tau``, and the previous *finite* cleaned
    value (if any) was ``<= tau``. The first finite value counts if it is
    already above ``tau``. Outlier spikes are not part of ``cleaned`` and
    therefore cannot create ground-truth events.
    """
    values = np.asarray(cleaned, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.asarray([], dtype=int)

    starts: list[int] = []
    prev_finite: float | None = None
    for t, flag in enumerate(finite):
        if not flag:
            continue
        current = float(values[t])
        if current > tau and (prev_finite is None or prev_finite <= tau):
            starts.append(t)
        prev_finite = current
    return np.asarray(starts, dtype=int)


def define_intel_excursions(
    df: pd.DataFrame,
    mote_id: int,
    tau_quantile: float = 0.95,
    H: int = 15,
) -> IntelExcursionSeries:
    """Build ``(y, m, event_times)`` for one mote with a data-driven ``tau``.

    ``tau`` is the ``tau_quantile`` quantile of the *cleaned baseline*
    (valid temperatures with ``m = 1``). Ground-truth crossings are
    upcrossings of that threshold on the same cleaned sequence, skipping
    ``NaN`` gaps: a new event begins when the series returns from
    ``<= tau`` to ``> tau`` at a valid index. Horizon ``H`` is stored for
    sequential detectors and is not used when labelling events.

    Args:
        df: Regularized Intel Lab frame.
        mote_id: Target mote.
        tau_quantile: Quantile in ``(0, 1)`` of the cleaned baseline.
        H: Forecast horizon in epochs.

    Returns:
        :class:`IntelExcursionSeries` (a named tuple).
    """
    if not 0.0 < tau_quantile < 1.0:
        raise ValueError(f"tau_quantile must lie in (0, 1), got {tau_quantile}.")
    if H < 1:
        raise ValueError("H must be >= 1.")

    mote = df.loc[df["moteid"] == mote_id].sort_values("epoch")
    if mote.empty:
        raise KeyError(f"mote_id {mote_id} is not present in the frame.")

    y_raw = mote["temperature"].to_numpy(dtype=float)
    m = mote["m"].to_numpy(dtype=np.int8)
    is_outlier = (
        mote["is_outlier"].to_numpy(dtype=bool)
        if "is_outlier" in mote.columns
        else np.zeros(len(mote), dtype=bool)
    )
    y = np.where(m == 1, y_raw, np.nan)
    cleaned = y.copy()
    finite = np.isfinite(cleaned)
    if not np.any(finite):
        raise ValueError(f"mote {mote_id} has no valid temperatures to set tau.")
    tau = float(np.nanquantile(cleaned, tau_quantile, method="linear"))
    event_times = _excursion_starts(cleaned, tau)
    timestamps = mote["timestamp"].to_numpy() if "timestamp" in mote.columns else np.array([])
    return IntelExcursionSeries(
        y=y,
        m=m,
        event_times=event_times,
        tau=tau,
        H=int(H),
        mote_id=int(mote_id),
        timestamps=timestamps,
        is_outlier=is_outlier,
        y_raw=y_raw,
    )


def split_data_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.5,
    val_ratio: float = 0.2,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = "timestamp",
) -> ChronologicalSplits:
    """Split by unique UTC timestamps with no temporal leakage.

    Cut points are taken on the sorted unique values of ``time_col`` so that
    every timestamp belongs to exactly one split. Ratios must sum to 1.
    Rows with null timestamps are excluded (they cannot be ordered).

    Args:
        df: Regularized (or raw) Intel Lab frame.
        train_ratio: Fraction of unique timestamps for training.
        val_ratio: Fraction for validation.
        calib_ratio: Fraction for conformal/e-process calibration.
        test_ratio: Fraction for testing.
        time_col: Timestamp column name.

    Returns:
        :class:`ChronologicalSplits` with disjoint frames and the unique
        timestamp cut boundaries.
    """
    ratios = np.asarray([train_ratio, val_ratio, calib_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0.0):
        raise ValueError("Split ratios must be non-negative.")
    if not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError(f"Split ratios must sum to 1, got {float(ratios.sum())}.")
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' is required for chronological splits.")

    ordered = df.loc[df[time_col].notna()].sort_values(time_col, kind="mergesort")
    unique_times = np.sort(pd.unique(ordered[time_col]))
    n = int(unique_times.size)
    empty = df.iloc[0:0].copy()
    if n == 0:
        return ChronologicalSplits(empty, empty, empty, empty, {"unique_timestamps": 0})

    edges = [
        0,
        int(np.floor(n * train_ratio)),
        int(np.floor(n * (train_ratio + val_ratio))),
        int(np.floor(n * (train_ratio + val_ratio + calib_ratio))),
        n,
    ]
    for i in range(1, len(edges)):
        edges[i] = int(min(max(edges[i], edges[i - 1]), n))
    edges[-1] = n

    ts = ordered[time_col]

    def _slice(i: int, j: int) -> pd.DataFrame:
        if i >= j:
            return ordered.iloc[0:0]
        t_start = unique_times[i]
        if j >= n:
            return ordered.loc[ts >= t_start]
        t_end = unique_times[j]
        return ordered.loc[(ts >= t_start) & (ts < t_end)]

    train = _slice(edges[0], edges[1])
    val = _slice(edges[1], edges[2])
    calib = _slice(edges[2], edges[3])
    test = _slice(edges[3], edges[4])

    def _edge_time(index: int):
        if 0 <= index < n:
            return unique_times[index]
        return None

    cutpoints: dict[str, Any] = {
        "unique_timestamps": n,
        "train_end": _edge_time(edges[1]) if edges[1] < n else None,
        "val_end": _edge_time(edges[2]) if edges[2] < n else None,
        "calib_end": _edge_time(edges[3]) if edges[3] < n else None,
        "t_min": unique_times[0],
        "t_max": unique_times[-1],
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "calib": calib_ratio,
            "test": test_ratio,
        },
        "index_edges": edges,
    }
    return ChronologicalSplits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        calib=calib.reset_index(drop=True),
        test=test.reset_index(drop=True),
        cutpoints=cutpoints,
    )
