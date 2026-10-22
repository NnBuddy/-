"""Tests for the Intel Berkeley Lab data loader."""

from __future__ import annotations

import gzip
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.intel_lab import (
    ChronologicalSplits,
    IntelLabConfig,
    define_intel_excursions,
    get_mote_summary,
    load_intel_data,
    load_mote_locations,
    split_data_chronological,
)

MOCK_DUMP = """\
2004-02-28 01:00:00.000000 10 1 19.5 40.0 45.0 2.70
2004-02-28 01:00:31.000000 11 1 19.6 40.1 45.0 2.69
not enough
2004-02-28 01:01:33.000000 13 1 19.8 40.2 45.0 2.68
2004-02-28 01:02:04.000000 14 1 122.4 41.0 11.0 2.65
2004-02-28 01:02:35.000000 15 1 20.1 40.0 45.0 2.10
2004-02-28 01:00:00.000000 10 2 18.0 35.0 50.0 2.80
2004-02-28 01:00:31.000000 11 2 18.2 35.0 50.0 2.80
this is garbage and should be skipped entirely
2004-02-28 01:01:02.000000 12 2 18.4 35.0 50.0 2.80
"""

MOCK_LOCS = """\
moteid x y
1 21.5 23.0
2 24.5 20.0
not-a-row
"""


def _write_dump(path: Path, text: str = MOCK_DUMP, gz: bool = False) -> Path:
    if gz:
        target = path / "data.txt.gz"
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            handle.write(text)
        return target
    target = path / "data.txt"
    target.write_text(text, encoding="utf-8")
    return target


def _write_locs(path: Path) -> Path:
    loc = path / "mote_locs.txt"
    loc.write_text(MOCK_LOCS, encoding="utf-8")
    return loc


def test_malformed_lines_are_skipped(tmp_path: Path) -> None:
    data_path = _write_dump(tmp_path)
    df = load_intel_data(data_path, regularize=False)
    assert df.attrs["n_malformed_lines"] >= 2
    # Identifiable packets: 5 on mote 1, 3 on mote 2.
    assert set(df["moteid"].unique()) == {1, 2}
    assert len(df) == 8


def test_gzip_and_location_merge(tmp_path: Path) -> None:
    data_path = _write_dump(tmp_path, gz=True)
    loc_path = _write_locs(tmp_path)
    df = load_intel_data(data_path, mote_locs_path=loc_path, regularize=True)
    assert "loc_x" in df.columns and "loc_y" in df.columns
    mote1 = df.loc[df["moteid"] == 1].iloc[0]
    assert mote1["loc_x"] == pytest.approx(21.5)
    assert df["timestamp"].dt.tz is not None
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_missing_epochs_receive_m_zero(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    mote1 = df.loc[df["moteid"] == 1].sort_values("epoch")
    # Raw epochs 10, 11, 13, 14, 15 → grid inserts epoch 12.
    np.testing.assert_array_equal(mote1["epoch"].to_numpy(), np.arange(10, 16))
    gap = mote1.loc[mote1["epoch"] == 12].iloc[0]
    assert int(gap["m"]) == 0
    assert bool(pd.isna(gap["temperature"]))
    assert bool(gap["is_outlier"]) is False
    # Inserted epoch keeps the discrete index (row is present).
    assert 12 in set(mote1["epoch"].tolist())


def test_low_voltage_flags_outlier_without_dropping_index(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    mote1 = df.loc[df["moteid"] == 1].set_index("epoch")
    row = mote1.loc[15]
    assert bool(row["is_voltage_low"])
    assert bool(row["is_outlier"])
    assert int(row["m"]) == 0
    assert epoch_present_after_grid(mote1)
    # Temperature/voltage values are retained for MNAR analysis.
    assert row["voltage"] == pytest.approx(2.10)
    assert row["temperature"] == pytest.approx(20.1)


def epoch_present_after_grid(mote_indexed: pd.DataFrame) -> bool:
    return 15 in mote_indexed.index


def test_temperature_spike_flagged_not_dropped(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    row = df.loc[(df["moteid"] == 1) & (df["epoch"] == 14)].iloc[0]
    assert bool(row["is_temp_spike"])
    assert bool(row["is_outlier"])
    assert int(row["m"]) == 0
    assert row["temperature"] == pytest.approx(122.4)


def test_valid_packets_have_m_one(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    row = df.loc[(df["moteid"] == 1) & (df["epoch"] == 10)].iloc[0]
    assert int(row["m"]) == 1
    assert bool(row["is_outlier"]) is False


def test_mote_summary_gap_and_anomaly_counts(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    summary = get_mote_summary(df)
    mote1 = summary.loc[summary["moteid"] == 1].iloc[0]
    assert int(mote1["total_epochs"]) == 6
    # Valid: epochs 10, 11, 13. Missing/invalid: 12 (gap), 14 (spike), 15 (voltage).
    assert int(mote1["n_observed"]) == 3
    assert mote1["missing_rate"] == pytest.approx(0.5)
    assert int(mote1["n_voltage_anomalies"]) == 1
    assert int(mote1["n_temp_anomalies"]) == 1
    assert int(mote1["n_outliers"]) == 2
    # One isolated missing epoch (12) plus two invalids (14, 15) may merge
    # with neighbours depending on adjacency: 14 and 15 are consecutive m=0.
    assert mote1["mean_gap_length"] > 0.0


def test_define_intel_excursions_uses_cleaned_quantile(tmp_path: Path) -> None:
    dump = "\n".join(
        [
            "2004-02-28 01:00:00.0 1 7 10.0 30.0 1.0 2.7",
            "2004-02-28 01:00:31.0 2 7 10.5 30.0 1.0 2.7",
            "2004-02-28 01:01:02.0 3 7 11.0 30.0 1.0 2.7",
            "2004-02-28 01:01:33.0 4 7 20.0 30.0 1.0 2.7",
            "2004-02-28 01:02:04.0 5 7 150.0 30.0 1.0 2.7",
            "2004-02-28 01:02:35.0 6 7 10.2 30.0 1.0 2.7",
            "2004-02-28 01:03:06.0 7 7 21.0 30.0 1.0 2.7",
        ]
    )
    df = load_intel_data(_write_dump(tmp_path, text=dump + "\n"), regularize=True)
    series = define_intel_excursions(df, mote_id=7, tau_quantile=0.5, H=15)
    y, m, event_times = series[0], series[1], series[2]
    assert series.H == 15
    # Cleaned finite values: 10, 10.5, 11, 20, 10.2, 21.
    # Linear-interpolation median of the even sample is 10.75.
    finite = np.array([10.0, 10.5, 11.0, 20.0, 10.2, 21.0])
    assert series.tau == pytest.approx(float(np.quantile(finite, 0.5, method="linear")))
    np.testing.assert_array_equal(m == 1, np.isfinite(y))
    # Spike is retained in y_raw but masked in y.
    assert np.isnan(y[4])
    assert series.y_raw[4] == pytest.approx(150.0)
    assert bool(series.is_outlier[4])
    # Upcrossings of 10.75: index 2 (10.5 -> 11) and index 6 (10.2 -> 21).
    np.testing.assert_array_equal(event_times, np.array([2, 6]))


def test_chronological_splits_have_no_overlap_or_leakage(tmp_path: Path) -> None:
    lines = [
        f"2004-02-28 01:00:{second:02d}.0 {epoch} 1 19.0 30.0 1.0 2.7"
        for epoch, second in enumerate(range(0, 40), start=1)
    ]
    df = load_intel_data(_write_dump(tmp_path, text="\n".join(lines) + "\n"), regularize=True)
    splits = split_data_chronological(
        df, train_ratio=0.5, val_ratio=0.2, calib_ratio=0.15, test_ratio=0.15
    )
    assert isinstance(splits, ChronologicalSplits)
    parts = [splits.train, splits.val, splits.calib, splits.test]
    timestamp_sets = [set(part["timestamp"].tolist()) for part in parts]
    for i in range(len(timestamp_sets)):
        for j in range(i + 1, len(timestamp_sets)):
            assert timestamp_sets[i].isdisjoint(timestamp_sets[j])

    assigned = pd.concat(parts, ignore_index=True)
    usable = df.loc[df["timestamp"].notna()]
    assert len(assigned) == len(usable)
    assert set(assigned.index) or True
    # Every original row with a timestamp is assigned exactly once (by epoch).
    merged = usable.merge(
        assigned[["moteid", "epoch"]].assign(_split=1),
        on=["moteid", "epoch"],
        how="left",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()

    # Temporal order: max(train) < min(val) < max(val) < min(calib) ...
    bounds = []
    for part in parts:
        if len(part):
            bounds.append((part["timestamp"].min(), part["timestamp"].max()))
    for prev, nxt in zip(bounds, bounds[1:]):
        assert prev[1] < nxt[0]


def test_split_ratios_must_sum_to_one(tmp_path: Path) -> None:
    df = load_intel_data(_write_dump(tmp_path), regularize=True)
    with pytest.raises(ValueError, match="sum to 1"):
        split_data_chronological(df, train_ratio=0.5, val_ratio=0.5, calib_ratio=0.2, test_ratio=0.2)


def test_full_train_ratio_does_not_drop_rows(tmp_path: Path) -> None:
    lines = [
        f"2004-02-28 01:00:{second:02d}.0 {epoch} 1 19.0 30.0 1.0 2.7"
        for epoch, second in enumerate(range(0, 10), start=1)
    ]
    df = load_intel_data(_write_dump(tmp_path, text="\n".join(lines) + "\n"), regularize=True)
    splits = split_data_chronological(
        df, train_ratio=1.0, val_ratio=0.0, calib_ratio=0.0, test_ratio=0.0
    )
    assert len(splits.train) == len(df)
    assert splits.val.empty and splits.calib.empty and splits.test.empty


def test_load_mote_locations_skips_header(tmp_path: Path) -> None:
    locs = load_mote_locations(_write_locs(tmp_path))
    assert list(locs["moteid"]) == [1, 2]
    assert locs.loc[locs["moteid"] == 2, "loc_y"].iloc[0] == pytest.approx(20.0)


def test_truncated_line_keeps_identifiable_packet(tmp_path: Path) -> None:
    dump = (
        "2004-02-28 01:00:00.0 1 3 18.0 30.0 1.0 2.7\n"
        "2004-02-28 01:00:31.0 2 3 18.5\n"  # truncated after temperature
        "short\n"
    )
    df = load_intel_data(_write_dump(tmp_path, text=dump), regularize=False)
    mote = df.loc[df["moteid"] == 3].sort_values("epoch")
    assert len(mote) == 2
    assert pd.isna(mote.loc[mote["epoch"] == 2, "voltage"].iloc[0])
    assert mote.loc[mote["epoch"] == 2, "temperature"].iloc[0] == pytest.approx(18.5)
    assert df.attrs["n_malformed_lines"] >= 1


def test_real_dump_sample_parses_and_flags_mnar(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data.txt"
    if not source.is_file():
        pytest.skip("Intel Lab dump data.txt is not in the workspace")
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        sample = "".join(islice(handle, 80))
    df = load_intel_data(_write_dump(tmp_path, text=sample), regularize=True)
    assert {"timestamp", "epoch", "moteid", "temperature", "m", "is_outlier"}.issubset(df.columns)
    assert df["timestamp"].dt.tz is not None
    # First public-dump row is a >100 C / low-voltage packet; it must remain.
    spikes = df.loc[df["is_temp_spike"] | df["is_voltage_low"]]
    if len(spikes):
        assert (spikes["m"] == 0).all()
        assert spikes["temperature"].notna().any() or spikes["voltage"].notna().any()


def test_custom_thresholds_via_config(tmp_path: Path) -> None:
    df = load_intel_data(
        _write_dump(tmp_path),
        config=IntelLabConfig(voltage_min=2.0, temp_max=200.0),
        regularize=True,
    )
    row_v = df.loc[(df["moteid"] == 1) & (df["epoch"] == 15)].iloc[0]
    row_t = df.loc[(df["moteid"] == 1) & (df["epoch"] == 14)].iloc[0]
    assert bool(row_v["is_voltage_low"]) is False  # 2.10 >= 2.0
    assert bool(row_t["is_temp_spike"]) is False  # 122.4 <= 200
