"""Tests for the CLI orchestration (run_generation)."""
from __future__ import annotations
import pickle
from pathlib import Path


from famail_temporal.data.source_generation.cli import run_generation


def _write_pkl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _minimal_raw_fixture(tmp_path):
    raw = tmp_path / "raw"
    for filename in ("taxi_record_07_50drivers.pkl",
                     "taxi_record_08_50drivers.pkl",
                     "taxi_record_09_50drivers.pkl"):
        _write_pkl(raw / filename, {})
    records: dict = {}
    for i in range(50):
        plate = f"PLATE_{i:02d}"
        records[plate] = [
            [plate, 22.5, 113.8, 0,    1, "2016-07-04 00:00:00"],
            [plate, 22.5, 113.8, 60,   0, "2016-07-04 00:01:00"],
            [plate, 22.5, 113.8, 120,  0, "2016-07-04 00:02:00"],
            [plate, 22.5, 113.8, 180,  0, "2016-07-04 00:03:00"],
            [plate, 22.5, 113.8, 240,  1, "2016-07-04 00:04:00"],
            [plate, 22.5, 113.8, 300,  1, "2016-07-04 00:05:00"],
            [plate, 22.5, 113.8, 360,  0, "2016-07-04 00:06:00"],
        ]
    _write_pkl(raw / "taxi_record_07_50drivers.pkl", records)
    return raw


def test_cli_runs_end_to_end(tmp_path):
    raw = _minimal_raw_fixture(tmp_path)
    out = tmp_path / "out"
    result = run_generation(input_dir=raw, output_dir=out)
    expected = [
        "pickup_dropoff_counts.pkl",
        "active_taxis_5x5_hourly.pkl",
        "passenger_seeking_trajs.pkl",
        "ms_seeking_trajs.pkl",
        "ms_driving_trajs.pkl",
        "ms_profile_features.pkl",
        "ms_seeking_calendar_days.pkl",
        "ms_driving_calendar_days.pkl",
        "calendar_day_map.pkl",
        "driver_index_mapping.pkl",
        "processing_metadata.json",
    ]
    for name in expected:
        assert (out / name).exists(), f"missing output: {name}"
    assert result.n_seeking_kept >= 1
    assert result.n_driving_kept >= 1
