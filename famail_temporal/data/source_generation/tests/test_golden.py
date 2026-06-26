"""End-to-end golden test on a hand-built fixture + slow real-data smoke test."""
from __future__ import annotations
import pickle
from pathlib import Path

import pytest

from famail_temporal.data.source_generation.cli import run_generation
from famail_temporal.data.source_generation.tests.golden_fixtures import (
    build_raw_fixture, expected_seeking_trajectories,
    expected_pickup_count_at_AAA_endpoint,
)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def test_golden_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    build_raw_fixture(raw)

    result = run_generation(raw, out, expect_n_drivers=2, apply_sink_filter=False)

    seeking = _load_pkl(out / "passenger_seeking_trajs.pkl")
    expected = expected_seeking_trajectories()
    assert set(seeking.keys()) == set(expected.keys())
    for plate, trajs in expected.items():
        assert seeking[plate] == trajs

    pd_counts = _load_pkl(out / "pickup_dropoff_counts.pkl")
    expected_pick = expected_pickup_count_at_AAA_endpoint()
    for key, value in expected_pick.items():
        assert key in pd_counts
        assert pd_counts[key][0] == value[0]

    # Invariant check: every seeking trajectory's endpoint has count >= 1.
    for plate, trajs in seeking.items():
        for traj in trajs:
            key = tuple(traj[-1])
            p, _ = pd_counts.get(key, (0, 0))
            assert p >= 1, f"endpoint {key} missing from pickup_counts"

    # Systemic invariant #5: total pickups == total seeking trajectories.
    total_pickups = sum(v[0] for v in pd_counts.values())
    total_seeking = sum(len(v) for v in seeking.values())
    assert total_pickups == total_seeking


@pytest.mark.slow
def test_smoke_on_real_raw_if_present(tmp_path):
    """Run on real raw GPS data if present under raw_data/; skip otherwise."""
    real_raw = Path("raw_data")
    required = [
        "taxi_record_07_50drivers.pkl",
        "taxi_record_08_50drivers.pkl",
        "taxi_record_09_50drivers.pkl",
    ]
    for name in required:
        if not (real_raw / name).exists():
            pytest.skip(f"Missing real raw file: {real_raw / name}")

    out = tmp_path / "smoke_out"
    # apply_sink_filter=True (production default): the STUCK_GPS_* constants are
    # now calibrated from the Stage-0 dry-run, so this exercises the real-data
    # hybrid guard (the run aborts if the flagged set drifts from EXPECTED_CELLS).
    result = run_generation(real_raw, out, expect_n_drivers=50, apply_sink_filter=True)
    assert result.n_seeking_kept >= 100
    assert result.n_driving_kept >= 100
    # the production guard flagged exactly the calibrated sink cells
    import json
    from famail_temporal.data.source_generation import config as _cfg
    meta = json.loads((out / "processing_metadata.json").read_text())
    flagged = {tuple(c) for c in meta["stuck_gps_sinks"]["flagged_cells"]}
    assert flagged == set(_cfg.STUCK_GPS_EXPECTED_CELLS)
    assert meta["stuck_gps_sinks"]["n_pickups_removed"] > 0
    for name in [
        "pickup_dropoff_counts.pkl", "active_taxis_5x5_hourly.pkl",
        "passenger_seeking_trajs.pkl", "ms_seeking_trajs.pkl",
        "ms_driving_trajs.pkl", "ms_profile_features.pkl",
        "ms_seeking_calendar_days.pkl", "ms_driving_calendar_days.pkl",
        "calendar_day_map.pkl",
        "driver_index_mapping.pkl", "processing_metadata.json",
    ]:
        assert (out / name).exists()
