"""Tests for writer.py."""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation.writer import (
    write_all_outputs, write_active_taxis_bundle, write_metadata_json,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)
from famail_temporal.data.source_generation.removal import (
    RemovalSummary, RemovalRecord,
)
from famail_temporal.data.source_generation import config


def test_writer_creates_all_files(tmp_path):
    pickup_dropoff = {(1, 1, 1, 1): (1, 0), (1, 1, 2, 1): (0, 1)}
    active_counts = {(1, 1, 0, 1): 1}
    trajs = TrajectoriesResult(
        seeking_by_plate={"A": [[[1, 1, 1, 1], [1, 1, 1, 1]]]},
        driving_by_plate={"A": [[[1, 1, 2, 1], [1, 1, 2, 1]]]},
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    ms_seeking = {0: trajs.seeking_by_plate["A"]}
    ms_driving = {0: trajs.driving_by_plate["A"]}
    profile = {
        "raw": np.ones((1, 11), dtype=float) * 42.0,     # obviously un-normalized
        "normalized": np.zeros((1, 11), dtype=float),    # z-score output
        "mean": np.ones(11) * 42.0, "std": np.ones(11),
        "feature_names": list(config.PROFILE_FEATURE_NAMES),
    }
    calendars = {
        "seeking": {0: [0]},
        "driving": {0: [0]},
        "calendar_day_map": {0: "2016-07-04"},
    }

    paths = write_all_outputs(
        out_dir=tmp_path,
        pickup_dropoff=pickup_dropoff,
        active_taxis=active_counts,
        passenger_seeking_trajs=trajs.seeking_by_plate,
        ms_seeking=ms_seeking,
        ms_driving=ms_driving,
        ms_profile=profile,
        ms_calendars=calendars,
        driver_mapping=mapping,
        removal_summary=RemovalSummary(),
        metadata_extras={
            "n_days": 3,
            "bounds": {"lat_min": 22.5, "lat_max": 22.9, "lon_min": 113.8, "lon_max": 114.5},
            "git_sha": "abc123",
            "config_snapshot": {},
        },
    )
    for p in [paths.pickup_dropoff, paths.active_taxis, paths.passenger_seeking,
              paths.ms_seeking, paths.ms_driving, paths.ms_profile,
              paths.ms_seeking_days, paths.ms_driving_days,
              paths.calendar_day_map,
              paths.driver_mapping, paths.metadata]:
        assert p.exists(), f"missing file {p}"


def test_profile_bundle_separates_raw_from_normalized(tmp_path):
    """Writer must store RAW features in 'features' and NORMALIZED in
    'features_normalized'. Consumers (discriminator dataset_generation) rely
    on this distinction; storing normalized in both leads to double-normalization
    at consumption time."""
    from famail_temporal.data.source_generation.writer import write_profile_bundle

    raw = np.array([[10.0, 20.0, 30.0]], dtype=float)
    normalized = np.array([[0.0, 0.0, 0.0]], dtype=float)
    mean = np.array([10.0, 20.0, 30.0])
    std = np.array([1.0, 1.0, 1.0])
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}

    path = tmp_path / "ms_profile_features.pkl"
    write_profile_bundle(
        path, raw=raw, normalized=normalized, mean=mean, std=std,
        feature_names=["a", "b", "c"], n_features=3,
        drivers_mapping=mapping,
    )
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    assert np.allclose(bundle["features"][0], [10.0, 20.0, 30.0]), \
        "'features' must hold RAW feature vectors"
    assert np.allclose(bundle["features_normalized"][0], [0.0, 0.0, 0.0]), \
        "'features_normalized' must hold NORMALIZED feature vectors"
    assert not np.allclose(
        bundle["features"][0], bundle["features_normalized"][0]
    ), "'features' and 'features_normalized' must be distinct"


def test_calendar_day_map_is_pickled(tmp_path):
    from famail_temporal.data.source_generation.writer import (
        write_all_outputs,
    )
    trajs = TrajectoriesResult(
        seeking_by_plate={"A": [[[1, 1, 1, 1], [1, 1, 1, 1]]]},
        driving_by_plate={"A": [[[1, 1, 2, 1], [1, 1, 2, 1]]]},
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    calendars = {
        "seeking": {0: [0]},
        "driving": {0: [0]},
        "calendar_day_map": {0: "2016-07-04"},
    }
    paths = write_all_outputs(
        out_dir=tmp_path,
        pickup_dropoff={(1, 1, 1, 1): (1, 0), (1, 1, 2, 1): (0, 1)},
        active_taxis={},
        passenger_seeking_trajs=trajs.seeking_by_plate,
        ms_seeking={0: trajs.seeking_by_plate["A"]},
        ms_driving={0: trajs.driving_by_plate["A"]},
        ms_profile={
            "raw": np.zeros((1, 11)),
            "normalized": np.zeros((1, 11)),
            "mean": np.zeros(11), "std": np.ones(11),
            "feature_names": list(config.PROFILE_FEATURE_NAMES),
        },
        ms_calendars=calendars,
        driver_mapping=mapping,
        removal_summary=RemovalSummary(),
        metadata_extras={"n_days": 1, "bounds": {}, "git_sha": "x", "config_snapshot": {}},
    )
    with open(paths.calendar_day_map, "rb") as f:
        cal_map = pickle.load(f)
    assert cal_map == {0: "2016-07-04"}


def test_active_taxis_bundle_format(tmp_path):
    counts = {(1, 1, 0, 1): 5}
    path = tmp_path / "active_taxis_5x5_hourly.pkl"
    write_active_taxis_bundle(
        path, counts, stats={"n_entries": 1}, config_snapshot={"neighborhood_dims": 5},
    )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    assert bundle["data"] == counts
    assert "stats" in bundle and "config" in bundle and "version" in bundle


def test_metadata_json_records_removals(tmp_path):
    summary = RemovalSummary(
        total_seeking_extracted=100, total_driving_extracted=100,
        removals=[RemovalRecord(
            driver_id="A", driver_idx=0, trajectory_index_within_driver=3,
            kind="seeking", which_invariant=1,
            failing_values={"endpoint": (99, 99, 1, 1)},
            n_states_before_removal=5, removal_reason_category="no_matching_count",
        )],
    )
    extras = {"n_days": 65, "bounds": {"lat_min": 22, "lat_max": 23, "lon_min": 113, "lon_max": 115}}
    path = tmp_path / "processing_metadata.json"
    write_metadata_json(path, summary, extras)
    with open(path) as f:
        m = json.load(f)
    assert m["n_days"] == 65
    assert m["removal_summary"]["total_extracted"] == 200
    assert m["removal_summary"]["removals"][0]["removal_reason_category"] == "no_matching_count"
    assert m["removal_summary"]["counts_by_category"]["no_matching_count"] == 1
