"""Tests for SHA-256 fingerprint in write_all_outputs (E30)."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation import writer as W
from famail_temporal.data.source_generation.views.trajectories import TrajectoriesResult
from famail_temporal.data.source_generation.removal import RemovalSummary
from famail_temporal.data.source_generation import config


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _minimal_payloads() -> dict:
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
        "raw": np.ones((1, 11), dtype=float) * 42.0,
        "normalized": np.zeros((1, 11), dtype=float),
        "mean": np.ones(11) * 42.0, "std": np.ones(11),
        "feature_names": list(config.PROFILE_FEATURE_NAMES),
    }
    calendars = {
        "seeking": {0: [0]},
        "driving": {0: [0]},
        "calendar_day_map": {0: "2016-07-04"},
    }
    return dict(
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


def test_metadata_records_data_sha256(tmp_path):
    paths = W.write_all_outputs(out_dir=tmp_path, **_minimal_payloads())
    meta = json.loads((tmp_path / "processing_metadata.json").read_text())
    assert "data_sha256" in meta
    # every recorded sha matches the file on disk
    for name, sha in meta["data_sha256"].items():
        assert _sha(tmp_path / name) == sha
    assert "pickup_dropoff_counts.pkl" in meta["data_sha256"]
