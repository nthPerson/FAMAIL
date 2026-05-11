"""End-to-end schema-contract tests.

These pin the on-disk contract consumed by:
  - ``famail_temporal/data/loader.py`` (training-time)
  - ``discriminator/multi_stream/dataset_generation/generation.py``
    (retraining-time)

Both consumers rely on invariants that would silently corrupt training data
if broken: per-driver parallel-list alignment between trajectories and
calendar-day indices, every index resolving to a real date in the map, and
``features`` vs ``features_normalized`` holding distinct (raw vs z-scored)
vectors.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from famail_temporal.data.source_generation.cli import run_generation
from famail_temporal.data.source_generation.tests.test_cli import (
    _write_pkl, _minimal_raw_fixture,
)


def _load_pkl(path: Path):
    import pickle as _pkl
    with open(path, "rb") as f:
        return _pkl.load(f)  # trusted pipeline output


def test_calendar_days_are_parallel_to_trajectories(tmp_path):
    """For every driver, len(seeking_calendar_days[d]) == len(seeking_trajs[d])
    and likewise for driving. Consumer pair-sampling uses zip(trajs, days) and
    silently mis-aligns if this breaks."""
    raw = _minimal_raw_fixture(tmp_path)
    out = tmp_path / "out"
    run_generation(raw, out)

    seeking = _load_pkl(out / "ms_seeking_trajs.pkl")
    driving = _load_pkl(out / "ms_driving_trajs.pkl")
    seek_days = _load_pkl(out / "ms_seeking_calendar_days.pkl")
    drive_days = _load_pkl(out / "ms_driving_calendar_days.pkl")

    for idx, trajs in seeking.items():
        assert len(seek_days[idx]) == len(trajs), (
            f"driver {idx}: seeking parallel-list mismatch"
        )
    for idx, trajs in driving.items():
        assert len(drive_days[idx]) == len(trajs), (
            f"driver {idx}: driving parallel-list mismatch"
        )


def test_every_calendar_day_idx_resolves_to_a_date(tmp_path):
    """Every index appearing in ms_*_calendar_days must exist as a key in
    calendar_day_map."""
    raw = _minimal_raw_fixture(tmp_path)
    out = tmp_path / "out"
    run_generation(raw, out)

    cal_map = _load_pkl(out / "calendar_day_map.pkl")
    for filename in ("ms_seeking_calendar_days.pkl",
                     "ms_driving_calendar_days.pkl"):
        per_driver = _load_pkl(out / filename)
        for idx, day_list in per_driver.items():
            for day_idx in day_list:
                assert day_idx in cal_map, (
                    f"{filename}: driver {idx} has cal_day_idx {day_idx} "
                    f"not in calendar_day_map"
                )


def test_profile_features_raw_differs_from_normalized(tmp_path):
    """'features' must be RAW; 'features_normalized' must be z-scored; the
    two must not be identical (else discriminator training double-normalizes)."""
    raw = _minimal_raw_fixture(tmp_path)
    out = tmp_path / "out"
    run_generation(raw, out)

    bundle = _load_pkl(out / "ms_profile_features.pkl")
    assert "features" in bundle and "features_normalized" in bundle
    assert set(bundle["features"].keys()) == set(
        bundle["features_normalized"].keys()
    )

    raw_stack = np.stack([bundle["features"][k]
                          for k in sorted(bundle["features"].keys())])
    norm_stack = np.stack([bundle["features_normalized"][k]
                           for k in sorted(bundle["features_normalized"].keys())])

    col_mean = norm_stack.mean(axis=0)
    col_std = norm_stack.std(axis=0)
    varying = col_std > 0.5
    assert np.allclose(col_mean, 0.0, atol=1e-5), (
        f"normalized column means not ~0: {col_mean}"
    )
    if varying.any():
        assert np.allclose(col_std[varying], 1.0, atol=1e-5), (
            f"normalized varying-column stds not ~1: {col_std}"
        )

    assert not (raw_stack.shape == norm_stack.shape
                and np.array_equal(raw_stack, norm_stack)), (
        "'features' and 'features_normalized' must not be identical"
    )
