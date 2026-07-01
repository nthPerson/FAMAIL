"""Tests for the SF multi-stream corpus + 11-dim profiles (Task 3.5)."""
import numpy as np
import pandas as pd

from famail_temporal.second_dataset.data.source_generation.sf_segmentation import SegmentationResult
from famail_temporal.second_dataset.data.source_generation.sf_config import GridSpec
from famail_temporal.second_dataset.data.source_generation.sf_multistream import (
    driver_profile, normalize_profiles, assemble_multistream, N_PROFILE_FEATURES,
    weekday_from_epoch_day, build_calendar_day_map,
)

GRID = GridSpec(lat_min=37.7, lon_min=-122.5, x_grid_max=32, y_grid_max=30, cell_deg=0.01)


def _seg():
    return SegmentationResult(
        seeking=[[[10, 10, 5, 0], [11, 10, 6, 0]]],
        driving=[[[10, 10, 7, 0], [12, 10, 8, 0]], [[5, 5, 9, 1], [6, 5, 10, 1]]],
        pickups=[[10, 10, 7, 0]], dropoffs=[[12, 10, 8, 0]],
        seeking_days=[0], driving_days=[0, 1],
    )


def test_driver_profile_shape_and_trips_per_day():
    df = pd.DataFrame({"driver_id": [0], "lat": [37.795], "lon": [-122.405],
                       "time_utc": [43200]})
    p = driver_profile(df, _seg(), GRID)
    assert p.shape == (N_PROFILE_FEATURES,) and N_PROFILE_FEATURES == 11
    assert np.isfinite(p).all()
    # 2 driving trajectories over 2 distinct days -> 1.0 trips/day (last feature)
    assert p[-1] == 1.0


def test_normalize_profiles_zscores_across_drivers():
    profiles = {0: np.zeros(11), 1: np.full(11, 4.0)}
    norm, mean, std = normalize_profiles(profiles)
    assert np.allclose(mean, 2.0) and np.allclose(std, 2.0)
    assert np.allclose(norm[0], -1.0) and np.allclose(norm[1], 1.0)


def test_weekday_from_epoch_day_known_dates():
    # epoch day 0 (1970-01-01) is a Thursday -> 4 (Mon=1 .. Sun=7)
    assert weekday_from_epoch_day(0) == 4
    # SF's first collection day 2008-05-17 = epoch day 14016 = Saturday -> 6
    assert weekday_from_epoch_day(14016) == 6
    # periodic with period 7; a full week covers all 7 weekdays exactly once
    assert weekday_from_epoch_day(14016 + 7) == 6
    assert {weekday_from_epoch_day(14016 + k) for k in range(7)} == set(range(1, 8))


def test_assemble_multistream_remaps_col3_to_weekday_sidecar_stays_absolute():
    df = pd.DataFrame({"driver_id": [0], "lat": [37.795], "lon": [-122.405],
                       "time_utc": [43200]})
    out = assemble_multistream({0: (df, _seg())}, GRID)
    # Discriminator/editor corpus col-3 is day-of-week (1..7), NOT the absolute 0.
    seek_day_vals = {s[3] for tr in out["ms_seeking"][0] for s in tr}
    assert seek_day_vals <= set(range(1, 8))
    assert weekday_from_epoch_day(0) in seek_day_vals            # 0 -> 4
    plate = out["driver_mapping"]["idx_to_plate"][0]
    assert {s[3] for tr in out["passenger_seeking"][plate] for s in tr} <= set(range(1, 8))
    # Calendar-day sidecars keep the ABSOLUTE day, parallel to the traj list
    # (this is what Ren pair-generation groups on).
    assert out["ms_seeking_days"][0] == [0]
    assert out["ms_driving_days"][0] == [0, 1]


def test_build_calendar_day_map_dates():
    # epoch day 14016 is 2008-05-17 (SF's first collection day)
    m = build_calendar_day_map([[14016, 14016, 14017], [14040]])
    assert m[14016] == "2008-05-17"
    assert m[14040] == "2008-06-10"
    assert set(m.keys()) == {14016, 14017, 14040}   # distinct, sorted internally


def test_assemble_multistream_emits_calendar_day_map():
    df = pd.DataFrame({"driver_id": [0], "lat": [37.795], "lon": [-122.405],
                       "time_utc": [43200]})
    out = assemble_multistream({0: (df, _seg())}, GRID)
    assert "calendar_day_map" in out
    # _seg() uses absolute days {0, 1} -> ISO dates near the epoch
    assert set(out["calendar_day_map"].keys()) == {0, 1}


def test_assemble_multistream_keys_by_driver_idx():
    df = pd.DataFrame({"driver_id": [0], "lat": [37.795], "lon": [-122.405],
                       "time_utc": [43200]})
    out = assemble_multistream({0: (df, _seg()), 1: (df, _seg())}, GRID)
    assert set(out["ms_seeking"].keys()) == {0, 1}
    assert set(out["ms_driving_days"].keys()) == {0, 1}
    assert set(out["profiles_normalized"].keys()) == {0, 1}
    # plate<->idx mapping round-trips
    assert out["driver_mapping"]["plate_to_idx"][out["driver_mapping"]["idx_to_plate"][0]] == 0
    # passenger_seeking_trajs is keyed by plate_id
    assert set(out["passenger_seeking"].keys()) == set(out["driver_mapping"]["idx_to_plate"].values())
