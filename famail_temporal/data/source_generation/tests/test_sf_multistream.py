"""Tests for the SF multi-stream corpus + 11-dim profiles (Task 3.5)."""
import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.sf_segmentation import SegmentationResult
from famail_temporal.data.source_generation.sf_config import GridSpec
from famail_temporal.data.source_generation.sf_multistream import (
    driver_profile, normalize_profiles, assemble_multistream, N_PROFILE_FEATURES,
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
