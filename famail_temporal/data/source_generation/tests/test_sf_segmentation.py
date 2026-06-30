"""Tests for SF occupancy/gap segmentation + gridding (Task 3.2)."""
import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.sf_config import GridSpec
from famail_temporal.data.source_generation.sf_segmentation import segment_driver


GRID = GridSpec(lat_min=37.7, lon_min=-122.5, x_grid_max=32, y_grid_max=30, cell_deg=0.01)


def test_gridspec_to_cell_is_1indexed_and_clipped():
    # mid-cell point avoids boundary float ambiguity
    assert GRID.to_cell(37.785, -122.405) == (9, 10)
    # out-of-extent clamps into [1, x_grid_max] / [1, y_grid_max]
    x, y = GRID.to_cell(40.0, -120.0)
    assert x == 32 and y == 30
    assert GRID.to_cell(37.701, -122.499) == (1, 1)


def _driver_df(occ, times, lat=37.785, lon=-122.405):
    return pd.DataFrame({
        "driver_id": 0,
        "lat": np.full(len(occ), lat),
        "lon": np.full(len(occ), lon),
        "occupancy": np.array(occ, dtype=np.int8),
        "time_utc": np.array(times, dtype=np.int64),
    })


def test_segment_splits_streams_and_counts_transitions():
    t0 = 1213084700
    df = _driver_df([0, 0, 1, 1, 0], [t0, t0 + 60, t0 + 120, t0 + 180, t0 + 240])

    res = segment_driver(df, GRID, gap_sec=300)

    # seeking run [0,1] is a trajectory; trailing single seeking point is dropped
    assert len(res.seeking) == 1 and len(res.seeking[0]) == 2
    assert len(res.driving) == 1 and len(res.driving[0]) == 2
    assert len(res.pickups) == 1            # 0->1 transition
    assert len(res.dropoffs) == 1           # 1->0 transition
    # state schema: [x, y, time_bucket, day], 1-indexed cell, 5-min bucket in 1..288
    x, y, tb, day = res.seeking[0][0]
    assert 1 <= x <= GRID.x_grid_max and 1 <= y <= GRID.y_grid_max
    assert 1 <= tb <= 288
    assert isinstance(x, int) and isinstance(tb, int)


def test_segment_splits_on_time_gap():
    t0 = 1213084700
    # all seeking, but a >gap jump between index 2 and 3 splits into two trajectories
    df = _driver_df([0, 0, 0, 0, 0], [t0, t0 + 60, t0 + 120, t0 + 120 + 9999, t0 + 120 + 9999 + 60])

    res = segment_driver(df, GRID, gap_sec=300)

    assert len(res.seeking) == 2
    assert [len(s) for s in res.seeking] == [3, 2]
    assert len(res.driving) == 0
