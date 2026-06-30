"""Tests for SF grid count artifacts (Task 3.4)."""
import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.sf_grid_counts import (
    count_pickup_dropoff, count_active_taxis_5x5, build_valid_mask,
)
from famail_temporal.data.source_generation.sf_config import GridSpec

GRID = GridSpec(lat_min=37.7, lon_min=-122.5, x_grid_max=32, y_grid_max=30, cell_deg=0.01)


def test_count_pickup_dropoff_aggregates_by_cell_bucket_day():
    pickups = [[1, 2, 3, 0], [1, 2, 3, 0], [5, 5, 10, 1]]
    dropoffs = [[1, 2, 3, 0]]
    out = count_pickup_dropoff(pickups, dropoffs)
    assert out[(1, 2, 3, 0)] == (2, 1)
    assert out[(5, 5, 10, 1)] == (1, 0)


def test_active_taxis_5x5_counts_distinct_drivers_in_neighborhood():
    # driver 0 @ cell (10,10); driver 1 @ (12,11) [inside the 5x5 of (10,10)]
    # and driver 1 also @ far cell (20,20); all hour 5, day 0 (local).
    df = pd.DataFrame(
        [(0, 37.795, -122.405, 43200),
         (1, 37.815, -122.395, 43200),
         (1, 37.895, -122.305, 43200)],
        columns=["driver_id", "lat", "lon", "time_utc"],
    )
    out = count_active_taxis_5x5(df, GRID)
    assert out[(10, 10, 5, 0)] == 2     # both drivers within the 5x5 window
    assert out[(20, 20, 5, 0)] == 1     # only driver 1 near the far cell


def test_build_valid_mask_shape():
    m = build_valid_mask(GRID)
    assert m.shape == (32, 30) and m.dtype == bool and m.all()
