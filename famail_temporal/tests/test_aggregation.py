"""Tests for data.aggregation."""
import pytest

from famail_temporal.data.aggregation import hour_to_block_index


@pytest.mark.parametrize("hour,expected", [
    (7, 0), (9, 0), (10, 1), (15, 1),
    (16, 2), (19, 2), (20, 3), (23, 3),
    (0, 3), (6, 3),
])
def test_hour_to_block_index(hour, expected):
    assert hour_to_block_index(hour) == expected


def test_invalid_hour_raises():
    with pytest.raises(ValueError):
        hour_to_block_index(24)


import numpy as np
from famail_temporal.data.aggregation import (
    aggregate_pickup_dropoff,
    aggregate_active_taxis,
    time_bucket_to_hour,
    block_n_hours,
)


def test_time_bucket_to_hour():
    assert time_bucket_to_hour(1) == 0
    assert time_bucket_to_hour(12) == 0
    assert time_bucket_to_hour(13) == 1
    assert time_bucket_to_hour(288) == 23


def test_block_n_hours():
    assert block_n_hours(0) == 3   # morning_peak (7-10)
    assert block_n_hours(1) == 6   # midday (10-16)
    assert block_n_hours(2) == 4   # evening_peak (16-20)
    assert block_n_hours(3) == 11  # night (20-31, wraparound)


def test_aggregate_pickup_dropoff_mean_scale():
    # cell (5, 10) at hour 7 (block 0, 3 hours), day 1: 6 pickups
    raw_data = {(5 + 1, 10 + 1, 85, 1): [6, 0]}
    n_days = 1
    pickup_3d, dropoff_3d = aggregate_pickup_dropoff(raw_data, n_days=n_days)
    assert pickup_3d.shape == (48, 90, 4)
    # mean hourly = 6 / (3 × 1) = 2.0
    assert np.isclose(pickup_3d[5, 10, 0], 2.0)
    assert pickup_3d.sum() == pickup_3d[5, 10, 0]


def test_aggregate_active_taxis_mean():
    raw_data = {
        (5 + 1, 10 + 1, 7, 1): 20,
        (5 + 1, 10 + 1, 8, 1): 10,
    }
    taxis_3d = aggregate_active_taxis(raw_data, n_days=1)
    assert taxis_3d.shape == (48, 90, 4)
    # mean hourly = (20 + 10) / (3 × 1) = 10.0
    assert np.isclose(taxis_3d[5, 10, 0], 10.0)
