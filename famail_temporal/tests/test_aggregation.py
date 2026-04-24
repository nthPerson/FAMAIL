"""Tests for data.aggregation."""
import pytest
import numpy as np

from famail_temporal.data.aggregation import (
    hour_to_block_index,
    aggregate_pickup_dropoff,
    aggregate_active_taxis,
    time_bucket_to_hour,
    block_n_hours,
    dataset_n_days,
)


@pytest.mark.parametrize("hour,expected", [
    (0, 0), (7, 7), (10, 10), (15, 15),
    (16, 16), (20, 20), (23, 23),
])
def test_hour_to_block_index(hour, expected):
    """At T=24, hour h maps to block h (identity). Pins the mapping
    against accidental off-by-one regressions."""
    assert hour_to_block_index(hour) == expected


def test_invalid_hour_raises():
    with pytest.raises(ValueError):
        hour_to_block_index(24)


def test_time_bucket_to_hour():
    assert time_bucket_to_hour(1) == 0
    assert time_bucket_to_hour(12) == 0
    assert time_bucket_to_hour(13) == 1
    assert time_bucket_to_hour(288) == 23


def test_time_bucket_to_hour_accepts_zero():
    """Real-data trajectories contain some states with time_bucket=0. The
    mapping must treat this as hour 0 (first bucket of the day) rather than
    raising via a downstream hour_to_block_index(-1) crash. Documented
    1-indexed inputs [1..288] behave unchanged.
    """
    from famail_temporal.data.aggregation import time_bucket_to_hour
    # The regression case: tb=0 used to give hour=-1 (invalid).
    assert time_bucket_to_hour(0) == 0
    # Backward-compatible: existing 1-indexed values unchanged.
    assert time_bucket_to_hour(1) == 0
    assert time_bucket_to_hour(12) == 0
    assert time_bucket_to_hour(13) == 1
    assert time_bucket_to_hour(288) == 23


def test_block_n_hours():
    """Each hourly block covers exactly 1 hour at T=24."""
    from famail_temporal import config
    for t in range(config.T):
        assert block_n_hours(t) == 1, f"block {t} should cover 1 hour at T=24"


def test_aggregate_pickup_dropoff_mean_scale():
    # cell (5, 10) at hour 7 (block 7 at T=24, 1 hour), day 1: 6 pickups
    # time_bucket 85 = (85-1)//12 = hour 7
    raw_data = {(5 + 1, 10 + 1, 85, 1): [6, 0]}
    n_days = 1
    pickup_3d, dropoff_3d = aggregate_pickup_dropoff(raw_data, n_days=n_days)
    assert pickup_3d.shape == (48, 90, 24)
    # mean hourly = 6 / (1 × 1) = 6.0 — block is 1 hour at T=24
    assert np.isclose(pickup_3d[5, 10, 7], 6.0)
    assert pickup_3d.sum() == pickup_3d[5, 10, 7]


def test_aggregate_active_taxis_mean():
    """At T=24, hour 7 and hour 8 are separate blocks. Each block is 1 hour,
    so the block-mean is the raw count (no averaging across hours)."""
    raw_data = {
        (5 + 1, 10 + 1, 7, 1): 20,
        (5 + 1, 10 + 1, 8, 1): 10,
    }
    taxis_3d = aggregate_active_taxis(raw_data, n_days=1)
    assert taxis_3d.shape == (48, 90, 24)
    # block 7 gets 20 / (1 × 1) = 20.0
    assert np.isclose(taxis_3d[5, 10, 7], 20.0)
    # block 8 gets 10 / (1 × 1) = 10.0
    assert np.isclose(taxis_3d[5, 10, 8], 10.0)


def test_aggregate_pickup_dropoff_multi_day():
    """5 days × 6 pickups in the same (cell, hour-block) averages to 6.0/hour
    (each block is 1 hour at T=24, so divisor = 1 × 5 days = 5)."""
    raw = {(5 + 1, 10 + 1, 85, d): [6, 0] for d in range(1, 6)}
    pickup, _ = aggregate_pickup_dropoff(raw, n_days=5)
    # sum = 5 × 6 = 30; divisor = 1 hour × 5 days = 5; mean = 6.0
    # time_bucket 85 maps to hour 7 at T=24 -> block 7
    assert np.isclose(pickup[5, 10, 7], 6.0)


def test_dataset_n_days():
    from famail_temporal.data.aggregation import dataset_n_days
    # 3 distinct day_index values
    raw = {(1, 1, 1, 1): 0, (1, 1, 2, 1): 0, (1, 1, 1, 2): 0, (2, 2, 1, 5): 0}
    assert dataset_n_days(raw) == 3
    assert dataset_n_days({}) == 0


def test_aggregate_active_taxis_supply_floor():
    from famail_temporal import config
    # Valid non-empty data but only one cell; all others should be floored to SUPPLY_FLOOR
    raw = {(5 + 1, 10 + 1, 7, 1): 20}
    taxis = aggregate_active_taxis(raw, n_days=1)
    assert np.all(taxis >= config.SUPPLY_FLOOR)
    # An untouched cell should be exactly SUPPLY_FLOOR — block 0 (hour 0)
    # wasn't touched; the single entry was at hour 7.
    assert np.isclose(taxis[0, 0, 0], config.SUPPLY_FLOOR)


def test_time_bucket_zero_maps_to_hour_zero():
    """Real trajectory data contains some states with time_bucket=0 despite the
    docstring claiming 1-indexed 1..288. Treat tb=0 as hour 0 (first 5 minutes
    of day) rather than raising downstream in hour_to_block_index.
    """
    from famail_temporal.data.aggregation import time_bucket_to_hour, hour_to_block_index
    assert time_bucket_to_hour(0) == 0
    # Downstream chain must not raise:
    hour_to_block_index(time_bucket_to_hour(0))


def test_time_bucket_out_of_range_warns_and_clamps():
    """Values outside [0, 288] should warn (once) and clamp to 0 rather than
    returning nonsense hours like -9 or 28."""
    import warnings
    from famail_temporal.data.aggregation import time_bucket_to_hour
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = time_bucket_to_hour(-5)
    assert h == 0
    assert len(w) == 1
    assert "outside expected range" in str(w[0].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h = time_bucket_to_hour(999)
    assert h == 0  # clamped
    assert len(w) == 1


def test_time_bucket_boundary_values_unchanged():
    """Ensure the fix for tb=0 does not alter behavior for any valid 1..288
    input. All existing call sites rely on this mapping, so behavior must be
    preserved for the entire documented range."""
    from famail_temporal.data.aggregation import time_bucket_to_hour
    # tb=1 -> hour 0 (first bucket of day)
    assert time_bucket_to_hour(1) == 0
    # tb=12 -> hour 0 (last bucket of hour 0)
    assert time_bucket_to_hour(12) == 0
    # tb=13 -> hour 1
    assert time_bucket_to_hour(13) == 1
    # tb=288 -> hour 23 (last bucket of day)
    assert time_bucket_to_hour(288) == 23
    # tb=276 -> hour 22 (first bucket of hour 22 in 1-indexed scheme: (276-1)//12 == 22)
    assert time_bucket_to_hour(276) == 22
