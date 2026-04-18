"""
Aggregation of raw .pkl data into (48, 90, T) tensors.

Handles the time-bucket-to-block mapping with night wraparound, and builds
the three base tensors (pickup_3d, dropoff_3d, active_taxis_3d) using the
unified mean-hourly aggregation rule.
"""

from __future__ import annotations
import warnings
from typing import Dict, Tuple
import numpy as np
from famail_temporal import config


def hour_to_block_index(hour: int) -> int:
    """Map hour [0, 24) → time block index [0, T)."""
    if not (0 <= hour < 24):
        raise ValueError(f"Hour must be in [0, 24), got {hour}")
    for i, (_, start, end) in enumerate(config.TIME_BLOCKS):
        if end > 24:
            if hour >= start or hour < (end - 24):
                return i
        else:
            if start <= hour < end:
                return i
    raise ValueError(f"Hour {hour} did not map to any time block")


def time_bucket_to_hour(time_bucket: int) -> int:
    """Map time_bucket to 0-indexed hour (0..23).

    Primary convention is 1-indexed (1..288, 5-min buckets) per the raw data
    schema. Real passenger_seeking_trajs data was also found to contain some
    states with time_bucket=0 — treated as hour 0 (first 5 minutes of day)
    so downstream pipelines don't raise on real-data inputs. All valid
    1-indexed inputs are unchanged.
    """
    return max(0, (time_bucket - 1) // 12)


def block_n_hours(block_idx: int) -> int:
    """Number of hours covered by block `block_idx`, handling wraparound."""
    _, start, end = config.TIME_BLOCKS[block_idx]
    return end - start


def dataset_n_days(raw_data: Dict[Tuple, object]) -> int:
    """Infer the number of distinct day_index values."""
    days = {key[3] for key in raw_data.keys() if len(key) >= 4}
    return len(days)


def aggregate_pickup_dropoff(
    raw_data: Dict[Tuple[int, int, int, int], object],
    n_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate raw counts dict → (48, 90, T) mean-hourly tensors.

    Raw keys use 1-indexed (x, y) and 1-indexed time_bucket.
    Aggregation: sum raw counts per (cell, block, day) combination, then
    divide by uniform n_obs = block_n_hours(t) × n_days.
    """
    assert n_days > 0, f"n_days must be positive, got {n_days}"

    pickup_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)
    dropoff_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)
    rejected_coords = 0

    for key, counts in raw_data.items():
        if len(key) < 4:
            raise ValueError(
                f"Raw data key {key} has fewer than 4 elements "
                f"(expected (x, y, time, day))"
            )
        x_raw, y_raw, time_bucket, _day = key
        x, y = int(x_raw) - 1, int(y_raw) - 1
        if not (0 <= x < config.GRID_DIMS[0] and 0 <= y < config.GRID_DIMS[1]):
            rejected_coords += 1
            continue
        if not isinstance(counts, (list, tuple)) or len(counts) < 2:
            raise ValueError(
                f"Raw counts value {counts!r} must be a list/tuple of at least "
                f"2 elements (pickup, dropoff)"
            )
        pickup = int(counts[0])
        dropoff = int(counts[1])
        hour = time_bucket_to_hour(int(time_bucket))
        t_block = hour_to_block_index(hour)
        pickup_3d[x, y, t_block] += pickup
        dropoff_3d[x, y, t_block] += dropoff

    if rejected_coords > 0:
        warnings.warn(
            f"aggregate_pickup_dropoff: {rejected_coords} keys had "
            f"out-of-bounds coordinates"
        )

    for t in range(config.T):
        divisor = block_n_hours(t) * n_days
        pickup_3d[:, :, t] /= divisor
        dropoff_3d[:, :, t] /= divisor

    return pickup_3d, dropoff_3d


def aggregate_active_taxis(
    raw_data: Dict[Tuple[int, int, int, int], int],
    n_days: int,
) -> np.ndarray:
    """Aggregate hourly active_taxis → (48, 90, T) mean-hourly tensor.

    Raw keys use 1-indexed (x, y) and 0-indexed hour.
    """
    assert n_days > 0, f"n_days must be positive, got {n_days}"

    active_3d = np.zeros((*config.GRID_DIMS, config.T), dtype=np.float32)
    rejected_coords = 0

    for key, count in raw_data.items():
        if len(key) < 4:
            raise ValueError(
                f"Raw data key {key} has fewer than 4 elements "
                f"(expected (x, y, time, day))"
            )
        x_raw, y_raw, hour, _day = key
        x, y = int(x_raw) - 1, int(y_raw) - 1
        if not (0 <= x < config.GRID_DIMS[0] and 0 <= y < config.GRID_DIMS[1]):
            rejected_coords += 1
            continue
        t_block = hour_to_block_index(int(hour))
        active_3d[x, y, t_block] += count

    if rejected_coords > 0:
        warnings.warn(
            f"aggregate_active_taxis: {rejected_coords} keys had "
            f"out-of-bounds coordinates"
        )

    for t in range(config.T):
        divisor = block_n_hours(t) * n_days
        active_3d[:, :, t] /= divisor

    active_3d = np.maximum(active_3d, config.SUPPLY_FLOOR)
    return active_3d
