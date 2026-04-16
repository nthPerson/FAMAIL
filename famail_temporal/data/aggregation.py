"""
Aggregation of raw .pkl data into (48, 90, T) tensors.

Handles the time-bucket-to-block mapping with night wraparound, and builds
the three base tensors (pickup_3d, dropoff_3d, active_taxis_3d) using the
unified mean-hourly aggregation rule.
"""

from __future__ import annotations
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
