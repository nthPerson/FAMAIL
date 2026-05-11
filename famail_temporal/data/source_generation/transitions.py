"""Per-driver passenger-indicator transition detection.

A pickup is a 0→1 transition; a dropoff is a 1→0 transition. Each transition
row is the FINAL (post-transition) state of its trajectory:
- The 1→0 row is the last state of a driving trajectory.
- The 0→1 row is the last state of a seeking trajectory.

`assign_segment_ids` gives each row a segment_id such that the transition row
is the LAST row of its segment and the next row starts a new segment.
"""
from __future__ import annotations
import pandas as pd


def add_transition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add `is_pickup`, `is_dropoff`, `is_transition` columns (per driver)."""
    out = df.copy()
    diff = out.groupby("plate_id")["passenger_indicator"].diff()
    out["is_pickup"] = diff == 1
    out["is_dropoff"] = diff == -1
    out["is_transition"] = out["is_pickup"] | out["is_dropoff"]
    return out


def assign_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign per-driver `segment_id` such that each transition row is the
    LAST row of its segment (cumsum of is_transition, shifted by 1)."""
    out = df.copy()
    out["segment_id"] = (
        out.groupby("plate_id")["is_transition"]
        .apply(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
        .reset_index(level=0, drop=True)
    )
    return out
