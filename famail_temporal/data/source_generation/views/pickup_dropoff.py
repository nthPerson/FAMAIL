"""View: produce pickup_dropoff_counts dictionary from the event stream.

Output schema: dict[(x, y, time_bucket, day_index)] -> (pickup_count, dropoff_count)
Only cells with at least one event appear. All counts are non-negative integers.
"""
from __future__ import annotations

import pandas as pd


def build_pickup_dropoff_counts(
    df: pd.DataFrame,
) -> dict[tuple[int, int, int, int], tuple[int, int]]:
    if len(df) == 0:
        return {}
    events = df[df["is_pickup"] | df["is_dropoff"]]
    if len(events) == 0:
        return {}
    grouped = (
        events
        .assign(_p=events["is_pickup"].astype(int),
                _d=events["is_dropoff"].astype(int))
        .groupby(["x_grid", "y_grid", "time_bucket", "day_index"], sort=False)
        .agg(pickup=("_p", "sum"), dropoff=("_d", "sum"))
        .reset_index()
    )
    out: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for row in grouped.itertuples(index=False):
        key = (int(row.x_grid), int(row.y_grid), int(row.time_bucket), int(row.day_index))
        out[key] = (int(row.pickup), int(row.dropoff))
    return out
