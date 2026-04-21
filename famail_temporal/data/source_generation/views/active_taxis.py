"""View: produce active_taxis counts (5x5 neighborhood, hourly, empty-only).

A driver counts as active at target cell (cx, cy) during (hour, day) if they
had >=1 GPS ping in the 5x5 neighborhood around (cx, cy) during that hour,
AND at least one of those pings had passenger_indicator == 0.
"""
from __future__ import annotations

import pandas as pd

from famail_temporal.data.source_generation import config


def build_active_taxis_counts(
    df: pd.DataFrame,
) -> dict[tuple[int, int, int, int], int]:
    if len(df) == 0:
        return {}

    empties = df[df["passenger_indicator"] == 0][
        ["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    ].copy()
    if len(empties) == 0:
        return {}

    empties = empties.drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    k = config.NEIGHBORHOOD_K
    pieces: list[pd.DataFrame] = []
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            exp = empties.copy()
            exp["x_grid"] = exp["x_grid"] + dx
            exp["y_grid"] = exp["y_grid"] + dy
            pieces.append(exp)
    expanded = pd.concat(pieces, ignore_index=True)

    expanded = expanded[
        (expanded["x_grid"] >= 1) & (expanded["x_grid"] <= config.X_GRID_MAX)
        & (expanded["y_grid"] >= 1) & (expanded["y_grid"] <= config.Y_GRID_MAX)
    ]

    expanded = expanded.drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    counts = (
        expanded
        .groupby(["x_grid", "y_grid", "hour", "day_index"], sort=False)
        .size()
        .reset_index(name="count")
    )

    return {
        (int(r.x_grid), int(r.y_grid), int(r.hour), int(r.day_index)): int(r.count)
        for r in counts.itertuples(index=False)
    }
