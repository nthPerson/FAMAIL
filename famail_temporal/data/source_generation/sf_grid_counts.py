"""SF gridded count artifacts (Task 3.4).

Produces the count dicts `preprocess.py` consumes:
- `pickup_dropoff_counts.pkl` <- count_pickup_dropoff: `(x,y,time_bucket,day) -> (pickup,dropoff)`
  (1-indexed x,y; 1-indexed 5-min time_bucket; calendar-day int).
- `active_taxis_5x5_hourly.pkl["data"]` <- count_active_taxis_5x5:
  `(x,y,hour,day) -> n_distinct_taxis` in the 5x5 cell neighborhood (0-indexed hour).
- `grid_to_district_mapping.pkl["valid_mask"]` <- build_valid_mask.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.sf_config import GridSpec, PDT_OFFSET_SEC

# 5x5 neighborhood radius (matches source_generation.config.NEIGHBORHOOD_K).
NEIGHBORHOOD_K: int = 2


def count_pickup_dropoff(
    pickups: List[List[int]], dropoffs: List[List[int]],
) -> Dict[Tuple[int, int, int, int], Tuple[int, int]]:
    """Aggregate pickup/dropoff events into `(x,y,time_bucket,day) -> (p,d)`."""
    acc: Dict[Tuple[int, int, int, int], List[int]] = defaultdict(lambda: [0, 0])
    for x, y, tb, day in pickups:
        acc[(int(x), int(y), int(tb), int(day))][0] += 1
    for x, y, tb, day in dropoffs:
        acc[(int(x), int(y), int(tb), int(day))][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def count_active_taxis_5x5(
    df: pd.DataFrame,
    grid: GridSpec,
    k: int = NEIGHBORHOOD_K,
    tz_offset_sec: int = PDT_OFFSET_SEC,
) -> Dict[Tuple[int, int, int, int], int]:
    """Distinct-taxi supply in each cell's 5x5 neighborhood per `(x,y,hour,day)`.

    A taxi present anywhere in the `(2k+1)x(2k+1)` window of cell `(x,y)` during a
    given local hour/day counts toward that cell's supply (0-indexed hour).
    """
    if len(df) == 0:
        return {}
    lat = df["lat"].to_numpy(np.float64)
    lon = df["lon"].to_numpy(np.float64)
    drv = df["driver_id"].to_numpy()
    t = df["time_utc"].to_numpy().astype(np.int64)

    x = np.clip(np.floor((lat - grid.lat_min) / grid.cell_deg).astype(int),
                0, grid.x_grid_max - 1) + 1
    y = np.clip(np.floor((lon - grid.lon_min) / grid.cell_deg).astype(int),
                0, grid.y_grid_max - 1) + 1
    local = t - tz_offset_sec
    hour = ((local % 86400) // 3600).astype(int)
    day = (local // 86400).astype(int)

    # Distinct drivers present in each source (cell, hour, day).
    g = pd.DataFrame({"cx": x, "cy": y, "h": hour, "d": day, "drv": drv})
    presence = g.groupby(["cx", "cy", "h", "d"])["drv"].apply(
        lambda s: set(s.tolist())
    )

    # Spread each source cell's drivers across its 5x5 target window.
    supply: Dict[Tuple[int, int, int, int], set] = defaultdict(set)
    for (sx, sy, h, d), drivers in presence.items():
        for tx in range(max(1, sx - k), min(grid.x_grid_max, sx + k) + 1):
            for ty in range(max(1, sy - k), min(grid.y_grid_max, sy + k) + 1):
                supply[(tx, ty, int(h), int(d))] |= drivers

    return {key: len(v) for key, v in supply.items()}


def build_valid_mask(grid: GridSpec) -> np.ndarray:
    """All grid cells are geometrically valid; the active mask filters further
    on supply + finite demographics (non-residential cells are NaN there)."""
    return np.ones((grid.x_grid_max, grid.y_grid_max), dtype=bool)
