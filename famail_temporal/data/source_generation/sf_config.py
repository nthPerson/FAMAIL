"""SF second-dataset grid + temporal config (Phase-2 D1/D4).

Faithful constant-0.01deg gridding (matches Shenzhen `source_generation/config.py`
`GRID_SIZE_DEG`). The grid DIMENSIONS are SF-specific (~32x30 over the SF taxi
footprint), NOT 48x90 — see docs/SF_PHASE2_DECISIONS.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CELL_DEG: float = 0.01           # faithful square cell (~1.1 x 0.88 km at SF lat)
DAYS_IN_WEEK: int = 7            # SF is 7-day (vs Shenzhen Mon-Fri)
N_TIME_BUCKETS: int = 288       # 5-min trajectory-state buckets (discriminator)
PDT_OFFSET_SEC: int = 7 * 3600  # May-Jun 2008 SF local = UTC-7 (PDT)
GAP_SEC: int = 300              # split a trajectory if consecutive pings > 5 min apart


@dataclass(frozen=True)
class GridSpec:
    """A constant-degree square grid over a lat/lon origin.

    `to_cell` returns **1-indexed** `(x, y)` (x = latitude axis, y = longitude
    axis), clipped into `[1, x_grid_max] x [1, y_grid_max]` — the same convention
    as Shenzhen `quantization.gps_to_grid` (0-indexed floor + clip, then +1).
    """
    lat_min: float
    lon_min: float
    x_grid_max: int
    y_grid_max: int
    cell_deg: float = CELL_DEG

    def to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        x0 = int(math.floor((lat - self.lat_min) / self.cell_deg))
        y0 = int(math.floor((lon - self.lon_min) / self.cell_deg))
        x0 = min(max(x0, 0), self.x_grid_max - 1)
        y0 = min(max(y0, 0), self.y_grid_max - 1)
        return (x0 + 1, y0 + 1)


def grid_from_points(lat, lon, p_lo: float = 0.5, p_hi: float = 99.5,
                     cell_deg: float = CELL_DEG) -> GridSpec:
    """Build a GridSpec from the percentile-trimmed bbox of observed points."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_min, lat_max = np.percentile(lat, p_lo), np.percentile(lat, p_hi)
    lon_min, lon_max = np.percentile(lon, p_lo), np.percentile(lon, p_hi)
    gx = math.ceil((lat_max - lat_min) / cell_deg)
    gy = math.ceil((lon_max - lon_min) / cell_deg)
    return GridSpec(float(lat_min), float(lon_min), int(gx), int(gy), cell_deg)
