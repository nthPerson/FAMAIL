"""Authoritative spatial and temporal quantization primitives.

Every module in the source-generation pipeline calls these functions (and only
these) for lat/lon → (x, y), seconds → time_bucket, timestamp → day.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Union

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation import config


Scalar = Union[int, float]
Array = np.ndarray


@dataclass(frozen=True)
class GlobalBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def compute_global_bounds(
    latitudes: pd.Series, longitudes: pd.Series,
) -> GlobalBounds:
    if len(latitudes) == 0 or len(longitudes) == 0:
        raise ValueError("compute_global_bounds: empty input")
    return GlobalBounds(
        lat_min=float(latitudes.min()),
        lat_max=float(latitudes.max()),
        lon_min=float(longitudes.min()),
        lon_max=float(longitudes.max()),
    )


def _bins(bound_min: float, bound_max: float) -> np.ndarray:
    return np.arange(bound_min, bound_max + config.GRID_SIZE_DEG, config.GRID_SIZE_DEG)


def gps_to_grid(
    lat: Union[Scalar, Array, pd.Series],
    lon: Union[Scalar, Array, pd.Series],
    bounds: GlobalBounds,
) -> tuple[Array, Array]:
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat_bins = _bins(bounds.lat_min, bounds.lat_max)
    lon_bins = _bins(bounds.lon_min, bounds.lon_max)
    x0 = np.digitize(lat_arr, lat_bins) - 1
    y0 = np.digitize(lon_arr, lon_bins) - 1
    x0 = np.clip(x0, 0, config.X_GRID_MAX - 1)
    y0 = np.clip(y0, 0, config.Y_GRID_MAX - 1)
    x = x0 + config.X_GRID_OFFSET
    y = y0 + config.Y_GRID_OFFSET
    return x, y


def seconds_to_time_bucket(seconds: Union[Scalar, Array, pd.Series]) -> Array:
    """Convert seconds-since-midnight to 1-indexed 5-min time_bucket.

    00:00:00 → 1; 00:04:59 → 1; 00:05:00 → 2; 23:59:59 → 288.
    """
    s_arr = np.asarray(seconds, dtype=int)
    bucket_0idx = s_arr // (config.TIME_INTERVAL_MIN * 60)
    bucket_1idx = bucket_0idx + 1
    return np.clip(bucket_1idx, 1, config.TIME_BUCKET_MAX)


def seconds_to_hour(seconds: Union[Scalar, Array, pd.Series]) -> Array:
    """Convert seconds-since-midnight to 0-indexed hour [0, 23]."""
    s_arr = np.asarray(seconds, dtype=int)
    h = s_arr // 3600
    return np.clip(h, 0, config.HOUR_MAX)


def timestamp_to_day(ts: str) -> int | None:
    """Convert a 'YYYY-MM-DD HH:MM:SS' string to a 1-indexed weekday index.

    Mon=1 .. Fri=5. Returns None for Sat, Sun, or unparseable input.
    """
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None
    dow = dt.weekday()
    if dow >= 5:
        return None
    return dow + 1
