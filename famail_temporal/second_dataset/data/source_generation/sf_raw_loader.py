"""SF Cabspotting raw loader (Task 3.1).

Parses the per-cab `new_<id>.txt` files (each line `lat lon occupancy time`,
occupancy 1 = with-fare/driving, 0 = free/seeking, time = UNIX epoch) into one
tidy DataFrame, integer-encoding each cab by sorted filename order and sorting
each cab's pings ascending in time (the source files are stored newest-first).
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import pandas as pd

# Bay Area validity window (drops null-island and far-stray GPS noise).
_LAT_MIN, _LAT_MAX = 36.5, 38.8
_LON_MIN, _LON_MAX = -123.2, -121.2


def load_sf_raw(data_dir: str) -> pd.DataFrame:
    """Load all `new_*.txt` cab traces under `data_dir` into a tidy DataFrame.

    Returns columns `[driver_id, lat, lon, occupancy, time_utc]`, invalid coords
    dropped, sorted by `(driver_id, time_utc)` ascending.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "new_*.txt")))
    frames: List[pd.DataFrame] = []
    for driver_id, path in enumerate(files):
        with open(path, "rb") as fh:
            flat = np.array(fh.read().split(), dtype=np.float64)
        if flat.size < 4:
            continue
        a = flat[: (flat.size // 4) * 4].reshape(-1, 4)
        frames.append(pd.DataFrame({
            "driver_id": np.full(a.shape[0], driver_id, dtype=np.int32),
            "lat": a[:, 0],
            "lon": a[:, 1],
            "occupancy": a[:, 2].astype(np.int8),
            "time_utc": a[:, 3].astype(np.int64),
        }))

    if not frames:
        return pd.DataFrame(
            columns=["driver_id", "lat", "lon", "occupancy", "time_utc"]
        )

    df = pd.concat(frames, ignore_index=True)
    valid = (
        (df["lat"] > _LAT_MIN) & (df["lat"] < _LAT_MAX)
        & (df["lon"] > _LON_MIN) & (df["lon"] < _LON_MAX)
    )
    df = df[valid]
    return df.sort_values(["driver_id", "time_utc"]).reset_index(drop=True)
