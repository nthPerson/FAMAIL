"""SF multi-stream corpus + 11-dim driver profiles (Task 3.5).

Aggregates per-driver segmentation into the discriminator's multi-stream inputs
(`ms_seeking_trajs`, `ms_driving_trajs`, calendar days), the editor's
`passenger_seeking_trajs` corpus, the `driver_index_mapping`, and the 11-dim
profile bundle (`features` raw + `features_normalized` z-scored). Profile feature
order matches `source_generation.config.PROFILE_FEATURE_NAMES`.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation.sf_config import GridSpec, PDT_OFFSET_SEC
from famail_temporal.data.source_generation.sf_segmentation import SegmentationResult
from famail_temporal.data.source_generation.config import (
    PROFILE_FEATURE_NAMES, N_PROFILE_FEATURES,
    PROFILE_SHIFT_LOW_PCT, PROFILE_SHIFT_HIGH_PCT,
)


def _path_len(tr) -> float:
    return float(sum(
        math.hypot(tr[i][0] - tr[i - 1][0], tr[i][1] - tr[i - 1][1])
        for i in range(1, len(tr))
    ))


def _avg(fn, trajs) -> float:
    return float(np.mean([fn(t) for t in trajs])) if trajs else 0.0


def driver_profile(df_driver: pd.DataFrame, seg: SegmentationResult,
                   grid: GridSpec) -> np.ndarray:
    """Compute the 11-dim profile vector for one driver (order =
    `PROFILE_FEATURE_NAMES`)."""
    lat = df_driver["lat"].to_numpy(np.float64)
    lon = df_driver["lon"].to_numpy(np.float64)
    t = df_driver["time_utc"].to_numpy().astype(np.int64)
    x = np.clip(np.floor((lat - grid.lat_min) / grid.cell_deg).astype(int),
                0, grid.x_grid_max - 1) + 1
    y = np.clip(np.floor((lon - grid.lon_min) / grid.cell_deg).astype(int),
                0, grid.y_grid_max - 1) + 1
    hour = (((t - PDT_OFFSET_SEC) % 86400) // 3600)

    if len(x):
        home = Counter(zip(x.tolist(), y.tolist())).most_common(1)[0][0]
        shift_start = float(np.percentile(hour, PROFILE_SHIFT_LOW_PCT))
        shift_end = float(np.percentile(hour, PROFILE_SHIFT_HIGH_PCT))
    else:
        home, shift_start, shift_end = (0, 0), 0.0, 0.0

    if seg.pickups:
        freq = Counter((p[0], p[1]) for p in seg.pickups).most_common(1)[0][0]
    else:
        freq = home

    n_days = len(set(seg.seeking_days) | set(seg.driving_days)) or 1
    vec = np.array([
        home[0], home[1], shift_start, shift_end, freq[0], freq[1],
        _avg(_path_len, seg.seeking), _avg(len, seg.seeking),
        _avg(_path_len, seg.driving), _avg(len, seg.driving),
        len(seg.driving) / n_days,
    ], dtype=np.float64)
    assert vec.shape == (N_PROFILE_FEATURES,)
    return vec


def normalize_profiles(
    profiles: Dict[int, np.ndarray],
) -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Z-score profiles across drivers; returns (normalized, mean, std)."""
    idxs = sorted(profiles)
    M = np.stack([profiles[i] for i in idxs])
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (M - mean) / std
    return {i: Z[k] for k, i in enumerate(idxs)}, mean, std


def assemble_multistream(
    per_driver: Dict[int, Tuple[pd.DataFrame, SegmentationResult]],
    grid: GridSpec,
) -> dict:
    """Build all multi-stream + corpus + mapping + profile artifacts."""
    ms_seeking, ms_driving = {}, {}
    ms_seeking_days, ms_driving_days = {}, {}
    profiles_raw = {}
    idx_to_plate, plate_to_idx, passenger_seeking = {}, {}, {}

    for idx, (df_d, seg) in per_driver.items():
        ms_seeking[idx] = seg.seeking
        ms_driving[idx] = seg.driving
        ms_seeking_days[idx] = list(seg.seeking_days)
        ms_driving_days[idx] = list(seg.driving_days)
        profiles_raw[idx] = driver_profile(df_d, seg, grid)
        plate = f"cab_{idx:04d}"
        idx_to_plate[idx] = plate
        plate_to_idx[plate] = idx
        passenger_seeking[plate] = seg.seeking

    profiles_norm, mean, std = normalize_profiles(profiles_raw)
    return {
        "ms_seeking": ms_seeking,
        "ms_driving": ms_driving,
        "ms_seeking_days": ms_seeking_days,
        "ms_driving_days": ms_driving_days,
        "profiles_raw": profiles_raw,
        "profiles_normalized": profiles_norm,
        "profile_mean": mean,
        "profile_std": std,
        "profile_feature_names": list(PROFILE_FEATURE_NAMES),
        "passenger_seeking": passenger_seeking,
        "driver_mapping": {"plate_to_idx": plate_to_idx, "idx_to_plate": idx_to_plate},
    }
