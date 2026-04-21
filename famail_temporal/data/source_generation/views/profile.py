"""View: compute the 11-dim driver profile features and z-score normalize."""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd

from famail_temporal.data.source_generation import config
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult, Trajectory,
)


def _mode_xy(sub: pd.DataFrame) -> tuple[int, int] | None:
    if len(sub) == 0:
        return None
    grouped = sub.groupby(["x_grid", "y_grid"]).size().reset_index(name="n")
    top = grouped.sort_values(
        ["n", "x_grid", "y_grid"], ascending=[False, True, True]
    ).iloc[0]
    return int(top.x_grid), int(top.y_grid)


def compute_home_xy_with_fallback(driver_df: pd.DataFrame) -> dict:
    if len(driver_df) == 0:
        raise ValueError("compute_home_xy_with_fallback: driver has no records")
    primary = driver_df[driver_df["time_bucket"] == 1]
    mode = _mode_xy(primary)
    if mode is not None:
        return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "none"}
    first_hour = driver_df[driver_df["time_bucket"].between(1, 12)]
    mode = _mode_xy(first_hour)
    if mode is not None:
        return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "first_hour"}
    mode = _mode_xy(driver_df)
    if mode is None:
        raise ValueError("compute_home_xy_with_fallback: driver has no records")
    return {"home_x": mode[0], "home_y": mode[1], "fallback_used": "all_records"}


def _trajectory_manhattan_length(traj: Trajectory) -> int:
    total = 0
    for a, b in zip(traj, traj[1:]):
        total += abs(a[0] - b[0]) + abs(a[1] - b[1])
    return total


def _trajectory_duration_minutes(
    driver_df: pd.DataFrame, traj: Trajectory,
) -> float | None:
    if len(traj) < 2:
        return None
    def find_s(state):
        matched = driver_df[
            (driver_df["x_grid"] == state[0])
            & (driver_df["y_grid"] == state[1])
            & (driver_df["time_bucket"] == state[2])
            & (driver_df["day_index"] == state[3])
        ]
        if len(matched) == 0:
            return None
        return matched.iloc[0]["seconds"]
    s_first = find_s(traj[0])
    s_last = find_s(traj[-1])
    if s_first is None or s_last is None:
        return None
    return max(0.0, (s_last - s_first) / 60.0)


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_profile_features(
    df: pd.DataFrame, trajs: TrajectoriesResult,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for plate, driver_df in df.groupby("plate_id", sort=False):
        home = compute_home_xy_with_fallback(driver_df)

        tbs = driver_df["time_bucket"].values
        shift_start = float(np.percentile(tbs, config.PROFILE_SHIFT_LOW_PCT))
        shift_end = float(np.percentile(tbs, config.PROFILE_SHIFT_HIGH_PCT))

        seeking = trajs.seeking_by_plate.get(plate, [])
        driving = trajs.driving_by_plate.get(plate, [])

        if seeking:
            pickup_cells = pd.DataFrame(
                [(t[-1][0], t[-1][1]) for t in seeking],
                columns=["x_grid", "y_grid"],
            )
            freq = _mode_xy(pickup_cells)
            freq_grid_x, freq_grid_y = freq if freq else (home["home_x"], home["home_y"])
        else:
            freq_grid_x, freq_grid_y = home["home_x"], home["home_y"]

        avg_seek_dist = _mean_or_zero([_trajectory_manhattan_length(t) for t in seeking])
        avg_drive_dist = _mean_or_zero([_trajectory_manhattan_length(t) for t in driving])
        avg_seek_time = _mean_or_zero([
            d for d in (_trajectory_duration_minutes(driver_df, t) for t in seeking)
            if d is not None
        ])
        avg_drive_time = _mean_or_zero([
            d for d in (_trajectory_duration_minutes(driver_df, t) for t in driving)
            if d is not None
        ])

        total_pickups = (
            int(driver_df["is_pickup"].sum())
            if "is_pickup" in driver_df.columns
            else len(seeking)
        )
        distinct_dates = driver_df["calendar_date"].nunique()
        num_trips_per_day = (
            total_pickups / distinct_dates if distinct_dates > 0 else 0.0
        )

        out[plate] = {
            "home_x": home["home_x"],
            "home_y": home["home_y"],
            "shift_start": shift_start,
            "shift_end": shift_end,
            "freq_grid_x": freq_grid_x,
            "freq_grid_y": freq_grid_y,
            "avg_seeking_dist": avg_seek_dist,
            "avg_seeking_time": avg_seek_time,
            "avg_driving_dist": avg_drive_dist,
            "avg_driving_time": avg_drive_time,
            "num_trips_per_day": float(num_trips_per_day),
            "fallback_used": home["fallback_used"],
        }
    return out


def zscore_normalize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = raw.mean(axis=0)
    std = raw.std(axis=0, ddof=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    normalized = (raw - mean) / std_safe
    return normalized, mean, std
