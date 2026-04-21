"""View: produce the three trajectory files + driver index mapping.

state[-1] convention:
  - Seeking trajectory: last state is the pickup-transition record (passenger=1).
  - Driving trajectory: last state is the dropoff-transition record (passenger=0).

Only complete segments (ending in a transition row) with length >= 2 are emitted.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


Trajectory = list[list[int]]


@dataclass
class TrajectoriesResult:
    seeking_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)
    driving_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)


def build_driver_index_mapping(df: pd.DataFrame) -> dict:
    plates = sorted(df["plate_id"].unique())
    plate_to_idx: dict[str, int] = {p: i for i, p in enumerate(plates)}
    idx_to_plate: dict[int, str] = {i: p for p, i in plate_to_idx.items()}
    return {"plate_to_idx": plate_to_idx, "idx_to_plate": idx_to_plate}


def _segment_is_seeking(segment: pd.DataFrame) -> bool:
    return bool(segment.iloc[-1]["is_pickup"])


def _segment_is_driving(segment: pd.DataFrame) -> bool:
    return bool(segment.iloc[-1]["is_dropoff"])


def _segment_to_trajectory(segment: pd.DataFrame) -> Trajectory:
    return [
        [int(r.x_grid), int(r.y_grid), int(r.time_bucket), int(r.day_index)]
        for r in segment.itertuples(index=False)
    ]


def build_trajectories(df: pd.DataFrame) -> TrajectoriesResult:
    result = TrajectoriesResult()
    for plate, driver_df in df.groupby("plate_id", sort=False):
        seeking: list[Trajectory] = []
        driving: list[Trajectory] = []
        for _, seg in driver_df.groupby("segment_id", sort=True):
            if len(seg) < 2:
                continue
            if _segment_is_seeking(seg):
                seeking.append(_segment_to_trajectory(seg))
            elif _segment_is_driving(seg):
                driving.append(_segment_to_trajectory(seg))
        if seeking:
            result.seeking_by_plate[plate] = seeking
        if driving:
            result.driving_by_plate[plate] = driving
    return result
