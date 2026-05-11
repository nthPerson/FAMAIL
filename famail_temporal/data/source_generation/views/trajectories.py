"""View: produce the three trajectory files + driver index mapping.

state[-1] convention:
  - Seeking trajectory: last state is the pickup-transition record (passenger=1).
  - Driving trajectory: last state is the dropoff-transition record (passenger=0).

Only complete segments (ending in a transition row) with length >= 2 are emitted.

Per-trajectory calendar_date sidecar
------------------------------------
`TrajectoriesResult` carries `seeking_dates_by_plate` and
`driving_dates_by_plate` as lists *parallel* to `seeking_by_plate` /
`driving_by_plate` — element ``i`` is the calendar date (``"YYYY-MM-DD"``) of
trajectory ``i``. This is required by the discriminator's Ren-style pair
sampling, which pairs *"same driver, 2 different calendar dates"* — it needs
a per-trajectory date, not just the set of weekdays the driver operated on.
The date is taken from the segment's first row (matches the legacy
extraction tool's convention).
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd


Trajectory = list[list[int]]


@dataclass
class TrajectoriesResult:
    seeking_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)
    driving_by_plate: dict[str, list[Trajectory]] = field(default_factory=dict)
    seeking_dates_by_plate: dict[str, list[str]] = field(default_factory=dict)
    driving_dates_by_plate: dict[str, list[str]] = field(default_factory=dict)


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


def _segment_calendar_date(segment: pd.DataFrame) -> str:
    return str(segment.iloc[0]["calendar_date"])


def build_trajectories(df: pd.DataFrame) -> TrajectoriesResult:
    result = TrajectoriesResult()
    has_dates = "calendar_date" in df.columns
    for plate, driver_df in df.groupby("plate_id", sort=False):
        seeking: list[Trajectory] = []
        driving: list[Trajectory] = []
        seeking_dates: list[str] = []
        driving_dates: list[str] = []
        for _, seg in driver_df.groupby("segment_id", sort=True):
            if len(seg) < 2:
                continue
            date = _segment_calendar_date(seg) if has_dates else ""
            if _segment_is_seeking(seg):
                seeking.append(_segment_to_trajectory(seg))
                seeking_dates.append(date)
            elif _segment_is_driving(seg):
                driving.append(_segment_to_trajectory(seg))
                driving_dates.append(date)
        if seeking:
            result.seeking_by_plate[plate] = seeking
            result.seeking_dates_by_plate[plate] = seeking_dates
        if driving:
            result.driving_by_plate[plate] = driving
            result.driving_dates_by_plate[plate] = driving_dates
    return result
