"""View: per-driver calendar-day lists for the ms_{seeking,driving}_calendar_days files.

Currently unused by famail_temporal (loaded but not consumed in the context
builder today); provided for forward-compatibility with same-day context sampling.
"""
from __future__ import annotations

from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def build_calendar_days_per_driver(
    trajs: TrajectoriesResult, mapping: dict,
) -> dict[str, dict[int, list[int]]]:
    seeking: dict[int, list[int]] = {}
    driving: dict[int, list[int]] = {}
    for plate, idx in mapping["plate_to_idx"].items():
        seek_days = sorted({t[0][3] for t in trajs.seeking_by_plate.get(plate, [])})
        drive_days = sorted({t[0][3] for t in trajs.driving_by_plate.get(plate, [])})
        seeking[idx] = seek_days
        driving[idx] = drive_days
    return {"seeking": seeking, "driving": driving}
