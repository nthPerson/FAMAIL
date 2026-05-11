"""View: per-trajectory calendar-day indexing for Ren-style pair sampling.

Given per-trajectory calendar date strings (in ``TrajectoriesResult``'s
``seeking_dates_by_plate`` / ``driving_dates_by_plate`` sidecar), this view
produces two int-keyed parallel lists plus a global date index map. Output
schema::

    {
        "seeking": {driver_idx: [cal_day_idx_0, cal_day_idx_1, ...]},
        "driving": {driver_idx: [cal_day_idx_0, ...]},
        "calendar_day_map": {cal_day_idx: "YYYY-MM-DD"},
    }

``seeking[idx]`` is parallel to ``ms_seeking_trajs[idx]`` (same length, same
ordering), and likewise for driving. The discriminator's
``dataset_generation`` consumer uses this to group trajectories by
``(driver, calendar_day)`` when sampling positive pairs
(*same driver, 2 different calendar dates*).

Indices are assigned in calendar-date sorted order, so indexing is stable
across re-runs of the tool.
"""
from __future__ import annotations

from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def _collect_unique_dates(trajs: TrajectoriesResult) -> list[str]:
    dates: set[str] = set()
    for dl in trajs.seeking_dates_by_plate.values():
        dates.update(dl)
    for dl in trajs.driving_dates_by_plate.values():
        dates.update(dl)
    return sorted(dates)


def build_per_trajectory_calendar_days(
    trajs: TrajectoriesResult, mapping: dict,
) -> dict:
    """Return the three-key dict described in the module docstring.

    Every driver in ``mapping["plate_to_idx"]`` appears as a key in both
    ``"seeking"`` and ``"driving"`` — a driver with no trajectories of a
    given kind maps to an empty list (preserves alignment with
    ``ms_seeking`` / ``ms_driving``, which use the same convention).
    """
    unique_dates = _collect_unique_dates(trajs)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    calendar_day_map = {i: d for i, d in enumerate(unique_dates)}

    seeking: dict[int, list[int]] = {}
    driving: dict[int, list[int]] = {}
    for plate, idx in mapping["plate_to_idx"].items():
        seek_dates = trajs.seeking_dates_by_plate.get(plate, [])
        drive_dates = trajs.driving_dates_by_plate.get(plate, [])
        seeking[idx] = [date_to_idx[d] for d in seek_dates]
        driving[idx] = [date_to_idx[d] for d in drive_dates]

    return {
        "seeking": seeking,
        "driving": driving,
        "calendar_day_map": calendar_day_map,
    }
