"""Tests for views/calendars.py — per-trajectory calendar-day indexing."""
from __future__ import annotations

from famail_temporal.data.source_generation.views.calendars import (
    build_per_trajectory_calendar_days,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def test_parallel_lists_map_each_trajectory_to_its_calendar_date():
    """Output must have len(seeking[idx]) == len(seeking_by_plate[plate]) for
    every driver — the discriminator's pair sampling needs 1:1 alignment."""
    result = TrajectoriesResult(
        seeking_by_plate={
            "A": [
                [[5, 10, 1, 1], [6, 11, 1, 1]],  # 2016-07-04
                [[5, 10, 1, 3], [6, 11, 1, 3]],  # 2016-07-06
                [[5, 10, 1, 1], [6, 11, 1, 1]],  # 2016-07-04
            ],
        },
        driving_by_plate={
            "A": [
                [[6, 11, 1, 2], [7, 12, 1, 2]],  # 2016-07-05
            ],
        },
        seeking_dates_by_plate={
            "A": ["2016-07-04", "2016-07-06", "2016-07-04"],
        },
        driving_dates_by_plate={
            "A": ["2016-07-05"],
        },
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}

    out = build_per_trajectory_calendar_days(result, mapping)

    # The index lists are parallel to ms_seeking / ms_driving, not sorted-unique.
    assert len(out["seeking"][0]) == 3
    assert len(out["driving"][0]) == 1

    # The calendar_day_map translates each index back to a date string.
    cal_map = out["calendar_day_map"]
    seeking_dates = [cal_map[i] for i in out["seeking"][0]]
    driving_dates = [cal_map[i] for i in out["driving"][0]]
    assert seeking_dates == ["2016-07-04", "2016-07-06", "2016-07-04"]
    assert driving_dates == ["2016-07-05"]


def test_missing_driver_produces_empty_list():
    """A driver with no kept trajectories maps to an empty parallel list."""
    result = TrajectoriesResult()
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    out = build_per_trajectory_calendar_days(result, mapping)
    assert out["seeking"] == {0: []}
    assert out["driving"] == {0: []}
    assert out["calendar_day_map"] == {}


def test_calendar_day_map_is_sorted_by_date():
    """Indices are assigned in calendar-date sorted order — needed for stable
    pair sampling across re-runs."""
    result = TrajectoriesResult(
        seeking_by_plate={
            "A": [[[1, 1, 1, 3]], [[1, 1, 1, 1]], [[1, 1, 1, 5]]],
        },
        driving_by_plate={},
        seeking_dates_by_plate={
            "A": ["2016-07-06", "2016-07-04", "2016-07-08"],
        },
        driving_dates_by_plate={"A": []},
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}

    out = build_per_trajectory_calendar_days(result, mapping)

    # Dates sorted → indices 0,1,2 assigned to 2016-07-04, 2016-07-06, 2016-07-08
    assert out["calendar_day_map"] == {
        0: "2016-07-04",
        1: "2016-07-06",
        2: "2016-07-08",
    }
    # Driver A's three seeking trajectories were on 07-06, 07-04, 07-08 in order
    assert out["seeking"][0] == [1, 0, 2]
