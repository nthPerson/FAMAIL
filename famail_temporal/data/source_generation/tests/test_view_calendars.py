"""Tests for views/calendars.py."""
from __future__ import annotations

from famail_temporal.data.source_generation.views.calendars import (
    build_calendar_days_per_driver,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def test_calendar_days_lists_unique_day_indices():
    result = TrajectoriesResult(
        seeking_by_plate={
            "A": [
                [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 1, 1]],
                [[5, 10, 1, 3], [5, 11, 1, 3], [6, 11, 1, 3]],
                [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 1, 1]],
            ],
        },
        driving_by_plate={
            "A": [
                [[6, 11, 1, 2], [7, 12, 1, 2], [7, 13, 1, 2]],
            ],
        },
    )
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    out = build_calendar_days_per_driver(result, mapping)
    assert out["seeking"] == {0: [1, 3]}
    assert out["driving"] == {0: [2]}


def test_missing_driver_produces_empty_list():
    result = TrajectoriesResult()
    mapping = {"plate_to_idx": {"A": 0}, "idx_to_plate": {0: "A"}}
    out = build_calendar_days_per_driver(result, mapping)
    assert out["seeking"] == {0: []}
    assert out["driving"] == {0: []}
