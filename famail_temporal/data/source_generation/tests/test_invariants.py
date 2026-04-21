"""Tests for invariants.py."""
from __future__ import annotations
import pytest

from famail_temporal.data.source_generation.invariants import (
    apply_per_trajectory_invariants, check_systemic_invariants,
    SystemicInvariantError,
)
from famail_temporal.data.source_generation.views.trajectories import (
    TrajectoriesResult,
)


def _valid_seeking_traj():
    return [[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 2, 1]]


def test_per_trajectory_drops_out_of_bounds():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [
            _valid_seeking_traj(),
            [[5, 10, 1, 1], [999, 999, 1, 1], [6, 11, 2, 1]],
        ],
    })
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    dropoff_counts: dict = {}
    kept, removals = apply_per_trajectory_invariants(
        trajs, pickup_counts, dropoff_counts,
    )
    assert len(kept.seeking_by_plate["A"]) == 1
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "out_of_bounds"


def test_per_trajectory_drops_no_matching_pickup_count():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[[5, 10, 1, 1], [5, 11, 1, 1], [6, 11, 2, 1]]],
    })
    kept, removals = apply_per_trajectory_invariants(trajs, {}, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "no_matching_count"


def test_per_trajectory_drops_degenerate_length():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[[5, 10, 1, 1]]],
    })
    kept, removals = apply_per_trajectory_invariants(trajs, {}, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "degenerate_length"


def test_per_trajectory_accepts_short_midnight_crossing():
    """A short overnight seeking episode (e.g., 5-min span across midnight)
    is well under the 10-hour duration threshold and must be kept."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 287, 1],   # day 1, 23:50
            [5, 10, 288, 1],   # day 1, 23:55
            [5, 10, 1,   2],   # day 2, 00:00  (midnight wrap)
            [6, 11, 2,   2],   # day 2, 00:05 — pickup
        ]],
    })
    # Duration: (288-287) + 0*288 + 2 = 3 buckets = 15 min. Well below 120.
    pickup_counts = {(6, 11, 2, 2): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert len(kept.seeking_by_plate.get("A", [])) == 1
    assert len(removals) == 0


def test_per_trajectory_accepts_long_overnight_shift_within_threshold():
    """An overnight seeking episode up to ~8 hours is within the 10-hour
    threshold and must be kept. Night-shift drivers do this regularly."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 264, 1],   # Mon 22:00
            [5, 10, 288, 1],   # Mon 23:55
            [5, 10, 1,   2],   # Tue 00:00
            [5, 10, 72,  2],   # Tue 05:55
            [6, 11, 73,  2],   # Tue 06:00 — pickup
        ]],
    })
    # Duration: (288-264) + 0*288 + 73 = 24 + 73 = 97 buckets = 485 min ≈ 8h 5m.
    pickup_counts = {(6, 11, 73, 2): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert len(kept.seeking_by_plate.get("A", [])) == 1
    assert len(removals) == 0


def test_per_trajectory_drops_temporal_order_within_day():
    """A trajectory whose time_bucket goes backward within the SAME day
    IS a genuine temporal_order violation and should still be dropped.
    """
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 10, 1],
            [5, 10, 5,  1],   # day 1, 10 → 5 within the same day (backward)
            [6, 11, 2,  1],
        ]],
    })
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "temporal_order"


def test_per_trajectory_drops_friday_to_monday_as_implausibly_long():
    """A Fri→Mon trajectory accumulates ≥48 elapsed hours — impossible for a
    single seeking/driving episode. Rejected as `implausibly_long`."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 287, 5],   # Fri 23:50
            [5, 10, 288, 5],   # Fri 23:55
            [5, 10, 1,   1],   # Mon 00:00 — day_index wrapped backward
            [6, 11, 2,   1],
        ]],
    })
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "implausibly_long"


def test_per_trajectory_drops_friday_to_midweek_as_implausibly_long():
    """A Fri→Tue trajectory (driver took additional days off) is also
    implausibly long for a single episode."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 288, 5],   # Fri 23:55
            [5, 10, 50,  2],   # Tue 04:05 (Mon was taken off too)
            [6, 11, 51,  2],
        ]],
    })
    pickup_counts = {(6, 11, 51, 2): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "implausibly_long"


def test_per_trajectory_drops_over_threshold_duration_within_week():
    """A same-week trajectory exceeding the duration threshold is also
    implausibly long. Here Mon 08:00 → Tue 08:00 is 24 hours."""
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [[
            [5, 10, 97, 1],    # Mon 08:00
            [5, 10, 288, 1],   # Mon 23:55
            [5, 10, 1,  2],    # Tue 00:00
            [6, 11, 97, 2],    # Tue 08:00 — 24h elapsed
        ]],
    })
    # Duration: (288-97) + 0*288 + 97 = 288 buckets = 1440 min = 24h.
    pickup_counts = {(6, 11, 97, 2): (1, 0)}
    kept, removals = apply_per_trajectory_invariants(trajs, pickup_counts, {})
    assert kept.seeking_by_plate.get("A", []) == []
    assert len(removals) == 1
    assert removals[0].removal_reason_category == "implausibly_long"


def test_systemic_count_mismatch_raises():
    trajs = TrajectoriesResult(seeking_by_plate={
        "A": [_valid_seeking_traj()],
    })
    pickup_counts = {(6, 11, 2, 1): (2, 0)}
    with pytest.raises(SystemicInvariantError):
        check_systemic_invariants(
            trajs, pickup_counts, {}, profile_matrix=None, n_drivers=1,
            expect_n_drivers=1,
        )


def test_systemic_wrong_driver_count_raises():
    trajs = TrajectoriesResult(seeking_by_plate={"A": [_valid_seeking_traj()]})
    pickup_counts = {(6, 11, 2, 1): (1, 0)}
    with pytest.raises(SystemicInvariantError, match="50"):
        check_systemic_invariants(
            trajs, pickup_counts, {}, profile_matrix=None, n_drivers=1,
            expect_n_drivers=50,
        )
