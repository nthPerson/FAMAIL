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
