"""Unit tests for famail_temporal.baselines.datasets."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import (
    active_units, make_traj_at, negative_attribution_units,
)
from famail_temporal.baselines import datasets as ds


def test_helper_pickup_unit_round_trips():
    bundle = _make_synthetic_bundle()
    (cx, cy, t_block) = active_units(bundle, 1)[0]
    traj = make_traj_at(cx, cy, t_block, traj_id=0)
    assert ds.pickup_unit_of(traj) == (cx, cy, t_block)


def test_pickup_mass_matches_modifier_formula():
    bundle = _make_synthetic_bundle()
    t_block = active_units(bundle, 1)[0][2]
    expected = 1.0 / (int(bundle.n_hours_per_block[t_block]) * bundle.n_days)
    assert ds.pickup_mass(bundle, t_block) == pytest.approx(expected)


def test_build_filtered_subtracts_mass_at_unit_only():
    bundle = _make_synthetic_bundle()
    (cx, cy, t_block) = active_units(bundle, 1)[0]
    traj = make_traj_at(cx, cy, t_block, traj_id=0)
    before = bundle.pickup_3d.copy()
    filtered = ds.build_filtered_pickup_3d(bundle, [traj])
    mass = ds.pickup_mass(bundle, t_block)
    # Target cell dropped by exactly one pickup mass.
    assert filtered[cx, cy, t_block] == pytest.approx(before[cx, cy, t_block] - mass)
    # Everything else identical.
    delta = before - filtered
    delta[cx, cy, t_block] = 0.0
    assert np.allclose(delta, 0.0)
    # Bundle's own grid is untouched (copy semantics).
    assert np.allclose(bundle.pickup_3d, before)


def test_build_filtered_floors_at_demand_floor():
    """Over-subtraction floors the cell at DEMAND_FLOOR (no negative demand)."""
    bundle = _make_synthetic_bundle()
    (cx, cy, t_block) = active_units(bundle, 1)[0]
    # Start the cell below one pickup mass so a single removal goes negative.
    bundle.pickup_3d[cx, cy, t_block] = ds.pickup_mass(bundle, t_block) * 0.5
    traj = make_traj_at(cx, cy, t_block, traj_id=0)
    filtered = ds.build_filtered_pickup_3d(bundle, [traj])
    assert filtered[cx, cy, t_block] == pytest.approx(config.DEMAND_FLOOR)


def test_rank_returns_only_negative_scores_most_unfair_first():
    bundle = _make_synthetic_bundle()
    # Place trajectories on the bundle's negative-attribution units (the only
    # valid filtering candidates) plus a few arbitrary active units, so the
    # ranking has real candidates and the test is not vacuous.
    neg_units = negative_attribution_units(bundle, 5)
    assert neg_units, "synthetic bundle has no negative-attribution units"
    units = neg_units + active_units(bundle, 10)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = ds.rank_unfair_trajectory_indices(bundle)
    assert ranked, "expected at least one strictly-unfair ranked trajectory"
    # All returned indices are valid and unique.
    assert len(set(ranked)) == len(ranked)
    assert all(0 <= i < len(bundle.trajectories) for i in ranked)
    # Recompute scores and confirm every returned idx is strictly negative
    # and the list is ascending (most-negative first).
    from famail_temporal.algorithm.attribution import (
        compute_per_unit_attribution, rank_trajectories,
    )
    attribution = compute_per_unit_attribution(bundle)
    scored = dict(rank_trajectories(bundle.trajectories, attribution, bundle.unit_map))
    returned_scores = [scored[i] for i in ranked]
    assert all(s < 0 for s in returned_scores)
    assert returned_scores == sorted(returned_scores)
