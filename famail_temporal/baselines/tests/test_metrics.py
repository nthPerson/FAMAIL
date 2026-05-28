"""Unit tests for famail_temporal.baselines.metrics."""
import numpy as np
import pytest

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines import metrics as m
from famail_temporal.baselines.tests._helpers import (
    active_units, make_traj_at, negative_attribution_units,
)
from famail_temporal.baselines import datasets as ds


def test_data_level_fairness_keys_and_ranges():
    bundle = _make_synthetic_bundle()
    out = m.data_level_fairness(bundle)
    assert set(out) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
    assert 0.0 <= out["f_spatial"] <= 1.0
    assert 0.0 <= out["f_causal"] <= 1.0


def test_data_level_fairness_default_matches_explicit_grid():
    bundle = _make_synthetic_bundle()
    out_default = m.data_level_fairness(bundle)
    out_explicit = m.data_level_fairness(bundle, pickup_3d=bundle.pickup_3d)
    assert out_default == out_explicit


def test_filtering_an_unfair_trajectory_returns_valid_fairness():
    """Mechanism check: filtering subtracts demand and yields a valid F_causal.

    (The empirical claim "filtering improves fairness" is validated on real
    data in the Task 8 smoke run, not asserted on tiny synthetic data where a
    single mass change is not guaranteed monotone through r^2.)
    """
    bundle = _make_synthetic_bundle()
    neg_units = negative_attribution_units(bundle, 5)
    assert neg_units, "synthetic bundle has no negative-attribution units"
    units = neg_units + active_units(bundle, 10)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    ranked = ds.rank_unfair_trajectory_indices(bundle)
    assert ranked, "expected at least one strictly-unfair trajectory"
    removed = [bundle.trajectories[ranked[0]]]
    filtered_grid = ds.build_filtered_pickup_3d(bundle, removed)
    f_filt = m.data_level_fairness(bundle, pickup_3d=filtered_grid)["f_causal"]
    assert 0.0 <= f_filt <= 1.0
    # The removed trajectory's demand mass was subtracted, so the grid changed.
    assert not np.allclose(filtered_grid, bundle.pickup_3d)
