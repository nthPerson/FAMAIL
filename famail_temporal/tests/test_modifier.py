"""Tests for algorithm.modifier TrajectoryModifier."""

import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _make_test_trajectory(driver_id=0, pickup_xy=(3, 4), time_bucket=90):
    """Make a trajectory with pickup at given coords, time_bucket in morning_peak."""
    states = [
        TrajectoryState(x_grid=0.0, y_grid=0.0,
                        time_bucket=time_bucket - 1, day_index=1),
        TrajectoryState(x_grid=float(pickup_xy[0]), y_grid=float(pickup_xy[1]),
                        time_bucket=time_bucket, day_index=1),
    ]
    return Trajectory(trajectory_id=0, driver_id=driver_id, states=states)


def _active_cell_and_bucket(bundle, active_idx=0):
    cell = bundle.unit_map.to_flat_cell(active_idx)
    t_block = bundle.unit_map.to_time_block(active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    return x, y, 1 + (start_hour * 12)


def test_modify_single_returns_history():
    """modify_single returns a ModificationHistory with iterations."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, multi_stream_builder=None,
        max_iterations=3,
    )
    # Pick an active unit
    any_active_idx = 0
    cell = bundle.unit_map.to_flat_cell(any_active_idx)
    t_block = bundle.unit_map.to_time_block(any_active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    tb = 1 + (start_hour * 12)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

    history = modifier.modify_single(traj)
    assert isinstance(history, ModificationHistory)
    assert history.total_iterations <= 3
    assert len(history.iterations) == history.total_iterations


def test_modify_single_respects_epsilon_ball():
    """After modification, pickup is within epsilon-ball of original."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle,
        max_iterations=50,
    )
    any_active_idx = 0
    cell = bundle.unit_map.to_flat_cell(any_active_idx)
    t_block = bundle.unit_map.to_time_block(any_active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    tb = 1 + (start_hour * 12)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

    history = modifier.modify_single(traj)
    orig = np.array([float(x), float(y)])
    final = np.array([
        history.modified.pickup_state.x_grid,
        history.modified.pickup_state.y_grid,
    ])
    diff = np.abs(final - orig)
    assert (diff <= config.EPSILON_BALL + 1e-5).all(), (
        f"Final pickup {final} strayed {diff} from original {orig}, "
        f"exceeding epsilon={config.EPSILON_BALL}"
    )


def test_current_pickup_3d_reflects_modifications():
    """current_pickup_3d() must return the post-modification pickup tensor as a
    numpy ndarray matching bundle.pickup_3d's shape."""
    import numpy as np
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective

    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    objective = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=objective, bundle=bundle, max_iterations=2,
    )
    before = modifier.current_pickup_3d()
    assert isinstance(before, np.ndarray)
    assert before.shape == bundle.pickup_3d.shape
    assert before.dtype == np.float32
    assert np.allclose(before, bundle.pickup_3d)

    snapshot = before.copy()
    before[0, 0, 0] = 999.0
    assert np.allclose(modifier.current_pickup_3d(), snapshot)


def test_modifier_resolves_config_at_init_not_at_import():
    """Regression: config overrides applied AFTER module import must still be
    picked up when the modifier is constructed without explicit kwargs.
    Previously, default args like `max_iterations: int = config.MAX_ITERATIONS`
    froze the value at import time."""
    from famail_temporal import config
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    objective = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)

    original = config.MAX_ITERATIONS
    try:
        config.MAX_ITERATIONS = 7
        modifier = TrajectoryModifier(objective=objective, bundle=bundle)
        assert modifier.max_iterations == 7, (
            f"Expected max_iterations=7 (from config mutation), got "
            f"{modifier.max_iterations} — default arg bug may have returned"
        )
    finally:
        config.MAX_ITERATIONS = original


def test_accept_rule_default_is_objective():
    """Default modifier keeps the historical objective gate."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3)
    assert modifier.accept_rule == "objective"


def test_non_regression_rejects_f_spatial_regression():
    """Under non-regression, an iterate that lifts F_causal but dips F_spatial
    below its iter-0 value is NOT persisted as best; objective rule may accept it.

    We drive this deterministically with a stub objective whose terms we control
    by iteration, so the test does not depend on bundle gradients.
    """
    import torch as _t
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    # Sequence of (f_spatial, f_causal) per iteration. iter0 = baseline.
    # iter1 improves BOTH; iter2 improves f_causal more but regresses f_spatial.
    seq = [(0.10, 0.50), (0.11, 0.55), (0.09, 0.70)]
    calls = {"i": 0}

    def fake_forward(soft_pickup_3d=None, **kw):
        i = min(calls["i"], len(seq) - 1)
        fs, fc = seq[i]
        calls["i"] += 1
        total = _t.tensor(fs + fc, requires_grad=True)
        terms = {
            "f_spatial": _t.tensor(fs),
            "f_causal": _t.tensor(fc),
            "f_fidelity": _t.tensor(0.0),
        }
        return total, terms

    # nn.Module dispatches obj(...) -> self.forward (dunder looked up on the
    # type), so override forward, NOT __call__. diagnostics_enabled=False below
    # selects the single-backward path (the decomposed path needs a real graph).
    obj.forward = fake_forward  # type: ignore[method-assign]

    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=3, patience=None,
        accept_rule="non-regression", diagnostics_enabled=False,
    )
    x, y, tb = _active_cell_and_bucket(bundle)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)
    history = modifier.modify_single(traj)
    # Best iterate must be iter1 (improves both), NOT iter2 (regresses f_spatial).
    assert history.best_iteration == 1


def test_epsilon_cap_default_equals_epsilon_ball():
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3)
    assert modifier.epsilon_cap == config.EPSILON_BALL


def test_epsilon_cap_is_respected_relative_to_original_cell():
    """modify_single keeps the pickup within epsilon_cap (L-inf) of original_cell.
    The cross-round anchor distinction (cap from the TRUE original, not the
    round-start cell) is covered at the engine level in test_editing_loop
    (test_bounded_cap_limits_total_displacement)."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=50, epsilon_cap=1.0,
    )
    x, y, tb = _active_cell_and_bucket(bundle)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)
    history = modifier.modify_single(traj, original_cell=(x, y))
    s = history.modified.pickup_state
    assert abs(s.x_grid - x) <= 1.0 + 1e-5
    assert abs(s.y_grid - y) <= 1.0 + 1e-5


def test_soft_neighborhood_size_override_reaches_soft_assign(monkeypatch):
    """A runtime SOFT_NEIGHBORHOOD_SIZE override must reach SoftCellAssignment.
    Regression: the size was previously frozen by SoftCellAssignment's
    import-time default arg, so `--override SOFT_NEIGHBORHOOD_SIZE` was silently
    ignored. The modifier now resolves it from config at construction time."""
    bundle = _make_synthetic_bundle()
    m_default = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m_default.soft_assign.k == config.SOFT_NEIGHBORHOOD_SIZE // 2
    monkeypatch.setattr(config, "SOFT_NEIGHBORHOOD_SIZE", 11)
    m_wide = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m_wide.soft_assign.k == 5  # 11 // 2
