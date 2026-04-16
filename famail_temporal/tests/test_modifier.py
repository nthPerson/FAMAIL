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
