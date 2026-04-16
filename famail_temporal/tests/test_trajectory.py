"""Tests for utils.trajectory."""
import numpy as np
import torch

from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _make_trajectory(n_states: int = 5) -> Trajectory:
    states = [
        TrajectoryState(x_grid=float(i), y_grid=float(i + 1),
                        time_bucket=100 + i, day_index=1)
        for i in range(n_states)
    ]
    return Trajectory(trajectory_id=0, driver_id=7, states=states)


def test_pickup_cell_is_last_state():
    traj = _make_trajectory(5)
    assert traj.pickup_cell == (4, 5)


def test_to_tensor_shape():
    traj = _make_trajectory(5)
    t = traj.to_tensor()
    assert t.shape == (5, 4)
    assert t.dtype == torch.float32


def test_clone_is_deep():
    traj = _make_trajectory(3)
    clone = traj.clone()
    clone.states[-1].x_grid = 99.0
    assert traj.states[-1].x_grid != 99.0


def test_apply_perturbation_clips_to_grid():
    traj = _make_trajectory(3)
    perturbed = traj.apply_perturbation(np.array([100.0, -100.0]), grid_dims=(48, 90))
    assert perturbed.states[-1].x_grid == 47.0
    assert perturbed.states[-1].y_grid == 0.0
