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
    # Mutating clone's metadata must not affect original
    clone.metadata["key"] = "value"
    assert "key" not in traj.metadata


def test_apply_perturbation_clips_to_grid():
    traj = _make_trajectory(3)
    perturbed = traj.apply_perturbation(np.array([100.0, -100.0]), grid_dims=(48, 90))
    assert perturbed.states[-1].x_grid == 47.0
    assert perturbed.states[-1].y_grid == 0.0
    # new object, not mutation
    assert perturbed is not traj
    # original state unchanged (for _make_trajectory(3), last state's x is 2.0)
    assert traj.states[-1].x_grid == 2.0
    # time_bucket preserved
    assert perturbed.states[-1].time_bucket == traj.states[-1].time_bucket
    # trajectory_id preserved
    assert perturbed.trajectory_id == traj.trajectory_id


def test_to_discriminator_format_column_order_and_n_states():
    traj = _make_trajectory(3)
    assert traj.n_states == 3
    arr = traj.to_discriminator_format()
    assert arr.shape == (3, 4)
    # For state i: x=i, y=i+1, time=100+i, day=1
    assert arr[2].tolist() == [2.0, 3.0, 102.0, 1.0]


def test_state_array_roundtrip():
    state = TrajectoryState(x_grid=1.5, y_grid=2.5, time_bucket=42, day_index=3)
    recovered = TrajectoryState.from_array(state.to_array())
    assert recovered == state
