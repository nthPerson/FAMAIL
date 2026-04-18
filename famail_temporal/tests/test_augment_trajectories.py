"""Tests for evaluation.augment.augment_trajectories."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _make_trajectory(tid, did, states_xyt):
    return Trajectory(
        trajectory_id=tid, driver_id=did,
        states=[TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0)
                for (x, y, tb) in states_xyt],
    )


def _active_cell_tb(bundle):
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    x, y, t_block = int(ix_x[0]), int(ix_y[0]), int(ix_t[0])
    start_hour = config.TIME_BLOCKS[t_block][1]
    time_bucket = start_hour * 12 + 1
    return x, y, time_bucket


def test_result_is_dict_keyed_by_driver_id():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    x, y, tb = _active_cell_tb(bundle)
    trajs = [
        _make_trajectory(0, did=7, states_xyt=[(x, y, tb), (x, y, tb)]),
        _make_trajectory(1, did=7, states_xyt=[(x, y, tb), (x, y, tb)]),
        _make_trajectory(2, did=9, states_xyt=[(x, y, tb), (x, y, tb)]),
    ]
    result = augment_trajectories(trajs, grid)
    assert set(result.keys()) == {7, 9}
    assert len(result[7]) == 2
    assert len(result[9]) == 1


def test_states_are_8_element_lists():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    grid = build_fairness_grid(bundle)
    x, y, tb = _active_cell_tb(bundle)
    trajs = [_make_trajectory(0, did=3, states_xyt=[(x, y, tb), (x, y, tb), (x, y, tb)])]
    result = augment_trajectories(trajs, grid)
    traj_out = result[3][0]
    assert len(traj_out) == 3  # state count preserved
    for state in traj_out:
        assert len(state) == 8


def test_on_disk_coords_are_1_indexed():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    grid = build_fairness_grid(bundle)
    x, y, tb = _active_cell_tb(bundle)
    trajs = [_make_trajectory(0, did=1, states_xyt=[(x, y, tb)])]
    result = augment_trajectories(trajs, grid)
    state = result[1][0][0]
    assert state[0] == x + 1
    assert state[1] == y + 1
    assert state[2] == tb
    assert state[3] == 0


def test_active_state_fairness_channels_match_grid():
    from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=3)
    grid = build_fairness_grid(bundle)
    x, y, tb = _active_cell_tb(bundle)
    t_block = hour_to_block_index(time_bucket_to_hour(tb))
    trajs = [_make_trajectory(0, did=1, states_xyt=[(x, y, tb)])]
    result = augment_trajectories(trajs, grid)
    state = result[1][0][0]
    assert state[4] == pytest.approx(float(grid[x, y, t_block, 0]), abs=1e-6)
    assert state[5] == pytest.approx(float(grid[x, y, t_block, 1]), abs=1e-6)
    assert state[6] == pytest.approx(float(grid[x, y, t_block, 2]), abs=1e-6)
    assert state[7] == pytest.approx(float(grid[x, y, t_block, 3]), abs=1e-6)


def test_inactive_state_fairness_channels_are_nan():
    bundle = _make_synthetic_bundle(N_cells_per_block=5, seed=4)
    grid = build_fairness_grid(bundle)
    ix = np.argwhere(~bundle.mask_3d)
    x, y, t_block = int(ix[0, 0]), int(ix[0, 1]), int(ix[0, 2])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    trajs = [_make_trajectory(0, did=1, states_xyt=[(x, y, tb)])]
    result = augment_trajectories(trajs, grid)
    state = result[1][0][0]
    for ch in range(4, 8):
        assert np.isnan(state[ch]), f"channel {ch} should be NaN on inactive cell"


def test_state_count_preserved_across_all_trajectories():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=5)
    grid = build_fairness_grid(bundle)
    x, y, tb = _active_cell_tb(bundle)
    trajs = [
        _make_trajectory(0, did=1, states_xyt=[(x, y, tb)] * 5),
        _make_trajectory(1, did=1, states_xyt=[(x, y, tb)] * 3),
        _make_trajectory(2, did=2, states_xyt=[(x, y, tb)] * 8),
    ]
    result = augment_trajectories(trajs, grid)
    all_out_trajs = [t for tlist in result.values() for t in tlist]
    state_counts = sorted(len(t) for t in all_out_trajs)
    assert state_counts == [3, 5, 8]


def test_empty_input_yields_empty_dict():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=6)
    grid = build_fairness_grid(bundle)
    result = augment_trajectories([], grid)
    assert result == {}
