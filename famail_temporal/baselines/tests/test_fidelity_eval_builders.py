"""fidelity_eval: discriminator tensor builders + trajectory statistics."""
import math
from types import SimpleNamespace

import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines import fidelity_eval as fe


def _state(x, y, t, d):
    return SimpleNamespace(x_grid=float(x), y_grid=float(y), time_bucket=t, day_index=d)


def _traj(states):
    return SimpleNamespace(states=states, driver_id=0)


def test_real_to_disc_tensor_adds_one_to_coords():
    traj = _traj([_state(0, 0, 5, 2), _state(3, 7, 5, 2)])
    out = fe.real_to_disc_tensor(traj)
    assert out.shape == (2, 4)
    # +1 coord conversion; time/day preserved
    assert out[0].tolist() == [1.0, 1.0, 5.0, 2.0]
    assert out[1].tolist() == [4.0, 8.0, 5.0, 2.0]


def test_generated_to_disc_tensor_unflattens_and_adds_one():
    # flat cell c -> (c // GY, c % GY); then +1
    c0 = 0                      # (0, 0) -> (1, 1)
    c1 = 2 * gc.GY + 3         # (2, 3) -> (3, 4)
    out = fe.generated_to_disc_tensor([c0, c1], time_bucket=9, day_index=4)
    assert out.shape == (2, 4)
    assert out[0].tolist() == [1.0, 1.0, 9.0, 4.0]
    assert out[1].tolist() == [3.0, 4.0, 9.0, 4.0]


def test_trajectory_statistics_from_cells():
    # cells (0,0) -> (0,1) -> (0,3): length 3, coverage 3,
    # displacements: |(0,1)-(0,0)|=1, |(0,3)-(0,1)|=2 -> mean 1.5
    cells = [0, 1, 3]
    s = fe.trajectory_statistics(cells)
    assert s["length"] == 3
    assert s["coverage"] == 3
    assert math.isclose(s["mean_displacement"], 1.5, rel_tol=1e-9)


def test_trajectory_statistics_from_trajectory_and_short_len():
    traj = _traj([_state(2, 2, 0, 0)])           # single state
    s = fe.trajectory_statistics(traj)
    assert s["length"] == 1
    assert s["coverage"] == 1
    assert s["mean_displacement"] == 0.0          # len < 2 -> 0
