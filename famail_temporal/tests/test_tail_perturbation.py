import numpy as np
import pytest
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj(cells, tb0=13):
    return Trajectory(trajectory_id=1, driver_id=1, states=[
        TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb0 + i, day_index=1)
        for i, (x, y) in enumerate(cells)])


def _king_ok(traj):
    return all(max(abs(b.x_grid - a.x_grid), abs(b.y_grid - a.y_grid)) <= 1
               for a, b in zip(traj.states, traj.states[1:]))


def test_all_stay_tail_full_delta_compliant():
    t = _traj([(10, 10)] * 6)                       # 5 stays then pickup
    out = t.apply_tail_perturbation(np.array([2.0, 2.0]), tail_len=4, grid_dims=(48, 90))
    assert out is not None
    assert (out.states[-1].x_grid, out.states[-1].y_grid) == (12, 12)   # pickup got full delta
    assert (out.states[0].x_grid, out.states[0].y_grid) == (10, 10)     # anchor untouched
    assert _king_ok(out)


def test_moving_tail_against_delta_still_compliant():
    t = _traj([(10, 10), (11, 10), (12, 10), (13, 10), (14, 10), (15, 10)])  # steps +1 in x
    out = t.apply_tail_perturbation(np.array([-2.0, 0.0]), tail_len=4, grid_dims=(48, 90))
    assert out is not None and _king_ok(out)
    assert out.states[-1].x_grid == 13              # pickup at 15-2


def test_time_and_day_preserved_and_original_unmodified():
    t = _traj([(5, 5)] * 4)
    out = t.apply_tail_perturbation(np.array([1.0, 0.0]), tail_len=4, grid_dims=(48, 90))
    assert [s.time_bucket for s in out.states] == [s.time_bucket for s in t.states]
    assert t.states[-1].x_grid == 5                 # original untouched (clone semantics)


def test_short_trajectory_uses_reduced_tail():
    t = _traj([(5, 5), (5, 5), (5, 5)])             # len 3 -> L_eff = 1
    out = t.apply_tail_perturbation(np.array([2.0, 0.0]), tail_len=4, grid_dims=(48, 90))
    assert out is not None and _king_ok(out)


def test_len2_returns_none_or_epsilon1():
    t = _traj([(5, 5), (5, 5)])                     # len 2: no tail room for |delta|=2
    out = t.apply_tail_perturbation(np.array([2.0, 0.0]), tail_len=4, grid_dims=(48, 90))
    assert out is None                              # skip; caller counts it


def test_property_random_trajectories_always_compliant_or_none():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(3, 15))
        cells = [(20, 20)]
        for _ in range(n - 1):                      # king-move random walk
            dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            cells.append((cells[-1][0] + dx, cells[-1][1] + dy))
        t = _traj(cells)
        d = rng.uniform(-2, 2, size=2)
        out = t.apply_tail_perturbation(d, tail_len=4, grid_dims=(48, 90))
        if out is not None:
            assert _king_ok(out)
            assert (out.states[-1].x_grid - t.states[-1].x_grid) == int(np.clip(round(d[0]), -2, 2))
