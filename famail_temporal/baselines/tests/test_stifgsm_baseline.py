"""Tests for the ST-iFGSM/FGSM/random baseline attack engine."""
import json
import math
import pickle

import numpy as np
import torch

from famail_temporal.algorithm.modifier import ModificationHistory
from famail_temporal.baselines.stifgsm_baseline import (
    AttackOutcome, adjacency_violation_rate, attack_trajectories,
    discretize_outcome, package_arm,
)
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


class StubDisc(torch.nn.Module):
    """Differentiable stand-in: p = sigmoid(-mean((x1-x2)^2 over xy)).

    Identical inputs -> p=0.5; moving x2 away lowers p, so a *descent* attack
    on p has a well-defined gradient signal, like the real Siamese scorer.
    """
    def forward(self, x1, x2, mask1=None, mask2=None, profile_1=None, profile_2=None):
        diff = (x1[..., :2] - x2[..., :2]) ** 2
        if mask2 is not None:
            m = mask2.unsqueeze(-1).float()
            diff = diff * m
            denom = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        else:
            denom = torch.tensor(float(diff[0].numel()), device=diff.device)
        return torch.sigmoid(-diff.sum(dim=(1, 2, 3)) / denom)


class AsymDisc(torch.nn.Module):
    """Asymmetric stand-in: p = sigmoid(mean over masked xy of (x2 - x1)).

    A linear +x/+y pull: p rises with x2, so its gradient at delta=0 is NONZERO
    (unlike the symmetric StubDisc). Descending p therefore drives x2 (= the
    perturbed trajectory) in the -x/-y direction — a signed step actually MOVES
    the trajectory, which is exactly what a discarded-step bug would fail to do.
    """
    def forward(self, x1, x2, mask1=None, mask2=None, profile_1=None, profile_2=None):
        diff = x2[..., :2] - x1[..., :2]
        if mask2 is not None:
            m = mask2.unsqueeze(-1).float()
            diff = diff * m
            denom = m.sum(dim=(1, 2, 3)).clamp(min=1.0)
        else:
            denom = torch.tensor(float(diff[0].numel()), device=diff.device)
        return torch.sigmoid(diff.sum(dim=(1, 2, 3)) / denom)


def _traj(tid, n_states=5, x0=10.0, y0=20.0, driver=7):
    states = [TrajectoryState(x_grid=x0 + i, y_grid=y0 + i, time_bucket=3, day_index=1)
              for i in range(n_states)]
    return Trajectory(trajectory_id=tid, driver_id=driver, states=states)


def _profiles():
    return {7: np.zeros(11, dtype=np.float32)}


def _run(mode, trajs=None, **kw):
    trajs = trajs or [_traj(1), _traj(2, n_states=3)]
    return attack_trajectories(trajs, StubDisc(), _profiles(), mode,
                               epsilon=2.0, step=0.1, max_iterations=8,
                               patience=3, convergence_tol=0.0, seed=0, **kw)


def test_epsilon_clip_invariant_all_modes():
    for mode in ("ifgsm", "fgsm", "random"):
        for out, traj in zip(_run(mode), [_traj(1), _traj(2, n_states=3)]):
            orig = np.array([[s.x_grid, s.y_grid] for s in traj.states])
            assert np.max(np.abs(out.perturbed_xy - orig)) <= 2.0 + 1e-9, mode


def test_fgsm_equals_ifgsm_single_fullstep():
    a = attack_trajectories([_traj(1)], StubDisc(), _profiles(), "fgsm",
                            epsilon=2.0, step=0.1, max_iterations=8, seed=0)
    b = attack_trajectories([_traj(1)], StubDisc(), _profiles(), "ifgsm",
                            epsilon=2.0, step=2.0, max_iterations=1, seed=0)
    np.testing.assert_array_equal(a[0].perturbed_xy, b[0].perturbed_xy)


def test_random_seed_determinism():
    r1 = _run("random")
    r2 = _run("random")
    r3 = attack_trajectories([_traj(1), _traj(2, n_states=3)], StubDisc(), _profiles(),
                             "random", epsilon=2.0, seed=1)
    np.testing.assert_array_equal(r1[0].perturbed_xy, r2[0].perturbed_xy)
    assert not np.array_equal(r1[0].perturbed_xy, r3[0].perturbed_xy)


def test_originals_never_mutated():
    trajs = [_traj(1)]
    before = [(s.x_grid, s.y_grid, s.time_bucket, s.day_index) for s in trajs[0].states]
    _ = attack_trajectories(trajs, StubDisc(), _profiles(), "ifgsm", epsilon=2.0,
                            step=0.1, max_iterations=4, seed=0)
    after = [(s.x_grid, s.y_grid, s.time_bucket, s.day_index) for s in trajs[0].states]
    assert before == after


def test_batched_equals_sequential():
    trajs = [_traj(i, n_states=3 + (i % 4)) for i in range(6)]
    big = attack_trajectories(trajs, StubDisc(), _profiles(), "ifgsm", epsilon=2.0,
                              step=0.1, max_iterations=6, seed=0, batch_size=6)
    one = attack_trajectories(trajs, StubDisc(), _profiles(), "ifgsm", epsilon=2.0,
                              step=0.1, max_iterations=6, seed=0, batch_size=1)
    for a, b in zip(big, one):
        np.testing.assert_allclose(a.perturbed_xy, b.perturbed_xy, atol=1e-6)


def test_ifgsm_attack_reduces_p():
    out = _run("ifgsm")[0]
    assert out.final_p < 0.5  # StubDisc gives p=0.5 at zero perturbation


def test_padding_states_untouched_and_shapes():
    trajs = [_traj(1, n_states=5), _traj(2, n_states=2)]
    outs = _run("ifgsm", trajs=trajs)
    assert outs[0].perturbed_xy.shape == (5, 2)
    assert outs[1].perturbed_xy.shape == (2, 2)  # no padding rows leak out


def test_random_mode_batched_equals_sequential():
    trajs = [_traj(i, n_states=3 + (i % 4)) for i in range(6)]
    big = attack_trajectories(trajs, StubDisc(), _profiles(), "random", epsilon=2.0,
                              seed=0, batch_size=6)
    one = attack_trajectories(trajs, StubDisc(), _profiles(), "random", epsilon=2.0,
                              seed=0, batch_size=1)
    for a, b in zip(big, one):
        np.testing.assert_array_equal(a.perturbed_xy, b.perturbed_xy)


def test_random_mode_multi_chunk_invariance():
    trajs = [_traj(i, n_states=3 + (i % 4)) for i in range(6)]
    a4 = attack_trajectories(trajs, StubDisc(), _profiles(), "random", epsilon=2.0,
                             seed=0, batch_size=4)
    a6 = attack_trajectories(trajs, StubDisc(), _profiles(), "random", epsilon=2.0,
                             seed=0, batch_size=6)
    for x, y in zip(a4, a6):
        np.testing.assert_array_equal(x.perturbed_xy, y.perturbed_xy)


def test_random_start_false_is_stationary_on_symmetric_stub():
    out = attack_trajectories([_traj(1)], StubDisc(), _profiles(), "ifgsm",
                              epsilon=2.0, step=0.1, max_iterations=8, patience=3,
                              convergence_tol=0.0, seed=0, random_start=False)[0]
    orig = np.array([[s.x_grid, s.y_grid] for s in _traj(1).states])
    np.testing.assert_array_equal(out.perturbed_xy, orig)
    assert out.final_p == 0.5


def test_fgsm_equals_ifgsm_single_fullstep_no_random_start():
    a = attack_trajectories([_traj(1)], StubDisc(), _profiles(), "fgsm",
                            epsilon=2.0, step=0.1, max_iterations=8, seed=0,
                            random_start=False)
    b = attack_trajectories([_traj(1)], StubDisc(), _profiles(), "ifgsm",
                            epsilon=2.0, step=2.0, max_iterations=1, seed=0,
                            random_start=False)
    np.testing.assert_array_equal(a[0].perturbed_xy, b[0].perturbed_xy)


def test_fgsm_applies_signed_step():
    # Asymmetric disc: p rises with x2, so one full-budget descent step moves
    # EVERY coordinate by -epsilon. random_start=False -> delta starts at 0, so
    # the ONLY signal that the step was applied (and kept, not discarded like the
    # init) is the trajectory actually moving by exactly epsilon in -x/-y.
    traj = _traj(1, n_states=5)
    orig = np.array([[s.x_grid, s.y_grid] for s in traj.states])
    out = attack_trajectories([traj], AsymDisc(), _profiles(), "fgsm",
                              epsilon=2.0, step=0.1, max_iterations=8, patience=3,
                              convergence_tol=0.0, seed=0, random_start=False)[0]
    np.testing.assert_allclose(out.perturbed_xy, orig - 2.0, atol=1e-6)
    assert out.final_p < 0.5   # p(original) = sigmoid(0) = 0.5 (identical inputs)


def test_ifgsm_descends_below_fgsm_or_equal():
    # Iterative descent must reach a p no worse than the single full-budget step
    # on the same asymmetric stub (both saturate at the epsilon-ball corner).
    traj = _traj(1, n_states=5)
    fgsm = attack_trajectories([traj], AsymDisc(), _profiles(), "fgsm",
                               epsilon=2.0, step=0.1, max_iterations=8, patience=3,
                               convergence_tol=0.0, seed=0, random_start=False)[0]
    ifgsm = attack_trajectories([traj], AsymDisc(), _profiles(), "ifgsm",
                                epsilon=2.0, step=0.5, max_iterations=8, patience=3,
                                convergence_tol=0.0, seed=0, random_start=False)[0]
    assert ifgsm.final_p <= fgsm.final_p + 1e-6


def test_discretize_rounds_clamps_and_freezes_time():
    traj = _traj(1, n_states=3, x0=0.4, y0=88.6)   # y walks past the 90-row edge
    out = AttackOutcome(trajectory_id=1,
                        perturbed_xy=np.array([[-1.2, 88.6], [0.4, 89.7], [2.6, 91.2]]),
                        final_p=0.1, iterations_run=3,
                        delta=np.zeros((3, 2)))
    d = discretize_outcome(traj, out, grid_dims=(48, 90))
    xs = [(s.x_grid, s.y_grid) for s in d.states]
    assert xs == [(0.0, 89.0), (0.0, 89.0), (3.0, 89.0)]        # round + clamp in-grid
    assert all(float(v[0]).is_integer() and float(v[1]).is_integer() for v in xs)
    assert [s.time_bucket for s in d.states] == [s.time_bucket for s in traj.states]
    assert d.trajectory_id == traj.trajectory_id and d.driver_id == traj.driver_id


def test_adjacency_violation_rate_crafted():
    ok = _traj(1, n_states=3)                       # unit steps -> compliant
    bad = Trajectory(trajectory_id=2, driver_id=7, states=[
        TrajectoryState(1.0, 1.0, 3, 1), TrajectoryState(4.0, 1.0, 3, 1)])  # dx=3
    assert adjacency_violation_rate([ok]) == 0.0
    assert adjacency_violation_rate([bad]) == 1.0
    assert adjacency_violation_rate([ok, bad]) == 0.5


def test_package_arm_roundtrip(tmp_path):
    trajs = [_traj(1), _traj(2, n_states=3)]
    outs = attack_trajectories(trajs, StubDisc(), _profiles(), "random",
                               epsilon=2.0, seed=0)
    arm_dir = package_arm(trajs, outs, tmp_path / "arm",
                          arm_config={"mode": "random", "epsilon": 2.0, "seed": 0})
    with open(arm_dir / "histories.pkl", "rb") as f:
        hists = pickle.load(f)
    assert len(hists) == 2 and isinstance(hists[0], ModificationHistory)
    assert hists[0].original.states[0].x_grid == trajs[0].states[0].x_grid
    assert float(hists[0].modified.states[-1].x_grid).is_integer()
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert meta["arm"]["mode"] == "random"
    assert meta["arm"]["n_edited"] == 2
    assert isinstance(meta["arm"]["mean_final_p"], float)
    assert math.isfinite(meta["arm"]["mean_final_p"])
    assert isinstance(meta["arm"]["mean_iterations"], float)
    assert meta["arm"]["mean_iterations"] >= 0
    assert 0.0 <= meta["arm"]["adjacency_violation_rate"] <= 1.0
