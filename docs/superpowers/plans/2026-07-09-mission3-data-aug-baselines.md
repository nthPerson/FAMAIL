# Mission 3 — Data-Augmentation Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the vanilla ST-iFGSM / plain-FGSM / random-jitter baseline editors (batched attack on the frozen HuMID discriminator over continuous float-grid seeking trajectories), score their edited corpora with the existing fidelity + fairness + external-metrics harnesses, and assemble the 5-row comparison table.

**Architecture:** A standalone module pair in `famail_temporal/baselines/` — a pure attack engine (`stifgsm_baseline.py`) and a runner CLI (`run_stifgsm_baseline.py`) — that emits results dirs in the standard `histories.pkl` + `metrics.json` format so every existing harness consumes them unchanged. Zero edits to `famail_temporal/algorithm/` or `famail_temporal/evaluation/runner.py`.

**Tech Stack:** Python 3 / PyTorch (the repo's existing stack); pytest; existing `famail_temporal` modules (`fidelity/`, `baselines/fidelity_eval.py`, `baselines/external_fairness*`, `data/loader.py`).

**Spec:** `docs/superpowers/specs/2026-07-09-mission3-data-aug-baselines-design.md` (read §2 decisions + §3 components first).

## Global Constraints

- Worktree: `/home/robert/FAMAIL/.claude/worktrees/mission3-baselines`, branch `mission3-baselines`. Run everything from the worktree root; `git` commands operate on this branch. **No merge to main.**
- **Frozen-algorithm gate:** `git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py` must stay **empty** through the whole plan.
- Parity constants (copy from `famail_temporal/config.py`, do not hardcode magic numbers): `EPSILON_BALL=2.0`, `STEP_SIZE_ALPHA=0.1`, `MAX_ITERATIONS=50`, `PATIENCE=10`, `CONVERGENCE_TOL`.
- Perturb ONLY the `x_grid, y_grid` floats of seeking states; `time_bucket` / `day_index` bitwise frozen. Originals never mutated.
- Default seed 0. Every results dir carries a config snapshot (mode, ε, step, iterations, seed, git context).
- Tests are CPU-only (tiny synthetic fixtures + a stub discriminator); no GPU needed for any plan task. The real corpus runs are documented commands executed AFTER the α-sweep frees the GPU — they are not plan tasks.
- Test command form: `python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py -v` (house layout: baseline tests live in `famail_temporal/baselines/tests/`).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Key existing interfaces (verified 2026-07-09 — consume, don't reimplement)

- `Trajectory(trajectory_id, driver_id, states: List[TrajectoryState], metadata)`; pickup = `states[-1]`; `TrajectoryState(x_grid: float, y_grid: float, time_bucket: int, day_index: int)` (`famail_temporal/utils/trajectory.py:96-129`).
- `ModificationHistory(original: Trajectory, modified: Trajectory, iterations=[], converged, total_iterations, final_objective, best_iteration, best_objective)` (`famail_temporal/algorithm/modifier.py:69`). Arm dirs write a `List[ModificationHistory]` pickle — that is the entire edit-dir contract `build_edited_pickup_3d(bundle, edit_dir)` needs (`famail_temporal/baselines/external_fairness_io.py:68`).
- Discriminator: `load_discriminator(path)` (`famail_temporal/fidelity/checkpoint.py:21`); checkpoint path = `config.DISCRIMINATOR_CHECKPOINT_DIR / config.DISCRIMINATOR_CHECKPOINT_FILENAME` (city-switched by `FAMAIL_CITY`). Call: `disc(x1, x2, mask1=, mask2=, profile_1=, profile_2=)` → same-agent probability; `x` is `[B, N, L, 4]` (N-set branches; N=1 works), features `[x_grid, y_grid, time_bucket, day_index]`.
- Fidelity-A: `fidelity_eval.humid_identity_fidelity(disc, pairs, batch_size=64, device=)`; pair = `((set_l [N,L,4], mask_l [N,L], prof_l [11]), (set_r, mask_r, prof_r))`; gate = `fidelity_eval.identity_validation_gate(disc, matched_pairs=, mismatched_pairs=, ...)`; feature helpers `real_to_disc_tensor(traj)`, `build_identity_branch(...)` (`famail_temporal/baselines/fidelity_eval.py:35,361,387,428,444`).
- Fidelity-B: `fidelity_eval.trajectory_statistics`, `stat_ranges`, `distributional_fidelity`, `terminal_cell_distribution_js` (same module, lines 70-222).
- Fairness rescore: `famail_temporal/baselines/metrics.py::data_level_fairness` (read its signature in Task 3 before wiring; it is the established pickup-grid → (F_spatial, F_causal) path used by `run_data_pareto`).
- Test fixtures: `famail_temporal/baselines/tests/_helpers.py::make_traj_at(cx, cy, t_block, traj_id)` + `active_units()`.
- Headline edit set: `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered/histories.pkl` (9,885 `ModificationHistory`; the baselines re-edit `h.original` trajectories).
- External metrics: `python -m famail_temporal.baselines.run_external_fairness --edit-dir <arm_dir> --dataset <label>`; supply recount: `python -m famail_temporal.analysis.supply_recount --edit-dir <arm_dir> --city shenzhen --persist-grids`.

---

### Task 1: Attack engine — `attack_trajectories` (ifgsm / fgsm / random)

**Files:**
- Create: `famail_temporal/baselines/stifgsm_baseline.py`
- Test: `famail_temporal/baselines/tests/test_stifgsm_baseline.py`

**Interfaces:**
- Consumes: `Trajectory`/`TrajectoryState`; a discriminator with the `disc(x1, x2, mask1=, mask2=, profile_1=, profile_2=)` call shape (tests use a stub).
- Produces (Tasks 2-4 rely on these exact names):
  - `@dataclass AttackOutcome: trajectory_id: Any; perturbed_xy: np.ndarray  # (S,2) float; final_p: float; iterations_run: int; delta: np.ndarray  # (S,2)`
  - `attack_trajectories(trajectories, disc, profiles, mode, *, epsilon=None, step=None, max_iterations=None, patience=None, convergence_tol=None, seed=0, device="cpu", batch_size=256) -> List[AttackOutcome]` — `None` parity args resolve to the `config.*` constants; `profiles: Dict[driver_id, np.ndarray (11,)]`.

- [ ] **Step 1: Write the failing tests** (stub disc + invariants)

```python
"""Tests for the ST-iFGSM/FGSM/random baseline attack engine."""
import numpy as np
import pytest
import torch

from famail_temporal.baselines.stifgsm_baseline import AttackOutcome, attack_trajectories
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
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /home/robert/FAMAIL/.claude/worktrees/mission3-baselines && python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py -v`
Expected: collection error / ImportError — `stifgsm_baseline` does not exist.

- [ ] **Step 3: Implement the engine**

```python
"""Baseline trajectory editors: vanilla ST-iFGSM, plain FGSM, random jitter.

Standalone Mission-3 module (Meeting-41 P0 #3). Attacks the frozen HuMID
discriminator on (original, perturbed) same-driver pairs over the CONTINUOUS
float-grid seeking states — the discriminator's native input space — with a
per-coordinate cumulative L-inf budget. Deliberately independent of
famail_temporal/algorithm/ (the frozen editor is untouched).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.utils.trajectory import Trajectory


@dataclass
class AttackOutcome:
    trajectory_id: Any
    perturbed_xy: np.ndarray   # (S, 2) float64 — attacked x,y per state
    final_p: float             # discriminator P(same driver) at the kept iterate
    iterations_run: int
    delta: np.ndarray          # (S, 2) applied perturbation (== perturbed - original)


def _features(traj: Trajectory) -> np.ndarray:
    return np.array(
        [[s.x_grid, s.y_grid, float(s.time_bucket), float(s.day_index)]
         for s in traj.states],
        dtype=np.float32,
    )


def _batch(trajs, profiles, device):
    """Pad to (B, 1, Lmax, 4) N=1 identity branches + masks + profiles."""
    lens = [len(t.states) for t in trajs]
    lmax = max(lens)
    b = len(trajs)
    x = torch.zeros(b, 1, lmax, 4, dtype=torch.float32, device=device)
    m = torch.zeros(b, 1, lmax, dtype=torch.bool, device=device)
    p = torch.zeros(b, 11, dtype=torch.float32, device=device)
    for i, t in enumerate(trajs):
        f = torch.from_numpy(_features(t)).to(device)
        x[i, 0, : lens[i]] = f
        m[i, 0, : lens[i]] = True
        p[i] = torch.from_numpy(np.asarray(profiles[t.driver_id], dtype=np.float32)).to(device)
    return x, m, p, lens


def attack_trajectories(
    trajectories: List[Trajectory],
    disc: torch.nn.Module,
    profiles: Dict[Any, np.ndarray],
    mode: str,
    *,
    epsilon: float | None = None,
    step: float | None = None,
    max_iterations: int | None = None,
    patience: int | None = None,
    convergence_tol: float | None = None,
    seed: int = 0,
    device: str = "cpu",
    batch_size: int = 256,
) -> List[AttackOutcome]:
    if mode not in ("ifgsm", "fgsm", "random"):
        raise ValueError(f"unknown mode '{mode}'")
    epsilon = config.EPSILON_BALL if epsilon is None else float(epsilon)
    step = config.STEP_SIZE_ALPHA if step is None else float(step)
    max_iterations = config.MAX_ITERATIONS if max_iterations is None else int(max_iterations)
    patience = config.PATIENCE if patience is None else int(patience)
    convergence_tol = (config.CONVERGENCE_TOL if convergence_tol is None
                       else float(convergence_tol))
    if mode == "fgsm":                    # single full-budget signed step
        max_iterations, step = 1, epsilon

    dev = torch.device(device)
    disc = disc.to(dev)
    disc.train(False)
    for prm in disc.parameters():
        prm.requires_grad_(False)

    outcomes: List[AttackOutcome] = []
    for start in range(0, len(trajectories), batch_size):
        chunk = trajectories[start : start + batch_size]
        x_orig, mask, prof, lens = _batch(chunk, profiles, dev)
        bsz, _, lmax, _ = x_orig.shape
        mask_f = mask.unsqueeze(-1).float()          # (B,1,L,1) freeze padding

        if mode == "random":
            g = torch.Generator(device="cpu").manual_seed(seed)
            signs = torch.randint(0, 2, (bsz, 1, lmax, 2), generator=g,
                                  dtype=torch.float32).mul_(2).sub_(1).to(dev)
            delta = (signs * epsilon) * mask_f
            x_adv = x_orig.clone()
            x_adv[..., :2] = x_orig[..., :2] + delta
            with torch.no_grad():
                p = disc(x_orig, x_adv, mask1=mask, mask2=mask,
                         profile_1=prof, profile_2=prof).reshape(-1)
            best_delta, best_p, iters = delta, p, torch.ones_like(p, dtype=torch.long)
        else:
            delta = torch.zeros(bsz, 1, lmax, 2, device=dev, requires_grad=True)
            best_p = torch.full((bsz,), float("inf"), device=dev)
            best_delta = torch.zeros_like(delta)
            iters = torch.zeros(bsz, dtype=torch.long, device=dev)
            stall = torch.zeros(bsz, dtype=torch.long, device=dev)
            for _ in range(max_iterations):
                x_adv = x_orig.clone()
                x_adv[..., :2] = x_orig[..., :2] + delta * mask_f
                p = disc(x_orig, x_adv, mask1=mask, mask2=mask,
                         profile_1=prof, profile_2=prof).reshape(-1)
                loss = p.sum()                      # descend P(same driver)
                grad = torch.autograd.grad(loss, delta)[0]
                with torch.no_grad():
                    live = stall < patience
                    improved = p < (best_p - convergence_tol)
                    upd = improved & live
                    best_delta[upd] = delta[upd]
                    best_p = torch.where(improved & live, p, best_p)
                    stall = torch.where(improved & live, torch.zeros_like(stall),
                                        stall + live.long())
                    iters += live.long()
                    d_new = (delta - step * grad.sign()).clamp_(-epsilon, epsilon)
                    delta.data = torch.where(live.view(-1, 1, 1, 1), d_new, delta.data)
                    delta.data *= mask_f            # padding stays zero
                if not bool((stall < patience).any()):
                    break
            best_delta = best_delta.detach() * mask_f

        x_final = x_orig.clone()
        x_final[..., :2] = x_orig[..., :2] + best_delta
        for i, traj in enumerate(chunk):
            s = lens[i]
            outcomes.append(AttackOutcome(
                trajectory_id=traj.trajectory_id,
                perturbed_xy=x_final[i, 0, :s, :2].detach().cpu().double().numpy(),
                final_p=float(best_p[i]),
                iterations_run=int(iters[i]),
                delta=best_delta[i, 0, :s].detach().cpu().double().numpy(),
            ))
    return outcomes
```

Note on the fgsm≡ifgsm test: with `mode="fgsm"` the function internally sets
`max_iterations=1, step=epsilon`, so the test compares it against an explicit
`ifgsm(step=epsilon, max_iterations=1)` call — they must be bitwise equal.

- [ ] **Step 4: Run tests — verify all pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/stifgsm_baseline.py famail_temporal/baselines/tests/test_stifgsm_baseline.py
git commit -m "feat(mission3): batched ST-iFGSM/FGSM/random baseline attack engine

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Discretize + package arm results dirs

**Files:**
- Modify: `famail_temporal/baselines/stifgsm_baseline.py` (append)
- Test: `famail_temporal/baselines/tests/test_stifgsm_baseline.py` (append)

**Interfaces:**
- Consumes: `AttackOutcome` (Task 1); `ModificationHistory` (`famail_temporal/algorithm/modifier.py:69` — import is read-only use of a dataclass, NOT an algorithm change).
- Produces:
  - `discretize_outcome(traj, outcome, grid_dims) -> Trajectory` — rounded/clamped integer-coordinate copy.
  - `adjacency_violation_rate(trajs) -> float` — fraction of trajectories with any consecutive step `max(|dx|,|dy|) > 1`.
  - `package_arm(originals, outcomes, out_dir, arm_config) -> Path` — writes `histories.pkl` (`List[ModificationHistory]`) + `metrics.json` skeleton (config snapshot, per-arm attack stats, adjacency rate).

- [ ] **Step 1: Write the failing tests**

```python
import json
import pickle

from famail_temporal.algorithm.modifier import ModificationHistory
from famail_temporal.baselines.stifgsm_baseline import (
    adjacency_violation_rate, discretize_outcome, package_arm,
)


def test_discretize_rounds_clamps_and_freezes_time(tmp_path):
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
    assert "adjacency_violation_rate" in meta["arm"]
```

- [ ] **Step 2: Run tests — verify the new ones fail**

Run: `python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py -v -k "discretize or adjacency or package"`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement**

```python
# append to stifgsm_baseline.py
import copy
import json
import pickle
from pathlib import Path

from famail_temporal.algorithm.modifier import ModificationHistory


def discretize_outcome(traj: Trajectory, outcome: AttackOutcome,
                       grid_dims) -> Trajectory:
    """Round attacked coords to grid ints, clamp in-grid. Vanilla: NO repair."""
    gx, gy = int(grid_dims[0]), int(grid_dims[1])
    mod = copy.deepcopy(traj)
    for i, s in enumerate(mod.states):
        s.x_grid = float(min(max(round(float(outcome.perturbed_xy[i, 0])), 0), gx - 1))
        s.y_grid = float(min(max(round(float(outcome.perturbed_xy[i, 1])), 0), gy - 1))
    return mod


def adjacency_violation_rate(trajs: List[Trajectory]) -> float:
    if not trajs:
        return 0.0
    bad = 0
    for t in trajs:
        for a, b in zip(t.states, t.states[1:]):
            if max(abs(b.x_grid - a.x_grid), abs(b.y_grid - a.y_grid)) > 1:
                bad += 1
                break
    return bad / len(trajs)


def package_arm(originals: List[Trajectory], outcomes: List[AttackOutcome],
                out_dir, arm_config: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_dims = config.GRID_DIMS
    histories, modified = [], []
    for traj, out in zip(originals, outcomes):
        mod = discretize_outcome(traj, out, grid_dims)
        modified.append(mod)
        histories.append(ModificationHistory(
            original=copy.deepcopy(traj), modified=mod,
            converged=True, total_iterations=out.iterations_run,
            final_objective=out.final_p,
        ))
    with open(out_dir / "histories.pkl", "wb") as f:
        pickle.dump(histories, f)
    meta = {
        "arm": {
            **arm_config,
            "n_edited": len(histories),
            "adjacency_violation_rate": adjacency_violation_rate(modified),
            "mean_final_p": float(np.mean([o.final_p for o in outcomes])) if outcomes else float("nan"),
            "mean_iterations": float(np.mean([o.iterations_run for o in outcomes])) if outcomes else 0.0,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    return out_dir
```

- [ ] **Step 4: Run the full test file — verify all pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/stifgsm_baseline.py famail_temporal/baselines/tests/test_stifgsm_baseline.py
git commit -m "feat(mission3): discretize + package arm results dirs (edit-dir contract)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Runner CLI with fairness rescoring

**Files:**
- Create: `famail_temporal/baselines/run_stifgsm_baseline.py`
- Test: `famail_temporal/baselines/tests/test_run_stifgsm_baseline.py`

**Interfaces:**
- Consumes: Tasks 1-2 (`attack_trajectories`, `package_arm`); `DataBundle.load()`; `load_discriminator`; the headline `histories.pkl`; `famail_temporal/baselines/metrics.py::data_level_fairness` (READ its exact signature first and adapt the call — it is the same rescore used by `run_data_pareto`); `external_fairness_io.build_edited_pickup_3d`.
- Produces: CLI `python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir <headline_dir> --mode {ifgsm,fgsm,random} [--epsilon --step --max-iterations --seed --device --batch-size --out-root --limit N]`, and `run_baseline(args) -> Path` (importable for tests). Output: `<out-root>/<ts>_baseline_<mode>_<city>/` with `histories.pkl` + `metrics.json` including `fairness: {f_spatial_before/after, f_causal_before/after, deltas}`.

Structure (implement; test via the synthetic path below):
1. Parse args; resolve city from `FAMAIL_CITY` (label only — loading already obeys it).
2. `bundle = DataBundle.load()`; `disc = load_discriminator(config.DISCRIMINATOR_CHECKPOINT_DIR / config.DISCRIMINATOR_CHECKPOINT_FILENAME)`.
3. Read `--edit-dir` `histories.pkl`; collect `h.original.trajectory_id` (ordered, deduped); map to bundle trajectories by `trajectory_id`; `--limit N` truncates (smoke runs).
4. Driver profiles: build the `Dict[driver_id, (11,)]` from the same profile source `fidelity_eval.build_identity_branch` uses (read that function; reuse its accessor rather than inventing one).
5. `attack_trajectories(...)` → `package_arm(...)`.
6. Rescore: `before = data_level_fairness(bundle.pickup_3d, ...)`; `after = data_level_fairness(build_edited_pickup_3d(bundle, arm_dir), ...)` — exact call per the read signature; write into `metrics.json["fairness"]`.
7. Print the one-line summary (house style): mode, n, ΔF_causal, ΔF_spatial, adjacency rate, mean final_p.

- [ ] **Step 1: Write the failing test** (synthetic end-to-end; no real bundle)

```python
"""CLI-level test: run_baseline on a synthetic bundle via monkeypatching."""
import json
import pickle

import numpy as np
import pytest

import famail_temporal.baselines.run_stifgsm_baseline as rb
from famail_temporal.algorithm.modifier import ModificationHistory
from famail_temporal.baselines.tests.test_stifgsm_baseline import (
    StubDisc, _profiles, _traj,
)


class _StubBundle:
    def __init__(self, trajs):
        self.trajectories = trajs
        self.pickup_3d = np.ones((48, 90, 24), dtype=np.float32)


def test_run_baseline_end_to_end(tmp_path, monkeypatch):
    trajs = [_traj(1), _traj(2, n_states=3)]
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    with open(seed_dir / "histories.pkl", "wb") as f:
        pickle.dump([ModificationHistory(original=t, modified=t) for t in trajs], f)

    monkeypatch.setattr(rb, "_load_bundle", lambda: _StubBundle(trajs))
    monkeypatch.setattr(rb, "_load_disc", lambda device: StubDisc())
    monkeypatch.setattr(rb, "_driver_profiles", lambda bundle: _profiles())
    monkeypatch.setattr(
        rb, "_rescore",
        lambda bundle, arm_dir: {"f_spatial_before": 0.1, "f_spatial_after": 0.1,
                                 "f_causal_before": 0.8, "f_causal_after": 0.8},
    )

    arm_dir = rb.run_baseline(rb.parse_args([
        "--edit-dir", str(seed_dir), "--mode", "random",
        "--out-root", str(tmp_path), "--seed", "0", "--device", "cpu",
    ]))
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert meta["arm"]["mode"] == "random" and meta["arm"]["n_edited"] == 2
    assert "fairness" in meta and "f_causal_before" in meta["fairness"]
    with open(arm_dir / "histories.pkl", "rb") as f:
        assert len(pickle.load(f)) == 2
```

- [ ] **Step 2: Run — verify it fails** (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `run_stifgsm_baseline.py`**

Seams the test monkeypatches MUST exist as module-level functions: `_load_bundle()`, `_load_disc(device)`, `_driver_profiles(bundle)`, `_rescore(bundle, arm_dir)`. Real implementations:

```python
def _load_bundle():
    from famail_temporal.data.loader import DataBundle
    return DataBundle.load()

def _load_disc(device):
    from famail_temporal.fidelity.checkpoint import load_discriminator
    path = config.DISCRIMINATOR_CHECKPOINT_DIR / config.DISCRIMINATOR_CHECKPOINT_FILENAME
    return load_discriminator(path).to(device)

def _driver_profiles(bundle):
    # Reuse the exact profile source fidelity_eval's identity branches use —
    # read fidelity_eval.build_identity_branch (fidelity_eval.py:361) and call
    # the same accessor; return {driver_id: np.ndarray (11,)}.
    ...

def _rescore(bundle, arm_dir):
    # data_level_fairness (baselines/metrics.py) on bundle.pickup_3d (before)
    # and build_edited_pickup_3d(bundle, arm_dir) (after); return the flat dict
    # {"f_spatial_before": ..., "f_spatial_after": ..., "f_causal_before": ...,
    #  "f_causal_after": ...} the CLI writes into metrics.json["fairness"].
    ...
```

`main(argv)`/`parse_args(argv)`/`run_baseline(args)` wire steps 1-7 above; `run_baseline` returns the arm dir. The `...` bodies are filled by READING the two named functions and calling them — the implementer must not invent a parallel profile source or fairness path; if either interface cannot support the call, STOP and surface it (do not fork the logic).

- [ ] **Step 4: Run both test files — verify pass**

Run: `python -m pytest famail_temporal/baselines/tests/test_stifgsm_baseline.py famail_temporal/baselines/tests/test_run_stifgsm_baseline.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_stifgsm_baseline.py famail_temporal/baselines/tests/test_run_stifgsm_baseline.py
git commit -m "feat(mission3): baseline runner CLI with fairness rescoring

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Fidelity-A/B scoring for an arm dir

**Files:**
- Modify: `famail_temporal/baselines/run_stifgsm_baseline.py` (add `score_fidelity(arm_dir, ...)` + `--score-fidelity` flag)
- Test: `famail_temporal/baselines/tests/test_run_stifgsm_baseline.py` (append)

**Interfaces:**
- Consumes: `fidelity_eval.humid_identity_fidelity`, `identity_validation_gate`, `real_to_disc_tensor`, `build_identity_branch`, `trajectory_statistics`, `stat_ranges`, `distributional_fidelity`, `terminal_cell_distribution_js` (all in `famail_temporal/baselines/fidelity_eval.py`); an arm dir's `histories.pkl`.
- Produces: `score_fidelity(arm_dir, disc, bundle, *, device) -> dict` writing `metrics.json["fidelity"] = {"fidelity_a": {...}, "gate": {...}, "fidelity_b": {...}}`. Pair construction mirrors the established protocol: matched pairs = (original branch, edited branch) same driver via `build_identity_branch`; mismatched = original branches across different drivers (gate anchor). READ `run_level1_table_v2.py:540-560` and reuse its pair-building calls verbatim-in-shape; if the helper needs inputs unavailable from an arm dir, STOP and surface it.

- [ ] **Step 1: Failing test** — stub-disc: `score_fidelity` on the Task-3 synthetic arm dir returns dict with keys `fidelity_a.mean`, `gate.passed`, `fidelity_b` (JS keys), and `metrics.json` gains the `fidelity` block:

```python
def test_score_fidelity_writes_block(tmp_path, monkeypatch):
    # ... reuse the test_run_baseline_end_to_end setup through run_baseline(...)
    arm_dir = ...  # as in the previous test
    out = rb.score_fidelity(arm_dir, StubDisc(), _StubBundle([_traj(1), _traj(2, n_states=3)]), device="cpu")
    assert "fidelity_a" in out and "fidelity_b" in out and "gate" in out
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert "fidelity" in meta
```

- [ ] **Step 2: Run — verify fails** (AttributeError).
- [ ] **Step 3: Implement** `score_fidelity` per the Interfaces block (pairs from histories originals/modifieds; Fidelity-B stats over modified vs original trajectory lists using `trajectory_statistics` + `distributional_fidelity` + `terminal_cell_distribution_js`).
- [ ] **Step 4: Run both test files — verify pass** (12 passed).
- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/run_stifgsm_baseline.py famail_temporal/baselines/tests/test_run_stifgsm_baseline.py
git commit -m "feat(mission3): Fidelity-A/B scoring for baseline arm dirs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Comparison-table assembler

**Files:**
- Create: `famail_temporal/baselines/assemble_baseline_table.py`
- Test: `famail_temporal/baselines/tests/test_assemble_baseline_table.py`

**Interfaces:**
- Consumes: arm `metrics.json` files (Tasks 2-4 schema) + the FAMAIL headline row sourced from `PAPER/supply-lift/data/shz_primary_filtered_metrics.json` + its fidelity numbers (pass via a small JSON argument file — do NOT recompute headline numbers).
- Produces: CLI `python -m famail_temporal.baselines.assemble_baseline_table --arm-dirs <d1> <d2> ... --famail-json <path> --raw-json <path> --out <dir>` writing `baseline_table.md` + `baseline_table.json` with rows [raw, FAMAIL, ifgsm, fgsm, random] × columns [Fidelity-A, gate, Fidelity-B(JS), ΔF_causal, ΔF_spatial, adjacency-violation %, mean final_p, n].

- [ ] **Step 1: Failing test** — feed three tiny synthetic arm dirs (write minimal `metrics.json` fixtures inline) + famail/raw JSON stubs; assert the markdown contains all five row labels and the json has 5 rows with the delta columns computed (`after - before`).
- [ ] **Step 2: Run — verify fails.**
- [ ] **Step 3: Implement** (pure JSON-in → md/json-out; no torch).
- [ ] **Step 4: Run — verify pass.**
- [ ] **Step 5: Commit**

```bash
git add famail_temporal/baselines/assemble_baseline_table.py famail_temporal/baselines/tests/test_assemble_baseline_table.py
git commit -m "feat(mission3): 5-row baseline comparison table assembler

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Lit-scan memo (non-blocking, gated)

**Files:**
- Create: `famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md`

Web-verify every citation (arXiv/DOI/publisher — the Mission-2 audit standard; no unverified quotes/metadata). Content: 3-5 candidate trajectory data-augmentation baselines (search themes: trajectory augmentation for deep mobility models; perturbation-based augmentation for GPS/spatio-temporal data; fairness-aware data augmentation for mobility; counterfactual trajectory generation). For each: verified citation, one-paragraph method summary, applicability to FAMAIL's discrete-grid seeking corpus, adopt/defer recommendation with cost estimate. End with: "**Decision gate: none of these are built without an explicit user go-ahead.**"

- [ ] **Step 1: Research + draft the memo** (WebSearch/WebFetch verification per citation).
- [ ] **Step 2: Verify no unverified metadata** — every entry carries its checked source URL.
- [ ] **Step 3: Commit**

```bash
git add famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md
git commit -m "docs(mission3): verified lit-scan memo of candidate data-aug baselines

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Suite gate + run-book

**Files:**
- Modify: `famail_temporal/baselines/STATUS.md` (append a Mission-3 section)

- [ ] **Step 1: Full-suite + frozen-algorithm gate**

```bash
python -m pytest famail_temporal/ -q 2>&1 | tail -3          # whole suite still green
git diff main -- famail_temporal/algorithm/ famail_temporal/evaluation/runner.py | wc -l   # expect 0
```
Expected: suite passes (same count as `main` + the new tests); diff lines = 0. Note: Task 2 *imports* `ModificationHistory` from `algorithm/modifier.py` but modifies nothing — the diff gate proves it.

- [ ] **Step 2: Append the run-book to `STATUS.md`** — a "Mission 3 baselines (built, awaiting GPU)" section with the three exact arm commands against the real headline dir:

```bash
# after the alpha-sweep frees the GPU (driver: famail_temporal/results/alpha_sweep/driver.sh --status)
H=famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered
for MODE in ifgsm fgsm random; do
  python -m famail_temporal.baselines.run_stifgsm_baseline \
    --edit-dir "$H" --mode "$MODE" --seed 0 --device auto --score-fidelity
done
# then per arm dir: run_external_fairness --edit-dir <arm_dir> --dataset baseline-<mode>
#                   supply_recount --edit-dir <arm_dir> --city shenzhen --persist-grids
# then assemble_baseline_table --arm-dirs <...> --famail-json <...> --raw-json <...>
```

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/baselines/STATUS.md
git commit -m "docs(mission3): run-book + suite/frozen-algorithm gate record

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review (author check against the spec)

**Spec coverage:** engine + 3 arms (Task 1 ↔ §3.1); discretize/no-repair/adjacency-rate/packaging (Task 2 ↔ §3.2); CLI + edit-set + rescore (Task 3 ↔ §3.3); Fidelity-A/B (Task 4 ↔ §3.4); external metrics + supply recount are existing CLIs invoked in the Task-7 run-book (↔ §3.4 — no new code needed); comparison table (Task 5 ↔ §3.4/§8); lit-scan gated (Task 6 ↔ §3.5); tests (Tasks 1-5 ↔ §5); frozen-algorithm gate + GPU deferral (Task 7 ↔ §6/§7/§8). Hypotheses (§4) are measurement-time, not build-time.

**Placeholders:** the two `...` bodies in Task 3 are explicit READ-THEN-CALL directives naming the exact source functions and the stop-and-surface rule — deliberate, since inventing those call sites without reading them is exactly how parallel logic forks. No TBD/TODO elsewhere.

**Type consistency:** `AttackOutcome`/`attack_trajectories` names match across Tasks 1-3; `package_arm` output schema (`metrics.json["arm"]`) matches what Tasks 3-5 read; the monkeypatch seams (`_load_bundle`, `_load_disc`, `_driver_profiles`, `_rescore`) are defined in Task 3's implementation step.
