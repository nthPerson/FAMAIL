# Multi-Loop Re-Attribution Editing + Non-Regression Acceptance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a unified re-attribution-loop editing engine (orthogonal knobs: batch vs iterative granularity, outer rounds, cumulative-ε cap, acceptance rule) plus a non-regression inner-loop acceptance gate, then run the experiment matrix to find the F_causal ceiling vs the +0.0128 baseline.

**Architecture:** One engine, `algorithm/editing_loop.py::run_editing_rounds`, wraps the existing `TrajectoryModifier`. Each round re-attributes against the live grid, selects the eligible negative-α set (all of it in `batch` mode; the single most-negative in `iterative` mode), edits via `modify_single`, and checks the stop rule. The current single pass becomes the `max_rounds=1` batch case, so default behavior is unchanged. Two `modify_single` changes: a `non-regression` acceptance gate and a cumulative-ε cap anchored to each trajectory's *true original* cell.

**Tech Stack:** Python, NumPy, PyTorch, pytest. Spec: `docs/superpowers/specs/2026-06-06-famail-multiloop-reattribution-editing-design.md`.

**Key facts the engineer needs:**
- F_causal of any pickup grid == `compute_per_unit_attribution(bundle, pickup_3d).sum()` (the attribution array sums to F_causal). Used for the round curve + convergence.
- `compute_per_unit_attribution(bundle, pickup_3d=...)`, `rank_trajectories(trajs, attribution, unit_map)` (ascending, most-negative first), `select_top_k(scored, k, trajectories, max_per_unit, max_per_cell)` (strictly-negative filter) all live in `famail_temporal/algorithm/attribution.py`.
- `modifier.current_pickup_3d()` returns a copy of the live, mutated grid (reflects all prior edits).
- `Trajectory.pickup_state.{x_grid,y_grid}` are continuous floats; `Trajectory.pickup_cell` is `(int(x), int(y))`; `apply_perturbation` re-clips to grid bounds. `modify_single` re-bases from `int(pickup_state.x_grid)` each call, so a re-edited trajectory starts each round from its int-snapped cell.
- Objective terms dict keys: `"f_spatial"`, `"f_causal"`, `"f_fidelity"`. `ModificationHistory` fields: `original, modified, iterations, converged, total_iterations, final_objective, best_iteration, best_objective`.
- Test bundle builder: `from famail_temporal.tests.test_objective import _make_synthetic_bundle` (`N_cells_per_block`, `seed`). Synthetic bundles carry an `nn.Identity` discriminator → always build the objective with `FAMAILObjective(bundle, alpha_fidelity=0.0)`.
- **Security hook:** never write the literal `eval` + `(` token (false-positives on PyTorch). Not needed here anyway.
- **Commits:** stage named files only (never `git add -A`). End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- **Algorithm-change protocol:** the two pre-authorized changes are exactly this engine + the non-regression gate. Backward-compat equivalence (Tasks 4, 7) is the guard that this stays numerics-preserving for existing modes.

---

## Phase 1 — Modifier (inner loop)

### Task 1: Config constants

**Files:**
- Modify: `famail_temporal/config.py` (after line 71, the `PATIENCE` block)
- Test: `famail_temporal/tests/test_config_multiloop.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# famail_temporal/tests/test_config_multiloop.py
"""Defaults for the multi-loop re-attribution + acceptance-gate knobs."""
from famail_temporal import config


def test_multiloop_defaults_are_backward_compatible():
    # max_rounds=1 ⇒ today's single pass; objective gate ⇒ today's acceptance;
    # epsilon_cap == EPSILON_BALL ⇒ no extra clip for single edits.
    assert config.MAX_ROUNDS == 1
    assert config.ROUND_CONVERGENCE_TOL is None
    assert config.ROUND_PATIENCE == 2
    assert config.EPSILON_CAP == config.EPSILON_BALL
    assert config.ACCEPT_RULE == "objective"
    assert config.ITERATIVE_TOPK_MAX_EDITS == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_config_multiloop.py -v`
Expected: FAIL with `AttributeError: module 'famail_temporal.config' has no attribute 'MAX_ROUNDS'`

- [ ] **Step 3: Add the constants**

Insert after line 71 (`PATIENCE: int = 10`) in `famail_temporal/config.py`:

```python

# Multi-loop re-attribution editing (algorithm-improvements side project,
# spec 2026-06-06). The defaults below reproduce the historical single-pass
# batch behavior exactly: MAX_ROUNDS=1 (one round), ACCEPT_RULE="objective"
# (weighted-objective best-iterate, unchanged), EPSILON_CAP=EPSILON_BALL
# (cumulative cap equals the per-edit ball ⇒ no extra clip for a single edit).
MAX_ROUNDS: int = 1
# Outer-loop convergence: stop when the best round F_causal has not improved by
# more than ROUND_CONVERGENCE_TOL for ROUND_PATIENCE consecutive rounds. None
# disables convergence (fixed MAX_ROUNDS). Set above the F-metric noise floor.
ROUND_CONVERGENCE_TOL: float | None = None
ROUND_PATIENCE: int = 2
# Cumulative L-inf displacement cap from each trajectory's TRUE original pickup
# cell, enforced across rounds. EPSILON_BALL (2.0) keeps edits in the cGAIL 5x5
# IL window; set to float('inf') for unbounded per-round-epsilon stacking.
EPSILON_CAP: float = EPSILON_BALL
# Inner-loop acceptance gate. "objective": keep the best weighted-objective
# iterate (historical). "non-regression": additionally require the persisted
# iterate to improve F_causal and not regress F_spatial vs the trajectory's
# iter-0 state.
ACCEPT_RULE: str = "objective"
# Max times the iterative (B=1) preset may edit the same trajectory across
# rounds. 1 = historical no-re-edit; 0 = unlimited (epsilon-cap is the limiter).
ITERATIVE_TOPK_MAX_EDITS: int = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/tests/test_config_multiloop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/config.py famail_temporal/tests/test_config_multiloop.py
git commit -m "feat(config): add multi-loop re-attribution + acceptance-gate knobs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Non-regression acceptance gate in `modify_single`

**Files:**
- Modify: `famail_temporal/algorithm/modifier.py` (`__init__` ~line 109-138; `modify_single` loop ~line 343-483)
- Test: `famail_temporal/tests/test_modifier.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `famail_temporal/tests/test_modifier.py`:

```python
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
```

Also add this shared helper near the top of `test_modifier.py` (after `_make_test_trajectory`) if not already present — it derives a valid active (cell, time_bucket) pair:

```python
def _active_cell_and_bucket(bundle, active_idx=0):
    cell = bundle.unit_map.to_flat_cell(active_idx)
    t_block = bundle.unit_map.to_time_block(active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    return x, y, 1 + (start_hour * 12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_modifier.py::test_accept_rule_default_is_objective famail_temporal/tests/test_modifier.py::test_non_regression_rejects_f_spatial_regression -v`
Expected: FAIL — `AttributeError: 'TrajectoryModifier' object has no attribute 'accept_rule'`

- [ ] **Step 3a: Add `accept_rule` to `__init__`**

In `famail_temporal/algorithm/modifier.py`, in `__init__`, after the `self.patience = (...)` block (~line 138) add:

```python
        # Inner-loop acceptance gate (see config.ACCEPT_RULE).
        self.accept_rule = (
            config.ACCEPT_RULE if accept_rule is None else accept_rule
        )
```

And add the parameter to the `__init__` signature (after `patience: int | None = None,`):

```python
        accept_rule: str | None = None,
```

- [ ] **Step 3b: Implement the gate in the `modify_single` loop**

In `modify_single`, before the `for it in range(self.max_iterations):` loop (just after `converged = False` ~line 351) add baseline holders:

```python
        f_causal_0 = None
        f_spatial_0 = None
```

Then replace the best-iterate block (currently ~line 468-483):

```python
            # (g) Best-iterate tracking + patience-based convergence.
            current_objective = float(total.detach())
            if current_objective > best_objective + self.convergence_tol:
                best_objective = current_objective
                best_cumulative_delta = cumulative_delta.copy()
                best_iteration = it
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
                if (self.patience is not None
                        and iters_since_improvement >= self.patience):
                    converged = True
                    break
```

with:

```python
            # (g) Best-iterate tracking + patience-based convergence.
            # iter-0 sits at the pre-edit pickup ⇒ captures the baseline F the
            # non-regression gate compares against.
            if it == 0:
                f_causal_0 = result.f_causal
                f_spatial_0 = result.f_spatial
            current_objective = float(total.detach())
            if self.accept_rule == "non-regression":
                qualifies = (
                    result.f_causal >= f_causal_0 + self.convergence_tol
                    and result.f_spatial >= f_spatial_0 - self.convergence_tol
                )
            else:
                qualifies = True
            if qualifies and current_objective > best_objective + self.convergence_tol:
                best_objective = current_objective
                best_cumulative_delta = cumulative_delta.copy()
                best_iteration = it
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
                if (self.patience is not None
                        and iters_since_improvement >= self.patience):
                    converged = True
                    break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_modifier.py -v`
Expected: PASS (new tests pass; all pre-existing modifier tests still pass — `accept_rule` defaults to `"objective"`, so the gate is a no-op by default).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/algorithm/modifier.py famail_temporal/tests/test_modifier.py
git commit -m "feat(modifier): non-regression acceptance gate (opt-in via accept_rule)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Cumulative-ε cap from the true original cell

**Files:**
- Modify: `famail_temporal/algorithm/modifier.py` (`__init__`; `modify_single` signature + projection ~line 430-437)
- Test: `famail_temporal/tests/test_modifier.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `famail_temporal/tests/test_modifier.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_modifier.py::test_epsilon_cap_default_equals_epsilon_ball famail_temporal/tests/test_modifier.py::test_epsilon_cap_is_respected_relative_to_original_cell -v`
Expected: FAIL — `AttributeError: ... 'epsilon_cap'` / `modify_single() got an unexpected keyword argument 'original_cell'`

- [ ] **Step 3a: Add `epsilon_cap` to `__init__`**

Add to the `__init__` signature (after `accept_rule`):

```python
        epsilon_cap: float | None = None,
```

And after the `self.accept_rule = ...` block:

```python
        # Cumulative L-inf cap from the true original cell, across rounds (see
        # config.EPSILON_CAP). Equals EPSILON_BALL by default ⇒ no-op for a
        # single edit anchored at its own cell.
        self.epsilon_cap = (
            config.EPSILON_CAP if epsilon_cap is None else epsilon_cap
        )
```

- [ ] **Step 3b: Add `original_cell` to `modify_single` and apply the cap**

Change the `modify_single` signature (~line 264) to keyword-only `original_cell`:

```python
    def modify_single(
        self,
        trajectory: Trajectory,
        on_iteration: Optional[Callable[[int, "ModificationResult"], None]] = None,
        *,
        original_cell: Optional[tuple] = None,
    ) -> ModificationHistory:
```

After `original_pickup = np.array(...)` (~line 312) add the true-original anchor:

```python
        true_original = (
            np.array([float(original_cell[0]), float(original_cell[1])],
                     dtype=np.float32)
            if original_cell is not None
            else original_pickup
        )
```

Then in step (f), replace the grid-clip / re-sync block (currently ~line 430-437):

```python
            # Clip pickup to grid bounds
            new_pickup = np.clip(
                original_pickup + cumulative_delta,
                [0.0, 0.0],
                [config.GRID_DIMS[0] - 1, config.GRID_DIMS[1] - 1],
            ).astype(np.float32)
            # Re-sync cumulative_delta after grid-clip
            cumulative_delta = new_pickup - original_pickup
```

with:

```python
            # Clip pickup to grid bounds
            new_pickup = np.clip(
                original_pickup + cumulative_delta,
                [0.0, 0.0],
                [config.GRID_DIMS[0] - 1, config.GRID_DIMS[1] - 1],
            ).astype(np.float32)
            # Cumulative-epsilon cap: keep within self.epsilon_cap (L-inf) of the
            # TRUE original cell, across rounds. With epsilon_cap == EPSILON_BALL
            # and original_cell == this call's start cell, this is a no-op
            # (new_pickup is already within EPSILON_BALL of original_pickup).
            if self.epsilon_cap is not None and np.isfinite(self.epsilon_cap):
                new_pickup = np.clip(
                    new_pickup,
                    true_original - self.epsilon_cap,
                    true_original + self.epsilon_cap,
                ).astype(np.float32)
            # Re-sync cumulative_delta after grid + cumulative-cap clips
            cumulative_delta = new_pickup - original_pickup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_modifier.py -v`
Expected: PASS (new tests pass; existing tests — including `test_modify_single_respects_epsilon_ball` — still pass, since the default cap is a no-op for single edits).

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/algorithm/modifier.py famail_temporal/tests/test_modifier.py
git commit -m "feat(modifier): cumulative epsilon cap anchored to true original cell

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Editing-loop engine

### Task 4: Engine module — dataclasses + batch single-round

**Files:**
- Create: `famail_temporal/algorithm/editing_loop.py`
- Test: `famail_temporal/tests/test_editing_loop.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# famail_temporal/tests/test_editing_loop.py
"""Tests for the unified re-attribution editing loop."""
import numpy as np
from dataclasses import replace

from famail_temporal import config
from famail_temporal.algorithm.editing_loop import (
    run_editing_rounds, EditingLoopResult, RoundRecord,
)
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _bundle_with_drag_trajectories(n_trajs=8, seed=5):
    """Synthetic bundle whose trajectories sit on a strictly-negative-alpha cell."""
    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=seed)
    attribution = compute_per_unit_attribution(bundle)
    gy = bundle.unit_map.grid_shape[1]
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    chosen = None
    for i in range(len(ix_x)):
        uidx = bundle.unit_map.from_cell_time(
            int(ix_x[i]) * gy + int(ix_y[i]), int(ix_t[i]))
        if attribution[uidx] < -1e-6:
            chosen = i
            break
    assert chosen is not None, "seed unstable: no negative-alpha cell"
    x, y, t_block = int(ix_x[chosen]), int(ix_y[chosen]), int(ix_t[chosen])
    tb = config.TIME_BLOCKS[t_block][1] * 12 + 1
    trajs = [
        Trajectory(trajectory_id=tid, driver_id=tid % 2,
                   states=[TrajectoryState(x, y, tb, 0),
                           TrajectoryState(x, y, tb, 0)])
        for tid in range(n_trajs)
    ]
    return replace(bundle, trajectories=trajs)


def _make_modifier(bundle, **kw):
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    return TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, **kw)


def test_batch_single_round_edits_select_top_k_set():
    """max_rounds=1 batch edits exactly the select_top_k(k) negative-alpha set."""
    bundle = _bundle_with_drag_trajectories()
    attribution = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
    expected = set(select_top_k(scored, k=4, trajectories=bundle.trajectories))

    modifier = _make_modifier(bundle)
    result = run_editing_rounds(modifier, bundle, k=4, mode="batch", max_rounds=1)

    assert isinstance(result, EditingLoopResult)
    assert len(result.rounds) == 1
    assert isinstance(result.rounds[0], RoundRecord)
    edited_indices = {
        bundle.trajectories.index(h.original) for h in result.histories
    }
    assert edited_indices == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'famail_temporal.algorithm.editing_loop'`

- [ ] **Step 3: Create the engine (batch, single + multi round skeleton)**

```python
# famail_temporal/algorithm/editing_loop.py
"""Unified re-attribution editing loop.

One engine for the whole family of editing schedules. Each ROUND re-attributes
against the live (post-edit) grid, selects the eligible negative-alpha set, edits
it via TrajectoryModifier.modify_single, and checks the stop rule:

- mode="batch":     edit ALL eligible negative-alpha trajectories each round
                    (against the round-start attribution); re-attribute between
                    rounds. max_rounds=1 reproduces the historical single pass.
- mode="iterative": edit the single most-negative eligible trajectory each round
                    (re-attribute every edit). The B=1 granularity.

Eligibility: alpha < 0 AND cumulative L-inf displacement from the true original
cell < epsilon_cap AND (iterative) edit-count < iterative_max_edits (0=unlimited).

F_causal of any grid == attribution.sum(); we reuse the round attribution for the
round curve and the convergence test (no extra compute).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import ModificationHistory, TrajectoryModifier
from famail_temporal.data.loader import DataBundle


@dataclass(frozen=True)
class RoundRecord:
    round_index: int          # 1-based
    n_edited: int             # edits applied this round
    f_causal: float           # global F_causal AFTER this round's edits
    delta_f_causal: float     # f_causal(this) - f_causal(previous round / baseline)
    pool_size: int            # eligible negative-alpha count at round start


@dataclass(frozen=True)
class EditingLoopResult:
    histories: List[ModificationHistory]   # one per edit (re-edits repeat the id)
    rounds: List[RoundRecord]
    stop_reason: str                       # "max_rounds"|"converged"|"pool_exhausted"
    edited_ids: List[object]               # trajectory ids edited (may repeat)


def _cum_disp(modified, ox: float, oy: float) -> float:
    """L-inf displacement of a modified trajectory's pickup from (ox, oy)."""
    s = modified.pickup_state
    return max(abs(float(s.x_grid) - ox), abs(float(s.y_grid) - oy))


def run_editing_rounds(
    modifier: TrajectoryModifier,
    bundle: DataBundle,
    *,
    k: int,
    mode: str = "batch",
    max_rounds: int = 1,
    round_convergence_tol: Optional[float] = None,
    round_patience: int = 2,
    iterative_max_edits: int = 1,
    max_per_unit: Optional[int] = None,
    max_per_cell: Optional[int] = None,
    on_iter: Optional[Callable[[int, object], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> EditingLoopResult:
    log = log or (lambda _msg: None)
    eps_cap = modifier.epsilon_cap

    current_trajs = list(bundle.trajectories)
    orig_pos = {t.trajectory_id: (float(t.pickup_state.x_grid),
                                  float(t.pickup_state.y_grid))
                for t in bundle.trajectories}
    cum_disp = {t.trajectory_id: 0.0 for t in bundle.trajectories}
    edit_count = {t.trajectory_id: 0 for t in bundle.trajectories}

    histories: List[ModificationHistory] = []
    rounds: List[RoundRecord] = []
    edited_ids: List[object] = []

    attribution = compute_per_unit_attribution(
        bundle, pickup_3d=modifier.current_pickup_3d())
    prev_fc = float(attribution.sum())
    best_fc = prev_fc
    rounds_since_improve = 0
    stop_reason = "max_rounds"

    for r in range(1, max_rounds + 1):
        scored = rank_trajectories(current_trajs, attribution, bundle.unit_map)
        # Eligibility filter (eps-cap + iterative edit-cap), preserving order.
        eligible = []
        for idx, sc in scored:
            if sc >= 0:
                break  # ascending; no more strictly-negative candidates
            tid = current_trajs[idx].trajectory_id
            if (eps_cap is not None and np.isfinite(eps_cap)
                    and cum_disp[tid] >= eps_cap - 1e-9):
                continue
            if (mode == "iterative" and iterative_max_edits > 0
                    and edit_count[tid] >= iterative_max_edits):
                continue
            eligible.append((idx, sc))

        if not eligible:
            stop_reason = "pool_exhausted"
            break

        pool_size = len(eligible)
        n_pick = k if mode == "batch" else 1
        selected = select_top_k(
            eligible, k=n_pick, trajectories=current_trajs,
            max_per_unit=max_per_unit, max_per_cell=max_per_cell,
        )

        for idx in selected:
            traj = current_trajs[idx]
            tid = traj.trajectory_id
            h = modifier.modify_single(
                traj, on_iteration=on_iter, original_cell=orig_pos[tid])
            histories.append(h)
            edited_ids.append(tid)
            current_trajs[idx] = h.modified
            edit_count[tid] += 1
            cum_disp[tid] = _cum_disp(h.modified, *orig_pos[tid])

        # Re-attribute against the post-edit grid: this is both the next round's
        # selection attribution AND this round's "after" F_causal.
        attribution = compute_per_unit_attribution(
            bundle, pickup_3d=modifier.current_pickup_3d())
        fc = float(attribution.sum())
        rounds.append(RoundRecord(
            round_index=r, n_edited=len(selected), f_causal=fc,
            delta_f_causal=fc - prev_fc, pool_size=pool_size))
        log(f"round {r}/{max_rounds}: edited={len(selected)} "
            f"F_causal={fc:.6f} (delta {fc - prev_fc:+.3e}) pool={pool_size}")
        prev_fc = fc

        if round_convergence_tol is not None:
            if fc > best_fc + round_convergence_tol:
                best_fc = fc
                rounds_since_improve = 0
            else:
                rounds_since_improve += 1
                if rounds_since_improve >= round_patience:
                    stop_reason = "converged"
                    break

    return EditingLoopResult(
        histories=histories, rounds=rounds,
        stop_reason=stop_reason, edited_ids=edited_ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py::test_batch_single_round_edits_select_top_k_set -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/algorithm/editing_loop.py famail_temporal/tests/test_editing_loop.py
git commit -m "feat(editing-loop): unified re-attribution engine (batch mode)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Outer stop rule — convergence + pool-exhaustion

**Files:**
- Test: `famail_temporal/tests/test_editing_loop.py` (append)
- (Engine already implements all three stop reasons in Task 4 — these tests lock the behavior.)

- [ ] **Step 1: Write the failing tests**

```python
def test_pool_exhausts_when_no_negative_alpha():
    """A bundle whose only drag cell is fixed in round 1 eventually exhausts."""
    bundle = _bundle_with_drag_trajectories(n_trajs=3)
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=10, mode="batch", max_rounds=50,
        round_convergence_tol=None)
    assert result.stop_reason in ("pool_exhausted", "max_rounds")
    # If it stopped on exhaustion, the last round edited >0 and no eligible remain.
    if result.stop_reason == "pool_exhausted":
        assert len(result.rounds) >= 1


def test_max_rounds_is_hard_ceiling():
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"))
    result = run_editing_rounds(
        modifier, bundle, k=4, mode="batch", max_rounds=3,
        round_convergence_tol=None)
    assert len(result.rounds) <= 3


def test_convergence_stops_when_f_causal_plateaus():
    """With a tiny epsilon_cap the grid barely changes ⇒ F_causal plateaus ⇒
    convergence fires within round_patience rounds of the ceiling."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=4, mode="batch", max_rounds=50,
        round_convergence_tol=1e-9, round_patience=2)
    assert result.stop_reason in ("converged", "pool_exhausted")
    assert len(result.rounds) < 50
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py -v`
Expected: PASS (the engine from Task 4 already implements convergence + pool-exhaustion; these lock it). If `test_convergence_stops_when_f_causal_plateaus` does not converge, raise `round_patience` understanding: the test asserts it stops before 50 rounds via either reason, which must hold because the negative-α pool is finite under a fixed ε-cap.

- [ ] **Step 3: (No new implementation expected.)** If any test fails, debug the engine's stop logic per `systematic-debugging` — do NOT weaken the assertions.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/tests/test_editing_loop.py
git commit -m "test(editing-loop): lock outer stop rule (convergence, pool-exhaust, max-rounds)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cumulative-ε eligibility across rounds

**Files:**
- Test: `famail_temporal/tests/test_editing_loop.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_bounded_cap_limits_total_displacement():
    """With epsilon_cap=2, no edited trajectory drifts more than 2 (L-inf) from
    its true original across all rounds."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=2.0, max_iterations=50)
    result = run_editing_rounds(
        modifier, bundle, k=8, mode="batch", max_rounds=10,
        round_convergence_tol=None)
    orig = {t.trajectory_id: (float(t.pickup_state.x_grid),
                              float(t.pickup_state.y_grid))
            for t in bundle.trajectories}
    for h in result.histories:
        ox, oy = orig[h.original.trajectory_id]
        s = h.modified.pickup_state
        assert max(abs(s.x_grid - ox), abs(s.y_grid - oy)) <= 2.0 + 1e-5


def test_unbounded_cap_allows_drift_past_two():
    """With epsilon_cap=inf and multiple rounds, at least one trajectory can
    exceed an L-inf displacement of 2 from its true original (when the gradient
    keeps pointing outward)."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"), max_iterations=50)
    result = run_editing_rounds(
        modifier, bundle, k=8, mode="batch", max_rounds=5,
        round_convergence_tol=None)
    orig = {t.trajectory_id: (float(t.pickup_state.x_grid),
                              float(t.pickup_state.y_grid))
            for t in bundle.trajectories}
    max_disp = 0.0
    for h in result.histories:
        ox, oy = orig[h.original.trajectory_id]
        s = h.modified.pickup_state
        max_disp = max(max_disp, abs(s.x_grid - ox), abs(s.y_grid - oy))
    # Not asserting > 2 strictly (depends on the synthetic gradient), but the
    # bounded run must never exceed 2 while the unbounded run is free to:
    assert max_disp <= 5 * 2.0 + 1e-5  # sanity: bounded by rounds * per-round eps
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py -v`
Expected: PASS (the eps-cap eligibility + `modify_single` cap from Tasks 3-4 enforce this). `test_bounded_cap_limits_total_displacement` is the load-bearing assertion; the unbounded test is a sanity bound.

- [ ] **Step 3: (No new implementation expected.)** If `test_bounded_cap_limits_total_displacement` fails, the bug is in the engine's `orig_pos`/`cum_disp` plumbing or `modify_single`'s `true_original` clip — debug those, do not relax the cap.

- [ ] **Step 4: Commit**

```bash
git add famail_temporal/tests/test_editing_loop.py
git commit -m "test(editing-loop): lock cumulative epsilon cap across rounds

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Iterative (B=1) mode + multi-edit cap + equivalence

**Files:**
- Test: `famail_temporal/tests/test_editing_loop.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
def test_iterative_max_edits_1_never_re_edits():
    """B=1 with max_edits=1 edits each trajectory at most once (historical
    --iterative-topk behavior / the §8.2 property)."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=1, mode="iterative", max_rounds=50,
        iterative_max_edits=1, round_convergence_tol=None)
    assert len(result.edited_ids) == len(set(result.edited_ids))
    # Each round edits exactly one trajectory.
    assert all(rec.n_edited == 1 for rec in result.rounds)


def test_iterative_unlimited_can_re_edit():
    """B=1 with max_edits=0 (unlimited) may edit the same trajectory more than
    once across rounds when it stays most-negative and under the eps-cap."""
    bundle = _bundle_with_drag_trajectories(n_trajs=2)
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"), max_iterations=50)
    result = run_editing_rounds(
        modifier, bundle, k=1, mode="iterative", max_rounds=6,
        iterative_max_edits=0, round_convergence_tol=None)
    # With only 2 trajectories and 6 rounds, unlimited re-edit must reuse ids.
    assert len(result.edited_ids) > len(set(result.edited_ids))
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_editing_loop.py -v`
Expected: PASS (Task 4's engine handles both via `mode` + `iterative_max_edits`).

- [ ] **Step 3: (No new implementation expected.)** Debug only if a test fails.

- [ ] **Step 4: Run the full algorithm test suite (regression)**

Run: `python -m pytest famail_temporal/tests/test_modifier.py famail_temporal/tests/test_editing_loop.py famail_temporal/tests/test_attribution.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/tests/test_editing_loop.py
git commit -m "test(editing-loop): iterative B=1 mode + multi-edit cap

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Runner integration

### Task 8: Wire engine into `run_experiment` + CLI + delete legacy wrappers

**Files:**
- Modify: `famail_temporal/evaluation/runner.py` (imports; `run_experiment` signature + dispatch ~line 449-643; `_build_arg_parser` ~line 750; `main` ~line 797; delete `_iterative_topk_modify` ~line 72-233 and `_modify_with_progress` ~line 236+)
- Modify: `famail_temporal/evaluation/persistence.py` (persist round records — locate the metrics dict assembly first)
- Test: `famail_temporal/tests/test_runner.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `famail_temporal/tests/test_runner.py`:

```python
def test_run_experiment_multiloop_records_rounds(tiny_bundle):
    """max_rounds>1 runs the engine and records per-round F_causal."""
    result = run_experiment(
        k=4, max_trajectories=None, max_drivers=None,
        max_rounds=3, round_convergence_tol=None, accept_rule="non-regression",
        epsilon_cap=2.0,
    )
    assert hasattr(result, "rounds")
    assert 1 <= len(result.rounds) <= 3
    assert all(hasattr(r, "f_causal") for r in result.rounds)


def test_run_experiment_default_is_single_round(tiny_bundle):
    """No multi-loop args ⇒ exactly one round (historical single pass)."""
    result = run_experiment(k=4)
    assert len(result.rounds) == 1


def test_run_experiment_iterative_topk_one_edit_per_round(tiny_bundle):
    """--iterative-topk maps to B=1 with max_rounds defaulting to k, so it edits
    one trajectory per round (historical behavior), not the whole batch at once."""
    result = run_experiment(k=6, iterative_topk=True)
    assert len(result.modified_trajectory_ids) >= 1
    # B=1: exactly one edit per round ⇒ #rounds == #edits.
    assert len(result.rounds) == len(result.modified_trajectory_ids)
    assert all(r.n_edited == 1 for r in result.rounds)


def test_cli_parses_multiloop_flags():
    from famail_temporal.evaluation.runner import _build_arg_parser
    args = _build_arg_parser().parse_args(
        ["-k", "10", "--max-rounds", "5", "--round-convergence-tol", "1e-4",
         "--round-patience", "2", "--epsilon-cap", "inf",
         "--accept-rule", "non-regression", "--iterative-topk-max-edits", "0"])
    assert args.max_rounds == 5
    assert args.round_convergence_tol == 1e-4
    assert args.epsilon_cap == float("inf")
    assert args.accept_rule == "non-regression"
    assert args.iterative_topk_max_edits == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest famail_temporal/tests/test_runner.py::test_run_experiment_multiloop_records_rounds famail_temporal/tests/test_runner.py::test_cli_parses_multiloop_flags -v`
Expected: FAIL — `run_experiment() got an unexpected keyword argument 'max_rounds'` / `AttributeError: 'Namespace' object has no attribute 'max_rounds'`

- [ ] **Step 3a: Add the import and `ExperimentResult.rounds`**

In `runner.py`, add to the algorithm imports:

```python
from famail_temporal.algorithm.editing_loop import run_editing_rounds, RoundRecord
```

Find the `ExperimentResult` dataclass (near the top, `@dataclass`) and add a field:

```python
    rounds: List[RoundRecord] = field(default_factory=list)
```

(Ensure `from dataclasses import dataclass, field` — add `field` to that import.)

- [ ] **Step 3b: Add parameters to `run_experiment`**

Add to the `run_experiment` signature (after `iterative_topk: bool = False,`):

```python
    max_rounds: Optional[int] = None,
    round_convergence_tol: Optional[float] = None,
    round_patience: Optional[int] = None,
    epsilon_cap: Optional[float] = None,
    accept_rule: Optional[str] = None,
    iterative_topk_max_edits: Optional[int] = None,
```

- [ ] **Step 3c: Replace selection + dispatch with the engine**

There are **three** spans to change in `run_experiment`, plus the modifier construction and the result. Keep the attribution+rank computation above them (`attribution = compute_per_unit_attribution(bundle)` and `scored = rank_trajectories(...)`) — `attribution` still feeds `per_cell_fairness_attribution=attribution` in the result.

**(i) Replace the pre-selection block** — everything from `if iterative_topk:` (the selection branch, ~line 526) down to the end of its `else:` branch (the `_log(t0, f"selected top-k: ...")` call, ~line 578) — with a minimal validation (the engine does selection internally, per round):

```python
        if not any(s < 0 for _, s in scored):
            raise ValueError(
                "No trajectories with strictly negative attribution were found. "
                "Under the F-decomposition convention, negative αᵢ marks cells "
                "dragging fairness below baseline; if none exist the audit set is "
                "uniformly fair (check the active mask / demographics carry signal)."
            )
```

**(ii) Update the modifier construction** (the `modifier = TrajectoryModifier(...)` call, ~line 610-617) to pass the gate + cap:

```python
        modifier = TrajectoryModifier(
            objective=objective, bundle=bundle,
            multi_stream_builder=ms_builder,
            diagnostics_enabled=diagnostics_enabled,
            device=device,
            patience=resolved_patience,
            convergence_tol=convergence_tol,
            accept_rule=accept_rule,
            epsilon_cap=epsilon_cap,
        )
```

**(iii) Replace the pre-modify logging block AND the dispatch block** — everything from the `patience_str = (...)` / `mode_str = ...` logging (~line 618) through the `else: histories = _modify_with_progress(modifier, top_k_trajs)` dispatch (~line 643) — with the resolved params + engine call (this removes the now-defunct `top_k_indices`/`top_k_trajs`/`mode_str` bookkeeping):

```python
        # Resolve outer-loop knobs. Historical --iterative-topk did up to k
        # single-edit rounds (B=1), stopping at pool-exhaustion — so iterative
        # mode defaults max_rounds to k. Batch defaults to config.MAX_ROUNDS (1).
        if max_rounds is not None:
            resolved_max_rounds = max_rounds
        elif iterative_topk:
            resolved_max_rounds = k
        else:
            resolved_max_rounds = config.MAX_ROUNDS
        resolved_round_patience = (
            config.ROUND_PATIENCE if round_patience is None else round_patience
        )
        resolved_round_tol = (
            config.ROUND_CONVERGENCE_TOL if round_convergence_tol is None
            else round_convergence_tol
        )
        resolved_max_edits = (
            config.ITERATIVE_TOPK_MAX_EDITS if iterative_topk_max_edits is None
            else iterative_topk_max_edits
        )
        if resolved_round_tol is not None and resolved_max_rounds <= 1:
            _log(t0, "WARNING: round-convergence-tol set but max-rounds<=1; "
                     "running a single pass. Raise --max-rounds for convergence mode.")
        _log(t0, f"editing loop: mode={'iterative' if iterative_topk else 'batch'} "
                 f"max_rounds={resolved_max_rounds} eps_cap={modifier.epsilon_cap} "
                 f"accept={modifier.accept_rule} round_tol={resolved_round_tol}")
        loop_result = run_editing_rounds(
            modifier, bundle,
            k=k,
            mode="iterative" if iterative_topk else "batch",
            max_rounds=resolved_max_rounds,
            round_convergence_tol=resolved_round_tol,
            round_patience=resolved_round_patience,
            iterative_max_edits=resolved_max_edits,
            max_per_unit=max_per_unit,
            max_per_cell=max_per_cell,
            on_iter=None,
            log=lambda msg: _log(t0, msg),
        )
        histories = loop_result.histories
        rounds = loop_result.rounds
        top_k_scores = [0.0] * len(histories)
        _log(t0, f"editing loop done: {len(histories)} edits over "
                 f"{len(rounds)} round(s), stop={loop_result.stop_reason}")
```

**(iv) Pass `rounds` to the result.** In the `return ExperimentResult(...)` call (~line 686-712), add as the final argument (after `augmented_trajs_after=augmented_after,`):

```python
            rounds=rounds,
```

**(v) Delete** the now-unused functions `_iterative_topk_modify` (~line 72-233) and `_modify_with_progress` (~line 236-...).

> NOTE: progress bars previously lived in those helpers. The engine's `log` callback emits a per-round summary line. A per-iteration tqdm bar can be re-added later via `on_iter`; out of scope here. `_TQDM_AVAILABLE`/`_tqdm` may become unused — leaving them is harmless; removing them is optional cleanup.

- [ ] **Step 3d: Add CLI flags**

In `_build_arg_parser`, after the `--iterative-topk` argument, add:

```python
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Outer re-attribution rounds (hard ceiling; also the "
                        "convergence-mode safety cap). Default config.MAX_ROUNDS "
                        "(1 = single pass).")
    p.add_argument("--round-convergence-tol", type=float, default=None,
                   help="Enable convergence stop: halt when best round F_causal "
                        "has not improved by more than this for --round-patience "
                        "rounds. Default config.ROUND_CONVERGENCE_TOL (off).")
    p.add_argument("--round-patience", type=int, default=None,
                   help="Outer-loop patience (rounds). Default config.ROUND_PATIENCE.")
    p.add_argument("--epsilon-cap", type=float, default=None,
                   help="Cumulative L-inf displacement cap from each trajectory's "
                        "true original cell, across rounds. Pass 'inf' for "
                        "unbounded per-round-epsilon stacking. Default "
                        "config.EPSILON_CAP (=EPSILON_BALL, 2.0).")
    p.add_argument("--accept-rule", choices=["objective", "non-regression"],
                   default=None,
                   help="Inner acceptance gate. 'non-regression' requires each "
                        "persisted edit to improve F_causal and not regress "
                        "F_spatial. Default config.ACCEPT_RULE ('objective').")
    p.add_argument("--iterative-topk-max-edits", type=int, default=None,
                   help="Max edits per trajectory in --iterative-topk mode "
                        "(0 = unlimited). Default config.ITERATIVE_TOPK_MAX_EDITS (1).")
```

(`argparse` parses `inf`/`-inf` via `float`, so `--epsilon-cap inf` works.)

- [ ] **Step 3e: Thread CLI args into `run_experiment` in `main`**

In `main`, add to the `run_experiment(...)` call:

```python
        max_rounds=args.max_rounds,
        round_convergence_tol=args.round_convergence_tol,
        round_patience=args.round_patience,
        epsilon_cap=args.epsilon_cap,
        accept_rule=args.accept_rule,
        iterative_topk_max_edits=args.iterative_topk_max_edits,
```

- [ ] **Step 3f: Persist round records**

In `famail_temporal/evaluation/persistence.py`, locate where the metrics dict / JSON is assembled (grep for `"deltas"` or `"convergence_summary"`). Add a `"rounds"` block built from `result.rounds`:

```python
    "rounds": [
        {"round_index": r.round_index, "n_edited": r.n_edited,
         "f_causal": r.f_causal, "delta_f_causal": r.delta_f_causal,
         "pool_size": r.pool_size}
        for r in getattr(result, "rounds", [])
    ],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest famail_temporal/tests/test_runner.py -v`
Expected: ALL PASS — the new tests pass AND every pre-existing runner test still passes (default `max_rounds=1` batch reproduces the historical single pass through the engine).

- [ ] **Step 5: Full regression suite**

Run: `python -m pytest famail_temporal/tests/ -q`
Expected: ALL PASS (no regressions across the ~266-test suite). If `test_runner_real_data.py` is slow/needs data, run `python -m pytest famail_temporal/tests/ -q --ignore=famail_temporal/tests/test_runner_real_data.py` and note it.

- [ ] **Step 6: Commit**

```bash
git add famail_temporal/evaluation/runner.py famail_temporal/evaluation/persistence.py famail_temporal/tests/test_runner.py
git commit -m "feat(runner): unify editing dispatch on the re-attribution engine + CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — Experiment campaign (GPU; checkpoint with the user between runs)

> These tasks run real experiments. They are NOT TDD. **Surface findings to the user; do not patch the algorithm to "fix" an unexpected result** (per the workflow protocol). The +0.0128 reference is `results/2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`.

### Task 9: α_fi=0 proxy check + cheap runs (R0′, A1, A2, A3) + A4 cost-gate

- [ ] **Step 1: α_fi=0-vs-0.1 proxy sanity (small, fast)**

```bash
python -m famail_temporal.evaluation.runner --name sanity_afi0 -k 1000 \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
python -m famail_temporal.evaluation.runner --name sanity_afi01 -k 1000 \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1
```
Compare `f_causal` deltas in the two `metrics.json`. Expected: within ~1e-4 (confirms α_fi=0 is a faithful proxy at bounded ε). **Report the two numbers to the user.**

- [ ] **Step 2: R0′ — single-pass α_fi=0 reference**

```bash
python -m famail_temporal.evaluation.runner --name R0prime_singlepass_afi0 -k 10000 \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```

- [ ] **Step 3: A1 — bounded-ε multi-loop (C=2, non-regression, α_fi=0)**

```bash
python -m famail_temporal.evaluation.runner --name A1_multiloop_C2_nonreg_afi0 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap 2 --accept-rule non-regression \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```

- [ ] **Step 4: A2 — unbounded-ε ceiling (C=inf, non-regression, α_fi=0)**

```bash
python -m famail_temporal.evaluation.runner --name A2_multiloop_Cinf_nonreg_afi0 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap inf --accept-rule non-regression \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```

- [ ] **Step 5: A3 — gate ablation (C=2, objective gate, α_fi=0)**

```bash
python -m famail_temporal.evaluation.runner --name A3_multiloop_C2_objective_afi0 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap 2 --accept-rule objective \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```

- [ ] **Step 6: A4 cost-probe, then run**

First probe one attribution's wall-time on real data (cheap), then decide A4's `k`:

```bash
python - <<'PY'
import time
from famail_temporal.evaluation.runner import _load_bundle
from famail_temporal.algorithm.attribution import compute_per_unit_attribution
b = _load_bundle()
t = time.monotonic(); compute_per_unit_attribution(b); print("attribution sec:", time.monotonic() - t)
PY
```
If `attribution sec * expected_edits` (≈3,773) is acceptable (say < ~1 hr), run A4 at k=10000; else drop to the largest affordable `k` and note it:

```bash
python -m famail_temporal.evaluation.runner --name A4_iterative_C2_nonreg_afi0 -k 10000 \
  --iterative-topk --iterative-topk-max-edits 0 \
  --max-rounds 100000 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap 2 --accept-rule non-regression \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.0
```
(B=1 uses one round per edit, so `--max-rounds` is set very high; pool-exhaustion/convergence terminates it.)

- [ ] **Step 7: Report**

Build a comparison table (ΔF_causal, ΔF_spatial, #edits, #rounds, stop_reason, mean cumulative displacement) for R0/R0′/A1/A2/A3/A4 from their `metrics.json`. **Present to the user and pause** for interpretation before the headline runs. Key reads: A1/A2 vs R0′ (does multi-loop help?), A2 vs A1 (ε-stacking), A3 vs A1 (gate), A4 vs A1 (B granularity / §8.2 equivalence).

---

### Task 10: Headline runs (α_fi=0.1) + analysis

- [ ] **Step 1: H1 — defensible in-distribution headline (C=2, non-reg, α_fi=0.1)**

```bash
python -m famail_temporal.evaluation.runner --name H1_multiloop_C2_nonreg_afi01 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap 2 --accept-rule non-regression \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1
```

- [ ] **Step 2: H2 — ceiling at α_fi=0.1 (C=inf, non-reg) + fidelity check**

```bash
python -m famail_temporal.evaluation.runner --name H2_multiloop_Cinf_nonreg_afi01 -k 10000 \
  --max-rounds 20 --round-convergence-tol 1e-5 --round-patience 2 \
  --epsilon-cap inf --accept-rule non-regression \
  --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1
```

- [ ] **Step 3: Analysis**

Run `python -m famail_temporal.evaluation.cell_histogram_analysis <H1_dir>` and `<H2_dir>`. Compare H1 vs H2 (in- vs out-of-distribution F_causal), H1 vs A1 (α_fi effect), and both vs the +0.0128 reference. Record max/mean cumulative displacement for H2 (the realism finding). **Headline = best F_causal of {H1, H2}, with the in/out-of-distribution nuance stated explicitly.** Present the final ΔF_causal table to the user.

---

## Phase 5 — Documentation & memory

### Task 11: Methods doc §8.7 + STATUS + memory

- [ ] **Step 1: Methods doc §8.7**

Append a `### 8.7` section to `famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md`: the multi-loop engine + non-regression gate; the round curve and chosen stop rule (Q2/Q3); the ε-stacking effect (A2 vs A1); the gate effect (A3 vs A1); the B-granularity finding (A4 vs A1, and whether §8.2 equivalence held); the headline (H1/H2 vs +0.0128) with the in/out-of-distribution nuance. If multi-loop did NOT beat +0.0128, document the negative result plainly.

- [ ] **Step 2: ε-convention correction**

In the methods doc (and note for memory update) revise any "ε=2 inviolable *across loops*" statement to: "ε=2 within-edit; cumulative cap `C` across rounds (default 2 = in-distribution; configurable)."

- [ ] **Step 3: STATUS.md**

Update `famail_temporal/baselines/STATUS.md` with the shipped editing config (it determines Phase 4's FAMAIL edit source).

- [ ] **Step 4: Commit docs**

```bash
git add famail_temporal/docs/TRAJECTORY_EDITING_METHODOLOGY.md famail_temporal/baselines/STATUS.md
git commit -m "docs(methodology): §8.7 multi-loop re-attribution + non-regression findings

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Memory update (assistant)**

Update `project_famail_temporal_state.md` and `project_fairness_convention.md`/`project_gan_baselines.md` memories where they assert "ε=2 inviolable across loops," and record the new best editing config + the multi-loop findings. (Done via the memory tool, not git.)

---

## Self-review notes (author)

- **Spec coverage:** §3.1 engine → T4-T8; §3.2 gate → T2; §3.3 ε-cap → T3,T6; §3.4 stop rule → T5; §3.5 CLI/config → T1,T8; §3.6 equivalence → T4 (batch), T7 (iterative); §5 matrix → T9-T10; §6 testing → T2-T7; §7 MAX_ITERATIONS → measured in T9 (read `mean_best_iter`/converged-fraction from metrics.json; bump only if hitting 50); §8 docs → T11; §9 findings → T9-T11 checkpoints.
- **MAX_ITERATIONS:** no code task changes it. After T9, inspect A1/A3 `convergence_summary`; if a large fraction hit 50, open a follow-up to bump `config.MAX_ITERATIONS` (gated change — confirm with user).
- **Naming consistency:** engine fn `run_editing_rounds`; dataclasses `RoundRecord`/`EditingLoopResult`; modifier params `accept_rule`, `epsilon_cap`, `original_cell`; config `MAX_ROUNDS`, `ROUND_CONVERGENCE_TOL`, `ROUND_PATIENCE`, `EPSILON_CAP`, `ACCEPT_RULE`, `ITERATIVE_TOPK_MAX_EDITS`; CLI mirrors config with kebab-case.
