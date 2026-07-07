# Supply-Lift Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **TWO HARD CHECKPOINTS:** after Task 1 (Stage-0 oracle → user reviews gate G0) and after Task 10
> (Shenzhen PRIMARY validation → user reviews gates G2/G4/G5/G6). Do NOT proceed past a checkpoint
> without explicit user approval.

**Goal:** Give the FAMAIL editor a supply lever — reroute selected trajectories' seeking tails (last `TAIL_LEN` states + pickup) toward under-served cells with supply endogenous to the objective — so fairness improves by *lifting up* the under-served group, not only leveling down the over-served one.

**Architecture:** A new pure module `algorithm/supply.py` (presence-mass ΔS math, soft/hard materialization, supply-gradient attribution, lift selection) + surgical integrations: `objective.forward` gains an optional `delta_supply_N`, `utils/trajectory.py` gains a taper/adjacency-repairing tail perturbation, `modifier.py` gains a lift mode mirroring its demand pattern (subtract-original → optimize-soft → persist). Trim stays byte-identical; legacy mode reproduces published numbers exactly.

**Tech Stack:** Python 3.12, PyTorch (existing), numpy, pytest. Branch: `supply-lift-editing`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-08-supply-lift-editing-design.md`. Gates G0–G6 are binding.
- **Never modify** `famail_temporal/fairness/*.py` (the fairness math is frozen).
- **G1 legacy byte-identity:** with `TAIL_LEN=0` and `LIFT_BUDGET=0`, every code path must be
  bit-for-bit identical to today (same selections, same pickups, same metrics).
- **G3 trim invariance:** in combined runs, trim-selected trajectories' final pickup cells and the
  demand grid must be identical to legacy; taper moves only their tails, and their tails' ΔS never
  enters *their* optimization loop.
- ΔS soft convention (exact values): one 5-min state = `1/12` hourly presence; per-state mass
  `= 1.0 / (12.0 * n_hours_per_block[t_block] * n_days)`; spread = 5×5 box kernel
  (`PRESENCE_KERNEL_SIZE = 5`); supply floor `config.SUPPLY_FLOOR = 0.1` clamps `S_base + ΔS`.
- King-move rule: every consecutive transition of an edited trajectory satisfies
  `max(|dx|,|dy|) <= 1` (G4, 100% in taper mode). Edits are spatial-only (time buckets and
  `day_index` never change). ε-ball `config.EPSILON_BALL = 2.0` unchanged.
- **Derive grid shapes from tensors, not `config.GRID_DIMS`** — the synthetic test bundle
  (`famail_temporal/tests/test_objective.py::_make_synthetic_bundle`) uses a small grid.
- New config constants (exact names): `TAIL_LEN = 4`, `TAIL_TAPER = (0.25, 0.5, 0.75, 1.0)`,
  `LIFT_BUDGET = None` (None → fill `k − n_trim`).
- Tests live in `famail_temporal/tests/` (algorithm) — module-level `test_*`, `pytest.approx`.
  Run: `python -m pytest <path> -v`.
- Commit after every task; messages `feat(supply-lift): ...` / `test(supply-lift): ...`.

## File Structure

- Create `famail_temporal/algorithm/supply.py` — ΔS math + supply gradient + lift selection (pure).
- Create `famail_temporal/analysis/supply_lift_oracle.py` — Stage-0 G0 oracle (standalone).
- Create `famail_temporal/analysis/supply_recount.py` — tier-2 distinct-count recount (Task 9).
- Modify `famail_temporal/utils/trajectory.py` — `apply_tail_perturbation`.
- Modify `famail_temporal/algorithm/objective.py` — `delta_supply_N` parameter.
- Modify `famail_temporal/algorithm/modifier.py` — lift mode + taper discretization + live fidelity tail.
- Modify `famail_temporal/config.py`, `famail_temporal/evaluation/runner.py`,
  `famail_temporal/evaluation/persistence.py` — wiring + persistence.
- Modify `famail_temporal/baselines/external_fairness_io.py`, `run_external_fairness.py` — supply override.
- Tests: `famail_temporal/tests/test_supply.py`, `test_tail_perturbation.py`, plus extensions to
  `test_objective.py` / `test_modifier.py`.

Key verified anchors (read them before editing): `utils/trajectory.py:76-87` (`apply_perturbation`),
`algorithm/modifier.py:309, 328-329, 350-370, 386-389, 402-409, 439-450, 479-498, 562-570`,
`algorithm/soft_cell_assignment.py:24-72, 90-167`, `algorithm/objective.py:66, 77-80, 93-99,
118-126, 135-141`, `algorithm/attribution.py:44-122, 125-191`, `data/aggregation.py:120-158`,
`active_taxis/generation.py:70-146`, `fidelity/context.py:102-159, 314-328`.

---

### Task 1: Stage-0 supply oracle (gate G0) — then STOP for user review

**Files:**
- Create: `famail_temporal/analysis/supply_lift_oracle.py`

**Interfaces:**
- Consumes: `DataBundle.load()`, `baselines.external_fairness_io.{_enriched_selected_grid, per_unit_demographics, service_ratio_Y}`, `baselines.external_fairness.region_extremes`, `baselines.datasets.pickup_unit_of`.
- Produces: `oracle.json` + `oracle_report.md` under `famail_temporal/analysis/supply_lift_oracle_out/` (gitignore it) with the achievable `Δmean(Y|D)` ceiling under two ΔS semantics.

**What it computes.** For each trajectory whose tail (last `TAIL_LEN+1` states) passes within
ε=2 (Chebyshev) of at least one disadvantaged-group cell (migrant axis, district extremes): the best
single discrete translation δ ∈ [−2,2]² of its tail, scoring the **net** effect on
`mean(Y|D)` — ΔS added at new tail neighborhoods (5×5 box, per-state presence mass), ΔS removed at
old ones, **and** the pickup's demand mass ΔD relocated with it (`Y = S′/max(D′, 0.5)`). Greedy over
trajectories (best-first, updating running S/D grids) up to budget `k = 10000` edits. Two semantics:
- `fraction`: every moved state's mass counts (optimistic; the soft convention's hard-tier-1 twin);
- `distinct-seeking`: a state's removal mass counts only if the driver has **no other seeking state**
  in the same 5×5-neighborhood-hour (computed from the seeking corpus itself), and its addition only
  if the driver is not already present there; report both. (Raw-GPS spot-check deferred to Task 9.)

- [ ] **Step 1: Write the script.** Structure (complete the obvious loops; all helpers named here
  exist or are defined in this file):

```python
"""Stage-0 supply-lift oracle (gate G0): achievable Delta mean(Y|D) ceiling.

Greedy upper bound on lifting-up via seeking-tail rerouting, BEFORE any build.
Run:  python -m famail_temporal.analysis.supply_lift_oracle [--budget 10000] [--tail-len 4]
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io

OUT = Path(__file__).resolve().parent / "supply_lift_oracle_out"

def state_mass(bundle, t_block):
    return 1.0 / (12.0 * float(bundle.n_hours_per_block[t_block]) * bundle.n_days)

def box5(grid_xy, x, y, gx, gy):
    """Coordinates of the clipped 5x5 neighborhood around (x, y)."""
    return [(i, j) for i in range(max(0, x-2), min(gx, x+3))
                   for j in range(max(0, y-2), min(gy, y+3))]

def tail_states(traj, L):
    """Last min(L, len-2)+1 states (tail + pickup); anchor untouched. [] if len < 3."""
    n = len(traj.states)
    if n < 3:
        return []
    L_eff = min(L, n - 2)
    return traj.states[-(L_eff + 1):]

def main():
    ...  # argparse: --budget 10000, --tail-len 4, --semantics both
    bundle = DataBundle.load()
    gx, gy, T = bundle.mask_3d.shape
    S = bundle.active_taxis_3d.astype(np.float64).copy()
    D = bundle.pickup_3d.astype(np.float64).copy()
    demo = io.per_unit_demographics(bundle)
    g_unit = ef.region_extremes(demo["MigrantRatio"], disadvantaged_high=True)
    # unit index grid for fast (x,y,t) -> unit lookup; D-group unit mask
    flat_idx = np.full(bundle.mask_3d.shape, -1, dtype=np.int64)
    flat_idx[bundle.mask_3d] = np.arange(int(bundle.mask_3d.sum()))
    N_D = int((g_unit == 1).sum())
    # D-group CELL set for candidate filtering (any t): from the group grid
    sel = io._enriched_selected_grid()
    cell_group = ef.region_extremes(sel[:, :, 2].ravel(), disadvantaged_high=True).reshape(gx, gy)
    # per-driver seeking-presence index for the distinct-seeking semantics:
    # driver -> set of (cx//1, cy//1 neighborhood-hour keys) built from ALL its states
    presence = defaultdict(set)   # (driver_id, hour_key(x, y, t_block)) membership
    for tr in bundle.trajectories:
        for s in tr.states:
            tb = hour_to_block_index(time_bucket_to_hour(s.time_bucket))
            presence[int(tr.driver_id)].add((int(s.x_grid), int(s.y_grid), tb))
    # candidate scan: for each trajectory with a tail near a D cell, evaluate all 24 deltas;
    # score = sum over affected D-group units of (S'/max(D',.5) - S/max(D,.5)) / N_D
    # (net: -mass at old box cells, +mass at new box cells, pickup demand mass moved too);
    # greedy: sort candidates by best score desc; apply sequentially re-scoring against the
    # RUNNING grids (skip if re-scored gain <= 0); stop at --budget edits.
    # Track both semantics in one pass (fraction always; distinct gated by the presence sets).
    ...
    result = {"budget": args.budget, "tail_len": args.tail_len,
              "n_candidates": n_cand, "n_applied": n_applied,
              "ceiling_fraction": total_gain_fraction / N_D,
              "ceiling_distinct_seeking": total_gain_distinct / N_D,
              "baseline_mean_Y_D": ..., "notes": "greedy upper bound; net of pickup demand move"}
    OUT.mkdir(exist_ok=True)
    (OUT / "oracle.json").write_text(json.dumps(result, indent=1))
    ...  # oracle_report.md: the two ceilings vs the G0 threshold (+0.3), top-20 example edits
```

Implementation notes the engineer must honor: score against **running** grids in the greedy loop
(interactions matter); the pickup's ΔD uses `pickup_mass = 1/(n_hours*n_days)` (12× a state's
presence mass — it is a count, not a presence fraction); `Y` recomputed only on affected units
(the ≤ 2×25×|tail| unit slices an edit touches); candidates where the tail is entirely inside
D-cells are allowed (δ=0 excluded); runtime target < 30 min (vectorize the 24-δ scan per candidate).

- [ ] **Step 2: Smoke-run with `--budget 500`** — expect finite ceilings, `n_candidates` in the
  thousands, report renders. Then full run (`--budget 10000`).
- [ ] **Step 3: Commit** (`feat(supply-lift): stage-0 supply oracle (G0)`) — commit the script
  only; add `famail_temporal/analysis/supply_lift_oracle_out/` to `.gitignore` in this commit.
- [ ] **Step 4: CHECKPOINT — STOP.** Report both ceilings vs the G0 threshold (`Δmean(Y|D) ≥ ~+0.3`)
  to the user. Do not start Task 2 without approval. (Fallback rung 1 if failed: the ceiling number
  goes to the paper's limitations; workstream stops.)

---

### Task 2: `algorithm/supply.py` — ΔS math (soft + hard tier-1)

**Files:**
- Create: `famail_temporal/algorithm/supply.py`
- Test: `famail_temporal/tests/test_supply.py`

**Interfaces:**
- Consumes: `SoftCellAssignment.forward(loc: (B,2), cell: (B,2)) -> (B, ns, ns)` probs; bundle arrays.
- Produces (exact signatures Tasks 4/7/8 rely on):
  - `PRESENCE_KERNEL_SIZE = 5`
  - `state_presence_mass(n_hours_per_block, n_days, t_block) -> float`
  - `soft_delta_supply(probs_batch, cells, t_blocks, masses, signs, grid_shape) -> torch.Tensor (gx,gy,T)`
    — differentiable; each entry b: place `probs_batch[b]` (ns,ns) at `cells[b]` (clipped), box-blur
    5×5, × `masses[b]` × `signs[b]` (+1 add / −1 remove), accumulate into `t_blocks[b]`'s slice.
  - `hard_delta_supply(positions_old, positions_new, t_blocks, masses, grid_shape) -> np.ndarray (gx,gy,T)`
    — same arithmetic at discrete cells (±mass over the clipped 5×5 box), fraction semantics.

- [ ] **Step 1: Write the failing tests**

```python
# famail_temporal/tests/test_supply.py
import numpy as np
import pytest
import torch

from famail_temporal.algorithm import supply as sp
from famail_temporal.algorithm.soft_cell_assignment import SoftCellAssignment


def test_state_presence_mass_convention():
    n_hours = np.ones(24, dtype=np.int32)
    assert sp.state_presence_mass(n_hours, 5, 0) == pytest.approx(1.0 / (12 * 1 * 5))


def test_hard_delta_supply_box_and_sign():
    gx, gy, T = 10, 10, 4
    d = sp.hard_delta_supply(
        positions_old=[(5, 5)], positions_new=[(7, 5)],
        t_blocks=[2], masses=[0.1], grid_shape=(gx, gy, T))
    assert d.shape == (gx, gy, T)
    assert d[5, 5, 2] == pytest.approx(0.0)      # overlap of -box(5,5) and +box(7,5): both cover (5..7,3..7)
    assert d[3, 5, 2] == pytest.approx(-0.1)     # only in old box
    assert d[9, 5, 2] == pytest.approx(+0.1)     # only in new box
    assert d[:, :, 0].sum() == pytest.approx(0.0)  # untouched block
    # clipped at edges: total added mass = 0.1 * |box(7,5)| (full 25 here), removed = 0.1 * 25
    assert d[:, :, 2].sum() == pytest.approx(0.0)


def test_hard_delta_supply_edge_clip():
    d = sp.hard_delta_supply([(0, 0)], [(9, 9)], [0], [1.0], (10, 10, 1))
    assert d[:, :, 0].min() == pytest.approx(-1.0)
    # old box at corner has 3x3=9 cells, new box 3x3=9
    assert (d[:, :, 0] < 0).sum() == 9 and (d[:, :, 0] > 0).sum() == 9


def test_soft_delta_supply_matches_hard_at_low_temperature():
    torch.manual_seed(0)
    gx, gy, T = 12, 12, 3
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    sa.temperature.fill_(1e-4)                    # near-one-hot softmax
    loc = torch.tensor([[6.0, 6.0]]); cell = torch.tensor([[6, 6]])
    probs = sa(loc, cell)                         # (1, ns, ns)
    d_soft = sp.soft_delta_supply(probs, cells=[(6, 6)], t_blocks=[1],
                                  masses=[0.2], signs=[+1], grid_shape=(gx, gy, T))
    d_hard = sp.hard_delta_supply([], [(6, 6)], [1], [0.2], (gx, gy, T))
    np.testing.assert_allclose(d_soft.detach().numpy(), d_hard, atol=1e-4)


def test_soft_delta_supply_differentiable():
    gx, gy, T = 12, 12, 2
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    loc = torch.tensor([[6.0, 6.0]], requires_grad=True)
    probs = sa(loc, torch.tensor([[6, 6]]))
    d = sp.soft_delta_supply(probs, [(6, 6)], [0], [0.2], [+1], (gx, gy, T))
    d.sum().backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
```

- [ ] **Step 2: Run to verify FAIL** (`ModuleNotFoundError: ...supply`).
- [ ] **Step 3: Implement.** `soft_delta_supply`: for each entry, embed the (ns,ns) probs into a
  zero (gx,gy) plane at the clipped window around `cells[b]` (mirror the padding logic of
  `inject_soft_counts_into_3d`, `soft_cell_assignment.py:90-167`), then
  `F.conv2d(plane[None,None], torch.ones(1,1,5,5), padding=2)[0,0] * masses[b] * signs[b]`, add into
  the `t_blocks[b]` slice of a (gx,gy,T) accumulator built with `torch.zeros`. `hard_delta_supply`:
  same via explicit clipped-box loops in numpy. Keep both loop-based (≤ ~10 states/edit; clarity over
  vectorization).
- [ ] **Step 4: Run to verify PASS** (5 tests).
- [ ] **Step 5: Commit** `feat(supply-lift): soft/hard delta-supply math (presence-fraction, 5x5 box)`.

---

### Task 3: objective `delta_supply_N` (byte-identical None path)

**Files:**
- Modify: `famail_temporal/algorithm/objective.py:93-99` (signature), `:118-126, 135-141` (call sites)
- Test: append to `famail_temporal/tests/test_objective.py`

**Interfaces:**
- Produces: `FAMAILObjective.forward(soft_pickup_3d, tau_features=None, tau_prime_features=None, multi_stream_kwargs=None, delta_supply_N=None)`. When `delta_supply_N` (torch (N,)) is given:
  `active_taxis_N = torch.clamp(self.active_taxis_N + delta_supply_N, min=config.SUPPLY_FLOOR)` feeds
  BOTH `compute_fspatial(pickup_N, dropoff_N, active_taxis_N)` (line 126) and
  `compute_fcausal_from_compact(..., supply_N=active_taxis_N, ...)` (line 139). When `None`: code path
  identical to today (do not clamp, do not add — use `self.active_taxis_N` directly).

- [ ] **Step 1: Write the failing tests**

```python
# append to famail_temporal/tests/test_objective.py
def test_forward_delta_supply_none_is_byte_identical():
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle)
    pick = torch.from_numpy(bundle.pickup_3d).float()
    l0, m0 = obj.forward(pick)
    l1, m1 = obj.forward(pick, delta_supply_N=None)
    assert float(l0) == float(l1)
    assert m0["f_causal"] == m1["f_causal"] and m0["f_spatial"] == m1["f_spatial"]


def test_forward_delta_supply_moves_both_terms_and_keeps_grad():
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle)
    pick = torch.from_numpy(bundle.pickup_3d).float()
    N = int(bundle.mask_3d.sum())
    ds = torch.zeros(N, requires_grad=True)
    l0, m0 = obj.forward(pick, delta_supply_N=ds)
    assert float(l0) == pytest.approx(float(obj.forward(pick)[0]))  # zeros == None numerically
    l0.backward()
    assert ds.grad is not None and torch.isfinite(ds.grad).all()
    ds2 = torch.full((N,), 0.5)
    l2, m2 = obj.forward(pick, delta_supply_N=ds2)
    assert m2["f_causal"] != m0["f_causal"]        # supply moved F_causal
    assert m2["f_spatial"] != m0["f_spatial"]      # and F_spatial (DSR/ASR denominators)
```

- [ ] **Step 2: FAIL** (unexpected kwarg). **Step 3: Implement** exactly per Interfaces (a 4-line
  change; `None` branch must not construct any new tensor). **Step 4: PASS** + run the full existing
  `test_objective.py` (no regressions). **Step 5: Commit**
  `feat(supply-lift): optional differentiable delta_supply_N in objective`.

---

### Task 4: supply-gradient attribution + lift scoring

**Files:**
- Modify: `famail_temporal/algorithm/supply.py`
- Test: append to `famail_temporal/tests/test_supply.py`

**Interfaces:**
- Consumes: Task 3's `delta_supply_N`; `attribution.py`-style bundle access.
- Produces:
  - `supply_gradient_N(bundle, objective) -> np.ndarray (N,)` — `∂L/∂S_i` at baseline: zero
    `delta_supply_N` leaf, one forward+backward, return grad (document: at units where
    `S_base == SUPPLY_FLOOR` the clamp subgradient may zero the grad — acceptable, those units
    can only gain).
  - `lift_candidates(bundle, grad_N, tail_len=config.TAIL_LEN, epsilon=config.EPSILON_BALL) ->
    List[Tuple[int, float]]` — `(trajectory_idx, score)` sorted descending; score = the best over
    δ ∈ [−ε,ε]² (integer) of `Σ_states grad_N[units in 5×5 box at new pos]·mass − same at old pos`
    (linearized gain; fast screen — the optimizer refines the actual δ). Skip trajectories with
    `len(states) < 3`.

- [ ] **Step 1: Write the failing tests**

```python
def test_supply_gradient_matches_analytic_fcausal():
    """alpha=(0,1,0): dL/dS_i must equal the analytic F_causal supply gradient."""
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    from famail_temporal.algorithm.objective import FAMAILObjective
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_spatial=0.0, alpha_causal=1.0, alpha_fidelity=0.0)
    g = sp.supply_gradient_N(bundle, obj)
    # analytic: R = Y - g0(D), Y = S/max(D,.5); dF/dS_i = (2/(R'MR)) * [((I-H)R)_i - F*(MR)_i] / max(D_i,.5)
    import torch
    D = torch.clamp(torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).double(), min=0.5)
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).double()
    Y = S / D
    g0 = torch.from_numpy(np.asarray(bundle.g0_func((D).numpy()))).double()
    R = Y - g0
    X = torch.from_numpy(bundle.hat_matrices["X_demo"]).double()
    XtX_inv = torch.from_numpy(bundle.hat_matrices["XtX_inv"]).double()
    IHR = R - X @ (XtX_inv @ (X.T @ R))
    MR = R - R.mean()
    RMR = float(R @ MR)
    F = float(R @ IHR) / RMR
    analytic = (2.0 / RMR) * (IHR - F * MR) / D
    mask_free = bundle.active_taxis_3d[bundle.mask_3d] > 0.1   # clamp-inactive units only
    np.testing.assert_allclose(g[mask_free], analytic.numpy()[mask_free], rtol=1e-3, atol=1e-6)


def test_lift_candidates_prefers_tails_near_positive_gradient():
    """Synthetic: gradient positive in one region; trajectory tails near it score higher."""
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    bundle = _make_synthetic_bundle()
    N = int(bundle.mask_3d.sum())
    grad = np.zeros(N); grad[: N // 4] = 1.0      # first units (low x) positive
    scored = sp.lift_candidates(bundle, grad, tail_len=2, epsilon=2)
    assert len(scored) > 0
    assert all(scored[i][1] >= scored[i + 1][1] for i in range(len(scored) - 1))
```

(For the second test, `_make_synthetic_bundle` carries real trajectories on active units — if its
trajectories are too few/short, build 10 synthetic `Trajectory` objects on active cells with the
helpers in `famail_temporal/baselines/tests/_helpers.py` and attach via `dataclasses.replace`.)

- [ ] **Step 2: FAIL.** **Step 3: Implement** (`supply_gradient_N`: build zero leaf, call
  `objective.forward(torch.from_numpy(bundle.pickup_3d).float(), delta_supply_N=leaf)`, backward,
  return `leaf.grad.numpy()`. `lift_candidates`: precompute `flat_idx` (x,y,t)→unit map; per
  trajectory, per δ (25 options), sum masked grad over each moved state's clipped 5×5 box ± —
  vectorize the box sums with a precomputed 5×5-box-summed gradient grid: `G_box[x,y,t] =
  box-sum of grad-grid` via `scipy.ndimage.uniform_filter`-style numpy cumsum, then a state's box
  contribution is one lookup: `G_box[new] − G_box[old]`). **Step 4: PASS.** **Step 5: Commit**
  `feat(supply-lift): supply-gradient attribution + lift candidate scoring`.

---

### Task 5: tapered tail perturbation + adjacency repair

**Files:**
- Modify: `famail_temporal/utils/trajectory.py` (after `apply_perturbation`, line ~88)
- Test: `famail_temporal/tests/test_tail_perturbation.py`

**Interfaces:**
- Produces: `Trajectory.apply_tail_perturbation(delta, tail_len, grid_dims) -> Trajectory` —
  integer-rounds the pickup offset from continuous `delta` (clip to grid), then assigns each of the
  `L_eff = min(tail_len, len(states)-2)` tail states an integer offset via **backward greedy repair**
  (pickup → anchor): target offsets = `round(taper_j · delta)`, adjusted so every consecutive
  transition of the RESULT satisfies `max(|dx|,|dy|) ≤ 1` given the ORIGINAL steps; anchor offset is
  (0,0) by construction. If no compliant assignment exists even after deepening the tail to the whole
  trajectory (anchor = states[0]), return `None` (caller skips the edit; counted).
  Also produces `taper_weights(L_eff) -> tuple` (linear `j/L_eff` for j=1..L_eff; equals
  `config.TAIL_TAPER` when `L_eff == 4`).

- [ ] **Step 1: Write the failing tests**

```python
# famail_temporal/tests/test_tail_perturbation.py
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
```

- [ ] **Step 2: FAIL.** **Step 3: Implement.** Backward greedy: `off[m] = rounded clipped delta`
  (pickup); for j = m−1 … first-tail-state: feasible off_j must satisfy
  `|orig_step_{j→j+1} + (off_{j+1} − off_j)|∞ ≤ 1` per axis — pick the feasible off_j closest to
  `round(taper_j·delta)` (per-axis independent: clamp target into
  `[off_{j+1}+orig_step−1 −? ...]` — derive the per-axis feasible interval and clamp); finally require
  the anchor constraint `|orig_step_{anchor→first} + off_first|∞ ≤ 1` with anchor offset 0; if
  violated, deepen (increase L_eff) and retry; if L_eff reaches `len−2` and still violated → `None`.
  **Step 4: PASS (6 tests).** **Step 5: Commit**
  `feat(supply-lift): tapered tail perturbation with king-move adjacency repair`.

---

### Task 6: lift selection wiring (budget fill, trim precedence)

**Files:**
- Modify: `famail_temporal/algorithm/supply.py` (selection assembly)
- Modify: `famail_temporal/config.py` (add `TAIL_LEN = 4`, `TAIL_TAPER = (0.25, 0.5, 0.75, 1.0)`,
  `LIFT_BUDGET = None` — grouped with the editing constants near `EPSILON_BALL`, config.py:83)
- Test: append to `famail_temporal/tests/test_supply.py`

**Interfaces:**
- Consumes: `attribution.rank_trajectories` + `attribution.select_top_k` outputs (trim indices);
  Task 4's `lift_candidates`.
- Produces: `assemble_edit_plan(trim_indices, lift_scored, k_total, lift_budget=None) ->
  List[Tuple[int, str]]` — `(trajectory_idx, mode)` with mode ∈ {"trim","lift"}; trim first
  (unchanged order); lift fills `lift_budget if lift_budget is not None else k_total − len(trim)`
  slots from `lift_scored` skipping any idx already in trim (trim precedence) and scores ≤ 0.

- [ ] **Step 1: Failing test**

```python
def test_assemble_edit_plan_trim_precedence_and_fill():
    trim = [3, 7]
    lift = [(7, 9.0), (1, 5.0), (2, 3.0), (9, 0.0), (4, -1.0)]
    plan = sp.assemble_edit_plan(trim, lift, k_total=5)
    assert plan[:2] == [(3, "trim"), (7, "trim")]
    assert plan[2:] == [(1, "lift"), (2, "lift")]     # 7 deduped to trim; 9 and 4 dropped (score<=0)

def test_assemble_edit_plan_explicit_budget():
    plan = sp.assemble_edit_plan([1], [(2, 4.0), (3, 2.0)], k_total=10, lift_budget=1)
    assert plan == [(1, "trim"), (2, "lift")]
```

- [ ] **Step 2: FAIL. Step 3: implement (10 lines). Step 4: PASS. Step 5: Commit**
  `feat(supply-lift): edit-plan assembly (trim precedence, budget fill) + config constants`.

---

### Task 7: modifier lift mode (the core integration)

**Files:**
- Modify: `famail_temporal/algorithm/modifier.py`
- Test: append to `famail_temporal/tests/test_modifier.py`

**Read first:** the whole per-trajectory loop `modifier.py:300-575`; the demand pattern to mirror is
documented in the module docstring (subtract original contribution :8, optimize soft :400-460,
persist :564-570).

**Interfaces:**
- Consumes: Tasks 2/4/5/6 APIs; Task 3's `delta_supply_N`.
- Produces: `TrajectoryModifier.modify_trajectory(trajectory, mode="trim")`; a shared accumulator
  `self._delta_supply_3d` (torch (gx,gy,T), init zeros) exposed via
  `current_delta_supply_3d() -> np.ndarray` (float64 copy) for persistence/eval.

**Behavioral contract (encode as written):**
1. **mode="trim"** — optimization loop byte-identical to today (no ΔS in ITS objective calls, pickup
   demand injection only). At discretization, when `config.TAIL_LEN > 0`: apply
   `apply_tail_perturbation(best_cumulative_delta, TAIL_LEN)` instead of `apply_perturbation`
   (falls back to legacy `apply_perturbation` when `TAIL_LEN == 0` → G1). If repair returns `None`,
   keep the legacy pickup-only move for trim (G3 guarantees the pickup lands; count
   `n_taper_infeasible_trim`). After discretization, compute the trajectory's hard tier-1 ΔS
   (`hard_delta_supply(old tail positions, new tail positions, ...)`) and add to
   `self._delta_supply_3d` — trim tails contribute to *evaluation* supply honestly, never to their
   own optimization.
2. **mode="lift"** — mirror the demand pattern for supply: at start, subtract the trajectory's
   original tail presence from the accumulator is NOT needed (the accumulator holds deltas, not
   totals — original presence lives in `S_base`); instead, per iteration: build the tail's soft
   positions `pos_j = orig_j + taper_j·delta_tensor` (torch, from `taper_weights(L_eff)`), batch-call
   `self.soft_assign(pos_stack, cell_stack)`, form
   `traj_soft_ds = soft_delta_supply(probs_new, ...,+1) + hard-anchored removal at ORIGINAL positions
   (constant, sign −1, computed once)`; call
   `self.objective(pickup_soft_3d, ..., delta_supply_N=(self._delta_supply_3d + traj_soft_ds)[mask])`.
   The pickup's soft demand injection stays exactly as today (lift moves demand too, by design).
   Fidelity: per iteration splice ALL moving rows, not just the last —
   `tau_prime_features[0, -(L_eff+1)+j, 0:2] = pos_j` and
   `ms_kwargs['x2'][0, 0, -(L_eff+1)+j, 0:2] = pos_j + 1` for each tail row j (extends
   modifier.py:439-450). Discretize via `apply_tail_perturbation` (repair `None` → skip edit
   entirely, count `n_taper_infeasible_lift`, revert demand). Persist: demand sub/add (existing
   :564-570) + `self._delta_supply_3d += hard tier-1 ΔS` of the final tail move.
3. ε-ball, temperature annealing, best-iterate selection: unchanged (shared code path).

- [ ] **Step 1: Failing tests** (synthetic bundle; keep runtimes small):

```python
def test_trim_mode_pickup_identical_to_legacy_and_demand_grid_unchanged():
    # run one trajectory through mode="trim" with TAIL_LEN=4 and with TAIL_LEN=0 (monkeypatch config);
    # assert identical final pickup cell, identical _base_pickup_3d, and (TAIL_LEN=4 case)
    # _delta_supply_3d nonzero only if the tail actually moved.
    ...

def test_lift_mode_moves_supply_toward_positive_gradient():
    # craft a bundle where one region has strongly under-served units (high demand, low supply);
    # a lift trajectory whose tail sits 2 cells away must end with delta_supply_3d mass
    # net-positive in that region and the edited trajectory king-move compliant.
    ...

def test_lift_skip_on_infeasible_repair_reverts_cleanly():
    # trajectory of len 2 (repair returns None): modifier must leave _base_pickup_3d and
    # _delta_supply_3d exactly unchanged and report the skip in the ModificationHistory/None return.
    ...
```

Write these three concretely against the synthetic-bundle helpers (see `test_modifier.py`'s existing
patterns for constructing the modifier; copy its setup lines rather than inventing new fixtures).

- [ ] **Step 2: FAIL. Step 3: implement per the contract. Step 4: PASS + full
  `test_modifier.py` suite green (legacy tests must not change).** **Step 5: Commit**
  `feat(supply-lift): modifier lift mode — batched tail soft-assign, endogenous dS, live fidelity tail`.

---

### Task 8: runner/persistence wiring + G1 legacy gate

**Files:**
- Modify: `famail_temporal/evaluation/runner.py` (selection → edit plan; pass mode; metrics with ΔS)
- Modify: `famail_temporal/evaluation/persistence.py` (persist `delta_supply_3d.npz` + counters)
- Test: `famail_temporal/tests/test_runner.py` (extend)

**Interfaces:**
- Runner: after the existing trim selection, compute `supply_gradient_N` + `lift_candidates` +
  `assemble_edit_plan`; loop the plan calling `modify_trajectory(traj, mode=mode)`; final metrics
  computed with `delta_supply_N = modifier.current_delta_supply_3d()[mask]` (torch) so
  `metrics_after` reflects endogenous supply. `metrics_before` unchanged (ΔS=0).
- Persistence: write `delta_supply_3d.npz` (key `delta_supply_3d`, float64) + add to `metrics.json`:
  `n_trim`, `n_lift`, `n_taper_infeasible_trim`, `n_taper_infeasible_lift`, and
  `supply_totals {added, removed}`.
- **G1 test (the load-bearing one):**

```python
def test_legacy_mode_end_to_end_byte_identical(monkeypatch):
    """TAIL_LEN=0, LIFT_BUDGET=0 must reproduce the pre-supply-lift pipeline exactly."""
    monkeypatch.setattr(config, "TAIL_LEN", 0)
    monkeypatch.setattr(config, "LIFT_BUDGET", 0)
    # run the runner's edit loop on the synthetic bundle twice: once through the new code path,
    # once through a pinned pre-change call sequence (attribution -> select_top_k ->
    # modify_trajectory(mode="trim")); assert identical metrics dicts, identical pickup grids,
    # and delta_supply_3d.sum() == 0.
```

- [ ] Steps: failing tests → FAIL → implement → PASS (incl. full `test_runner.py`) → commit
  `feat(supply-lift): runner edit-plan wiring + delta-supply persistence + G1 legacy gate`.

---

### Task 9: tier-2 distinct-count recount tool

**Files:**
- Create: `famail_temporal/analysis/supply_recount.py`

**Interfaces:**
- CLI: `python -m famail_temporal.analysis.supply_recount --edit-dir <results_dir> [--city shenzhen]`.
- Reads the edit's `histories.pkl` (original + modified seeking states) and the raw GPS
  (`raw_data/taxi_record_0*_50drivers.pkl`, records `[plate_id, lat, lon, seconds,
  passenger_indicator, timestamp]` — reuse the quantization in `active_taxis/processor.py` and the
  distinct-set logic in `active_taxis/generation.py:70-146`). Recomputes hour-level distinct-taxi
  counts twice — with original vs edited seeking states substituted for the edited trajectories'
  pings — aggregates both to (gx,gy,T) mean-hourly (mirror `data/aggregation.py:120-158`), and
  reports: `S_tier2_before`, `S_tier2_after`, their metric-level effect (F_causal + mean(Y|D) via the
  external harness's conventions), and the tier1-vs-tier2 gap. Writes
  `<edit_dir>/supply_recount_report.md` + `.json`. SF variant: `--city sf12` reads the Cabspotting
  ping source used by `second_dataset/data/source_generation/sf_build.py` (budget a day; if the SF
  ping-path needs new plumbing, report rather than improvise).
- [ ] Steps: write → smoke on the Task-10 validation run's edit dir → commit
  `feat(supply-lift): tier-2 distinct-count supply recount tool`.

---

### Task 10: external-metrics supply override + Shenzhen PRIMARY validation — then STOP

**Files:**
- Modify: `famail_temporal/baselines/external_fairness_io.py` — `service_ratio_Y(pickup_3d, bundle,
  supply_3d=None)` (backward-compatible default) using `supply_3d` when given.
- Modify: `famail_temporal/baselines/run_external_fairness.py` — `--delta-supply <path.npz>` flag:
  loads the edit's `delta_supply_3d.npz`, computes `S′ = clip(S_base + ΔS, SUPPLY_FLOOR, None)`,
  passes as `supply_3d` for the AFTER side (before side stays `S_base`).
- Tests: extend `famail_temporal/baselines/tests/test_external_fairness_io.py` (+2: override used;
  default path unchanged) and `test_run_external_fairness.py` (+1 flag plumbing).

**Then run the validation (commands, in order):**

```bash
# 1. trim+lift edit run, Shenzhen PRIMARY (defaults TAIL_LEN=4, LIFT_BUDGET=None)
python -m famail_temporal.evaluation.runner --name supply_lift_v1_shz_primary -k 10000 \
  --device cuda --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1
# 2. gates on its results dir <RES>:
python -m famail_temporal.analysis.supply_recount --edit-dir <RES>                  # G2 tier-2
python -m famail_temporal.baselines.run_external_fairness --edit-dir <RES> \
  --delta-supply <RES>/delta_supply_3d.npz --dataset shz-primary-supplylift          # G6 metrics
# 3. adjacency sweep (G4): one-off assert over <RES>/histories.pkl — every modified trajectory
#    king-move compliant (script inline or add --check-adjacency to supply_recount).
# 4. fidelity (G5): the runner's metrics.json fidelity numbers vs the trim-only run's.
```

**Gate review package for the user (CHECKPOINT — STOP):** G2 soft-vs-hard-tier1 gap + tier-2 gap;
G4 = 100%; G5 Fidelity-A/B vs trim-only; G6 `Δmean(Y|D)` with bootstrap CI + external metrics table
(the group-levels rows are the lifting-up evidence) + F_causal not regressed; `n_lift`,
`n_taper_infeasible_*` counts. Do not start Task 11 without approval.

---

### Task 11: headline execution playbook (post-approval)

No new code. In order, daemonized (`setsid nohup ... > run.log`, PID file, resumable):

1. Trim+lift edit runs on the remaining 3 datasets (gdp-comp, logpop feature-set configs; SF via
   `FAMAIL_CITY=sf12`, same overrides as its published run).
2. External metrics with `--delta-supply` on all 4 → `--combine` cross-dataset table.
3. Weighted-BC sweeps on the trim+lift corpora (existing
   `run_weighted_bc_smoke` machinery, `--edit-dir` = new results dirs; ~10h GPU each — Shenzhen
   PRIMARY first, others as time allows per the fallback ladder).
4. Option-A allocation re-run (`PAPER/external-metrics/scripts/option_a_rollout_eval.py` with
   `EDIT_DIR` pointed at the new Shenzhen PRIMARY edit): poor-area pickup + state shares must rise.
5. Curate into `PAPER/supply-lift/` mirroring the external-metrics bundle (FINDINGS.md, tables/,
   figures/; trim-only published rows carried as the ablation), and update
   `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` §5/§6 status lines.
6. Fallback ladder is binding: build slipping past ~Jul 16 → BC re-run drops (trim-only cited);
   weak metrics → additive subsection.

---

## Self-Review

- **Spec coverage:** §3 mechanism→T5/T7; §4 ΔS soft/hard-1/hard-2→T2/T9; §5 attribution/selection→T4/T6;
  §6 integration+fidelity-live-tail→T3/T7; §7 G0→T1(checkpoint), G1→T8, G2→T9+T10, G3→T7/T8 tests,
  G4→T5 property test+T10 sweep, G5/G6→T10; §8 evaluation→T10/T11. Covered.
- **Placeholder scan:** T1/T7/T8 contain intentional `...` in *scripts the implementer completes
  against named, verified anchors with the algorithm fully specified in prose* — each names exact
  inputs, outputs, and the loop semantics; no "TBD"/"handle edge cases" anywhere. T5/T2/T3/T4/T6
  carry complete code.
- **Type consistency:** `soft_delta_supply(probs_batch, cells, t_blocks, masses, signs, grid_shape)`
  and `hard_delta_supply(positions_old, positions_new, t_blocks, masses, grid_shape)` used
  identically in T2/T7/T8/T9; `apply_tail_perturbation(delta, tail_len, grid_dims) -> Trajectory|None`
  consistent T5/T7; `assemble_edit_plan(trim_indices, lift_scored, k_total, lift_budget)` T6/T8;
  `delta_supply_N` torch (N,) T3/T7/T8/T10.
