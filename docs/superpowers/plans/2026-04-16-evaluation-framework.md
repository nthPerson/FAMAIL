# Evaluation Framework Implementation Plan

> **Serialization note:** Several artifacts in this plan use Python pickle (`.pkl`) format. This is an explicit, user-approved requirement — the augmented trajectory output must be a drop-in structural replacement for the existing `passenger_seeking_trajs_45-800.pkl` consumed by downstream tooling. Pickle inputs are only ever loaded from paths the framework itself writes; they are never loaded from untrusted sources. See §1 of the design spec for the full rationale.
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Subagent model policy (hard requirement):** Every implementation subagent and every code-review subagent dispatched by this plan MUST be run on the Opus model. The orchestrator must pass `model: "opus"` when launching subagents via the Agent tool.
>
> **Subagent skill policy (hard requirement):** Every subagent must be instructed in its prompt to invoke the following superpowers skills:
> - **Implementation subagents:** `superpowers:test-driven-development` (write the failing test first; red -> green -> refactor), `superpowers:verification-before-completion` (run the test/build and show passing output before claiming done), and `superpowers:systematic-debugging` (when a test fails, diagnose the root cause rather than patch the symptom).
> - **Review subagents:** `superpowers:requesting-code-review` / `superpowers:code-reviewer` workflow at every phase checkpoint; `superpowers:receiving-code-review` for the follow-up pass that acts on feedback.
> Every subagent's prompt must explicitly name the skills it is required to use.

**Goal:** Build the `famail_temporal/evaluation/` package that runs the FAMAIL trajectory-modification pipeline end-to-end and produces two priority artifacts — a `(48, 90, T, 4)` fairness-aware state-space grid and a fairness-augmented trajectory dataset — plus an experiment runner, persistence layer, gradient diagnostics, and a tables-only markdown report.

**Architecture:** Ten sequential phases. Phases 1–3 produce the two priority artifacts and can be used from a notebook without any orchestration layer. Phases 4–5 add the plumbing (public modifier accessor + Tier A diagnostics) needed for the runner. Phases 6–9 compose the runner, persistence, report, and Tier C sensitivity grids. Phase 10 is documentation and changelog. Every phase ends at a commit with all tests green.

**Tech Stack:** Python 3.10, PyTorch >=2.0, NumPy, pytest. No new third-party dependencies.

**Spec reference:** `docs/superpowers/specs/2026-04-16-evaluation-framework-design.md`

---

## File Structure

### Files to create

| Path | Responsibility |
|---|---|
| `famail_temporal/evaluation/__init__.py` | Re-exports: `run_experiment`, `ExperimentResult`, `build_fairness_grid`, `augment_trajectories`, `compute_gradient_sensitivity` |
| `famail_temporal/evaluation/grid.py` | `build_fairness_grid(bundle, pickup_3d=None) -> np.ndarray` — (48, 90, T, 4) grid |
| `famail_temporal/evaluation/augment.py` | `augment_trajectories(trajectories, grid) -> dict[int, list[list[list]]]` |
| `famail_temporal/evaluation/diagnostics.py` | `compute_gradient_sensitivity(bundle, pickup_3d) -> np.ndarray` |
| `famail_temporal/evaluation/runner.py` | `run_experiment(...)`, `ExperimentResult`, CLI entry point |
| `famail_temporal/evaluation/persistence.py` | `write(result, output_root) -> Path` with conditional gzip |
| `famail_temporal/evaluation/report.py` | `render(output_dir) -> Path` producing `report.md` |
| `famail_temporal/evaluation/README.md` | Usage docs |
| `famail_temporal/results/.gitkeep` | Ensures the output root exists |
| `famail_temporal/tests/test_per_unit_gini_decomposition.py` | Tests for the new fairness primitive |
| `famail_temporal/tests/test_compute_spatial_attribution.py` | Tests for the 3-channel wrapper |
| `famail_temporal/tests/test_fairness_grid.py` | Tests for `build_fairness_grid` |
| `famail_temporal/tests/test_augment_trajectories.py` | Tests for `augment_trajectories` |
| `famail_temporal/tests/test_gradient_diagnostics.py` | Tests for Tier A decomposition + fallback parity |
| `famail_temporal/tests/test_persistence.py` | Round-trip tests, gzip threshold |
| `famail_temporal/tests/test_runner.py` | End-to-end synthetic test, override restore |
| `famail_temporal/tests/test_runner_real_data.py` | `@pytest.mark.slow` real-data smoke |
| `famail_temporal/tests/test_gradient_sensitivity.py` | Tests for `compute_gradient_sensitivity` |
| `famail_temporal/tests/test_report.py` | Tests for `report.render` |

### Files to modify

| Path | Change |
|---|---|
| `famail_temporal/config.py` | Add `DIAGNOSTICS_ENABLED: bool = True` |
| `famail_temporal/fairness/spatial.py` | Add `per_unit_gini_decomposition`, `compute_spatial_attribution`; refactor `pairwise_gini` to use the new primitive |
| `famail_temporal/algorithm/modifier.py` | Add `current_pickup_3d()` accessor; add Tier A gradient-decomposition branch gated by `diagnostics_enabled`; extend `ModificationResult` with new fields |
| `famail_temporal/tests/test_math_invariants.py` | Add invariant: `sum(per_unit_gini_decomposition(x)) == pairwise_gini(x)` |
| `CHANGELOG.md` | Entry describing the evaluation framework addition |
| `.gitignore` | Ignore `famail_temporal/results/` contents (keep `.gitkeep`) |

---

## Subagent Dispatch Template

Every subagent dispatch in this plan must follow this template so the Opus model + skill requirements are consistent. The orchestrator should copy-paste and fill in the `{TASK CONTENT}` slot:

```
Agent tool call:
  description: "<phase N, task N: short title>"
  subagent_type: "general-purpose"
  model: "opus"
  prompt: |
    You are executing a task from the famail_temporal evaluation framework
    implementation plan at docs/superpowers/plans/2026-04-16-evaluation-framework.md.

    REQUIRED SKILLS - invoke each via the Skill tool before and during your work:
    - superpowers:test-driven-development  (red -> green -> refactor; failing test first)
    - superpowers:verification-before-completion  (run the test/build and show passing output before reporting done)
    - superpowers:systematic-debugging  (if a test fails, diagnose root cause; no symptom patching)

    Constraint: Do NOT modify files outside the scope listed in the task.
    Constraint: Commit only the files the task names.
    Constraint: Run the exact test command shown and paste the passing tail
    of its output into your final report.

    {TASK CONTENT - paste the full task block from the plan here}

    When done, reply with: files changed, test command you ran, and the
    final lines of its output.
```

Review subagents use the same template but with `subagent_type: "superpowers:code-reviewer"` and these required skills in the prompt:

- `superpowers:requesting-code-review` (the review workflow)
- `superpowers:receiving-code-review` (if the review finds issues, this skill guides the act-on-feedback pass)

Both agent types MUST run on Opus (`model: "opus"`).

---

## Phase 0: Preparation (orchestrator-only, no subagent)

**Files:**
- Modify: `.gitignore`
- Create: `famail_temporal/results/.gitkeep`

- [ ] **Step 1: Verify baseline tests pass**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q`
Expected: 173 tests pass (fast suite). No failures.

- [ ] **Step 2: Add `.gitignore` entry for results directory**

Append to `.gitignore`:

```
# Evaluation framework output (per-experiment directories)
famail_temporal/results/*
!famail_temporal/results/.gitkeep
```

- [ ] **Step 3: Create the output root placeholder**

```bash
mkdir -p famail_temporal/results
touch famail_temporal/results/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore famail_temporal/results/.gitkeep
git commit -m "chore: prepare famail_temporal/results output root"
```

---

## Phase 1: Per-unit Gini decomposition + spatial attribution

**Files:**
- Modify: `famail_temporal/fairness/spatial.py`
- Modify: `famail_temporal/tests/test_math_invariants.py`
- Create: `famail_temporal/tests/test_per_unit_gini_decomposition.py`
- Create: `famail_temporal/tests/test_compute_spatial_attribution.py`

### Task 1.1: Write the failing test for `per_unit_gini_decomposition`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/tests/test_per_unit_gini_decomposition.py`:

```python
"""Tests for fairness.spatial.per_unit_gini_decomposition."""
import torch
import pytest

from famail_temporal.fairness.spatial import (
    per_unit_gini_decomposition, pairwise_gini,
)


def test_decomposition_sums_to_gini_random():
    torch.manual_seed(0)
    values = torch.rand(50) * 10.0 + 0.1
    decomp = per_unit_gini_decomposition(values)
    assert decomp.shape == values.shape
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_equal_values_zero():
    values = torch.full((20,), 3.0)
    decomp = per_unit_gini_decomposition(values)
    assert torch.allclose(decomp, torch.zeros_like(values), atol=1e-6)


def test_decomposition_one_hot_concentrated_on_outlier():
    values = torch.zeros(10)
    values[0] = 100.0
    decomp = per_unit_gini_decomposition(values)
    assert decomp[0] > 10 * decomp[1]
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_single_element_zero():
    values = torch.tensor([5.0])
    decomp = per_unit_gini_decomposition(values)
    assert decomp.shape == (1,)
    assert float(decomp.sum()) == 0.0


def test_decomposition_two_elements_sum_matches_gini():
    values = torch.tensor([1.0, 3.0])
    decomp = per_unit_gini_decomposition(values)
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_all_nonnegative():
    torch.manual_seed(1)
    values = torch.rand(30) * 5.0 + 0.1
    decomp = per_unit_gini_decomposition(values)
    assert (decomp >= 0.0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_per_unit_gini_decomposition.py -v`
Expected: `ImportError` / `AttributeError: module 'famail_temporal.fairness.spatial' has no attribute 'per_unit_gini_decomposition'`.

### Task 1.2: Implement `per_unit_gini_decomposition` and refactor `pairwise_gini`

- [ ] **Step 1: Edit `famail_temporal/fairness/spatial.py`**

Replace the existing `pairwise_gini` function with this pair:

```python
def per_unit_gini_decomposition(values: torch.Tensor) -> torch.Tensor:
    """Row-sum decomposition of the pairwise Gini on an N-vector.

    For each i in [0, N):  contrib_i = sum_j |x_i - x_j| / (2 * N^2 * mean(x))
    so that sum_i contrib_i == pairwise_gini(values) exactly (modulo float
    precision). Callers are responsible for passing only the active-unit
    subset - this function operates on 1-D N-vectors with no mask handling.
    """
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)
    mean_val = values.mean() + config.EPS
    diff = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))  # (N, N)
    row_sums = diff.sum(dim=1)                                    # (N,)
    return row_sums / (2 * n * n * mean_val)


def pairwise_gini(values: torch.Tensor) -> torch.Tensor:
    """Differentiable pairwise Gini.

    Implemented as sum(per_unit_gini_decomposition(values)) so the per-unit
    decomposition and the aggregate stay numerically linked by construction.
    Clamped to [0, 1] to guard against float drift at the upper boundary.
    """
    gini = per_unit_gini_decomposition(values).sum()
    return torch.clamp(gini, 0.0, 1.0)
```

- [ ] **Step 2: Run both test files to verify green**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_per_unit_gini_decomposition.py famail_temporal/tests/test_spatial_fairness.py -v`
Expected: all tests pass. The existing `pairwise_gini` tests must still pass unchanged.

### Task 1.3: Add math invariant

- [ ] **Step 1: Read current invariants file**

```bash
cat famail_temporal/tests/test_math_invariants.py | head -40
```

- [ ] **Step 2: Append the new invariant**

Add to `famail_temporal/tests/test_math_invariants.py`:

```python
def test_spatial_gini_decomposition_sums_to_gini():
    """sum(per_unit_gini_decomposition(x)) == pairwise_gini(x) for random x."""
    import torch
    from famail_temporal.fairness.spatial import (
        per_unit_gini_decomposition, pairwise_gini,
    )
    torch.manual_seed(17)
    for _ in range(5):
        n = int(torch.randint(2, 100, (1,)).item())
        values = torch.rand(n) * 10.0 + 0.01
        assert torch.isclose(
            per_unit_gini_decomposition(values).sum(),
            pairwise_gini(values),
            atol=1e-6,
        )
```

- [ ] **Step 3: Run to verify**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_math_invariants.py -v`
Expected: all invariants pass, including the new one.

### Task 1.4: Write failing test for `compute_spatial_attribution`

- [ ] **Step 1: Write the failing test**

Create `famail_temporal/tests/test_compute_spatial_attribution.py`:

```python
"""Tests for fairness.spatial.compute_spatial_attribution."""
import torch
import pytest

from famail_temporal.fairness.spatial import (
    compute_spatial_attribution, compute_fspatial, pairwise_gini,
)


def _synth(N, seed=0):
    torch.manual_seed(seed)
    pickup = torch.rand(N) * 5.0 + 0.1
    dropoff = torch.rand(N) * 5.0 + 0.1
    active = torch.rand(N) * 3.0 + 1.0
    return pickup, dropoff, active


def test_returns_three_channels_of_length_N():
    N = 40
    pickup, dropoff, active = _synth(N)
    result = compute_spatial_attribution(pickup, dropoff, active)
    assert set(result.keys()) == {"gini_decomp_dsr", "gini_decomp_asr", "spatial_attr"}
    for key, vec in result.items():
        assert vec.shape == (N,), f"{key} shape {vec.shape} != ({N},)"


def test_spatial_attr_sums_to_one_minus_fspatial():
    N = 60
    pickup, dropoff, active = _synth(N, seed=3)
    result = compute_spatial_attribution(pickup, dropoff, active)
    f_spatial, _ = compute_fspatial(pickup, dropoff, active)
    assert torch.isclose(
        result["spatial_attr"].sum(),
        1.0 - f_spatial,
        atol=1e-5,
    )


def test_dsr_decomp_sums_to_dsr_gini():
    N = 50
    pickup, dropoff, active = _synth(N, seed=5)
    result = compute_spatial_attribution(pickup, dropoff, active)
    from famail_temporal import config
    dsr = pickup / (active + config.EPS)
    assert torch.isclose(result["gini_decomp_dsr"].sum(), pairwise_gini(dsr), atol=1e-6)


def test_asr_decomp_sums_to_asr_gini():
    N = 50
    pickup, dropoff, active = _synth(N, seed=7)
    result = compute_spatial_attribution(pickup, dropoff, active)
    from famail_temporal import config
    asr = dropoff / (active + config.EPS)
    assert torch.isclose(result["gini_decomp_asr"].sum(), pairwise_gini(asr), atol=1e-6)


def test_spatial_attr_equals_half_sum_of_components():
    N = 30
    pickup, dropoff, active = _synth(N, seed=11)
    result = compute_spatial_attribution(pickup, dropoff, active)
    expected = 0.5 * (result["gini_decomp_dsr"] + result["gini_decomp_asr"])
    assert torch.allclose(result["spatial_attr"], expected, atol=1e-7)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_compute_spatial_attribution.py -v`
Expected: `AttributeError: module 'famail_temporal.fairness.spatial' has no attribute 'compute_spatial_attribution'`.

### Task 1.5: Implement `compute_spatial_attribution`

- [ ] **Step 1: Append the wrapper to `famail_temporal/fairness/spatial.py`**

```python
def compute_spatial_attribution(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> dict:
    """Per-unit spatial attribution (3 N-vector channels).

    Returns:
        dict with keys 'gini_decomp_dsr', 'gini_decomp_asr', 'spatial_attr'.
        spatial_attr = 0.5 * (gini_decomp_dsr + gini_decomp_asr), so that
        sum(spatial_attr) == 1 - F_spatial (same canonical decomposition
        consumed by the fairness-aware grid).

    Input validation mirrors compute_fspatial (1-D, matching shapes, non-negative).
    """
    if pickup_N.dim() != 1 or dropoff_N.dim() != 1 or active_taxis_N.dim() != 1:
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must be 1-D tensors."
        )
    if not (pickup_N.shape == dropoff_N.shape == active_taxis_N.shape):
        raise ValueError("pickup_N, dropoff_N, and active_taxis_N must have the same shape.")
    if (pickup_N < 0).any() or (dropoff_N < 0).any() or (active_taxis_N < 0).any():
        raise ValueError("pickup_N, dropoff_N, and active_taxis_N must not contain negatives.")

    dsr = pickup_N / (active_taxis_N + config.EPS)
    asr = dropoff_N / (active_taxis_N + config.EPS)
    gini_decomp_dsr = per_unit_gini_decomposition(dsr)
    gini_decomp_asr = per_unit_gini_decomposition(asr)
    spatial_attr = 0.5 * (gini_decomp_dsr + gini_decomp_asr)
    return {
        "gini_decomp_dsr": gini_decomp_dsr,
        "gini_decomp_asr": gini_decomp_asr,
        "spatial_attr": spatial_attr,
    }
```

- [ ] **Step 2: Run to verify green**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_compute_spatial_attribution.py famail_temporal/tests/test_per_unit_gini_decomposition.py famail_temporal/tests/test_math_invariants.py famail_temporal/tests/test_spatial_fairness.py -v`
Expected: all tests pass.

### Task 1.6: Commit Phase 1

- [ ] **Step 1: Run the full fast test suite**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q`
Expected: all baseline tests + new tests pass.

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/fairness/spatial.py \
        famail_temporal/tests/test_per_unit_gini_decomposition.py \
        famail_temporal/tests/test_compute_spatial_attribution.py \
        famail_temporal/tests/test_math_invariants.py
git commit -m "feat(fairness): add per-unit Gini decomposition + spatial attribution"
```

### Phase 1 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`, `superpowers:requesting-code-review`) against this commit. The review prompt must ask for: (a) correctness of the row-sum decomposition, (b) preservation of existing `pairwise_gini` behavior, (c) adequacy of edge-case tests (n=0, n=1, equal-values, one-hot). Address any high-priority findings before proceeding.

---

## Phase 2: `evaluation/grid.py` — fairness-aware state-space grid

**Files:**
- Create: `famail_temporal/evaluation/__init__.py`
- Create: `famail_temporal/evaluation/grid.py`
- Create: `famail_temporal/tests/test_fairness_grid.py`

### Task 2.1: Scaffold the package

- [ ] **Step 1: Create the package init**

Create `famail_temporal/evaluation/__init__.py`:

```python
"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.grid import build_fairness_grid

__all__ = ["build_fairness_grid"]
```

This file will be extended in later phases as modules are added. Do not import modules that don't yet exist.

### Task 2.2: Write failing tests for `build_fairness_grid`

- [ ] **Step 1: Write the tests**

Create `famail_temporal/tests/test_fairness_grid.py`:

```python
"""Tests for evaluation.grid.build_fairness_grid."""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.fairness.spatial import compute_fspatial
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fairness.causal import per_unit_attribution
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_returns_correct_shape():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    gx, gy = bundle.pickup_3d.shape[:2]
    assert grid.shape == (gx, gy, config.T, 4)
    assert grid.dtype == np.float32


def test_inactive_cells_are_nan():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    inactive = ~bundle.mask_3d
    assert np.isnan(grid[inactive]).all()


def test_active_cells_are_finite():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    active = bundle.mask_3d
    for c in range(4):
        assert np.isfinite(grid[active, c]).all(), f"channel {c} has NaN on active cells"


def test_spatial_attr_channel_sums_to_one_minus_fspatial():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    grid = build_fairness_grid(bundle)
    pickup_N = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_N)
    assert np.isclose(np.nansum(grid[..., 0]), 1.0 - float(f_spatial), atol=1e-5)


def test_causal_attr_channel_sums_to_one_minus_fcausal():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    grid = build_fairness_grid(bundle)
    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D_clamped = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D_clamped
    g0_D = torch.from_numpy(np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32))
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    expected = float(per_unit_attribution(R, tensors["I_minus_H_demo"], tensors["M"]).sum())
    assert np.isclose(np.nansum(grid[..., 1]), expected, atol=1e-5)


def test_gini_dsr_channel_sums_to_dsr_gini():
    from famail_temporal.fairness.spatial import pairwise_gini
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=4)
    grid = build_fairness_grid(bundle)
    pickup_N = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    dsr = pickup_N / (active_N + config.EPS)
    assert np.isclose(np.nansum(grid[..., 2]), float(pairwise_gini(dsr)), atol=1e-6)


def test_gini_asr_channel_sums_to_asr_gini():
    from famail_temporal.fairness.spatial import pairwise_gini
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=6)
    grid = build_fairness_grid(bundle)
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    asr = dropoff_N / (active_N + config.EPS)
    assert np.isclose(np.nansum(grid[..., 3]), float(pairwise_gini(asr)), atol=1e-6)


def test_channel_0_equals_half_sum_of_channels_2_3_on_active():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=8)
    grid = build_fairness_grid(bundle)
    active = bundle.mask_3d
    lhs = grid[..., 0][active]
    rhs = 0.5 * (grid[..., 2][active] + grid[..., 3][active])
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_pickup_override_changes_grid():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=9)
    grid_default = build_fairness_grid(bundle)
    pickup_mod = bundle.pickup_3d.copy()
    active_ix = np.argwhere(bundle.mask_3d)
    x0, y0, t0 = active_ix[0]
    x1, y1, t1 = active_ix[1]
    pickup_mod[x0, y0, t0] = max(0.0, pickup_mod[x0, y0, t0] - 0.5)
    pickup_mod[x1, y1, t1] += 0.5
    grid_mod = build_fairness_grid(bundle, pickup_3d=pickup_mod)
    assert not np.allclose(
        grid_default[..., 0][bundle.mask_3d],
        grid_mod[..., 0][bundle.mask_3d],
    ), "Channel 0 should change when pickup_3d changes"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_fairness_grid.py -v`
Expected: `ModuleNotFoundError: No module named 'famail_temporal.evaluation.grid'`.

### Task 2.3: Implement `build_fairness_grid`

- [ ] **Step 1: Create `famail_temporal/evaluation/grid.py`**

```python
"""Fairness-aware state-space grid builder.

Produces a (grid_x, grid_y, T, 4) tensor whose channels are:
    0: spatial_attr       (sums to 1 - F_spatial)
    1: causal_attr        (sums to 1 - F_causal)
    2: gini_decomp_dsr    (sums to Gini(DSR))
    3: gini_decomp_asr    (sums to Gini(ASR))

Inactive units are NaN on all channels.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.spatial import compute_spatial_attribution
from famail_temporal.fairness.causal import per_unit_attribution
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch


def build_fairness_grid(
    bundle: DataBundle,
    pickup_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the (grid_x, grid_y, T, 4) fairness-aware grid.

    Args:
        bundle: DataBundle - provides dropoff_3d, active_taxis_3d, mask_3d,
                hat_matrices, g0_func.
        pickup_3d: Optional override for the pickup tensor. If None, uses
                   bundle.pickup_3d (the before-state). For the after-state
                   grid, pass TrajectoryModifier.current_pickup_3d().

    Returns:
        (grid_x, grid_y, T, 4) float32 ndarray. Inactive cells are NaN on
        all 4 channels.
    """
    if pickup_3d is None:
        pickup_3d = bundle.pickup_3d
    if pickup_3d.shape != bundle.pickup_3d.shape:
        raise ValueError(
            f"pickup_3d shape {pickup_3d.shape} != bundle.pickup_3d shape "
            f"{bundle.pickup_3d.shape}"
        )

    mask = bundle.mask_3d

    # Project 3D -> N in canonical order (numpy boolean indexing iterates
    # in C order, matching UnitIndexMap's cell-major/time-within-cell ordering).
    pickup_N = torch.from_numpy(pickup_3d[mask]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()

    # Channels 0, 2, 3 - spatial attribution.
    sp = compute_spatial_attribution(pickup_N, dropoff_N, active_N)
    spatial_attr = sp["spatial_attr"].detach().numpy()
    gini_dsr = sp["gini_decomp_dsr"].detach().numpy()
    gini_asr = sp["gini_decomp_asr"].detach().numpy()

    # Channel 1 - causal attribution (sums to 1 - F_causal).
    D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
    Y = active_N / D_clamped
    g0_D = torch.from_numpy(
        np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32)
    )
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    causal_attr = per_unit_attribution(
        R, tensors["I_minus_H_demo"], tensors["M"],
    ).detach().numpy()

    # Scatter back to (gx, gy, T, 4) with NaN on inactive cells.
    gx, gy = bundle.pickup_3d.shape[:2]
    grid = np.full((gx, gy, config.T, 4), np.nan, dtype=np.float32)
    ix_x, ix_y, ix_t = np.where(mask)
    grid[ix_x, ix_y, ix_t, 0] = spatial_attr
    grid[ix_x, ix_y, ix_t, 1] = causal_attr
    grid[ix_x, ix_y, ix_t, 2] = gini_dsr
    grid[ix_x, ix_y, ix_t, 3] = gini_asr
    return grid
```

- [ ] **Step 2: Run tests to verify green**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_fairness_grid.py -v`
Expected: all 9 tests pass.

### Task 2.4: Commit Phase 2

- [ ] **Step 1: Run the full fast suite**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q`
Expected: all tests pass.

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/evaluation/__init__.py \
        famail_temporal/evaluation/grid.py \
        famail_temporal/tests/test_fairness_grid.py
git commit -m "feat(evaluation): add fairness-aware state-space grid builder"
```

### Phase 2 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) correctness of the grid->N->grid projection (does numpy `arr[mask]` ordering match `UnitIndexMap.from_mask`?), (b) NaN handling on inactive cells, (c) `pickup_3d` override equivalence with a modified bundle.

---

## Phase 3: `evaluation/augment.py` — trajectory augmentation

**Files:**
- Create: `famail_temporal/evaluation/augment.py`
- Create: `famail_temporal/tests/test_augment_trajectories.py`
- Modify: `famail_temporal/evaluation/__init__.py`

### Task 3.1: Write failing tests

- [ ] **Step 1: Write the tests**

Create `famail_temporal/tests/test_augment_trajectories.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_augment_trajectories.py -v`
Expected: `ModuleNotFoundError: No module named 'famail_temporal.evaluation.augment'`.

### Task 3.2: Implement `augment_trajectories`

- [ ] **Step 1: Create `famail_temporal/evaluation/augment.py`**

```python
"""Trajectory augmentation - widens 4-element states to 8-element states.

Output format is a drop-in replacement for passenger_seeking_trajs_45-800.pkl:
a dict keyed by driver_id, values are lists of trajectories, each trajectory
is a list of 8-element state lists. Indices 0-3 are 1-indexed on disk.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

import numpy as np

from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.utils.trajectory import Trajectory


def augment_trajectories(
    trajectories: List[Trajectory],
    grid: np.ndarray,
) -> Dict[int, List[List[list]]]:
    """Produce the driver-keyed augmented dataset.

    Each state is widened from 4 to 8 elements:
        [x_grid, y_grid, time_bucket, day_index,
         spatial_attr, causal_attr, gini_decomp_dsr, gini_decomp_asr]

    Indices 0-3 written 1-indexed (drop-in compatibility with
    passenger_seeking_trajs_45-800.pkl). Indices 4-7 come from
    grid[x, y, t_block, :] (NaN for inactive cells).

    Every input trajectory is included in the output.
    """
    if grid.ndim != 4 or grid.shape[3] != 4:
        raise ValueError(
            f"grid must have shape (gx, gy, T, 4); got {grid.shape}"
        )

    out: Dict[int, List[List[list]]] = defaultdict(list)
    for traj in trajectories:
        augmented_states: List[list] = []
        for st in traj.states:
            x = int(st.x_grid)
            y = int(st.y_grid)
            t_block = hour_to_block_index(time_bucket_to_hour(st.time_bucket))
            fairness = grid[x, y, t_block, :]
            augmented_states.append([
                x + 1,
                y + 1,
                int(st.time_bucket),
                int(st.day_index),
                float(fairness[0]),
                float(fairness[1]),
                float(fairness[2]),
                float(fairness[3]),
            ])
        out[int(traj.driver_id)].append(augmented_states)

    return dict(out)
```

- [ ] **Step 2: Extend `famail_temporal/evaluation/__init__.py`**

Replace with:

```python
"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.augment import augment_trajectories

__all__ = ["build_fairness_grid", "augment_trajectories"]
```

- [ ] **Step 3: Run tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_augment_trajectories.py -v`
Expected: all 7 tests pass.

### Task 3.3: Commit Phase 3

- [ ] **Step 1: Run fast suite**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q`
Expected: all tests pass.

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/evaluation/__init__.py \
        famail_temporal/evaluation/augment.py \
        famail_temporal/tests/test_augment_trajectories.py
git commit -m "feat(evaluation): trajectory augmentation producing 8-element states"
```

### Phase 3 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) 1-indexed-on-disk correctness (reader-side subtraction would restore 0-indexed values), (b) NaN preservation on inactive cells, (c) no trajectories are silently dropped.

---

## Phase 4: Public `TrajectoryModifier.current_pickup_3d()` accessor

**Files:**
- Modify: `famail_temporal/algorithm/modifier.py`
- Modify: `famail_temporal/tests/test_modifier.py`

### Task 4.1: Add failing test

- [ ] **Step 1: Append test to `famail_temporal/tests/test_modifier.py`**

```python
def test_current_pickup_3d_reflects_modifications():
    """current_pickup_3d() must return the post-modification pickup tensor as a
    numpy ndarray matching bundle.pickup_3d's shape."""
    import numpy as np
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective

    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    objective = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=objective, bundle=bundle, max_iterations=2,
    )
    before = modifier.current_pickup_3d()
    assert isinstance(before, np.ndarray)
    assert before.shape == bundle.pickup_3d.shape
    assert before.dtype == np.float32
    assert np.allclose(before, bundle.pickup_3d)

    snapshot = before.copy()
    before[0, 0, 0] = 999.0
    assert np.allclose(modifier.current_pickup_3d(), snapshot)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_modifier.py::test_current_pickup_3d_reflects_modifications -v`
Expected: `AttributeError`.

### Task 4.2: Implement the accessor

- [ ] **Step 1: Add the method to `TrajectoryModifier`**

Insert immediately after `__init__` (before `_get_annealed_temperature`) in `famail_temporal/algorithm/modifier.py`:

```python
    def current_pickup_3d(self) -> np.ndarray:
        """Return the post-modification pickup tensor as a numpy ndarray.

        Shape (grid_x, grid_y, T), float32. Returns a copy so callers
        cannot mutate modifier state.
        """
        return self._base_pickup_3d.detach().cpu().numpy().copy().astype(np.float32)
```

- [ ] **Step 2: Run the new test to verify green**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_modifier.py::test_current_pickup_3d_reflects_modifications -v`
Expected: PASS.

- [ ] **Step 3: Run all modifier tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_modifier.py famail_temporal/tests/test_modifier_integration.py -v`
Expected: all existing tests still pass.

### Task 4.3: Commit Phase 4

```bash
git add famail_temporal/algorithm/modifier.py famail_temporal/tests/test_modifier.py
git commit -m "feat(algorithm): expose current_pickup_3d accessor on TrajectoryModifier"
```

### Phase 4 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) is the returned ndarray detached from internal state (copy-on-read), (b) does it reflect mutations from prior `modify_single` calls, (c) method placement consistent with class ordering.

---

## Phase 5: Tier A gradient decomposition in the modifier

**Files:**
- Modify: `famail_temporal/config.py`
- Modify: `famail_temporal/algorithm/modifier.py`
- Create: `famail_temporal/tests/test_gradient_diagnostics.py`

### Task 5.1: Add `DIAGNOSTICS_ENABLED` config flag

- [ ] **Step 1: Edit `famail_temporal/config.py`**

Add after the `CONVERGENCE_TOL` line (around line 53):

```python
# Gradient diagnostics
DIAGNOSTICS_ENABLED: bool = True
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q`

### Task 5.2: Write failing tests for Tier A decomposition

- [ ] **Step 1: Create `famail_temporal/tests/test_gradient_diagnostics.py`**

```python
"""Tests for Tier A gradient decomposition in TrajectoryModifier."""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationResult
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _first_active_traj(bundle):
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    x, y, t_block = int(ix_x[0]), int(ix_y[0]), int(ix_t[0])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    return Trajectory(
        trajectory_id=0, driver_id=1,
        states=[
            TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
            TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
        ],
    )


def test_modification_result_has_diagnostic_fields_when_enabled():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=2, diagnostics_enabled=True,
    )
    hist = modifier.modify_single(_first_active_traj(bundle))
    for r in hist.iterations:
        assert r.grad_spatial_norm is not None
        assert r.grad_causal_norm is not None
        assert r.grad_fidelity_norm is not None
        assert r.grad_cosine_spatial_causal is not None
        assert r.dominant_term in {"spatial", "causal", "fidelity"}
        assert isinstance(r.sign_flipped, bool)


def test_modification_result_has_none_diagnostics_when_disabled():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=2, diagnostics_enabled=False,
    )
    hist = modifier.modify_single(_first_active_traj(bundle))
    for r in hist.iterations:
        assert r.grad_spatial_norm is None
        assert r.grad_causal_norm is None
        assert r.grad_fidelity_norm is None
        assert r.grad_cosine_spatial_causal is None
        assert r.dominant_term is None


def test_decomposed_gradients_produce_same_first_step():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    obj = FAMAILObjective(bundle, alpha_spatial=0.4, alpha_causal=0.4, alpha_fidelity=0.0)
    modifier_diag = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=True,
    )
    modifier_plain = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=False,
    )
    traj = _first_active_traj(bundle)
    h_diag = modifier_diag.modify_single(traj)
    h_plain = modifier_plain.modify_single(traj)
    delta_diag = h_diag.iterations[0].cumulative_delta
    delta_plain = h_plain.iterations[0].cumulative_delta
    assert np.allclose(delta_diag, delta_plain, atol=1e-5)


def test_no_diagnostics_path_preserves_final_trajectory():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    traj = _first_active_traj(bundle)
    mod_a = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, diagnostics_enabled=True,
    )
    mod_b = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, diagnostics_enabled=False,
    )
    h_a = mod_a.modify_single(traj)
    h_b = mod_b.modify_single(traj)
    assert h_a.modified.pickup_cell == h_b.modified.pickup_cell


def test_diagnostics_default_from_config():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=3)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=1)
    assert modifier.diagnostics_enabled is True
```

- [ ] **Step 2: Run to verify failures**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_gradient_diagnostics.py -v`
Expected: multiple failures (constructor arg missing, new fields missing).

### Task 5.3: Extend `ModificationResult` + add decomposition path

- [ ] **Step 1: Edit `famail_temporal/algorithm/modifier.py` — replace `ModificationResult`**

```python
@dataclass
class ModificationResult:
    """Single iteration record from the ST-iFGSM loop."""
    iteration: int
    objective_value: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    gradient_norm: float
    cumulative_delta: np.ndarray
    # Tier A diagnostics - None when diagnostics_enabled=False.
    grad_spatial_norm: float | None = None
    grad_causal_norm: float | None = None
    grad_fidelity_norm: float | None = None
    grad_cosine_spatial_causal: float | None = None
    grad_cosine_fairness_fidelity: float | None = None
    sign_flipped: bool | None = None
    dominant_term: str | None = None
```

- [ ] **Step 2: Add `diagnostics_enabled` kwarg to `__init__`**

Update the `__init__` signature:

```python
    def __init__(
        self,
        objective: FAMAILObjective,
        bundle: DataBundle,
        multi_stream_builder=None,
        alpha: float = config.STEP_SIZE_ALPHA,
        epsilon: float = config.EPSILON_BALL,
        max_iterations: int = config.MAX_ITERATIONS,
        convergence_tol: float = config.CONVERGENCE_TOL,
        diagnostics_enabled: bool | None = None,
    ):
        self.objective = objective
        self.bundle = bundle
        self.multi_stream_builder = multi_stream_builder
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.diagnostics_enabled = (
            config.DIAGNOSTICS_ENABLED if diagnostics_enabled is None
            else diagnostics_enabled
        )

        self.soft_assign = SoftCellAssignment()
        self._base_pickup_3d = torch.from_numpy(bundle.pickup_3d).float().clone()
```

- [ ] **Step 3: Add `_compute_decomposed_gradient` helper method**

Add immediately before `modify_single`:

```python
    def _compute_decomposed_gradient(
        self,
        f_spatial: torch.Tensor,
        f_causal: torch.Tensor,
        f_fidelity: torch.Tensor,
        pickup_tensor: torch.Tensor,
    ):
        """Return (grad_combined_ndarray, diagnostics_dict)."""
        grad_spatial = torch.autograd.grad(
            f_spatial, pickup_tensor, retain_graph=True,
        )[0].detach().cpu().numpy()
        grad_causal = torch.autograd.grad(
            f_causal, pickup_tensor, retain_graph=True,
        )[0].detach().cpu().numpy()
        alpha_sp = self.objective.alpha_spatial
        alpha_ca = self.objective.alpha_causal
        alpha_fi = self.objective.alpha_fidelity

        if alpha_fi > 0:
            grad_fidelity = torch.autograd.grad(
                f_fidelity, pickup_tensor, retain_graph=True,
            )[0].detach().cpu().numpy()
        else:
            grad_fidelity = np.zeros_like(grad_spatial)

        grad_combined = (
            alpha_sp * grad_spatial
            + alpha_ca * grad_causal
            + alpha_fi * grad_fidelity
        )

        def _cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

        norms = {
            "spatial":  float(np.linalg.norm(grad_spatial)),
            "causal":   float(np.linalg.norm(grad_causal)),
            "fidelity": float(np.linalg.norm(grad_fidelity)),
        }
        weighted = {
            "spatial":  alpha_sp * norms["spatial"],
            "causal":   alpha_ca * norms["causal"],
            "fidelity": alpha_fi * norms["fidelity"],
        }
        dominant = max(weighted, key=weighted.get)

        fairness_grad = alpha_sp * grad_spatial + alpha_ca * grad_causal
        diagnostics = {
            "grad_spatial_norm":              norms["spatial"],
            "grad_causal_norm":               norms["causal"],
            "grad_fidelity_norm":             norms["fidelity"],
            "grad_cosine_spatial_causal":     _cos(grad_spatial, grad_causal),
            "grad_cosine_fairness_fidelity":  _cos(fairness_grad, grad_fidelity),
            "dominant_term":                  dominant,
        }
        return grad_combined, diagnostics
```

- [ ] **Step 4: Replace the backward block in `modify_single`**

In the for-loop inside `modify_single`, replace:

```python
            # (e) Backward — zero_grad before backward to clear accumulated gradients
            self.objective.zero_grad()
            total.backward(retain_graph=True)

            if pickup_tensor.grad is None:
                grad = np.zeros(2)
            else:
                grad = pickup_tensor.grad.detach().cpu().numpy()
            grad_norm = float(np.linalg.norm(grad))
```

with:

```python
            # (e) Backward - decomposed if diagnostics_enabled, else single-backward
            self.objective.zero_grad()
            tier_a_metrics = None
            if self.diagnostics_enabled:
                grad, tier_a_metrics = self._compute_decomposed_gradient(
                    terms["f_spatial"], terms["f_causal"], terms["f_fidelity"],
                    pickup_tensor,
                )
            else:
                total.backward(retain_graph=True)
                if pickup_tensor.grad is None:
                    grad = np.zeros(2)
                else:
                    grad = pickup_tensor.grad.detach().cpu().numpy()
            grad_norm = float(np.linalg.norm(grad))
```

- [ ] **Step 5: Track sign flips and build extended `ModificationResult`**

At the very top of `modify_single` (before the "pickup_state = ..." line), add:

```python
        self._prev_grad_sign = None
```

Replace the `ModificationResult(...)` construction inside the loop with:

```python
            prev_sign = self._prev_grad_sign
            cur_sign = np.sign(grad)
            sign_flipped = (
                bool(np.any(prev_sign != cur_sign))
                if (self.diagnostics_enabled and prev_sign is not None)
                else (False if self.diagnostics_enabled else None)
            )
            self._prev_grad_sign = cur_sign

            result = ModificationResult(
                iteration=it,
                objective_value=float(total.detach()),
                f_spatial=float(terms["f_spatial"].detach()),
                f_causal=float(terms["f_causal"].detach()),
                f_fidelity=float(terms["f_fidelity"].detach()),
                gradient_norm=grad_norm,
                cumulative_delta=cumulative_delta.copy(),
                grad_spatial_norm=(tier_a_metrics or {}).get("grad_spatial_norm"),
                grad_causal_norm=(tier_a_metrics or {}).get("grad_causal_norm"),
                grad_fidelity_norm=(tier_a_metrics or {}).get("grad_fidelity_norm"),
                grad_cosine_spatial_causal=(tier_a_metrics or {}).get("grad_cosine_spatial_causal"),
                grad_cosine_fairness_fidelity=(tier_a_metrics or {}).get("grad_cosine_fairness_fidelity"),
                sign_flipped=sign_flipped,
                dominant_term=(tier_a_metrics or {}).get("dominant_term"),
            )
            iterations.append(result)
```

- [ ] **Step 6: Run diagnostics tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_gradient_diagnostics.py -v`
Expected: all 5 tests pass.

- [ ] **Step 7: Run modifier regression tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_modifier.py famail_temporal/tests/test_modifier_integration.py -v`
Expected: all existing tests still pass.

### Task 5.4: Commit Phase 5

```bash
git add famail_temporal/config.py famail_temporal/algorithm/modifier.py \
        famail_temporal/tests/test_gradient_diagnostics.py
git commit -m "feat(algorithm): Tier A gradient decomposition in TrajectoryModifier"
```

### Phase 5 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) are the three `autograd.grad` calls with `retain_graph=True` correct, (b) does the `sign_flipped` tracker reset per trajectory and not leak across `modify_batch`, (c) does the `diagnostics_enabled=False` path produce bit-identical behavior to the pre-Phase-5 code, (d) does the "decomposed first-step equals combined first-step" test actually validate the math?

---

## Phase 6: `evaluation/runner.py` + `ExperimentResult`

**Files:**
- Create: `famail_temporal/evaluation/runner.py`
- Create: `famail_temporal/tests/test_runner.py`
- Modify: `famail_temporal/evaluation/__init__.py`

### Task 6.1: Write failing tests

- [ ] **Step 1: Create `famail_temporal/tests/test_runner.py`**

```python
"""Tests for evaluation.runner.run_experiment (synthetic, fast)."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.runner import (
    ExperimentResult, run_experiment, _parse_override_value, _apply_config_overrides,
)


@pytest.fixture
def tiny_bundle(monkeypatch):
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=0)
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    x, y, t_block = int(ix_x[0]), int(ix_y[0]), int(ix_t[0])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    trajs = []
    for tid in range(6):
        trajs.append(Trajectory(
            trajectory_id=tid, driver_id=tid % 2,
            states=[
                TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
                TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
            ],
        ))
    from dataclasses import replace
    bundle = replace(bundle, trajectories=trajs)
    monkeypatch.setattr(
        "famail_temporal.evaluation.runner._load_bundle",
        lambda **kwargs: bundle,
    )
    return bundle


def test_parse_override_value_tries_int_then_float_then_str():
    assert _parse_override_value("42") == 42
    assert _parse_override_value("1.5") == 1.5
    assert _parse_override_value("hello") == "hello"


def test_apply_config_overrides_raises_on_unknown_key():
    with pytest.raises(KeyError, match="NOT_A_REAL_KEY"):
        _apply_config_overrides({"NOT_A_REAL_KEY": 1})


def test_apply_config_overrides_restores_on_exit():
    original = config.EPSILON_BALL
    restore_fn = _apply_config_overrides({"EPSILON_BALL": 9.9})
    assert config.EPSILON_BALL == 9.9
    restore_fn()
    assert config.EPSILON_BALL == original


def test_run_experiment_returns_result_dataclass(tiny_bundle):
    result = run_experiment(k=2, max_trajectories=6)
    assert isinstance(result, ExperimentResult)
    assert result.grid_before.shape == (*tiny_bundle.pickup_3d.shape[:2], config.T, 4)
    assert result.grid_after.shape == result.grid_before.shape
    assert len(result.histories) <= 2
    assert set(result.augmented_trajs_before.keys()) == {0, 1}
    assert set(result.augmented_trajs_after.keys())  == {0, 1}


def test_run_experiment_overrides_restore(tiny_bundle):
    original = config.MAX_ITERATIONS
    _ = run_experiment(
        k=2, max_trajectories=6,
        config_overrides={"MAX_ITERATIONS": 2},
    )
    assert config.MAX_ITERATIONS == original


def test_run_experiment_unknown_override_raises(tiny_bundle):
    with pytest.raises(KeyError):
        run_experiment(k=2, config_overrides={"NOT_REAL": 7})


def test_run_experiment_no_diagnostics_produces_none_fields(tiny_bundle):
    result = run_experiment(k=2, max_trajectories=6, diagnostics_enabled=False)
    assert result.gradient_sensitivity_before is None
    assert result.gradient_sensitivity_after is None
    for hist in result.histories:
        for r in hist.iterations:
            assert r.grad_spatial_norm is None


def test_experiment_id_format_with_name(tiny_bundle):
    result = run_experiment(k=2, max_trajectories=6, name="my-run")
    assert "my-run" in result.experiment_id
    assert result.experiment_id.startswith("2")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_runner.py -v`
Expected: `ModuleNotFoundError`.

### Task 6.2: Implement `run_experiment`

- [ ] **Step 1: Create `famail_temporal/evaluation/runner.py`**

```python
"""Experiment runner: orchestrates the full FAMAIL pipeline."""

from __future__ import annotations
import argparse
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.fidelity.context import MultiStreamContextBuilder


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    config_snapshot: dict
    config_overrides: dict
    diagnostics_enabled: bool

    f_spatial_before: float
    f_spatial_after: float
    f_causal_before: float
    f_causal_after: float
    gini_dsr_before: float
    gini_dsr_after: float
    gini_asr_before: float
    gini_asr_after: float

    grid_before: np.ndarray
    grid_after: np.ndarray
    per_unit_attribution_before: np.ndarray
    per_unit_attribution_signed_before: np.ndarray

    gradient_sensitivity_before: Optional[np.ndarray]
    gradient_sensitivity_after: Optional[np.ndarray]

    modified_trajectory_ids: List[int]
    histories: List[ModificationHistory]
    top_k_scores: List[float]

    augmented_trajs_before: Dict[int, list]
    augmented_trajs_after: Dict[int, list]


def _parse_override_value(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_config_overrides(overrides: Dict[str, Any]):
    if overrides is None:
        return lambda: None
    for key in overrides:
        if not hasattr(config, key):
            raise KeyError(
                f"Unknown config override '{key}'. Only existing config.* "
                f"attributes can be overridden."
            )
    originals: Dict[str, Any] = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    def restore():
        for key, value in originals.items():
            setattr(config, key, value)

    return restore


def _load_bundle(max_trajectories: Optional[int], max_drivers: Optional[int]) -> DataBundle:
    return DataBundle.load(
        max_trajectories=max_trajectories, max_drivers=max_drivers,
    )


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name).strip("-")


def _generate_experiment_id(name: Optional[str]) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if name:
        return f"{timestamp}_{_slugify(name)}"
    return timestamp


def _scalar_metrics_from_grid(grid: np.ndarray) -> dict:
    return {
        "f_spatial": float(1.0 - np.nansum(grid[..., 0])),
        "f_causal":  float(1.0 - np.nansum(grid[..., 1])),
        "gini_dsr":  float(np.nansum(grid[..., 2])),
        "gini_asr":  float(np.nansum(grid[..., 3])),
    }


def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    output_root: Optional[Path] = None,
    max_trajectories: Optional[int] = None,
    max_drivers: Optional[int] = None,
    k: int = 100,
    diagnostics_enabled: bool = True,
) -> ExperimentResult:
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")

    restore_config = _apply_config_overrides(config_overrides or {})
    try:
        experiment_id = _generate_experiment_id(name)
        bundle = _load_bundle(max_trajectories, max_drivers)

        grid_before = build_fairness_grid(bundle)
        metrics_before = _scalar_metrics_from_grid(grid_before)
        augmented_before = augment_trajectories(bundle.trajectories, grid_before)
        attr_unsigned, attr_signed = compute_per_unit_attribution(bundle)

        scored = rank_trajectories(bundle.trajectories, attr_unsigned, bundle.unit_map)
        if k > len(scored):
            raise ValueError(
                f"k={k} exceeds ranked trajectory count {len(scored)}. "
                f"Reduce k or widen max_trajectories."
            )
        top_k_indices = select_top_k(scored, k=k)
        if not top_k_indices:
            raise ValueError(
                "Top-k is empty - no trajectories with strictly positive "
                "attribution were found. Inspect per_unit_attribution_before; "
                "if all zeros, demographics carry no signal on this bundle."
            )
        top_k_scores = [scored[i][1] for i in range(len(top_k_indices))]
        top_k_trajs = [bundle.trajectories[i] for i in top_k_indices]

        objective = FAMAILObjective(bundle)
        try:
            ms_builder = MultiStreamContextBuilder(bundle.multi_stream)
        except Exception:
            ms_builder = None
        modifier = TrajectoryModifier(
            objective=objective, bundle=bundle,
            multi_stream_builder=ms_builder,
            diagnostics_enabled=diagnostics_enabled,
        )
        histories = modifier.modify_batch(top_k_trajs)

        pickup_after = modifier.current_pickup_3d()
        grid_after = build_fairness_grid(bundle, pickup_3d=pickup_after)
        metrics_after = _scalar_metrics_from_grid(grid_after)

        modified_by_tid = {h.original.trajectory_id: h.modified for h in histories}
        trajs_after = [
            modified_by_tid.get(t.trajectory_id, t) for t in bundle.trajectories
        ]
        augmented_after = augment_trajectories(trajs_after, grid_after)

        snapshot = {
            k: getattr(config, k) for k in dir(config)
            if k.isupper() and not k.startswith("_")
        }

        return ExperimentResult(
            experiment_id=experiment_id,
            config_snapshot=snapshot,
            config_overrides=dict(config_overrides or {}),
            diagnostics_enabled=diagnostics_enabled,
            f_spatial_before=metrics_before["f_spatial"],
            f_spatial_after=metrics_after["f_spatial"],
            f_causal_before=metrics_before["f_causal"],
            f_causal_after=metrics_after["f_causal"],
            gini_dsr_before=metrics_before["gini_dsr"],
            gini_dsr_after=metrics_after["gini_dsr"],
            gini_asr_before=metrics_before["gini_asr"],
            gini_asr_after=metrics_after["gini_asr"],
            grid_before=grid_before,
            grid_after=grid_after,
            per_unit_attribution_before=attr_unsigned,
            per_unit_attribution_signed_before=attr_signed,
            gradient_sensitivity_before=None,
            gradient_sensitivity_after=None,
            modified_trajectory_ids=[h.original.trajectory_id for h in histories],
            histories=histories,
            top_k_scores=top_k_scores,
            augmented_trajs_before=augmented_before,
            augmented_trajs_after=augmented_after,
        )
    finally:
        restore_config()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="famail_temporal.evaluation.runner")
    p.add_argument("--name", default=None)
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--max-drivers", type=int, default=None)
    p.add_argument("-k", type=int, default=100)
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--override", action="append", default=[],
                   help="KEY=VALUE override (repeatable)")
    return p


def _parse_cli_overrides(raw: list[str]) -> dict:
    out: Dict[str, Any] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"Invalid --override entry '{entry}', expected KEY=VALUE")
        k, v = entry.split("=", 1)
        out[k] = _parse_override_value(v)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_cli_overrides(args.override)
    result = run_experiment(
        config_overrides=overrides,
        name=args.name,
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
        k=args.k,
        diagnostics_enabled=not args.no_diagnostics,
    )
    print(f"[runner] experiment_id = {result.experiment_id}")
    print(f"[runner]   F_spatial: {result.f_spatial_before:.4f} -> {result.f_spatial_after:.4f}")
    print(f"[runner]   F_causal:  {result.f_causal_before:.4f} -> {result.f_causal_after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Extend `famail_temporal/evaluation/__init__.py`**

```python
"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.runner import ExperimentResult, run_experiment

__all__ = [
    "ExperimentResult",
    "augment_trajectories",
    "build_fairness_grid",
    "run_experiment",
]
```

- [ ] **Step 3: Run tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_runner.py -v`
Expected: all tests pass.

### Task 6.3: Commit Phase 6

```bash
git add famail_temporal/evaluation/__init__.py \
        famail_temporal/evaluation/runner.py \
        famail_temporal/tests/test_runner.py
git commit -m "feat(evaluation): add run_experiment orchestration + CLI skeleton"
```

### Phase 6 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) does the override restore path survive exceptions (try/finally), (b) does the swap-in-modified list comprehension preserve bundle.trajectories, (c) is `config_snapshot` captured post-override (intended), (d) are error messages actionable?

---

## Phase 7: `evaluation/persistence.py` + gzip fallback + provenance

**Files:**
- Create: `famail_temporal/evaluation/persistence.py`
- Create: `famail_temporal/tests/test_persistence.py`
- Modify: `famail_temporal/evaluation/runner.py` (CLI hookup)

### Task 7.1: Write failing tests for persistence

- [ ] **Step 1: Create `famail_temporal/tests/test_persistence.py`**

```python
"""Tests for evaluation.persistence."""
import json
import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.persistence import (
    write, _conditional_gzip_pickle,
)
from famail_temporal.evaluation.runner import ExperimentResult


def _fake_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="2026-04-16T00-00-00_test",
        config_snapshot={"EPSILON_BALL": 2.0, "T": 4},
        config_overrides={"EPSILON_BALL": 2.0},
        diagnostics_enabled=True,
        f_spatial_before=0.3, f_spatial_after=0.4,
        f_causal_before=0.5,  f_causal_after=0.55,
        gini_dsr_before=0.7,  gini_dsr_after=0.6,
        gini_asr_before=0.8,  gini_asr_after=0.8,
        grid_before=np.ones((4, 4, 2, 4), dtype=np.float32),
        grid_after=np.ones((4, 4, 2, 4), dtype=np.float32) * 2.0,
        per_unit_attribution_before=np.arange(10, dtype=np.float32),
        per_unit_attribution_signed_before=np.arange(10, dtype=np.float32),
        gradient_sensitivity_before=None,
        gradient_sensitivity_after=None,
        modified_trajectory_ids=[0, 1],
        histories=[],
        top_k_scores=[0.9, 0.5],
        augmented_trajs_before={0: [[[1, 2, 3, 0, 0.1, 0.2, 0.3, 0.4]]]},
        augmented_trajs_after={0:  [[[1, 2, 3, 0, 0.2, 0.3, 0.4, 0.5]]]},
    )


def test_write_creates_directory(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert out_dir.is_dir()
    assert out_dir.name == "2026-04-16T00-00-00_test"


def test_write_produces_metrics_json_with_provenance(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    data = json.loads((out_dir / "metrics.json").read_text())
    assert data["experiment_id"] == "2026-04-16T00-00-00_test"
    assert "git_sha" in data
    assert "git_dirty" in data
    assert "command_line" in data
    assert "timestamp_utc" in data
    assert data["diagnostics_enabled"] is True
    assert data["metrics_before"]["f_spatial"] == pytest.approx(0.3)
    assert data["metrics_after"]["f_spatial"] == pytest.approx(0.4)
    assert data["deltas"]["f_spatial"] == pytest.approx(0.1, abs=1e-6)


def test_write_produces_grid_pickles_with_dict_schema(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    for name in ("grid_before.pkl", "grid_after.pkl"):
        with open(out_dir / name, "rb") as f:
            obj = pickle.load(f)
        assert set(obj.keys()) == {"grid", "channel_names", "time_blocks", "active_mask"}
        assert obj["grid"].shape == (4, 4, 2, 4)


def test_write_produces_modified_trajectory_ids_json(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    data = json.loads((out_dir / "modified_trajectory_ids.json").read_text())
    assert data["modified_trajectory_ids"] == [0, 1]


def test_write_skips_sensitivity_pickles_when_diagnostics_disabled(tmp_path):
    result = _fake_result()
    from dataclasses import replace
    result = replace(result, diagnostics_enabled=False)
    out_dir = write(result, output_root=tmp_path)
    assert not (out_dir / "gradient_sensitivity_before.pkl").exists()
    assert not (out_dir / "gradient_sensitivity_after.pkl").exists()


def test_conditional_gzip_uncompressed_when_small(tmp_path):
    obj = {"a": list(range(100))}
    path = tmp_path / "small.pkl"
    written = _conditional_gzip_pickle(obj, path)
    assert written.suffix == ".pkl"
    assert path.exists()


def test_conditional_gzip_compressed_when_large(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "famail_temporal.evaluation.persistence._gzip_threshold_bytes",
        lambda: 10,
    )
    obj = {"a": list(range(100))}
    path = tmp_path / "big.pkl"
    written = _conditional_gzip_pickle(obj, path)
    assert written.suffix == ".gz"
    assert written.exists()
    with gzip.open(written, "rb") as f:
        roundtrip = pickle.load(f)
    assert roundtrip == obj


def test_write_csv_files_exist(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert (out_dir / "per_unit_attribution.csv").exists()
    assert (out_dir / "trajectories.csv").exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_persistence.py -v`
Expected: `ModuleNotFoundError`.

### Task 7.2: Implement `persistence.py`

- [ ] **Step 1: Create `famail_temporal/evaluation/persistence.py`**

```python
"""Persistence layer for ExperimentResult."""

from __future__ import annotations
import csv
import datetime as _dt
import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from famail_temporal import config
from famail_temporal.evaluation.runner import ExperimentResult


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def _command_line() -> str:
    return " ".join(sys.argv)


def _gzip_threshold_bytes() -> int:
    return 500 * 1024 * 1024


def _conditional_gzip_pickle(obj: Any, path: Path) -> Path:
    data = pickle.dumps(obj, protocol=4)
    if len(data) > _gzip_threshold_bytes():
        gz_path = path.with_suffix(".pkl.gz")
        with gzip.open(gz_path, "wb") as f:
            f.write(data)
        return gz_path
    path.write_bytes(data)
    return path


def _grid_payload(grid: np.ndarray, active_mask: np.ndarray) -> dict:
    return {
        "grid": grid,
        "channel_names": ["spatial_attr", "causal_attr", "gini_decomp_dsr", "gini_decomp_asr"],
        "time_blocks": list(config.TIME_BLOCKS),
        "active_mask": active_mask,
    }


def _sensitivity_payload(grid: np.ndarray, active_mask: np.ndarray) -> dict:
    return {
        "grid": grid,
        "channel_names": ["dF_spatial_dp", "dF_causal_dp"],
        "time_blocks": list(config.TIME_BLOCKS),
        "active_mask": active_mask,
    }


def _diagnostics_summary(result: ExperimentResult) -> dict | None:
    if not result.diagnostics_enabled or not result.histories:
        return None
    all_iters = [r for h in result.histories for r in h.iterations]
    if not all_iters:
        return None
    def _mean(attr):
        vals = [getattr(r, attr) for r in all_iters if getattr(r, attr) is not None]
        return float(np.mean(vals)) if vals else None
    dom = [r.dominant_term for r in all_iters if r.dominant_term is not None]
    total = len(dom) or 1
    return {
        "mean_grad_spatial_norm":       _mean("grad_spatial_norm"),
        "mean_grad_causal_norm":        _mean("grad_causal_norm"),
        "mean_grad_fidelity_norm":      _mean("grad_fidelity_norm"),
        "mean_cos_spatial_causal":      _mean("grad_cosine_spatial_causal"),
        "mean_cos_fairness_fidelity":   _mean("grad_cosine_fairness_fidelity"),
        "frac_iters_spatial_dominant":  dom.count("spatial") / total,
        "frac_iters_causal_dominant":   dom.count("causal") / total,
        "frac_iters_fidelity_dominant": dom.count("fidelity") / total,
    }


def _convergence_summary(result: ExperimentResult) -> dict:
    if not result.histories:
        return {"n_converged": 0, "n_max_iter": 0,
                "mean_total_iterations": 0.0, "mean_final_grad_norm": 0.0}
    n_conv = sum(1 for h in result.histories if h.converged)
    n_max = len(result.histories) - n_conv
    total_iters = [h.total_iterations for h in result.histories]
    finals = [h.iterations[-1].gradient_norm for h in result.histories if h.iterations]
    return {
        "n_converged": n_conv,
        "n_max_iter": n_max,
        "mean_total_iterations": float(np.mean(total_iters)) if total_iters else 0.0,
        "mean_final_grad_norm": float(np.mean(finals)) if finals else 0.0,
    }


def _write_trajectories_csv(result: ExperimentResult, path: Path) -> None:
    from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
    headers = [
        "trajectory_id", "driver_id",
        "original_pickup_cell_x", "original_pickup_cell_y",
        "modified_pickup_cell_x", "modified_pickup_cell_y",
        "pickup_t_block", "delta_x", "delta_y",
        "attribution_score", "rank",
        "converged", "total_iterations",
        "initial_objective", "final_objective",
        "f_spatial_initial", "f_spatial_final",
        "f_causal_initial", "f_causal_final",
        "f_fidelity_initial", "f_fidelity_final",
        "mean_grad_spatial_norm", "mean_grad_causal_norm", "mean_grad_fidelity_norm",
        "frac_iters_spatial_dominant", "frac_iters_causal_dominant", "frac_iters_fidelity_dominant",
        "mean_cos_spatial_causal", "mean_cos_fairness_fidelity",
        "sign_flip_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rank, (h, score) in enumerate(zip(result.histories, result.top_k_scores), start=1):
            orig = h.original.pickup_cell
            modc = h.modified.pickup_cell
            tb = hour_to_block_index(time_bucket_to_hour(h.original.pickup_state.time_bucket))
            iters = h.iterations
            def _first(attr):
                return getattr(iters[0], attr) if iters else 0.0
            def _last(attr):
                return getattr(iters[-1], attr) if iters else 0.0
            def _mean_none(attr):
                vals = [getattr(r, attr) for r in iters if getattr(r, attr) is not None]
                return float(np.mean(vals)) if vals else ""
            def _frac(term):
                if not iters or iters[0].dominant_term is None:
                    return ""
                return sum(1 for r in iters if r.dominant_term == term) / len(iters)
            sign_flip_rate = (
                sum(1 for r in iters if r.sign_flipped) / len(iters)
                if iters and iters[0].sign_flipped is not None else ""
            )
            writer.writerow([
                h.original.trajectory_id, h.original.driver_id,
                orig[0], orig[1], modc[0], modc[1],
                tb, modc[0] - orig[0], modc[1] - orig[1],
                score, rank,
                h.converged, h.total_iterations,
                _first("objective_value"), _last("objective_value"),
                _first("f_spatial"),       _last("f_spatial"),
                _first("f_causal"),        _last("f_causal"),
                _first("f_fidelity"),      _last("f_fidelity"),
                _mean_none("grad_spatial_norm"),
                _mean_none("grad_causal_norm"),
                _mean_none("grad_fidelity_norm"),
                _frac("spatial"), _frac("causal"), _frac("fidelity"),
                _mean_none("grad_cosine_spatial_causal"),
                _mean_none("grad_cosine_fairness_fidelity"),
                sign_flip_rate,
            ])


def _write_per_unit_attribution_csv(result: ExperimentResult, path: Path, mask_3d: np.ndarray) -> None:
    ix_x, ix_y, ix_t = np.where(mask_3d)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "unit_idx", "cell_x", "cell_y", "t_block", "flat_cell_id",
            "spatial_attr_before", "spatial_attr_after",
            "causal_attr_before",  "causal_attr_after",
            "causal_attr_signed_before",
            "gini_dsr_contrib_before", "gini_dsr_contrib_after",
            "gini_asr_contrib_before", "gini_asr_contrib_after",
        ])
        for i, (x, y, t) in enumerate(zip(ix_x, ix_y, ix_t)):
            writer.writerow([
                i, int(x), int(y), int(t), int(x) * config.GRID_DIMS[1] + int(y),
                float(result.grid_before[x, y, t, 0]),
                float(result.grid_after [x, y, t, 0]),
                float(result.grid_before[x, y, t, 1]),
                float(result.grid_after [x, y, t, 1]),
                float(result.per_unit_attribution_signed_before[i]),
                float(result.grid_before[x, y, t, 2]),
                float(result.grid_after [x, y, t, 2]),
                float(result.grid_before[x, y, t, 3]),
                float(result.grid_after [x, y, t, 3]),
            ])


def _coerce_json(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_coerce_json(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def write(result: ExperimentResult, output_root: Path, bundle=None) -> Path:
    output_root = Path(output_root)
    out_dir = output_root / result.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    active_mask = bundle.mask_3d if bundle is not None else ~np.isnan(result.grid_before[..., 0])

    artifact_paths: Dict[str, str] = {}
    file_sizes: Dict[str, int] = {}

    for name, grid in [("grid_before", result.grid_before),
                       ("grid_after",  result.grid_after)]:
        path = out_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(_grid_payload(grid, active_mask), f, protocol=4)
        artifact_paths[name] = path.name
        file_sizes[name] = path.stat().st_size

    for name, obj in [("augmented_trajs_before", result.augmented_trajs_before),
                      ("augmented_trajs_after",  result.augmented_trajs_after)]:
        base = out_dir / f"{name}.pkl"
        written = _conditional_gzip_pickle(obj, base)
        artifact_paths[name] = written.name
        file_sizes[name] = written.stat().st_size

    mod_ids_payload = {
        "modified_trajectory_ids": list(result.modified_trajectory_ids),
        "original_pickup_cells": {
            str(h.original.trajectory_id): list(h.original.pickup_cell)
            for h in result.histories
        },
        "modified_pickup_cells": {
            str(h.original.trajectory_id): list(h.modified.pickup_cell)
            for h in result.histories
        },
    }
    path = out_dir / "modified_trajectory_ids.json"
    path.write_text(json.dumps(mod_ids_payload, indent=2))
    artifact_paths["modified_trajectory_ids"] = path.name
    file_sizes["modified_trajectory_ids"] = path.stat().st_size

    path = out_dir / "histories.pkl"
    with open(path, "wb") as f:
        pickle.dump(result.histories, f, protocol=4)
    artifact_paths["histories"] = path.name
    file_sizes["histories"] = path.stat().st_size

    if result.diagnostics_enabled and result.gradient_sensitivity_before is not None:
        for name, grid in [
            ("gradient_sensitivity_before", result.gradient_sensitivity_before),
            ("gradient_sensitivity_after",  result.gradient_sensitivity_after),
        ]:
            path = out_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(_sensitivity_payload(grid, active_mask), f, protocol=4)
            artifact_paths[name] = path.name
            file_sizes[name] = path.stat().st_size

    path = out_dir / "trajectories.csv"
    _write_trajectories_csv(result, path)
    artifact_paths["trajectories_csv"] = path.name
    file_sizes["trajectories_csv"] = path.stat().st_size

    path = out_dir / "per_unit_attribution.csv"
    _write_per_unit_attribution_csv(result, path, active_mask)
    artifact_paths["per_unit_attribution_csv"] = path.name
    file_sizes["per_unit_attribution_csv"] = path.stat().st_size

    metrics = {
        "experiment_id": result.experiment_id,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "command_line": _command_line(),
        "config_snapshot": {k: _coerce_json(v) for k, v in result.config_snapshot.items()},
        "config_overrides": {k: _coerce_json(v) for k, v in result.config_overrides.items()},
        "diagnostics_enabled": result.diagnostics_enabled,
        "dataset": {
            "n_trajectories": sum(len(v) for v in result.augmented_trajs_before.values()),
            "n_drivers": len(result.augmented_trajs_before),
            "n_active_units": int(np.sum(active_mask)),
        },
        "k_modified": len(result.histories),
        "metrics_before": {
            "f_spatial": result.f_spatial_before, "f_causal": result.f_causal_before,
            "gini_dsr":  result.gini_dsr_before,  "gini_asr":  result.gini_asr_before,
        },
        "metrics_after": {
            "f_spatial": result.f_spatial_after,  "f_causal": result.f_causal_after,
            "gini_dsr":  result.gini_dsr_after,   "gini_asr":  result.gini_asr_after,
        },
        "deltas": {
            "f_spatial": result.f_spatial_after - result.f_spatial_before,
            "f_causal":  result.f_causal_after  - result.f_causal_before,
            "gini_dsr":  result.gini_dsr_after  - result.gini_dsr_before,
            "gini_asr":  result.gini_asr_after  - result.gini_asr_before,
        },
        "convergence_summary": _convergence_summary(result),
        "diagnostics_summary": _diagnostics_summary(result),
        "artifact_paths": artifact_paths,
        "file_sizes_bytes": file_sizes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return out_dir
```

- [ ] **Step 2: Run the persistence tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_persistence.py -v`
Expected: all 8 tests pass.

### Task 7.3: Wire persistence into the CLI

- [ ] **Step 1: Update `main()` in `famail_temporal/evaluation/runner.py`**

```python
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_cli_overrides(args.override)
    result = run_experiment(
        config_overrides=overrides,
        name=args.name,
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
        k=args.k,
        diagnostics_enabled=not args.no_diagnostics,
    )
    from famail_temporal.evaluation.persistence import write
    output_root = Path(config.PACKAGE_ROOT) / "results"
    out_dir = write(result, output_root=output_root)
    print(f"[runner] experiment_id = {result.experiment_id}")
    print(f"[runner] results_dir  = {out_dir}")
    print(f"[runner]   F_spatial: {result.f_spatial_before:.4f} -> {result.f_spatial_after:.4f}")
    print(f"[runner]   F_causal:  {result.f_causal_before:.4f} -> {result.f_causal_after:.4f}")
    return 0
```

- [ ] **Step 2: Re-run tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_runner.py famail_temporal/tests/test_persistence.py -v`
Expected: all tests pass.

### Task 7.4: Commit Phase 7

```bash
git add famail_temporal/evaluation/persistence.py \
        famail_temporal/evaluation/runner.py \
        famail_temporal/tests/test_persistence.py
git commit -m "feat(evaluation): persistence layer + conditional gzip + provenance"
```

### Phase 7 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) is `metrics.json` written LAST so `artifact_paths`/`file_sizes` are accurate, (b) does the gzip threshold behave correctly under monkeypatch, (c) does the sensitivity-pickle gate work, (d) are CSV rows human-readable?

---

## Phase 8: `evaluation/report.py` — tables-only markdown

**Files:**
- Create: `famail_temporal/evaluation/report.py`
- Create: `famail_temporal/tests/test_report.py`
- Modify: `famail_temporal/evaluation/runner.py`

### Task 8.1: Write failing tests

- [ ] **Step 1: Create `famail_temporal/tests/test_report.py`**

```python
"""Tests for evaluation.report.render."""
import json
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.report import render
from famail_temporal.evaluation.persistence import write
from famail_temporal.evaluation.runner import ExperimentResult


def _fake_result() -> ExperimentResult:
    return ExperimentResult(
        experiment_id="2026-04-16T00-00-00_test",
        config_snapshot={"EPSILON_BALL": 2.0, "T": 4, "MAX_ITERATIONS": 50},
        config_overrides={"EPSILON_BALL": 2.0},
        diagnostics_enabled=True,
        f_spatial_before=0.3, f_spatial_after=0.4,
        f_causal_before=0.5,  f_causal_after=0.55,
        gini_dsr_before=0.7,  gini_dsr_after=0.6,
        gini_asr_before=0.8,  gini_asr_after=0.8,
        grid_before=np.ones((4, 4, 2, 4), dtype=np.float32),
        grid_after=np.ones((4, 4, 2, 4), dtype=np.float32) * 2.0,
        per_unit_attribution_before=np.arange(10, dtype=np.float32),
        per_unit_attribution_signed_before=np.arange(10, dtype=np.float32),
        gradient_sensitivity_before=None,
        gradient_sensitivity_after=None,
        modified_trajectory_ids=[], histories=[], top_k_scores=[],
        augmented_trajs_before={}, augmented_trajs_after={},
    )


def test_render_produces_report_md(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report_path = render(out_dir)
    assert report_path.exists()
    assert report_path.name == "report.md"


def test_report_contains_header_and_sections(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report = render(out_dir).read_text()
    assert result.experiment_id in report
    assert "Config" in report
    assert "Fairness" in report
    assert "Artifact" in report


def test_report_marks_overridden_config_values_bold(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    report = render(out_dir).read_text()
    assert "**EPSILON_BALL**" in report or "**2.0**" in report


def test_report_reads_only_from_disk(tmp_path):
    result = _fake_result()
    out_dir = write(result, output_root=tmp_path)
    assert render.__code__.co_varnames[0] == "output_dir"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_report.py -v`
Expected: `ModuleNotFoundError`.

### Task 8.2: Implement `report.py`

- [ ] **Step 1: Create `famail_temporal/evaluation/report.py`**

```python
"""Report generator: reads {output_dir} from disk and writes report.md."""

from __future__ import annotations
import csv
import json
from pathlib import Path


def render(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text())

    lines: list[str] = []
    _header(lines, metrics)
    _config_table(lines, metrics)
    _dataset_summary(lines, metrics)
    _fairness_table(lines, metrics)
    _convergence_summary(lines, metrics)
    if metrics.get("diagnostics_enabled"):
        _diagnostics_summary(lines, metrics)
    _top_k_table(lines, output_dir / "trajectories.csv")
    _key_findings(lines, metrics)
    _artifact_index(lines, metrics)

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _header(lines, m):
    lines.append(f"# Experiment Report - `{m['experiment_id']}`\n")
    lines.append(f"- **Timestamp (UTC):** {m['timestamp_utc']}")
    lines.append(f"- **Git SHA:** `{m['git_sha']}`"
                 + ("  **(dirty)**" if m.get("git_dirty") else ""))
    lines.append(f"- **Command line:** `{m['command_line']}`")
    lines.append("")


def _config_table(lines, m):
    lines.append("## Config\n")
    lines.append("| Param | Value |")
    lines.append("|---|---|")
    overridden = set(m.get("config_overrides", {}).keys())
    for k, v in sorted(m["config_snapshot"].items()):
        key_cell = f"**{k}**" if k in overridden else k
        val_cell = f"**{v}**"  if k in overridden else str(v)
        lines.append(f"| {key_cell} | {val_cell} |")
    lines.append("")


def _dataset_summary(lines, m):
    ds = m.get("dataset", {})
    lines.append("## Dataset\n")
    lines.append("| n_trajectories | n_drivers | n_active_units | k_modified |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {ds.get('n_trajectories', '-')} | {ds.get('n_drivers', '-')} "
                 f"| {ds.get('n_active_units', '-')} | {m.get('k_modified', '-')} |")
    lines.append("")


def _fairness_table(lines, m):
    mb = m["metrics_before"]; ma = m["metrics_after"]; d = m["deltas"]
    def _arrow(delta):
        if delta > 1e-6:  return "up"
        if delta < -1e-6: return "down"
        return "-"
    lines.append("## Fairness\n")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k in ("f_spatial", "f_causal", "gini_dsr", "gini_asr"):
        lines.append(f"| `{k}` | {mb[k]:.4f} | {ma[k]:.4f} | "
                     f"{d[k]:+.4f} {_arrow(d[k])} |")
    lines.append("")


def _convergence_summary(lines, m):
    cs = m.get("convergence_summary", {})
    total = cs.get("n_converged", 0) + cs.get("n_max_iter", 0)
    lines.append("## Convergence\n")
    lines.append(f"- Converged: {cs.get('n_converged')} / {total}")
    lines.append(f"- Mean total iterations: {cs.get('mean_total_iterations', 0.0):.2f}")
    lines.append(f"- Mean final gradient norm: {cs.get('mean_final_grad_norm', 0.0):.4f}")
    lines.append("")


def _diagnostics_summary(lines, m):
    ds = m.get("diagnostics_summary") or {}
    lines.append("## Gradient diagnostics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in ("mean_grad_spatial_norm", "mean_grad_causal_norm", "mean_grad_fidelity_norm",
              "mean_cos_spatial_causal", "mean_cos_fairness_fidelity",
              "frac_iters_spatial_dominant", "frac_iters_causal_dominant",
              "frac_iters_fidelity_dominant"):
        v = ds.get(k)
        lines.append(f"| `{k}` | {'' if v is None else f'{v:.4f}'} |")
    lines.append("")


def _top_k_table(lines, csv_path: Path):
    lines.append("## Top 10 modified trajectories\n")
    if not csv_path.exists():
        lines.append("_No trajectories._\n")
        return
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        lines.append("_No trajectories._\n")
        return
    cols = ["rank", "trajectory_id", "driver_id",
            "original_pickup_cell_x", "original_pickup_cell_y",
            "modified_pickup_cell_x", "modified_pickup_cell_y",
            "delta_x", "delta_y",
            "converged", "total_iterations",
            "initial_objective", "final_objective"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows[:10]:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")


def _key_findings(lines, m):
    d = m["deltas"]
    findings: list[str] = []
    if d["f_spatial"] > 0:
        findings.append(f"F_spatial improved by {d['f_spatial']:+.4f}.")
    elif d["f_spatial"] < 0:
        findings.append(f"F_spatial regressed by {d['f_spatial']:+.4f}.")
    if d["f_causal"] > 0:
        findings.append(f"F_causal improved by {d['f_causal']:+.4f}.")
    elif d["f_causal"] < 0:
        findings.append(f"F_causal regressed by {d['f_causal']:+.4f}.")
    if abs(d["gini_asr"]) < 1e-6:
        findings.append("ASR Gini unchanged - only pickups are modified by the framework.")
    if m.get("diagnostics_enabled") and m.get("diagnostics_summary"):
        ds = m["diagnostics_summary"]
        dom = max(
            [("spatial", ds.get("frac_iters_spatial_dominant") or 0.0),
             ("causal",  ds.get("frac_iters_causal_dominant")  or 0.0),
             ("fidelity",ds.get("frac_iters_fidelity_dominant")or 0.0)],
            key=lambda kv: kv[1],
        )
        findings.append(f"Dominant gradient term: `{dom[0]}` in {dom[1]:.1%} of iterations.")
    lines.append("## Key findings\n")
    if not findings:
        lines.append("_No notable findings._\n")
    else:
        for f in findings:
            lines.append(f"- {f}")
    lines.append("")


def _artifact_index(lines, m):
    lines.append("## Artifacts\n")
    paths = m.get("artifact_paths", {})
    sizes = m.get("file_sizes_bytes", {})
    lines.append("| Artifact | Path | Size (bytes) |")
    lines.append("|---|---|---:|")
    for name, path in sorted(paths.items()):
        lines.append(f"| {name} | `{path}` | {sizes.get(name, '-')} |")
    lines.append("")
```

- [ ] **Step 2: Run the tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_report.py -v`
Expected: all 4 tests pass.

### Task 8.3: Hook the report into the CLI

- [ ] **Step 1: Update `main()` in `famail_temporal/evaluation/runner.py`**

```python
def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_cli_overrides(args.override)
    result = run_experiment(
        config_overrides=overrides,
        name=args.name,
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
        k=args.k,
        diagnostics_enabled=not args.no_diagnostics,
    )
    from famail_temporal.evaluation.persistence import write
    from famail_temporal.evaluation.report import render
    output_root = Path(config.PACKAGE_ROOT) / "results"
    out_dir = write(result, output_root=output_root)
    render(out_dir)
    print(f"[runner] experiment_id = {result.experiment_id}")
    print(f"[runner] results_dir  = {out_dir}")
    print(f"[runner] report       = {out_dir / 'report.md'}")
    print(f"[runner]   F_spatial: {result.f_spatial_before:.4f} -> {result.f_spatial_after:.4f}")
    print(f"[runner]   F_causal:  {result.f_causal_before:.4f} -> {result.f_causal_after:.4f}")
    return 0
```

### Task 8.4: Commit Phase 8

```bash
git add famail_temporal/evaluation/report.py \
        famail_temporal/evaluation/runner.py \
        famail_temporal/tests/test_report.py
git commit -m "feat(evaluation): tables-only markdown report"
```

### Phase 8 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) `render` reads only from disk, (b) overridden config values are visually distinct, (c) key-findings list is grounded in actual deltas, (d) empty-trajectory cases don't crash.

---

## Phase 9: Tier C `compute_gradient_sensitivity`

**Files:**
- Create: `famail_temporal/evaluation/diagnostics.py`
- Create: `famail_temporal/tests/test_gradient_sensitivity.py`
- Modify: `famail_temporal/evaluation/__init__.py`
- Modify: `famail_temporal/evaluation/runner.py`

### Task 9.1: Write failing tests

- [ ] **Step 1: Create `famail_temporal/tests/test_gradient_sensitivity.py`**

```python
"""Tests for evaluation.diagnostics.compute_gradient_sensitivity."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_returns_correct_shape():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    gx, gy = bundle.pickup_3d.shape[:2]
    assert sens.shape == (gx, gy, config.T, 2)
    assert sens.dtype == np.float32


def test_inactive_cells_are_nan():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    inactive = ~bundle.mask_3d
    assert np.isnan(sens[inactive]).all()


def test_active_cells_are_finite():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    active = bundle.mask_3d
    for c in range(2):
        assert np.isfinite(sens[active, c]).all()


def test_sensitivity_changes_under_pickup_modification():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=3)
    sens_a = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    pickup_mod = bundle.pickup_3d.copy()
    active_ix = np.argwhere(bundle.mask_3d)
    x0, y0, t0 = active_ix[0]
    pickup_mod[x0, y0, t0] += 1.0
    sens_b = compute_gradient_sensitivity(bundle, pickup_mod)
    assert not np.allclose(
        sens_a[bundle.mask_3d], sens_b[bundle.mask_3d],
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_gradient_sensitivity.py -v`
Expected: `ModuleNotFoundError`.

### Task 9.2: Implement `compute_gradient_sensitivity`

- [ ] **Step 1: Create `famail_temporal/evaluation/diagnostics.py`**

```python
"""Tier C global gradient sensitivity field."""

from __future__ import annotations
import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.causal import compute_fcausal
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fairness.spatial import compute_fspatial


def compute_gradient_sensitivity(
    bundle: DataBundle,
    pickup_3d: np.ndarray,
) -> np.ndarray:
    """Global dF/dp sensitivity grid of shape (gx, gy, T, 2).

    Channels:
        0: d F_spatial / d pickup[x, y, t]
        1: d F_causal  / d pickup[x, y, t]

    Inactive cells are NaN. Fidelity channel omitted - F_fidelity is
    per-trajectory and has no global per-cell gradient.
    """
    mask = bundle.mask_3d
    mask_t = torch.from_numpy(mask)
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()

    # Channel 0 - F_spatial
    pickup_tensor_a = torch.from_numpy(pickup_3d.copy()).float().requires_grad_(True)
    pickup_N = pickup_tensor_a[mask_t]
    f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_N)
    grad_sp = torch.autograd.grad(f_spatial, pickup_tensor_a)[0].detach().numpy()

    # Channel 1 - F_causal
    pickup_tensor_b = torch.from_numpy(pickup_3d.copy()).float().requires_grad_(True)
    pickup_N_b = pickup_tensor_b[mask_t]
    D_clamped = torch.clamp(pickup_N_b, min=config.DEMAND_FLOOR)
    with torch.no_grad():
        g0_D = torch.from_numpy(
            np.asarray(bundle.g0_func(D_clamped.detach().numpy()), dtype=np.float32),
        )
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal, _ = compute_fcausal(
        demand_N=pickup_N_b, supply_N=active_N,
        g0_D_N=g0_D,
        I_minus_H_demo=tensors["I_minus_H_demo"], M=tensors["M"],
    )
    grad_ca = torch.autograd.grad(f_causal, pickup_tensor_b)[0].detach().numpy()

    gx, gy = bundle.pickup_3d.shape[:2]
    sens = np.full((gx, gy, config.T, 2), np.nan, dtype=np.float32)
    sens[..., 0][mask] = grad_sp[mask]
    sens[..., 1][mask] = grad_ca[mask]
    return sens
```

- [ ] **Step 2: Run tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_gradient_sensitivity.py -v`
Expected: all 4 tests pass.

### Task 9.3: Wire sensitivity into the runner

- [ ] **Step 1: Edit `famail_temporal/evaluation/runner.py`**

After `grid_before = build_fairness_grid(bundle)`, add:

```python
        if diagnostics_enabled:
            from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
            sensitivity_before = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
        else:
            sensitivity_before = None
```

After `grid_after = build_fairness_grid(bundle, pickup_3d=pickup_after)`, add:

```python
        if diagnostics_enabled:
            sensitivity_after = compute_gradient_sensitivity(bundle, pickup_after)
        else:
            sensitivity_after = None
```

Update the `ExperimentResult(...)` construction to pass `sensitivity_before` and `sensitivity_after` in place of the existing `None` placeholders.

- [ ] **Step 2: Run runner tests**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_runner.py -v`
Expected: tests pass; `--no-diagnostics` test still sees `None` sensitivity grids.

- [ ] **Step 3: Extend `famail_temporal/evaluation/__init__.py`**

```python
"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.runner import ExperimentResult, run_experiment

__all__ = [
    "ExperimentResult",
    "augment_trajectories",
    "build_fairness_grid",
    "compute_gradient_sensitivity",
    "run_experiment",
]
```

### Task 9.4: Verify persistence handles sensitivity pickles

- [ ] **Step 1: Re-run persistence test**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_persistence.py -v`
Expected: still green. Persistence already gates on `result.diagnostics_enabled and result.gradient_sensitivity_before is not None`.

### Task 9.5: Commit Phase 9

```bash
git add famail_temporal/evaluation/__init__.py \
        famail_temporal/evaluation/diagnostics.py \
        famail_temporal/evaluation/runner.py \
        famail_temporal/tests/test_gradient_sensitivity.py
git commit -m "feat(evaluation): Tier C global gradient sensitivity grid"
```

### Phase 9 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Review prompt: (a) is the grid->N projection consistent with `FAMAILObjective.forward`, (b) are active-cell gradients non-zero on meaningful inputs, (c) is the persistence gate correct?

---

## Phase 10: Real-data slow test + README + CHANGELOG

**Files:**
- Create: `famail_temporal/tests/test_runner_real_data.py`
- Create: `famail_temporal/evaluation/README.md`
- Modify: `CHANGELOG.md`

### Task 10.1: Real-data slow smoke test

- [ ] **Step 1: Create `famail_temporal/tests/test_runner_real_data.py`**

```python
"""Real-data end-to-end smoke test for evaluation.runner (slow)."""
import json
from pathlib import Path

import numpy as np
import pytest

from famail_temporal.evaluation.runner import run_experiment
from famail_temporal.evaluation.persistence import write
from famail_temporal.evaluation.report import render


@pytest.mark.slow
def test_real_data_end_to_end(tmp_path):
    result = run_experiment(
        max_trajectories=200, k=5,
        config_overrides={"MAX_ITERATIONS": 5},
        diagnostics_enabled=True,
    )
    out_dir = write(result, output_root=tmp_path)
    report_path = render(out_dir)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "grid_before.pkl").exists()
    assert (out_dir / "grid_after.pkl").exists()
    assert (out_dir / "trajectories.csv").exists()
    assert (out_dir / "per_unit_attribution.csv").exists()
    assert (out_dir / "modified_trajectory_ids.json").exists()
    assert report_path.exists()

    m = json.loads((out_dir / "metrics.json").read_text())
    assert np.isclose(
        m["metrics_before"]["f_spatial"],
        1.0 - float(np.nansum(result.grid_before[..., 0])),
        atol=1e-5,
    )
    assert np.isclose(
        m["metrics_before"]["f_causal"],
        1.0 - float(np.nansum(result.grid_before[..., 1])),
        atol=1e-5,
    )
```

- [ ] **Step 2: Run the slow test**

Run: `cd /home/robert/FAMAIL && pytest famail_temporal/tests/test_runner_real_data.py -v --run-slow`
Expected: PASS within ~2 minutes.

### Task 10.2: README

- [ ] **Step 1: Create `famail_temporal/evaluation/README.md`**

```markdown
# famail_temporal.evaluation

End-to-end evaluation framework for the FAMAIL trajectory-modification pipeline.

## Quickstart

CLI:

    python -m famail_temporal.evaluation.runner --name demo

Programmatic:

    from famail_temporal.evaluation import run_experiment

    result = run_experiment(
        name="tighter-epsilon",
        config_overrides={"EPSILON_BALL": 1.5, "MAX_ITERATIONS": 20},
        k=100,
    )
    print(result.experiment_id, result.f_spatial_before, result.f_spatial_after)

## CLI flags

| Flag | Purpose |
|---|---|
| `--name <slug>` | Appended to the experiment ID for readability |
| `--max-trajectories N` | Limit the dataset (useful for quick iterations) |
| `--max-drivers N` | Limit the number of drivers loaded |
| `-k N` | Number of top-attribution trajectories to modify (default 100) |
| `--no-diagnostics` | Skip Tier A gradient decomposition and Tier C sensitivity grids |
| `--override KEY=VALUE` | Override any `famail_temporal.config` attribute. Repeat the flag. |

## What gets written

See `docs/superpowers/specs/2026-04-16-evaluation-framework-design.md` for the
authoritative artifact list and schemas. Summary, per run (under
`famail_temporal/results/{experiment_id}/`):

- `metrics.json` - config snapshot + provenance + before/after scalars
- `grid_before.pkl` / `grid_after.pkl` - (48, 90, T, 4) fairness grids
- `augmented_trajs_before.pkl[.gz]` / `augmented_trajs_after.pkl[.gz]` - full 8-element-state datasets
- `modified_trajectory_ids.json` - which trajectories were modified + cell moves
- `histories.pkl` - full per-iteration modification history
- `trajectories.csv` - one row per top-k modified trajectory
- `per_unit_attribution.csv` - one row per active unit
- `gradient_sensitivity_{before,after}.pkl` - (48, 90, T, 2) when diagnostics are on
- `report.md` - tables-only human-readable summary
```

### Task 10.3: CHANGELOG entry

- [ ] **Step 1: Prepend to `CHANGELOG.md`** (after the intro lines):

```markdown
---

## 2026-04-16 - FAMAIL Temporal Evaluation Framework

**Files**: `famail_temporal/evaluation/*`, `famail_temporal/fairness/spatial.py`,
`famail_temporal/algorithm/modifier.py`, `famail_temporal/config.py`,
`famail_temporal/tests/test_*`

**Why**: The `famail_temporal/` algorithm was complete (178 tests) but lacked
reproducible end-to-end experiment orchestration. A downstream team member also
needed a fairness-augmented version of `passenger_seeking_trajs_45-800.pkl` and
a `(48, 90, T, 4)` fairness-aware state-space grid for dashboard and analysis
work. The framework bundles both priority artifacts, the orchestration to
produce them, and gradient-level diagnostics for investigating whether the new
cell-level fairness formulation is actually driving optimization.

**What**: Added `famail_temporal.evaluation` with seven modules - `grid`,
`augment`, `diagnostics`, `runner`, `persistence`, `report`, `__init__`. Added
`per_unit_gini_decomposition` and `compute_spatial_attribution` to
`fairness/spatial.py` (refactored `pairwise_gini` to route through the
decomposition primitive). Added public `TrajectoryModifier.current_pickup_3d()`.
Extended `ModificationResult` with Tier A gradient-decomposition fields gated
by `config.DIAGNOSTICS_ENABLED` and a `--no-diagnostics` CLI opt-out. Runner
produces a timestamped output directory with provenance-stamped `metrics.json`,
two grid pickles, two augmented-trajectory pickles (gzipped automatically above
500 MB), two CSVs, a sidecar JSON of modified IDs, a histories pickle, and a
tables-only `report.md`.
```

### Task 10.4: Final verification + commit

- [ ] **Step 1: Full fast + slow suites**

Run:
```bash
cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q
cd /home/robert/FAMAIL && pytest famail_temporal/tests/ -q --run-slow
```
Expected: all tests pass in both runs.

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/evaluation/README.md \
        famail_temporal/tests/test_runner_real_data.py \
        CHANGELOG.md
git commit -m "docs: evaluation framework README + CHANGELOG entry + real-data smoke test"
```

### Phase 10 Review Checkpoint

- [ ] Dispatch a review subagent (Opus, `superpowers:code-reviewer`). Final review prompt: (a) does the real-data slow test cover at least one invariant beyond "files exist," (b) does the README let a new engineer run their first experiment without reading the spec, (c) does the CHANGELOG entry explain the *why* per CLAUDE.md?

---

## Post-implementation: `superpowers:finishing-a-development-branch`

- [ ] Dispatch the `superpowers:finishing-a-development-branch` skill to decide on merge strategy, PR vs. direct merge, and any remaining cleanup. Do NOT merge/push without explicit user approval.
