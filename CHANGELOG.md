# Changelog

Notable changes to the FAMAIL project — algorithm modifications, architectural decisions,
and non-trivial edits. Minor bugfixes and UI tweaks are omitted.

---

## 2026-03-18 — Experiment Framework + Gradient Flow Fixes

### Experiment & Analysis Framework (New Module)

**Files**: New `experiment_framework/` directory (7 files, ~1,500 lines)

The existing Streamlit dashboard is effective for interactive exploration but lacks systematic
instrumentation for understanding how each objective term contributes to trajectory edits.
Built a new experiment framework that provides reproducible batch experiments with structured
output.

- `experiment_config.py` — `ExperimentConfig` and `SweepConfig` dataclasses consolidating
  all pipeline parameters (attribution, ST-iFGSM, objective weights, discriminator, etc.)
- `experiment_runner.py` — `ExperimentRunner` orchestrates the full Phase 1 (attribution) +
  Phase 2 (modification) pipeline with per-trajectory instrumentation and cumulative global
  fairness tracking via `GlobalMetrics` snapshots.
- `gradient_decomposition.py` — `GradientDecomposer` computes per-term gradient vectors
  (∂F_spatial/∂pos, ∂F_causal/∂pos, ∂F_fidelity/∂pos) via separate backward passes with
  `retain_graph=True`. Configurable flag (`record_gradient_decomposition`) since it requires
  4 backward passes per iteration instead of 1.
- `experiment_result.py` — Result types (`ExperimentResult`, `TrajectoryResult`,
  `IterationRecord`, `GradientDecomposition`) with JSON/CSV serialization and auto-generated
  markdown reports. Each run produces 8 output files: config.json, results.json, summary.json,
  trajectories.csv, iterations.csv, global_snapshots.csv, attribution_scores.csv, report.md.
- `analysis_dashboard.py` — Streamlit dashboard for loading experiment results with 7 views:
  experiment overview, global fairness evolution, per-trajectory deep dive, gradient
  decomposition analysis, cross-run comparison, attribution effectiveness, spatial heatmap.
- `cli.py` — Command-line interface with `run`, `sweep`, `dashboard`, and `summarize`
  subcommands.

### Gradient Flow Fixes (Critical Bug Fixes)

**Why**: The gradient decomposition framework revealed that the ST-iFGSM optimization was
driven **entirely** by F_spatial. F_causal and F_fidelity produced zero gradients in every
iteration of every trajectory, meaning the algorithm was effectively single-objective despite
the multi-objective formulation `L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity`.

**Root causes and fixes**:

1. **F_causal gradient = 0** — The causal fairness computation received `causal_demand`
   (a constant tensor set once in `set_global_state()`) instead of `pickup_counts` (the soft
   cell output with gradient flow from `pickup_tensor`). The constant had no connection to
   the computational graph. Additionally, `pickup_counts` is sum-aggregated while g₀(D) was
   fitted on mean-aggregated data, requiring a scale correction factor (÷144 observation
   periods) to keep g₀(D) inputs at the correct scale.
   - `trajectory_modification/objective.py` — Changed `demand_for_causal` to use
     `pickup_counts` (soft cell output) with lazy-computed `_causal_demand_scale` factor.

2. **F_fidelity gradient = 0** — The modified trajectory features (`tau_prime_features`)
   were built from `modified.to_tensor()`, which converts numpy→tensor and breaks the
   computational graph. Fixed by constructing `tau_prime_features` from the original
   trajectory tensor with the pickup coordinates replaced by the differentiable
   `pickup_tensor` values. Same fix applied for V3 multi-stream x2 tensor slot 0.
   - `trajectory_modification/modifier.py` — Build `tau_prime_features` via `.clone()` +
     coordinate replacement instead of `modified.to_tensor()`.
   - `experiment_framework/gradient_decomposition.py` — Same fix in the decomposition loop.

3. **cuDNN LSTM backward in eval mode** — Once fidelity gradients flowed, backward through
   the discriminator's LSTM failed with "cudnn RNN backward can only be called in training
   mode". Fixed by disabling cuDNN during the discriminator forward pass
   (`torch.backends.cudnn.flags(enabled=False)`), which uses PyTorch's native LSTM that
   supports backward in eval mode.
   - `trajectory_modification/objective.py` — Added `torch.backends.cudnn.flags(enabled=False)`
     context manager around discriminator call in `compute_fidelity()`.

**After fix**: Gradient decomposition shows all three terms contributing: spatial ≈ 12%,
causal ≈ 9%, fidelity ≈ 80%. The spatial-causal alignment is consistently negative (-0.47
to -1.0), indicating the two fairness terms push modifications in opposing directions — an
important finding about the algorithm's multi-objective dynamics.

---

## 2026-03-17 — V3 Multi-Stream Discriminator Integration

### Why

The V3 discriminator (Ren et al. KDD 2020 multi-stream architecture) was trained on three
data streams — seeking trajectories, driving trajectories, and driver profile features —
but the modification pipeline only passed single seeking trajectories, zero-padding the
other two streams. This put the model far outside its training distribution, producing
poorly calibrated fidelity scores that weakened the F_fidelity gradient signal.

### What Changed

Wired real driving trajectory and profile feature data through the entire modification
pipeline so the V3 discriminator operates in-distribution during trajectory editing.

**New files**:
- `trajectory_modification/multi_stream_context.py` — `MultiStreamContextBuilder` class
  that assembles V3 discriminator kwargs (x1/x2, driving, profile tensors with masks)
  for each trajectory being modified. Implements three seeking fill strategies: "sample"
  (default: 1 target + 4 context trajectories from same driver), "replicate", and "single".
  Handles 0-indexed→1-indexed coordinate conversion. Documents key design decisions:
  same-driver branch construction, seeking fill strategy rationale, gradient flow through
  slot 0 only.
- `trajectory_modification/compare_discriminators.py` — Comparison script for systematic
  V2 vs V3 evaluation on identical trajectory sets, with V3 ablation (with/without
  multi-stream context).

**Modified files**:
- `trajectory_modification/data_loader.py` — Added `MultiStreamDataLoader` class and
  extended `DataBundle` with multi-stream fields (`ms_driving_trajs`, `ms_seeking_trajs`,
  `ms_profile_features`, `ms_seeking_days`, `ms_driving_days`). `load_default()` gains
  `load_multi_stream=True` parameter with graceful fallback.
- `trajectory_modification/objective.py` — `forward()` accepts `**fidelity_kwargs` and
  routes multi-stream tensors (x1/x2 keys) to `compute_fidelity()`, bypassing the
  single-trajectory `tau_features`/`tau_prime_features` path.
- `trajectory_modification/modifier.py` — `TrajectoryModifier` accepts optional
  `multi_stream_context` and builds fidelity kwargs per iteration in `modify_single()`.
- `trajectory_modification/discriminator_adapter.py` — Fixed config merging for V3
  checkpoints (`model_config` takes precedence over `config`). Added `is_multi_stream`
  property.
- `objective_function/fidelity/config.py` — Added `multi_stream_data_dir`,
  `n_trajs_per_stream`, `seeking_fill_strategy` fields.
- `trajectory_modification/dashboard.py` — Auto-detects V3, loads multi-stream context,
  reports stream status in sidebar.
- `trajectory_modification/__init__.py` — Exports `MultiStreamDataLoader`,
  `MultiStreamContextBuilder`.

---

## 2026-03-09 — GlobalMetrics F_causal: Option B Alignment + Dashboard Results Overhaul

### F_causal Always Zero in Summary Metrics (Bug Fix)

**Files**: `trajectory_modification/metrics.py`

`GlobalMetrics.compute_r_squared()` used the baseline variance-ratio formula
`F = max(0, 1 − var(R)/var(Y))` with the isotonic `g(D)`. This is the wrong formulation
when Option B is selected — and when the isotonic fit is poor, `var(R) > var(Y)` yields
R² < 0, which `max(0, ...)` clamps to exactly 0.0000. Meanwhile, per-iteration F_causal
values (computed via `FAMAILObjective._compute_option_b_causal()`) used the correct
hat-matrix formula and returned non-zero values.

**Fix**: Replaced `compute_r_squared()` with `compute_f_causal()`. When `hat_matrices`,
`active_cell_indices`, and `g0_power_basis_func` are provided, it delegates to
`compute_fcausal_option_b_numpy()` — the same `R'(I-H)R / R'MR` quadratic form used in
the differentiable objective. Falls back to the baseline formula only when hat matrices
are unavailable. Updated `compute_snapshot()` to call the new method directly.

Dashboard initialization (`dashboard.py`) now passes `hat_matrices`,
`active_cell_indices`, and `g0_power_basis_func` from session state to `GlobalMetrics()`.

### Dashboard Results Section Rework

**Files**: `trajectory_modification/dashboard.py`

- **Summary Metrics**: Replaced single-value metric cards with a Before/After two-row
  layout showing F_spatial, F_causal, F_fidelity, and Combined L. Removed standalone
  Gini Coefficient card (redundant with F_spatial = 1 − Gini). Added tooltips describing
  each metric and its interpretation.
- **Modification Statistics**: Renamed from "Convergence Statistics". Added min/max/average
  perturbation magnitude metrics alongside existing Converged, Avg Iterations, Mean Fidelity.
- **Modification Details labels**: Changed per-trajectory fairness labels from generic
  `SPATIAL`/`CAUSAL` to `F_spatial`/`F_causal` for consistency with objective function
  terminology.
- **Iteration Details precision**: Increased display precision for Objective L, F_spatial,
  F_causal, and F_fidelity columns from default (~4 digits) to 8 decimal places via
  `df.style.format()`.
- **Before/After viz tooltip**: Updated Fairness Metric radio help text to describe
  the Option B DCD formula (`R̂ = H_demo·R`).

---

## 2026-03-09 — Trajectory Modification Algorithm: 5-Issue Fix Set

Addresses five known issues (7.1–7.5) identified during pseudocode documentation of
the ST-iFGSM trajectory editing pipeline.

### 7.1 — Epsilon Constraint Bug (Critical)

**Files**: `trajectory_modification/modifier.py`

The ST-iFGSM loop applied `cumulative_delta` (a total offset from the original position)
to the *already-shifted* `modified` trajectory each iteration, compounding the perturbation.
With ε=2 and 50 iterations, pickups could drift to the grid boundary (~50ε) instead of
staying within ±ε cells of the original.

**Fix**: Apply `cumulative_delta` to the original `trajectory` parameter (never mutated)
rather than the running `modified` copy:
```python
# Before (buggy):  modified = modified.apply_perturbation(cumulative_delta)
# After (correct): modified = trajectory.apply_perturbation(cumulative_delta)
```

### 7.2 — Soft/Hard Count Divergence in Batch Mode

**Files**: `trajectory_modification/modifier.py`

In batch modification, `_base_pickup_counts` was set once in `set_global_state()` and
never updated between trajectories. Trajectory N+1's soft count computation subtracted
from trajectory N's *original* cell rather than its final modified position, causing the
differentiable counts to diverge from the hard counts.

**Fix**: Sync base counts after each trajectory's ST-iFGSM loop:
```python
self._base_pickup_counts = self.pickup_counts.clone()
self._base_dropoff_counts = self.dropoff_counts.clone()
```

### 7.3 — DCD Attribution Aligned with Option B Causal Formulation

**Files**: `trajectory_modification/dashboard.py`

The Demand-Conditional Deviation (DCD) attribution metric always used the baseline formula
`DCD_c = |Y_c − g(D_c)|`, even when Option B was selected for the objective function.
Under Option B, causal fairness measures demographic residual independence — the relevant
signal is how much demographics explain the demand-controlled residuals, not the raw
deviation from expected service ratio.

**Change**: Extended `compute_cell_dcd_scores` with a `causal_formulation` parameter.
When `option_b` is selected, DCD is computed as:
```
R = Y − g₀(D)           (demand-controlled residuals)
R̂ = H_demo @ R          (project onto demographic subspace)
DCD_c = |R̂_c|           (magnitude of demographically-explained residual)
```
Propagated through `compute_trajectory_attribution_scores`, `_compute_cell_fairness_scores`,
and all dashboard call sites. Added `key="causal_formulation"` to the sidebar selectbox
for cross-function session state access.

### 7.4 — Computational Complexity Analysis (Documentation)

**Files**: `trajectory_modification/pseudocode_docs/algorithm_pseudocode.md`

No code changes. Added a complexity analysis to the pseudocode document comparing the
cost of re-attribution after each trajectory vs. the current approach (attribute once,
then modify k trajectories sequentially).

Key finding: pairwise Gini at O(n²) dominates per-iteration cost (~4M ops with n≈2000
service-active cells). Re-attribution costs O(C + N log N) ≈ 10K ops — approximately
0.0025% of modification cost. Re-attribution after each trajectory is effectively free
and recommended for future implementation.

### 7.5 — Spatial Fairness Gini Over Service-Active Cells Only

**Files**: `trajectory_modification/objective.py`, `trajectory_modification/metrics.py`

The Gini coefficient was computed over all 4320 grid cells. Because `data_loader.py`
floors `active_taxis_grid` to 0.1 (preventing division by zero), the existing
`active_taxis > eps` filter was True for *all* cells and filtered nothing. ~2300 cells
with no real taxi activity contributed zero-vs-zero differences, diluting the metric.

**Change**: Added a service mask to identify cells with real taxi activity:
```
service_mask = (pickup_counts > ε) | (dropoff_counts > ε) | (active_taxis > 0.5)
```
The 0.5 threshold separates real taxi activity (mean-aggregated from integer counts ≥ 1)
from the artificial 0.1 floor. Yields ~2000 service-active cells. Applied in both
`objective.py` (differentiable PyTorch path) and `metrics.py` (NumPy evaluation path).

Side effect: pairwise Gini drops from O(4320²) ≈ 18.7M to O(2000²) ≈ 4M operations
per call — a ~4.7× speedup. Gini absolute values will shift (no longer diluted) but
more accurately reflect inequality among cells that actually receive taxi service.

### Documentation

Updated `trajectory_modification/pseudocode_docs/algorithm_pseudocode.md` throughout:
- Section 2.2: DCD pseudocode now shows both baseline and Option B code paths
- Section 2.4: Marked as IMPLEMENTED
- Section 3.1: Removed BUG annotation, updated to show correct trajectory offset
- Section 3.2: Added base count sync step after each trajectory in batch pseudocode
- Section 5.1: Noted Gini computed over service-active cells only
- Section 7.1–7.5: All marked as FIXED with concise descriptions
- Section 7.4: Replaced with computational complexity analysis

---

