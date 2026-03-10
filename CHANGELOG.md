# Changelog

Notable changes to the FAMAIL project — algorithm modifications, architectural decisions,
and non-trivial edits. Minor bugfixes and UI tweaks are omitted.

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
