# Evaluation Framework Handoff

**Date:** 2026-04-16
**Context:** This document captures all decisions made during the implementation session for a fresh planning agent to pick up the evaluation framework design without needing the original conversation's context.

---

## 1. What exists in `famail_temporal/`

A standalone, 178-test, fully-functional implementation of the FAMAIL trajectory modification algorithm with temporally-aware fairness metrics. Key modules:

- **`config.py`** — single source of truth (T=4 time blocks, grid 48x90, all hyperparams)
- **`data/`** — DataBundle, 3D aggregation (mean-hourly), UnitIndexMap (canonical N-vector ordering), active mask, preprocessing, demographics enrichment
- **`fairness/`** — `compute_fspatial` (pooled pairwise Gini over DSR/ASR), `compute_fcausal` (Option B hat-matrix R'(I-H)R/R'MR), `per_unit_attribution` (sums to 1-F_causal), `G0Function` (fitted g0(D)), hat matrices
- **`fidelity/`** — ported V3 MultiStreamSiameseDiscriminator (4 classes), checkpoint loader, MultiStreamContextBuilder (4 design decisions), compute_ffidelity with cuDNN workaround
- **`algorithm/`** — SoftCellAssignment (differentiable Gaussian kernel), inject_soft_counts_into_3d (delta-tensor pattern), FAMAILObjective (orchestrator), attribution pipeline (per-unit -> per-trajectory ranking), TrajectoryModifier (ST-iFGSM loop)
- **`preprocess.py`** — one-time raw data -> cache pipeline
- **`tests/`** — 178 tests (173 fast + 5 real-data slow), all passing including real Shenzhen data

**Design spec:** `docs/superpowers/specs/2026-04-16-famail-temporal-design.md`
**Implementation plan:** `docs/superpowers/plans/2026-04-16-famail-temporal*.md` (4 files)

---

## 2. What needs to be built: Evaluation Framework

### 2.1 Three-layer evaluation module

Create `famail_temporal/evaluation/` with:

**Layer 1 — `runner.py`:** A `run_experiment(config_overrides=None)` function that executes the full pipeline:
- Load DataBundle
- Compute "before" fairness metrics (F_spatial, F_causal)
- **Generate the fairness-aware state-space grid** (see Section 3 below)
- **Generate augmented trajectory dataset** (see Section 4 below)
- Compute per-unit attribution
- Rank trajectories, select top-k
- Run TrajectoryModifier.modify_batch on selected trajectories
- Compute "after" fairness metrics
- Return structured `ExperimentResult` dataclass

**Layer 2 — `persistence.py`:** Writes results to a timestamped output directory `results/{experiment_id}/`:
- `metrics.json` — aggregate before/after F_spatial, F_causal, config snapshot
- `trajectories.csv` — per-trajectory: id, driver, pickup_before, pickup_after, displacement, attribution_score, per-iteration metrics
- `per_unit_attribution.csv` — the N-vector heatmap data
- `grid_before.pkl` — fairness-aware state-space grid (Section 3)
- `grid_after.pkl` — post-modification version of the same grid
- `augmented_trajs_before.pkl` — fairness-augmented trajectories (Section 4)

**Layer 3 — `report.py`:** Generates a self-contained `report.md` with:
- Experiment config table
- Before/after fairness comparison
- Top-10 modified trajectories table
- Convergence summary
- "Key findings" section auto-populated from metric deltas

### 2.2 Entry point

```bash
python -m famail_temporal.evaluation.runner [--config-override key=value ...]
```

Or programmatic:
```python
from famail_temporal.evaluation.runner import run_experiment
result = run_experiment(config_overrides={"MAX_ITERATIONS": 20, "EPSILON_BALL": 1.5})
```

---

## 3. Fairness-aware state-space grid (CRITICAL ARTIFACT)

### Specification

A persistent `.pkl` file containing a grid of shape `(48, 90, T, 4)` where T=4 time blocks and the 4 channels are:

| Channel | Name | Derivation | Sum property |
|---|---|---|---|
| 0 | `spatial_attr` | `0.5 * (gini_decomp_dsr + gini_decomp_asr)` | `sum = 1 - F_spatial` |
| 1 | `causal_attr` | `((MR)_i^2 - ((I-H)R)_i^2) / R'MR` | `sum = 1 - F_causal` |
| 2 | `gini_decomp_dsr` | `row_sum_dsr_i / (2 * n^2 * mean_dsr)` | `sum = Gini(DSR)` |
| 3 | `gini_decomp_asr` | `row_sum_asr_i / (2 * n^2 * mean_asr)` | `sum = Gini(ASR)` |

**Inactive units** (those not in the active mask) are set to `NaN` — forces downstream code to handle missing data explicitly.

### Pairwise Gini per-unit decomposition (AGREED UPON)

The pairwise Gini `G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)` admits a per-unit decomposition:

```
gini_contrib_i = sum_j |x_i - x_j| / (2 * n^2 * mu)
```

This is:
- O(N^2) — same cost as Gini itself (reuses the pairwise distance matrix)
- Exact: `sum_i gini_contrib_i = G`
- Analogous to the causal attribution's sum property

For F_spatial = 1 - 0.5 * (Gini(DSR) + Gini(ASR)):
```
spatial_attr_i = 0.5 * (gini_dsr_contrib_i + gini_asr_contrib_i)
sum_i spatial_attr_i = 0.5 * (Gini(DSR) + Gini(ASR)) = 1 - F_spatial
```

### Implementation location

This should be a new function in `fairness/spatial.py`:
```python
def per_unit_gini_decomposition(values: torch.Tensor) -> torch.Tensor:
    """Per-unit contribution to pairwise Gini. Sums to Gini(values)."""
    ...
```

Then a higher-level function that computes all 4 channels:
```python
def compute_spatial_attribution(pickup_N, dropoff_N, active_taxis_N) -> dict:
    """Returns gini_decomp_dsr, gini_decomp_asr, spatial_attr (all N-vectors)."""
    ...
```

---

## 4. Trajectory augmentation dataset (CRITICAL ARTIFACT)

### Specification

A persistent `.pkl` file structured identically to `passenger_seeking_trajs_45-800.pkl` (dict keyed by driver_id, values are lists of trajectories), but with each state augmented from 4 elements to 8:

**Original state:** `[x_grid, y_grid, time_bucket, day_index]`
**Augmented state:** `[x_grid, y_grid, time_bucket, day_index, spatial_attr, causal_attr, gini_decomp_dsr, gini_decomp_asr]`

### Generation process

1. Build the fairness-aware grid (Section 3) — shape `(48, 90, T, 4)`
2. For each trajectory state `(x, y, time_bucket, day)`:
   - `t_block = hour_to_block_index(time_bucket_to_hour(time_bucket))`
   - Look up `grid[x, y, t_block, :]` -> 4 fairness scores
   - Append to state vector
3. States in inactive `(x, y, t_block)` units get NaN for all 4 fairness scores
4. Save as `.pkl` in the same dict-of-lists structure

### Use cases (from the user)

- Trajectory visualizations with fairness coloring ("this driver spent 80% of seeking time in high-unfairness units")
- Team member needs this exact dataset for downstream analysis
- Before/after comparison (generate `augmented_trajs_before.pkl` + `augmented_trajs_after.pkl`)

---

## 5. Key decisions already made

| Decision | Choice | Rationale |
|---|---|---|
| Spatial per-unit attribution | Pairwise Gini row-sum decomposition | O(N^2), exact sum property, analogous to causal attribution |
| Both Gini components stored | Yes — gini_decomp_dsr AND gini_decomp_asr as separate channels | User requested both raw components + the composite |
| Inactive unit fill value | NaN | Semantically correct, forces explicit handling |
| Grid shape | (48, 90, T, 4) | 4 channels: spatial_attr, causal_attr, gini_dsr, gini_asr |
| Trajectory augmentation structure | Same as passenger_seeking_trajs but 8-element states | Requested by team member; drop-in replacement |
| Evaluation runner scope | Single-run first, sweep later | Start simple, the CSV/JSON output feeds a future sweep orchestrator |
| Results format | Timestamped directory with metrics.json + CSVs + .pkl artifacts + report.md | Persistent, programmatically evaluable, human-readable |

---

## 6. Technical context for the planning agent

### How to run the existing algorithm

```python
from famail_temporal.data import DataBundle
from famail_temporal.algorithm import (
    FAMAILObjective, TrajectoryModifier,
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.fidelity import MultiStreamContextBuilder

bundle = DataBundle.load(max_trajectories=1000)
attribution, signed = compute_per_unit_attribution(bundle)
scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
top_k = select_top_k(scored, k=100)

objective = FAMAILObjective(bundle)
ms_builder = MultiStreamContextBuilder(bundle.multi_stream)
modifier = TrajectoryModifier(objective=objective, bundle=bundle,
                               multi_stream_builder=ms_builder)
histories = modifier.modify_batch([bundle.trajectories[i] for i in top_k])
```

### Key data structures

- `DataBundle` — frozen dataclass with `pickup_3d (48,90,T)`, `mask_3d`, `unit_map`, `hat_matrices`, `g0_func`, etc.
- `UnitIndexMap` — canonical ordering of N active `(cell, t)` units. `from_cell_time(flat_cell, t_block) -> unit_idx`
- `ModificationHistory` — per-trajectory result with iteration-by-iteration metrics
- `per_unit_attribution(R, I_minus_H_demo, M) -> (N,)` — sums to `1 - F_causal`

### Existing test infrastructure

- `@pytest.mark.slow` for real-data tests (need `--run-slow` flag)
- `_make_synthetic_bundle()` in `tests/test_objective.py` for fast synthetic tests
- `conftest.py` with `--run-slow` marker support and `seeded` autouse fixture

### Codebase conventions

- `config.py` is the single source of truth
- Fail-loud: `ValueError` for bad inputs, not silent drops
- Frozen dataclasses for immutable data containers
- `hat_matrices_to_torch()` for warning-free numpy->torch conversion of read-only arrays
- Per-field shape/dtype comments on dataclass fields
- Every mathematical invariant is pinned by a test in `test_math_invariants.py`

---

## 7. Suggested plan structure for the evaluation framework

1. Add `per_unit_gini_decomposition` to `fairness/spatial.py` (new function + test)
2. Add `compute_spatial_attribution` wrapper (returns 3 N-vectors for DSR, ASR, composite)
3. Create `evaluation/__init__.py`, `evaluation/grid.py` (fairness-aware grid builder)
4. Create `evaluation/augment.py` (trajectory augmentation via grid lookup)
5. Create `evaluation/runner.py` (experiment orchestration)
6. Create `evaluation/persistence.py` (JSON + CSV + .pkl output)
7. Create `evaluation/report.py` (markdown generation)
8. Integration tests with real data
9. Documentation (evaluation/README.md)

---

## 8. Branch and commit state

- **Branch:** `famail_temporal`
- **Latest commit:** `4afae0a` (fix: load model_config from checkpoint + relative mass-balance tolerance)
- **All 178 tests passing** (including real-data slow tests)
- **Preprocessing cache populated** (raw data + cache artifacts present)
