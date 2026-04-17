# FAMAIL Temporal — Evaluation Framework Design

**Date:** 2026-04-16
**Status:** Approved (pending user review of this document)
**Scope:** Design for the evaluation framework that runs the FAMAIL trajectory-modification pipeline end-to-end, produces reproducible artifacts, and generates the two highest-priority research outputs: a fairness-aware state-space grid and a trajectory-augmented dataset.

---

## 1. Context and intent

`famail_temporal/` contains a complete, 178-test implementation of the FAMAIL trajectory modification algorithm with temporally-aware fairness metrics. This spec describes the evaluation framework layered on top of it.

**Research questions the framework must answer:**

1. *Did we improve fairness?* — Before/after F_spatial, F_causal, and per-cell decompositions.
2. *What changes were necessary to improve fairness?* — Per-trajectory modification history, pickup-cell moves, convergence curves.
3. *Which trajectories were modified, and why?* — Per-unit attribution driving the ranking, per-trajectory attribution scores.

**Two artifacts are designated highest priority** because a downstream team member depends on them:

- **Fairness-aware state-space grid** — `(48, 90, T, 4)` tensor with per-cell fairness decompositions.
- **Fairness-augmented trajectory dataset** — `passenger_seeking_trajs_45-800.pkl` widened from 4-element states to 8-element states.

**Serialization note:** Several artifacts use Python pickle format. This is an explicit requirement — the augmented trajectory output must be a drop-in structural replacement for the existing `passenger_seeking_trajs_45-800.pkl` consumed by downstream tooling. Pickle inputs will only ever be loaded from paths the framework itself writes, never from untrusted sources.

---

## 2. Architecture

### 2.1 New package

`famail_temporal/evaluation/` alongside existing `algorithm/`, `fairness/`, `fidelity/`, `data/`.

### 2.2 One edit to existing code

`fairness/spatial.py` gains two functions:

- `per_unit_gini_decomposition(values) -> Tensor` — row-sum decomposition of the pairwise Gini.
- `compute_spatial_attribution(pickup_N, dropoff_N, active_taxis_N) -> dict` — 3-channel wrapper producing gini_decomp_dsr, gini_decomp_asr, spatial_attr.

`pairwise_gini` is refactored to call `per_unit_gini_decomposition(values).sum()`, so the two stay numerically linked by construction.

### 2.3 One public-accessor addition

`TrajectoryModifier.current_pickup_3d() -> np.ndarray` — exposes the post-modification pickup tensor (currently the private `_base_pickup_3d` field). Converts an implementation detail into an explicit, tested contract so the evaluation framework doesn't tight-couple to modifier internals.

### 2.3a One config addition

`config.py` gains `DIAGNOSTICS_ENABLED: bool = True`. The modifier reads this flag (along with an optional constructor override) to select between the 3-backward Tier A path and the 1-backward fallback. The runner sets it via the override system (same mechanism as any other config field).

### 2.4 New modules

| Module | Purpose |
|---|---|
| `evaluation/__init__.py` | Re-export `run_experiment`, `ExperimentResult`, `build_fairness_grid`, `augment_trajectories`. |
| `evaluation/grid.py` | Build the (48, 90, T, 4) fairness-aware grid from a `DataBundle` + optional post-modification `pickup_3d` override. |
| `evaluation/augment.py` | Look up grid channels per trajectory state; produce the driver-keyed dict-of-lists with 8-element states. |
| `evaluation/diagnostics.py` | Tier A/B/C gradient diagnostics (per-iteration decomposition, per-trajectory summaries, global sensitivity field). |
| `evaluation/runner.py` | Orchestrate the full pipeline; return `ExperimentResult` dataclass. |
| `evaluation/persistence.py` | Write `ExperimentResult` to `famail_temporal/results/{experiment_id}/`; conditional gzip fallback >500 MB. |
| `evaluation/report.py` | Render `metrics.json` + CSVs -> `report.md` (tables only). |

### 2.5 Data flow

```
DataBundle (from cache)
   |
   +-> build_fairness_grid(bundle, pickup_3d=bundle.pickup_3d)
   |     -> grid_before (48, 90, T, 4)
   |        +-> augment_trajectories(trajectories, grid_before) -> augmented_trajs_before
   |        +-> metrics_before (from channel sums)
   |        +-> compute_gradient_sensitivity(bundle, bundle.pickup_3d) -> sensitivity_before (48, 90, T, 2)  [diagnostics only]
   |
   +-> per_unit_attribution -> rank_trajectories -> select_top_k
   |
   +-> TrajectoryModifier.modify_batch(top_k_trajs) -> histories (list[ModificationHistory])
   |     +-> modifier.current_pickup_3d() -> pickup_after (48, 90, T)
   |
   +-> build_fairness_grid(bundle, pickup_3d=pickup_after)
   |     -> grid_after (48, 90, T, 4)
   |        +-> augment_trajectories(trajs_after, grid_after) -> augmented_trajs_after
   |        +-> metrics_after (from channel sums)
   |        +-> compute_gradient_sensitivity(bundle, pickup_after) -> sensitivity_after  [diagnostics only]
   |
   +-> ExperimentResult(config_snapshot, metrics_before, metrics_after,
                        grid_before, grid_after, sensitivity_before, sensitivity_after,
                        augmented_trajs_before, augmented_trajs_after,
                        histories, per_unit_attribution_before, modified_trajectory_ids)
         |
         +-> persistence.write(result, output_root) -> report.render(output_dir)
```

---

## 3. Fairness-aware state-space grid (highest-priority artifact #1)

### 3.1 Shape and channels

`(48, 90, T, 4)` float32, with `T=4` time blocks. For any `(x, y, t_block)`:

| Ch. | Name | Formula | Sum over active units |
|---|---|---|---|
| 0 | `spatial_attr` | `0.5 * (gini_decomp_dsr + gini_decomp_asr)` | `1 - F_spatial` |
| 1 | `causal_attr` | `((MR)_i^2 - ((I-H)R)_i^2) / R'MR` | `1 - F_causal` |
| 2 | `gini_decomp_dsr` | `sum_j |DSR_i - DSR_j| / (2 * n^2 * mu_DSR)` | `Gini(DSR)` |
| 3 | `gini_decomp_asr` | `sum_j |ASR_i - ASR_j| / (2 * n^2 * mu_ASR)` | `Gini(ASR)` |

Inactive units (outside `bundle.mask_3d`) -> `NaN` on all 4 channels.

### 3.2 New fairness primitive

```python
# fairness/spatial.py

def per_unit_gini_decomposition(values: torch.Tensor) -> torch.Tensor:
    """Row-sum decomposition of pairwise Gini. Returns (N,) tensor.

    sum(per_unit_gini_decomposition(x)) == pairwise_gini(x) exactly (modulo float precision).
    Inactive-unit handling is the caller's responsibility (operates on 1-D N-vectors only).
    """
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)
    mean_val = values.mean() + config.EPS
    diff = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))  # (N, N)
    row_sums = diff.sum(dim=1)                                    # (N,)
    return row_sums / (2 * n * n * mean_val)


def compute_spatial_attribution(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Returns {'gini_decomp_dsr', 'gini_decomp_asr', 'spatial_attr'}, each an N-vector."""
```

`pairwise_gini` is refactored to `return per_unit_gini_decomposition(values).sum().clamp(0.0, 1.0)`.

### 3.3 Grid builder

```python
# evaluation/grid.py

def build_fairness_grid(
    bundle: DataBundle,
    pickup_3d: np.ndarray | None = None,   # default: bundle.pickup_3d (before-state)
) -> np.ndarray:
    """Returns (48, 90, T, 4) float32 ndarray.

    For the 'after' grid, pass modifier.current_pickup_3d().
    Inactive units are NaN on all 4 channels.
    """
```

**Implementation steps:**

1. Project `pickup_3d`, `bundle.dropoff_3d`, `bundle.active_taxis_3d` to N-vectors via `bundle.unit_map` (canonical active-unit ordering).
2. Call `compute_spatial_attribution` -> 3 N-vectors for channels 0, 2, 3.
3. Compute `Y = supply/demand`, `R = Y - g_0(D)`, `per_unit_attribution(R, hat_matrices['I_minus_H_demo'], hat_matrices['M'])` -> N-vector for channel 1.
4. Scatter the 4 N-vectors back to `(48, 90, T, 4)` pre-filled with `NaN`, using `unit_map.cell_indices` + `unit_map.time_block_indices`.

A small helper `project_3d_to_N(tensor_3d, unit_map) -> tensor_N` is added to `data/active_mask.py` if no equivalent already exists.

### 3.4 Invariants (pinned by tests)

- `np.nansum(grid[..., 0]) == 1 - F_spatial` within float tol
- `np.nansum(grid[..., 1]) == 1 - F_causal` within float tol
- `np.nansum(grid[..., 2]) == Gini(DSR)` within float tol
- `np.nansum(grid[..., 3]) == Gini(ASR)` within float tol
- Inactive cells are `NaN`; no active cell is `NaN`
- Channel-0 identity: `grid[..., 0] == 0.5 * (grid[..., 2] + grid[..., 3])` on active cells
- `sum(per_unit_gini_decomposition(x)) == pairwise_gini(x)` (math invariant, pinned in `test_math_invariants.py`)

### 3.5 On-disk format

`grid_before.pkl` / `grid_after.pkl` each contain a self-describing dict:

```python
{
    "grid": np.ndarray,            # (48, 90, T, 4) float32
    "channel_names": ["spatial_attr", "causal_attr", "gini_decomp_dsr", "gini_decomp_asr"],
    "time_blocks": config.TIME_BLOCKS,  # snapshotted so the file is self-describing
    "active_mask": np.ndarray,     # (48, 90, T) bool - duplicated from bundle.mask_3d for portability
}
```

Dict-not-ndarray so a reader doesn't need to import `config` to interpret the file.

---

## 4. Trajectory augmentation (highest-priority artifact #2)

### 4.1 Output structure

A dict keyed by `driver_id: int`, values are lists of trajectories. Each trajectory is a list of states. Each state is an **8-element list**:

```
state = [x_grid, y_grid, time_bucket, day_index,
         spatial_attr, causal_attr, gini_decomp_dsr, gini_decomp_asr]
```

- Indices 0-3 are **1-indexed on disk** (matching `passenger_seeking_trajs_45-800.pkl`; `data/loader.py:78-84` already subtracts 1 on read).
- Indices 4-7 are `float32`; `NaN` if the state's `(x, y, t_block)` is inactive.

The output is a full augmented dataset - every trajectory from the input list is included, grouped by driver. Drop-in replacement for `passenger_seeking_trajs_45-800.pkl`, just wider per state.

### 4.2 API

```python
# evaluation/augment.py

def augment_trajectories(
    trajectories: list[Trajectory],   # the complete list to augment
    grid: np.ndarray,                 # (48, 90, T, 4) from build_fairness_grid
) -> dict[int, list[list[list]]]:
    """Produce driver-keyed dict-of-lists. Includes every input trajectory.

    Each state is widened from 4 to 8 elements. Indices 0-3 written as 1-indexed
    for drop-in compatibility with passenger_seeking_trajs_45-800.pkl.
    """
```

Single-purpose function. Runner call sites:

```python
augmented_before = augment_trajectories(bundle.trajectories, grid_before)

modified_by_tid = {h.original.trajectory_id: h.modified for h in histories}
trajs_after = [
    modified_by_tid.get(t.trajectory_id, t) for t in bundle.trajectories
]
augmented_after = augment_trajectories(trajs_after, grid_after)
```

The "swap modified trajectories in" logic is a one-line comprehension at the runner site - visible, inspectable, and separate from the augmentation function's responsibility.

### 4.3 Sidecar

`modified_trajectory_ids.json`:

```json
{
  "modified_trajectory_ids": [17, 42, 89],
  "original_pickup_cells": {"17": [12, 34]},
  "modified_pickup_cells": {"17": [12, 35]}
}
```

Lets downstream code separate modified vs. unmodified populations without loading the full pickle.

### 4.4 Size and compression

Estimated size for the full ~44K trajectory dataset (avg ~25 states/traj):

- Typical pickle: ~100-200 MB (dominated by per-list-element overhead).
- Upper bound if avg doubles: ~400 MB.
- Pathological max-length-800 all trajectories: ~1.1 GB (not expected).

**Persistence rule:** When writing an augmented-trajectories pickle, log the uncompressed size. If it exceeds 500 MB, gzip-compress and write as `.pkl.gz`; update the path recorded in `metrics.json`. Always-uncompressed and always-compressed were both rejected; the conditional path keeps small runs snappy and large runs safe.

### 4.5 Invariants (pinned by tests)

- Driver-key set matches `{t.driver_id for t in trajectories}`.
- Per-trajectory state count is preserved (augmentation widens, never drops).
- Active-cell fairness channels equal `grid[x, y, t_block, :]` exactly.
- Inactive-cell fairness channels are all `NaN`.
- On-disk coords: `state[0] in [1, 48]`, `state[1] in [1, 90]`.
- Sidecar `modified_trajectory_ids` matches the ID set whose modified `Trajectory` was passed in.

---

## 5. Experiment runner

### 5.1 Public entry points

```python
# evaluation/runner.py

@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    config_snapshot: dict
    config_overrides: dict
    diagnostics_enabled: bool

    # Fairness metrics
    f_spatial_before: float
    f_spatial_after: float
    f_causal_before: float
    f_causal_after: float
    gini_dsr_before: float
    gini_dsr_after: float
    gini_asr_before: float
    gini_asr_after: float   # == gini_asr_before (dropoffs unchanged)

    # Full-dataset artifacts
    grid_before: np.ndarray                 # (48, 90, T, 4)
    grid_after: np.ndarray                  # (48, 90, T, 4)
    per_unit_attribution_before: np.ndarray # (N,)
    per_unit_attribution_signed_before: np.ndarray  # (N,)

    # Diagnostics (None if diagnostics_enabled is False)
    gradient_sensitivity_before: np.ndarray | None   # (48, 90, T, 2) [spatial, causal]
    gradient_sensitivity_after: np.ndarray | None    # (48, 90, T, 2)

    # Top-k modification artifacts
    modified_trajectory_ids: list[int]
    histories: list[ModificationHistory]
    top_k_scores: list[float]

    # Full-dataset augmented trajectories
    augmented_trajs_before: dict[int, list]
    augmented_trajs_after:  dict[int, list]


def run_experiment(
    config_overrides: dict | None = None,
    name: str | None = None,
    output_root: Path | None = None,        # default: famail_temporal/results/
    max_trajectories: int | None = None,
    max_drivers: int | None = None,
    k: int = 100,
    diagnostics_enabled: bool = True,
) -> ExperimentResult: ...
```

### 5.2 Orchestration stages

1. **Apply config overrides** (mutate `famail_temporal.config` attributes, capture originals for restoration in a `finally` block). Unknown keys -> `KeyError`.
2. **Generate `experiment_id`**: `f"{iso_timestamp}_{slug(name)}"` if `name` is given, else timestamp only. Format: `YYYY-MM-DDTHH-MM-SS[_slug]`.
3. **Load `DataBundle`** from cache via `DataBundle.load(max_trajectories, max_drivers)`.
4. **Build `grid_before`**; derive scalar before-metrics from channel sums.
5. **Augment before**: `augment_trajectories(bundle.trajectories, grid_before)`.
6. **Compute before-sensitivity** (if `diagnostics_enabled`): Tier C global sensitivity grid via autograd.
7. **Rank & select top-k** via existing `compute_per_unit_attribution` -> `rank_trajectories` -> `select_top_k`.
8. **Build objective + multi-stream + modifier.**
9. **`histories = modifier.modify_batch(top_k_trajs)`.** The modifier uses Tier A gradient decomposition (3 `autograd.grad` calls) if `diagnostics_enabled`; otherwise the existing single-backward path.
10. **Extract `pickup_after`** via `modifier.current_pickup_3d()`.
11. **Build `grid_after`**; derive scalar after-metrics.
12. **Compute after-sensitivity** (if `diagnostics_enabled`).
13. **Augment after**: build a new list — not an in-place mutation of `bundle.trajectories` — where each top-k trajectory is replaced by its modified counterpart (via the `{tid: modified}` map and a list comprehension, as shown in §4.2). Call `augment_trajectories(trajs_after, grid_after)`.
14. **Assemble `ExperimentResult`** and return.

### 5.3 CLI

```bash
python -m famail_temporal.evaluation.runner \
    [--name <slug>] \
    [--max-trajectories N] \
    [--max-drivers N] \
    [-k N] \
    [--no-diagnostics] \
    [--override KEY=VALUE ...]
```

- `--override` parses `KEY=VALUE` pairs; tries `int`, then `float`, then falls back to `str`. Fails loudly if `KEY` is not an existing `config.*` attribute.
- `--no-diagnostics` sets `diagnostics_enabled=False`: skips Tier A decomposition, Tier B derived columns, and Tier C sensitivity grids.
- After `run_experiment` returns, the CLI calls `persistence.write(...)` then `report.render(...)` and prints the result directory path.

### 5.4 Config override guardrails

- Only symbols defined in `famail_temporal.config` can be overridden.
- Original values restored in a `finally` block (safe for notebook reuse).
- Overrides persisted separately from the full snapshot -> trivial to diff runs later.

### 5.5 Error-handling philosophy

Fail loudly at the runner boundary:
- Unknown override key -> `KeyError` before loading anything.
- `k <= 0` or `k > len(scored_trajectories)` -> `ValueError` with remediation hint.
- Scored trajectories list empty after ranking (can occur if all attribution scores are zero — e.g., demographics carry no explanatory power on the given dataset) -> `ValueError` distinct from the `k` check, with a hint to inspect `per_unit_attribution_before`.
- Missing cache -> propagate `DataBundle.load`'s existing error.

No silent fallbacks. Matches the fail-loud convention in `data/loader.py`, `data/aggregation.py`, etc.

---

## 6. Gradient diagnostics

Three tiers, unified behind the `diagnostics_enabled` flag (CLI: `--no-diagnostics`). All three are on by default.

### 6.1 Tier A - Per-iteration, per-trajectory

Inside the modifier's ST-iFGSM loop, replace the single `total.backward()` with three `torch.autograd.grad` calls:

```python
grad_spatial  = torch.autograd.grad(f_spatial,  pickup_tensor, retain_graph=True)[0]
grad_causal   = torch.autograd.grad(f_causal,   pickup_tensor, retain_graph=True)[0]
grad_fidelity = torch.autograd.grad(f_fidelity, pickup_tensor, retain_graph=True)[0]
grad = (alpha_spatial  * grad_spatial
      + alpha_causal   * grad_causal
      + alpha_fidelity * grad_fidelity)
```

Added to `ModificationResult`:

| Field | Value |
|---|---|
| `grad_spatial_norm`, `grad_causal_norm`, `grad_fidelity_norm` | `\|\|grad_term\|\|_2` per term |
| `grad_cosine_spatial_causal` | `cos(grad_F_spatial, grad_F_causal)` |
| `grad_cosine_fairness_fidelity` | `cos(alpha1*grad_spatial + alpha2*grad_causal, grad_fidelity)` |
| `sign_flipped` | bool - did `sign(grad)` of any component flip from prev iter? |
| `dominant_term` | `argmax({alpha1*norm1, alpha2*norm2, alpha3*norm3})` |

When `diagnostics_enabled=False`, these fields are `None` and the modifier uses the single-backward path unchanged.

Cost: ~3x backward time per iteration. For `k=100 x 50 iters` this is seconds to minutes. Acceptable default.

### 6.2 Tier B - Per-trajectory summaries

Aggregated from Tier A for flat querying in `trajectories.csv`:

- `mean_grad_spatial_norm`, `mean_grad_causal_norm`, `mean_grad_fidelity_norm`
- `frac_iters_spatial_dominant`, `frac_iters_causal_dominant`, `frac_iters_fidelity_dominant`
- `mean_cos_spatial_causal`, `mean_cos_fairness_fidelity`
- `sign_flip_rate`
- `final_grad_norm_total`

When `diagnostics_enabled=False`, these columns are empty.

### 6.3 Tier C - Global sensitivity field

One-shot autograd pass on the full `pickup_3d` as a leaf tensor. Answers "where across the 48x90xT grid would an infinitesimal pickup-mass perturbation most change each fairness term?"

```python
def compute_gradient_sensitivity(bundle, pickup_3d) -> np.ndarray:
    """Returns (48, 90, T, 2) float32.

    Channels = [dF_spatial/dp, dF_causal/dp]. Inactive cells are NaN.
    Fidelity sensitivity is intentionally omitted because F_fidelity is
    defined per-trajectory, not per-cell.
    """
```

Persist as `gradient_sensitivity_before.pkl` / `gradient_sensitivity_after.pkl`, same dict-with-metadata schema as the fairness grids. Shape convention shared with the fairness grid -> a dashboard rendering one can render the other.

---

## 7. Persistence layout

### 7.1 Directory structure

```
famail_temporal/results/{experiment_id}/
|-- metrics.json
|-- trajectories.csv
|-- per_unit_attribution.csv
|-- grid_before.pkl
|-- grid_after.pkl
|-- augmented_trajs_before.pkl[.gz]     # conditional gzip if > 500 MB
|-- augmented_trajs_after.pkl[.gz]      # conditional gzip if > 500 MB
|-- modified_trajectory_ids.json
|-- histories.pkl
|-- gradient_sensitivity_before.pkl     # diagnostics only
|-- gradient_sensitivity_after.pkl      # diagnostics only
`-- report.md
```

### 7.2 `metrics.json` schema

```jsonc
{
  "experiment_id": "2026-04-16T14-30-12_tighter-epsilon",
  "timestamp_utc": "2026-04-16T21:30:12Z",
  "git_sha": "4afae0a",
  "git_dirty": false,                       // true iff `git status --porcelain` had output at run time
  "command_line": "python -m famail_temporal.evaluation.runner --name tighter-epsilon --override EPSILON_BALL=1.5",
  "config_snapshot": { /* every config.* value */ },
  "config_overrides": { "EPSILON_BALL": 1.5 },
  "diagnostics_enabled": true,
  "dataset": {
    "n_trajectories": 44000,
    "n_drivers": 50,
    "n_active_units": 8234,
    "max_trajectories_cap": null,
    "max_drivers_cap": null
  },
  "k_modified": 100,
  "metrics_before": { "f_spatial": 0.312, "f_causal": 0.478, "gini_dsr": 0.544, "gini_asr": 0.832 },
  "metrics_after":  { "f_spatial": 0.341, "f_causal": 0.491, "gini_dsr": 0.502, "gini_asr": 0.832 },
  "deltas":         { "f_spatial": 0.029, "f_causal": 0.013, "gini_dsr": -0.042, "gini_asr": 0.000 },
  "convergence_summary": {
    "n_converged": 87, "n_max_iter": 13,
    "mean_total_iterations": 42.1,
    "mean_final_grad_norm": 0.023
  },
  "diagnostics_summary": null,              // populated dict if diagnostics_enabled else null
  "artifact_paths": { /* relative paths, keyed by artifact name */ },
  "file_sizes_bytes": { /* one entry per artifact */ }
}
```

Provenance fields (`git_sha`, `git_dirty`, `command_line`) captured automatically via `subprocess`. `git_dirty=true` runs are legal but flagged.

### 7.3 `trajectories.csv` columns

One row per top-k modified trajectory.

Identity and modification:
- `trajectory_id`, `driver_id`
- `original_pickup_cell_x`, `original_pickup_cell_y`
- `modified_pickup_cell_x`, `modified_pickup_cell_y`
- `pickup_t_block` (time block of the pickup state, i.e. `hour_to_block_index(time_bucket_to_hour(pickup_state.time_bucket))`)
- `delta_x`, `delta_y` (signed integer moves)
- `attribution_score`, `rank`

Convergence:
- `converged`, `total_iterations`
- `initial_objective`, `final_objective`
- `f_spatial_initial`, `f_spatial_final`
- `f_causal_initial`, `f_causal_final`
- `f_fidelity_initial`, `f_fidelity_final`

Diagnostics (empty when `diagnostics_enabled=False`):
- `mean_grad_spatial_norm`, `mean_grad_causal_norm`, `mean_grad_fidelity_norm`
- `frac_iters_spatial_dominant`, `frac_iters_causal_dominant`, `frac_iters_fidelity_dominant`
- `mean_cos_spatial_causal`, `mean_cos_fairness_fidelity`
- `sign_flip_rate`

### 7.4 `per_unit_attribution.csv` columns

One row per active unit (N rows total):

- `unit_idx` (0..N-1, canonical ordering)
- `cell_x`, `cell_y`, `t_block`
- `flat_cell_id` (`x * 90 + y`, matches the project's grid-conventions memory)
- `spatial_attr_before`, `spatial_attr_after`
- `causal_attr_before`, `causal_attr_after`
- `causal_attr_signed_before`
- `gini_dsr_contrib_before`, `gini_dsr_contrib_after`
- `gini_asr_contrib_before`, `gini_asr_contrib_after`

### 7.5 `histories.pkl`

Raw `list[ModificationHistory]` pickled directly. Contains the full per-iteration timeseries (gradient decomposition, convergence trajectory). `trajectories.csv` captures aggregates; `histories.pkl` captures everything. Expected size ~1-10 MB for `k=100`.

### 7.6 Write ordering

1. Compute all in-memory artifacts -> return `ExperimentResult`.
2. `persistence.write(result, output_root)` writes every file **except `report.md`**.
3. `report.render(output_dir)` reads files back from disk (not the in-memory object) and produces `report.md`.

Report reads from disk deliberately: forces the persistence format to be self-sufficient, so a reviewer with only the output directory can regenerate the report.

---

## 8. Report generation (scope A: tables only)

`report.md` reads from `metrics.json` + the two CSVs and produces a single markdown file:

1. **Header.** Experiment ID, timestamp (local + UTC), git SHA, git dirty flag, command line.
2. **Config table.** `param`/`value` columns; overridden values in `**bold**`.
3. **Dataset summary.** 1-row table (n_trajectories, n_drivers, n_active_units, k_modified).
4. **Fairness before/after.** 4-row x 3-col (metric, before, after, delta). Arrows on delta column.
5. **Convergence summary.** Converged count, mean iterations, mean final grad norm.
6. **Gradient diagnostics summary** (omitted when `diagnostics_enabled=False`).
7. **Top-10 modified trajectories** - first 10 rows of `trajectories.csv` sorted by rank.
8. **Key findings.** 3-5 auto-generated bullets from metric deltas and diagnostics.
9. **Artifact index.** File list with sizes.

No plots, no images. Extending to scope B (embedded static plots) is deferred until we know which visuals matter.

---

## 9. Testing

New test files:

| File | Covers | Kind |
|---|---|---|
| `tests/test_per_unit_gini_decomposition.py` | New fairness primitive | Fast synthetic |
| `tests/test_compute_spatial_attribution.py` | 3-channel wrapper, NaN handling | Fast synthetic |
| `tests/test_fairness_grid.py` | `build_fairness_grid` shape, channel sums, NaN on inactive, channel-0 identity | Fast synthetic |
| `tests/test_augment_trajectories.py` | Driver-dict structure, 1-indexed on disk, 8-element states, count preservation, inactive -> NaN | Fast synthetic |
| `tests/test_gradient_diagnostics.py` | Tier A decomposition: weighted sum of decomposed gradients equals the original combined gradient (float tol); `--no-diagnostics` path produces identical final trajectory as diagnostics-on path (modulo per-iter metadata) | Fast synthetic |
| `tests/test_persistence.py` | Round-trip of each artifact kind, conditional gzip threshold, `metrics.json` schema stability | Fast synthetic |
| `tests/test_runner.py` | `run_experiment` end-to-end on synthetic bundle with `k=2`, `max_iters=3`; config-override restoration; unknown-override key raises | Fast synthetic |
| `tests/test_runner_real_data.py` | `@pytest.mark.slow`; real-data run with `max_trajectories=200`, `k=5`, `max_iters=5`; asserts all artifacts written, grid sums match scalars | Slow real data |

New math invariant in `tests/test_math_invariants.py`:

```python
def test_spatial_gini_decomposition_sums_to_gini():
    """sum(per_unit_gini_decomposition(x)) == pairwise_gini(x)."""
```

---

## 10. Phasing

Every phase leaves the codebase in a shippable state.

| Phase | Content | Phase gate |
|---|---|---|
| 1 | `per_unit_gini_decomposition` + `compute_spatial_attribution` + `pairwise_gini` refactor + tests | Math invariants pass; existing `pairwise_gini` tests unchanged |
| 2 | `evaluation/grid.py` + tests | All grid invariants pass; NaN handling verified |
| 3 | `evaluation/augment.py` + sidecar JSON + tests | 1-indexed-on-disk passes; count-preservation passes |
| 4 | `data/active_mask.py::project_3d_to_N` helper (if missing) + public `TrajectoryModifier.current_pickup_3d()` | Existing modifier tests unchanged |
| 5 | Tier A gradient decomposition in modifier + new `ModificationResult` fields | `test_gradient_diagnostics.py` passes; existing `test_modifier.py`/`test_modifier_integration.py` unchanged |
| 6 | `evaluation/runner.py` + `ExperimentResult` + CLI | Synthetic end-to-end test passes |
| 7 | `evaluation/persistence.py` + gzip fallback + provenance capture | `test_persistence.py` passes; real-data slow test passes |
| 8 | `evaluation/report.py` | Real-data slow test produces valid `report.md` |
| 9 | Tier C `compute_gradient_sensitivity` (2-channel) + persistence integration | Sanity test: sensitivity grid has no NaN on active cells |
| 10 | `evaluation/README.md` + CHANGELOG entry per CLAUDE.md | - |

**Phases 1-3 are the highest-priority artifacts.** If time pressure forces a cut, everything from Phase 4 onward is still valuable but can be deferred; the grid and augmented datasets stand alone for notebook-based analysis.

---

## 11. Known risks and mitigations

1. **Tier A 3x backward cost.** Mitigated by `--no-diagnostics` fallback. Expected cost of a few minutes on top of the base run is acceptable as a default.
2. **Pickle size at full scale.** Mitigated by conditional gzip >500 MB in `persistence.py`.
3. **`_base_pickup_3d` coupling.** Mitigated by the public `current_pickup_3d()` accessor and tests that pin its behavior.
4. **Config mutation restoration.** Any exception inside `run_experiment` must go through the `finally` block. Tests explicitly raise inside the runner body and assert post-run config equals the pre-run snapshot.

---

## 12. Out of scope

Deliberately not part of this framework:

- **Sweep orchestration** (multiple runs with varied `config_overrides`). Single-run first; a sweep driver can be written later on top of this framework's `run_experiment` + the per-run `metrics.json`.
- **Embedded plots in `report.md`** (scope B). Deferred until we know which visuals matter.
- **Fidelity channel in Tier C sensitivity grid.** F_fidelity is inherently per-trajectory, not per-cell; no global sensitivity field makes sense for it.
- **A dashboard.** Mentioned as a consumer of Tier C; not built here.
- **CI integration.** The existing project has no CI; adding one is out of scope.

---

## 13. Implementation model

**All implementation work must use Opus**, per the user's directive. The implementation plan will mirror this requirement in each phase's subagent delegation.
