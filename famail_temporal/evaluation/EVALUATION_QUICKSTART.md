# FAMAIL Temporal — Evaluation Quickstart

A researcher's guide to running experiments with `famail_temporal.evaluation` and interpreting the results. Focused on answering three core research questions:

1. **Did we improve fairness?**
2. **What changes were necessary to improve fairness?**
3. **Which trajectories were modified, and why?**

For authoritative artifact schemas and architectural rationale, see
[`docs/superpowers/specs/2026-04-16-evaluation-framework-design.md`](../docs/superpowers/specs/2026-04-16-evaluation-framework-design.md).
For the terse CLI reference, see [`evaluation/README.md`](evaluation/README.md).

> **Serialization note.** Several run artifacts use Python pickle (`.pkl`) format for structural compatibility with the existing `passenger_seeking_trajs_45-800.pkl` dataset and for preserving numpy tensor shapes. The examples in this guide load those artifacts back with `pickle.load` — this is only safe because every `.pkl` in a run directory was written by this same framework on the same machine. Never load `.pkl` artifacts from untrusted sources.

---

## Prerequisites

Before your first run, confirm:

| Requirement | How to check |
|---|---|
| Preprocess cache exists | `ls famail_temporal/cache/` shows cached artifacts. If empty, run `python -m famail_temporal.preprocess`. |
| Source trajectory data exists | `ls famail_temporal/source_data/passenger_seeking_trajs.pkl` |
| Discriminator checkpoint (optional but recommended) | `ls famail_temporal/discriminator_checkpoints/default/best.pt`. If absent, the runner falls back to `nn.Identity` and silently forces `alpha_fidelity=0.0` (see "Common pitfalls" below). |
| Conda env active | `conda activate famail` (or your project env) |
| Tests pass | `pytest famail_temporal/tests/ -q` → expect all fast tests green. Run `--run-slow` if you want to validate the end-to-end pipeline on real data (~2 minutes). |

---

## Your first experiment

Run a small, fast experiment to verify the pipeline works:

```bash
python -m famail_temporal.evaluation.runner \
    --name first-run \
    --max-trajectories 200 \
    -k 5 \
    --override MAX_ITERATIONS=5
```

This loads 200 trajectories, modifies the top 5 by attribution, runs only 5 ST-iFGSM iterations per trajectory, and writes results under:

```
famail_temporal/results/2026-04-17T09-15-22_first-run/
```

(The timestamp will be your run's actual start time.) You'll see:

```
[runner] experiment_id = 2026-04-17T09-15-22_first-run
[runner] results_dir  = /home/.../famail_temporal/results/2026-04-17T09-15-22_first-run
[runner] report       = /home/.../famail_temporal/results/2026-04-17T09-15-22_first-run/report.md
[runner]   F_spatial: 0.5842 -> 0.5891
[runner]   F_causal:  0.3117 -> 0.3122
```

Open `report.md` first — it's the 30-second summary of the run.

---

## What gets produced

Every run directory contains these files. Each is described in "how to read it" form.

### `report.md` — start here

A tables-only human-readable summary with these sections:

1. **Header** — experiment ID, timestamp, git SHA + dirty flag, exact command line. Use the git SHA + command line to reproduce the run.
2. **Config** — every `config.*` value that was active. Overridden values are **bolded**.
3. **Effective-alpha override** (conditional) — appears only if runtime alphas differ from config snapshot. If you see this, your fidelity term was silently disabled (see "Common pitfalls"). This line is *load-bearing*: a run with `config.ALPHA_FIDELITY=0.34` but effective `0.0` is NOT comparable to a run with a trained discriminator.
4. **Dataset** — `n_trajectories` (total), `n_drivers`, `n_active_units` (cells that passed the activity filter), `k_modified`.
5. **Fairness** — the four-row before/after/Δ table. This is the direct answer to Research Q1.
6. **Convergence** — how many top-k trajectories converged vs. hit `MAX_ITERATIONS`, mean iteration count, mean final gradient norm.
7. **Gradient diagnostics** (conditional, when diagnostics are on) — the Tier B summary aggregated across all modified trajectories.
8. **Top 10 modified trajectories** — per-trajectory rank, driver, original/modified pickup cells, delta, convergence.
9. **Key findings** — 3–5 auto-generated bullets derived from the deltas and diagnostics. Watch for "Dominant gradient term: `fidelity` in X%" — if this is >50%, the fidelity term is driving the step rather than the fairness terms, which is a signal to lower `ALPHA_FIDELITY`.
10. **Artifacts** — index of every file written, with sizes.

### `metrics.json` — programmatic source of truth

All scalar-level information about the run, in a single JSON file. Schema:

```jsonc
{
  "experiment_id": "2026-04-17T09-15-22_first-run",
  "timestamp_utc": "2026-04-17T16:15:45+00:00",

  // Provenance — enough to reproduce the run from clean state
  "git_sha": "c8a45b5",
  "git_dirty": false,
  "command_line": "python -m famail_temporal.evaluation.runner --name first-run ...",

  // Every config.UPPERCASE value at run time
  "config_snapshot": { "T": 4, "ALPHA_SPATIAL": 0.33, ... },
  "config_overrides": { "MAX_ITERATIONS": 5 },
  "diagnostics_enabled": true,

  // The alpha values actually used (differs from config_snapshot on
  // nn.Identity-stub runs — see "Common pitfalls")
  "effective_alphas": { "alpha_spatial": 0.33, "alpha_causal": 0.33, "alpha_fidelity": 0.0 },

  "dataset": { "n_trajectories": 200, "n_drivers": 50, "n_active_units": 3124 },
  "k_modified": 5,

  // Before/after/delta — the Research Q1 answer
  "metrics_before": { "f_spatial": 0.5842, "f_causal": 0.3117, "gini_dsr": 0.411, "gini_asr": 0.385 },
  "metrics_after":  { "f_spatial": 0.5891, "f_causal": 0.3122, "gini_dsr": 0.409, "gini_asr": 0.385 },
  "deltas":         { "f_spatial": 0.0049, "f_causal": 0.0005, "gini_dsr": -0.002, "gini_asr": 0.0 },

  "convergence_summary": {
    "n_converged": 3, "n_max_iter": 2,
    "mean_total_iterations": 4.2, "mean_final_grad_norm": 0.018
  },
  "diagnostics_summary": { ... },      // Tier B aggregates (null when diagnostics off)
  "artifact_paths": { ... },           // {name: filename}
  "file_sizes_bytes": { ... }
}
```

**Tip for scripting comparisons across runs:** use `glob.glob("famail_temporal/results/*/metrics.json")`, load each, pivot on `(config_overrides, deltas)`. The schema is stable.

### `grid_before.pkl` / `grid_after.pkl` — the fairness-aware state-space grid

Each is a dict (deserialized via standard pickle loading):

```python
{
    "grid": np.ndarray,                        # (48, 90, T, 4) float32
    "channel_names": ["spatial_attr", "causal_attr",
                      "gini_decomp_dsr", "gini_decomp_asr"],
    "time_blocks": [("morning_peak", 7, 10),
                    ("midday", 10, 16),
                    ("evening_peak", 16, 20),
                    ("night", 20, 31)],
    "active_mask": np.ndarray,                 # (48, 90, T) bool
}
```

**Channel semantics (each sums over active units):**

| Channel | Value at unit (x, y, t) | Sum (over active) equals |
|---|---|---|
| 0 `spatial_attr` | contribution of unit to spatial unfairness | `1 − F_spatial` |
| 1 `causal_attr` | contribution of unit to demographic-explained variance | `1 − F_causal` |
| 2 `gini_decomp_dsr` | row-sum decomposition of DSR Gini | `Gini(DSR)` |
| 3 `gini_decomp_asr` | row-sum decomposition of ASR Gini | `Gini(ASR)` |

**Inactive cells are `NaN`** on all channels. Use `np.nansum`, `np.nanmax`, etc.

```python
import pickle, numpy as np
with open("grid_before.pkl", "rb") as f:
    before = pickle.load(f)
with open("grid_after.pkl", "rb") as f:
    after = pickle.load(f)

# Cells whose spatial unfairness contribution shrunk the most:
delta_spatial = after["grid"][..., 0] - before["grid"][..., 0]
most_improved_mask = np.nan_to_num(delta_spatial, nan=0.0) < np.nanpercentile(delta_spatial, 5)
# most_improved_mask[x, y, t] == True where cell became much more fair.
```

### `augmented_trajs_before.pkl[.gz]` / `augmented_trajs_after.pkl[.gz]`

Drop-in replacement for `passenger_seeking_trajs_45-800.pkl` with each state widened from 4 to 8 elements:

```
[x_grid, y_grid, time_bucket, day_index,
 spatial_attr, causal_attr, gini_decomp_dsr, gini_decomp_asr]
```

- Indices 0–3 are **1-indexed on disk** (matching the raw file's convention).
- Indices 4–7 are looked up from `grid_before.pkl` (for `_before`) or `grid_after.pkl` (for `_after`).
- Every trajectory is included — unmodified trajectories keep their states; top-k modified trajectories have their final pickup state moved.
- Automatically gzipped when uncompressed size exceeds 500 MB (check `artifact_paths` in `metrics.json` to see the actual filename).

**Use case:** a downstream team member loads `augmented_trajs_before.pkl` and plots each driver's seeking trajectory colored by `spatial_attr`. Then loads the `_after` version to see where the intervention moved drivers' experienced fairness landscape.

### `modified_trajectory_ids.json`

Sidecar that tells you *which* trajectories were modified without loading the full pickle:

```json
{
  "modified_trajectory_ids": [17, 42, 89, 103, 177],
  "original_pickup_cells": { "17": [12, 34], "42": [8, 61], ... },
  "modified_pickup_cells": { "17": [12, 35], "42": [9, 61], ... }
}
```

### `trajectories.csv` — per-trajectory summary for top-k

One row per modified trajectory, sorted by rank (highest attribution first). Key columns:

| Column | Meaning |
|---|---|
| `rank`, `trajectory_id`, `driver_id` | identity |
| `original_pickup_cell_x/y`, `modified_pickup_cell_x/y` | before/after pickup coordinates |
| `pickup_t_block` | time block of the pickup (0–3) |
| `delta_x`, `delta_y` | signed integer moves |
| `attribution_score` | this trajectory's share of `1 − F_causal` via its pickup cell |
| `converged`, `total_iterations` | did ST-iFGSM converge inside `MAX_ITERATIONS`? |
| `initial_objective`, `final_objective` | scalar objective values |
| `f_spatial_initial/final`, `f_causal_initial/final`, `f_fidelity_initial/final` | per-term before/after |
| `mean_grad_spatial_norm`, `mean_grad_causal_norm`, `mean_grad_fidelity_norm` | average per-term gradient magnitudes (diagnostics only) |
| `frac_iters_{spatial,causal,fidelity}_dominant` | fraction of iterations where each term dominated the step |
| `mean_cos_spatial_causal`, `mean_cos_fairness_fidelity` | gradient-alignment cosines |
| `sign_flip_rate` | how often the signed gradient flipped between iterations (high rate → oscillation) |

**This is where Research Q2 lives.** Pandas-friendly:

```python
import pandas as pd
df = pd.read_csv("trajectories.csv")
# How much did each top-k trajectory move?
df[["rank", "delta_x", "delta_y", "converged", "total_iterations"]]
# Which trajectories didn't converge and why?
df[~df["converged"]][["rank", "total_iterations", "final_objective", "sign_flip_rate"]]
```

### `per_unit_attribution.csv` — per-active-unit view

One row per active `(cell, time-block)` unit. Columns pair `_before` with `_after` so you can diff:

- `unit_idx`, `cell_x`, `cell_y`, `t_block`, `flat_cell_id`
- `spatial_attr_before/after`, `causal_attr_before/after`
- `causal_attr_signed_before` — sign indicates direction (positive = over-serviced relative to demographics, negative = under-serviced)
- `gini_dsr_contrib_before/after`, `gini_asr_contrib_before/after`

**This is where Research Q3 lives** — the cells that ranked highest in `causal_attr_before` are the cells whose trajectories the ranker prioritized.

### `histories.pkl` — full per-iteration timeseries

The most verbose artifact. Contains a list of `ModificationHistory` objects — one entry per modified trajectory with every iteration's complete state:

```python
import pickle
with open("histories.pkl", "rb") as f:
    histories = pickle.load(f)

h = histories[0]
h.original          # Trajectory before modification
h.modified          # Trajectory after modification
h.converged         # bool
h.total_iterations  # int
h.iterations        # list of per-iteration records
h.iterations[-1].objective_value        # final objective
h.iterations[-1].grad_spatial_norm      # diagnostics only
h.iterations[-1].dominant_term          # 'spatial' | 'causal' | 'fidelity' | None
h.iterations[-1].sign_flipped           # bool | None
```

Use this when you need to plot a per-iteration convergence curve or see exactly how one trajectory's gradient decomposed step by step.

### `gradient_sensitivity_before.pkl` / `gradient_sensitivity_after.pkl`

Written only when `--no-diagnostics` is NOT set. Same payload shape as the fairness grid pickles but 2-channel:

```python
{
    "grid": np.ndarray,                       # (48, 90, T, 2) float32
    "channel_names": ["dF_spatial_dp", "dF_causal_dp"],
    "time_blocks": [...],
    "active_mask": np.ndarray,
}
```

Channel 0 is `∂F_spatial/∂pickup[x, y, t]` — "how much would F_spatial change if we added infinitesimal pickup mass at this cell?" Channel 1 is the same for F_causal. Fidelity is intentionally omitted (it's per-trajectory, not per-cell).

**Use case:** before running a full `k=100` experiment, compute this on the before-state to identify the cells with the highest leverage. These are the cells where a small perturbation would move the needle the most.

---

## Answering the three research questions

### Q1: Did we improve fairness?

**Primary signal:** `metrics.json.deltas` → `f_spatial`, `f_causal`, `gini_dsr`, `gini_asr`. Positive `f_*` deltas = improvement. `gini_asr` will always be ~0 because dropoffs aren't modified.

**Secondary signal:** `grid_after.pkl[..., 0]` − `grid_before.pkl[..., 0]` shows *where* the improvement happened spatially. Heatmap this to see if the intervention moved fairness toward under-served areas.

**Trajectory-level reframe:** "Did the fairness landscape that drivers' seeking paths actually traverse become more equitable?" Average `spatial_attr` across each driver's states in `augmented_trajs_before` vs `_after`. Drivers whose modified grid makes their unchanged seeking paths pass through lower-unfairness cells are experiencing the intervention indirectly — this is a legitimate finding.

### Q2: What changes were necessary?

**Primary signal:** `trajectories.csv`.

- `delta_x`, `delta_y` — the actual spatial moves. Distribution over top-k tells you whether modifications were small (tuning) or large (relocating).
- `total_iterations` + `converged` — if most runs hit `MAX_ITERATIONS` without converging, raise it or reduce `EPSILON_BALL`.
- `initial_objective` vs `final_objective` — scalar improvement per trajectory.

**Diagnostic signal:** `histories.pkl` — per-iteration convergence curves. Plot `objective_value` vs iteration for each top-k trajectory. Look for:
- Monotonic descent → healthy optimization.
- Oscillation → step size too large; check `sign_flip_rate` in `trajectories.csv`.
- Plateau without convergence → step size too small or in a flat region of the loss.

### Q3: Which trajectories were modified, and why?

**Who:** `modified_trajectory_ids.json` gives the complete list. `trajectories.csv.rank` + `trajectory_id` gives the ranking.

**Why:** Every top-k trajectory was picked because its pickup cell has high `causal_attr_before` (high per-unit contribution to `1 − F_causal`). To see the cell-level justification:

1. Open `trajectories.csv` — note the trajectory's `original_pickup_cell_x/y` and `pickup_t_block`.
2. Open `per_unit_attribution.csv` — find the matching `(cell_x, cell_y, t_block)` row.
3. `spatial_attr_before` and `causal_attr_before` tell you that cell's share of total unfairness.
4. `causal_attr_signed_before` tells you the *direction* of that unfairness (positive = this cell is over-serviced relative to its demographics; negative = under-serviced).

Cross-reference with `gradient_sensitivity_before.pkl` to understand *where the leverage was*: a trajectory whose pickup is in a high-attribution but low-sensitivity cell will move less than one in a high-attribution, high-sensitivity cell.

---

## Parameter sweeps and ablations

Every `famail_temporal.config` attribute is overridable via `--override KEY=VALUE`. Repeat the flag.

### Common variations

```bash
# Tighter epsilon-ball (smaller allowed moves):
--override EPSILON_BALL=1.0

# More iterations per trajectory:
--override MAX_ITERATIONS=100

# Different alpha weighting (e.g., emphasize spatial):
--override ALPHA_SPATIAL=0.6 --override ALPHA_CAUSAL=0.3 --override ALPHA_FIDELITY=0.1

# Larger top-k:
-k 500

# Subset the data for faster iteration:
--max-trajectories 1000 --max-drivers 10
```

### Naming convention for sweep runs

Use `--name` to give each run a human-readable label:

```bash
for eps in 0.5 1.0 1.5 2.0; do
    python -m famail_temporal.evaluation.runner \
        --name "epsilon-sweep-${eps}" \
        --override EPSILON_BALL=${eps}
done
```

Post-hoc comparison:

```python
import json, glob, pandas as pd
rows = []
for m_path in glob.glob("famail_temporal/results/*/metrics.json"):
    m = json.load(open(m_path))
    rows.append({
        "experiment_id": m["experiment_id"],
        "epsilon_ball":  m["config_snapshot"]["EPSILON_BALL"],
        "f_spatial_delta": m["deltas"]["f_spatial"],
        "f_causal_delta":  m["deltas"]["f_causal"],
        "n_converged":     m["convergence_summary"]["n_converged"],
    })
pd.DataFrame(rows).sort_values("epsilon_ball")
```

### Turning diagnostics off for faster runs

Tier A gradient decomposition costs ~3× per-iteration backward time. For large sweeps where you only care about final deltas:

```bash
python -m famail_temporal.evaluation.runner --name fast-sweep --no-diagnostics
```

This also skips writing `gradient_sensitivity_*.pkl` artifacts.

---

## Reading the gradient diagnostics

The `diagnostics_summary` section in `metrics.json` (Tier B, aggregated) and the per-trajectory columns in `trajectories.csv` answer: **is the new cell-level fairness formulation actually driving optimization, or is it being dominated by the fidelity term?**

### The key metrics

| Metric | Interpretation |
|---|---|
| `mean_grad_spatial_norm` | Average L2 of ∇F_spatial per iteration |
| `mean_grad_causal_norm` | Average L2 of ∇F_causal per iteration |
| `mean_grad_fidelity_norm` | Average L2 of ∇F_fidelity per iteration |
| `mean_cos_spatial_causal` | Alignment of spatial and causal gradients. Near +1 = they pull the same direction (healthy); near 0 = orthogonal; near −1 = fighting |
| `mean_cos_fairness_fidelity` | Alignment of the weighted (α₁∇F_sp + α₂∇F_ca) and ∇F_fidelity. Negative means fidelity is vetoing fairness |
| `frac_iters_spatial_dominant` | Fraction of iterations where α₁‖∇F_sp‖ was the largest weighted term |
| `frac_iters_causal_dominant`, `frac_iters_fidelity_dominant` | Same for the other terms |

### Interpreting common patterns

**Fidelity dominates:** `frac_iters_fidelity_dominant > 0.6` — the fidelity term is driving the step more than half the time. Either (a) the discriminator is the binding constraint (your trajectory-realism floor is too strict), or (b) `ALPHA_FIDELITY` is too high relative to fairness alphas. Lower `ALPHA_FIDELITY`.

**Fairness terms fight each other:** `mean_cos_spatial_causal < 0` — spatial and causal gradients are anti-aligned. This means improving F_spatial hurts F_causal and vice versa. This is a design-level signal: either the dataset has inherent tension between these notions of fairness, or one of the formulations has a bug. Worth investigating the per-unit attribution to see which cells are contributing.

**High sign-flip rate:** `sign_flip_rate > 0.5` on many top-k rows of `trajectories.csv` — the signed-gradient step is oscillating. `STEP_SIZE_ALPHA` is too large relative to the objective's local curvature. Lower it.

**All gradients are zero:** `dominant_term` is `None` for many iterations. This can happen at convergence (objective has plateaued) or at degenerate configurations (all alphas zero, all gradients vanish simultaneously). The framework returns `None` rather than silently picking one via dict-order tiebreak, so aggregates like `frac_iters_*_dominant` won't be inflated. If you see this for most iterations, optimization effectively stopped — raise tolerances or investigate.

---

## Common pitfalls

### 1. `nn.Identity` discriminator silently disables fidelity

If `famail_temporal/discriminator_checkpoints/default/best.pt` is absent, `DataBundle.load()` falls back to `nn.Identity()` as the discriminator. The runner detects this and sets `alpha_fidelity=0.0` on `FAMAILObjective` (because `nn.Identity.forward(tau, tau_prime)` raises `TypeError`).

**How you'll know:**
- `metrics.json.effective_alphas.alpha_fidelity == 0.0` but `metrics.json.config_snapshot.ALPHA_FIDELITY == 0.34`.
- `report.md` header shows an `Effective-alpha override:` line.

**What to do:** Either (a) put a trained checkpoint in place and re-run, or (b) accept the two-term run and remember that results aren't directly comparable to three-term runs.

### 2. `time_bucket=0` warning

About 1,090 out of 904,116 real Shenzhen trajectory states have `time_bucket=0`, though the schema docstring claims 1-indexed 1..288. The framework maps these to hour 0 and emits one `UserWarning`:

```
time_bucket=0 outside expected range [0, 288]; clamping to hour 0.
This likely indicates a data-quality issue in the raw trajectory file.
```

**What to do:** The warning is informational — results are still valid for the overwhelmingly typical case. If you see this warning for values far outside `[0, 288]` (e.g., negative or >300), something is wrong upstream.

### 3. `ValueError: Top-k is empty`

If `per_unit_attribution` is all zeros (demographics carry no explanatory power on this dataset), no trajectory has positive attribution and `select_top_k` returns an empty list. The runner fails loudly.

**What to do:** Inspect `per_unit_attribution_before` in a debugger or a quick script. Most common cause: the bundle was too small for the demographic regression to fit meaningful coefficients. Raise `max_trajectories` or use the full dataset.

### 4. `ValueError: k exceeds ranked trajectory count`

You asked for a larger top-k than the dataset has valid trajectories.

**What to do:** Lower `-k` or raise `--max-trajectories`.

### 5. Out-of-memory during augmentation

For the full ~44K-trajectory Shenzhen dataset, `augmented_trajs_*.pkl` can exceed 500 MB and will be automatically gzipped. Reading it back requires gzip-aware pickle loading:

```python
import gzip, pickle
with gzip.open("augmented_trajs_before.pkl.gz", "rb") as f:
    data = pickle.load(f)
```

Check `metrics.json.artifact_paths` to see whether gzip was applied to your run's files.

### 6. Modifier hit `MAX_ITERATIONS` without converging

`metrics.json.convergence_summary.n_max_iter` is close to `k_modified` (most trajectories didn't converge). Either raise `MAX_ITERATIONS`, lower `EPSILON_BALL` (smaller moves converge faster), or lower `CONVERGENCE_TOL` (accept coarser convergence).

---

## Recipes

### Recipe 1: Compare alpha weightings

```bash
for fid in 0.1 0.34 0.5 0.7; do
    python -m famail_temporal.evaluation.runner \
        --name "fidelity-${fid}" \
        --override ALPHA_FIDELITY=${fid}
done
```

Then plot `deltas.f_spatial` and `deltas.f_causal` vs `ALPHA_FIDELITY` across the four runs. Inflection point tells you where fidelity starts dominating.

### Recipe 2: Study which cells became more/less unfair

```python
import pickle, numpy as np
before = pickle.load(open("results/<id>/grid_before.pkl", "rb"))
after  = pickle.load(open("results/<id>/grid_after.pkl",  "rb"))

# Per-cell change in spatial unfairness contribution:
delta = after["grid"][..., 0] - before["grid"][..., 0]  # (48, 90, T)

# Aggregate across time-blocks for a single cell-level heatmap:
cell_delta = np.nansum(delta, axis=2)                   # (48, 90)

# Plot cell_delta. Negative = cell became more fair.
```

### Recipe 3: Identify drivers whose experience improved despite not being modified

```python
import pickle, json
before = pickle.load(open("results/<id>/augmented_trajs_before.pkl", "rb"))
after  = pickle.load(open("results/<id>/augmented_trajs_after.pkl",  "rb"))
mod_ids = set(json.load(open("results/<id>/modified_trajectory_ids.json"))["modified_trajectory_ids"])

# For each driver, average spatial_attr across all their trajectory states.
def avg_spatial_attr(trajs_dict):
    out = {}
    for did, traj_list in trajs_dict.items():
        values = []
        for traj in traj_list:
            for state in traj:
                v = state[4]  # spatial_attr channel
                if not (v != v):  # skip NaN
                    values.append(v)
        out[did] = sum(values) / max(len(values), 1)
    return out

before_avg = avg_spatial_attr(before)
after_avg  = avg_spatial_attr(after)

# Drivers whose average spatial_attr dropped (their seeking paths now pass
# through fairer cells, even though their trajectories are unchanged):
indirect_improvers = sorted(
    [(did, after_avg[did] - before_avg[did]) for did in before_avg],
    key=lambda kv: kv[1],
)
```

### Recipe 4: Plot per-trajectory convergence curves

```python
import pickle, matplotlib.pyplot as plt
histories = pickle.load(open("results/<id>/histories.pkl", "rb"))

fig, ax = plt.subplots(figsize=(10, 6))
for h in histories[:10]:
    objs = [r.objective_value for r in h.iterations]
    ax.plot(objs, label=f"traj {h.original.trajectory_id}")
ax.set_xlabel("Iteration")
ax.set_ylabel("Objective value (higher = more fair)")
ax.legend(fontsize="small")
```

### Recipe 5: Identify cells where Tier C sensitivity is highest

```python
import pickle, numpy as np
sens = pickle.load(open("results/<id>/gradient_sensitivity_before.pkl", "rb"))
# Channel 0 is ∂F_spatial/∂pickup; high positive = adding pickup mass there
# would most improve F_spatial.
spatial_sens = sens["grid"][..., 0]

# Top 10 most-leverageable cells at the first time block:
t = 0
flat = spatial_sens[:, :, t].ravel()
top_idx = np.nanargpartition(-flat, 10)[:10]
for idx in top_idx:
    x, y = idx // 90, idx % 90
    print(f"cell ({x}, {y}), t={t}: sensitivity {flat[idx]:.4f}")
```

These are the cells where, if you could nudge any trajectory's pickup there by one unit, you'd get the most F_spatial improvement per unit of change.

---

## Where to look next

- **Authoritative schemas:** [design spec](../docs/superpowers/specs/2026-04-16-evaluation-framework-design.md) §§3–8
- **Implementation phases:** [implementation plan](../docs/superpowers/plans/2026-04-16-evaluation-framework.md)
- **Terse CLI reference:** [`evaluation/README.md`](evaluation/README.md)
- **Changelog entry:** `CHANGELOG.md` → `2026-04-16 — FAMAIL Temporal Evaluation Framework`
- **The canonical fairness math:** [`fairness/spatial.py`](fairness/spatial.py), [`fairness/causal.py`](fairness/causal.py), [`fairness/hat_matrices.py`](fairness/hat_matrices.py)
- **The ST-iFGSM modifier:** [`algorithm/modifier.py`](algorithm/modifier.py)

Questions or oddities in a run's output? Open `metrics.json` first — the provenance triple (`git_sha`, `git_dirty`, `command_line`) is usually enough to reproduce and debug the run from clean state.
