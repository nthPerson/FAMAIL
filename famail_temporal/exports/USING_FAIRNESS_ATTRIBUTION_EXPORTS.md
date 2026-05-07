# Using FAMAIL Fairness-Attribution Exports

Assumes you have read **`../docs/RESEARCHER_HANDOFF.md`**. If you have not, start there.

## TL;DR

This document tells you how to load a FAMAIL fairness-attribution export and feed its per-cell fairness signal into a GAIL, GAN, or generic offline-RL training loop. It covers loading patterns, the sign convention you must respect, the axis semantics that have a broadcast trap inside them, three self-contained training-method recipes, a numbered pitfalls catalogue, and a sanity-check checklist.

It does **not** re-derive the fairness math (that lives in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md)), it does **not** prescribe a framework, and it does **not** opine on which metric to use, how to normalize, or how to weight your fairness term against your other losses. It shows you how to apply each option correctly; the choice is yours.

## What you have

Each export directory at `famail_temporal/exports/<timestamp>_<name>/` contains:

- **Three `.pkl` artifacts** — `fairness_attribution_dense.pkl` (block-level tensors for fast lookup), `fairness_attribution_long.pkl` (pandas DataFrame for filtering), `fairness_attribution_tuples.pkl` (dependency-free row iteration). Algebraically equivalent; pick by convenience.
- **`metadata.json`** — provenance sidecar (git SHA, source-data SHA, config snapshot, overall F values, active-cell counts per block).
- **`README.md`** — the auto-generated reference card for that specific export. Carries the export's actual F values and `n_days`. This how-to is the prescriptive companion to that reference card.

---

## §1. Shared preamble

Read this once before any recipe. The recipes in §2 link back here rather than restating shared content.

### §1.1 Loading the export

Three artifacts; pick by convenience. They carry the same data.

```text
# Dense — fastest tensor lookup; best for a training loop
load fairness_attribution_dense.pkl as dense
spatial = dense["spatial"]            # (gx, gy, T) float, NaN on inactive
causal  = dense["causal"]             # (gx, gy, T) float, NaN on inactive
mask    = dense["active_mask"]        # (gx, gy, T) bool
metadata = dense["metadata"]

# Long — pandas DataFrame; best for filtering and analysis
load fairness_attribution_long.pkl as payload
df       = payload["dataframe"]
metadata = payload["metadata"]

# Tuples — list of row-tuples; best for dependency-free iteration
load fairness_attribution_tuples.pkl as payload
columns  = payload["columns"]
rows     = payload["rows"]
metadata = payload["metadata"]
```

Decision rule: **dense for tensor lookups inside a training loop, long for pandas filtering, tuples for dependency-free iteration**. The metadata sidecar (`metadata.json`) carries the same dict that is embedded inside each `.pkl`; either source is fine.

### §1.2 Sign convention and scale

Per-cell α is **signed** and **unbounded**. The sum over active cells equals F ∈ [0, 1]:

```text
Σ over active cells  spatial_fairness_attribution  =  F_spatial
Σ over active cells  causal_fairness_attribution   =  F_causal
```

Reading: positive α means "the cell contributes more than the 1/N baseline to fairness"; negative α means the cell drags fairness below baseline. Magnitude is unbounded in both directions.

Two anti-patterns to avoid:

- **Do not clamp per-cell α to [0, 1] without intent.** Only the overall metric is in [0, 1]. Clamping per-cell α silently discards the negative-fair signal and turns a signed reward into a one-sided one.
- **Do not treat per-cell magnitudes as probabilities.** They are signed contributions to a sum, not weights. Anything that requires a [0, 1] or simplex constraint needs explicit normalization on your side; this document does not prescribe one.

The full derivation of the 1/N-shifted decomposition lives in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md).

### §1.3 Axis semantics: the broadcast trap

The export's row schema looks four-dimensional: `(x_grid, y_grid, time_bucket, day)`. It is not. Two of those axes are **broadcast duplicates**, not independent samples:

- **`time_bucket` within an hour-block.** Fairness is computed at `(x, y, time_block)` granularity (24 hourly blocks). Each block contains 12 five-minute `time_bucket` values. The same per-block α is duplicated across all 12 buckets in the block.
- **`day`.** Fairness is computed pooled across the dataset's `n_days` days, not per-day. The same pooled α is duplicated across every value of the `day` index.

Three concrete consequences:

- **IID sampling along broadcast axes inflates effective sample size.** Drawing N rows uniformly from `df` and treating them as independent observations gives you up to `12 × n_days =` 60 duplicates per (cell, block) in the manuel-handoff export. Variance estimates and standard errors computed against that count are wrong by an order of magnitude or more.
- **"Per-bucket" or "per-day" features are pass-throughs.** Any feature you compute by picking a single (bucket, day) value is identical to its sibling values in the same block; there is no per-bucket or per-day signal to extract.
- **Aggregations along broadcast axes are pass-throughs of the per-block / pooled value.** `mean` and `std` over a broadcast axis return the per-block / pooled value with zero spread.

If your training loop needs a per-state reward at finer granularity than `(cell, hour-block)`, you are looking up a duplicated value, not a finer signal. That is fine — it just means your model is being trained on the same target that the audit measured. Decisions to break the broadcast (e.g., per-day fairness) require recomputing the audit, which is out of scope for this export.

### §1.4 Active vs inactive cells

Every (x, y, time_bucket, day) appears in the export — including cells that are not part of the fairness audit. The `is_active` boolean (long / tuples) and `active_mask` array (dense) tell you which is which. A cell is **inactive** when any of three conditions holds:

- The cell has insufficient supply: mean active taxis below `ACTIVE_SUPPLY_THRESHOLD = 0.5` per hour.
- The cell is outside the Shenzhen administrative boundary.
- Any required demographic feature for the cell is NaN.

NaN propagation rule: at inactive cells, **every numeric column is NaN** — both attribution columns and the context columns `demand_D`, `supply_S`, `service_rate_Y`.

Recommended masking pattern:

```text
# Preferred: mask before reducing
active = mask                                           # bool array same shape as spatial
total_spatial = sum(spatial[active])                    # equals overall_F_spatial

# Acceptable: NaN-aware reductions
total_spatial = nansum(spatial)                         # equals overall_F_spatial

# Wrong: replace NaN with 0 without intent
spatial = where(isnan(spatial), 0, spatial)             # ambiguates "inactive" with "exactly zero α"
```

Zero is a valid attribution value (a cell at the negative-fair / anti-fair boundary, see §1.2); replacing NaN with zero throws away the inactive-versus-boundary distinction.

### §1.5 Two metrics, briefly

Each export carries two attribution columns; they measure different things and are not interchangeable.

- **`spatial_fairness_attribution`** decomposes `F_spatial`, a Gini-based measure of equity in service exposure across active `(cell, hour-block)` units. Its per-cell α captures how a cell's service-rate ratios (DSR = pickup/supply, ASR = dropoff/supply) compare to the rest of the active set.
- **`causal_fairness_attribution`** decomposes `F_causal = 1 − r²_demo`, where `r²_demo` is the share of the demand-adjusted residual variance explained by neighborhood demographics. Its per-cell α captures whether the cell's residual service rate aligns with neighborhood wealth (negative α) or is uncorrelated with it (positive α).

Full derivation in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md). Causal-specific methodology in [`../docs/F_CAUSAL_METHODOLOGY_NOTES.md`](../docs/F_CAUSAL_METHODOLOGY_NOTES.md).

**This document does not prescribe which metric to use.** Both are exported because they measure different things; the choice depends on what your model is trying to optimize and is yours to make.

---

## §2. Recipes

Each recipe is self-contained; pick the one that matches your training method. All three assume you have read §1 (sign convention, broadcast trap, NaN handling) and use the loading patterns from §1.1.

### §2.1 Recipe: GAIL / imitation learning reward shaping

**Where α enters:** as a per-state additive bonus on top of the discriminator-derived reward. The agent is trained to prefer cells with positive α and avoid cells with negative α.

**Pseudocode for the per-step reward computation:**

```text
# Pre-load once before the training loop
load fairness_attribution_dense.pkl as dense
α_grid = dense["spatial"]                    # or dense["causal"]; pick one (see §1.5)
mask   = dense["active_mask"]
λ      = a scalar weight you choose          # not prescribed here

# Inside the training loop, for each visited state s = (x, y, time_block):
def fairness_bonus(state):
    x, y, t_block = state.cell_x, state.cell_y, state.time_block
    if not mask[x, y, t_block]:
        return OFF_SUPPORT_PENALTY           # explicit; do not silently zero
    return λ * α_grid[x, y, t_block]

reward(state, action) = discriminator_reward(state, action) + fairness_bonus(state)
```

The natural framing — drawn directly from [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md) §3 — is: "agents that randomly visit any cell average a fairness bonus of λ · F/N; only agents that preferentially visit positive-α cells beat that." This makes α a meaningful gradient signal without further engineering.

**Active-mask and broadcast notes.** `OFF_SUPPORT_PENALTY` is your call — pick a value that pushes the policy back onto the active set without dominating the discriminator term. Per §1.3, an agent that steps at 5-minute resolution and reads `α_grid[x, y, t_block]` is reading the same per-block value 12 times within an hour; that is intentional, not a bug.

**Relevant pitfalls:** §3 items 1, 2, 4, 5.

### §2.2 Recipe: GAN training

**Where α enters:** three options, in order of increasing intrusiveness on the existing pipeline. Pick the lightest fit.

**Option A — α as an evaluation diagnostic (no training change).** Score generated trajectories' pickup distribution against α offline.

```text
# After generating a batch of trajectories
gen_pickup_grid = histogram_2d_per_block(generated_trajectories)   # (gx, gy, T)
gen_alpha_mass  = nansum(gen_pickup_grid * α_grid)                 # NaN at inactive cells
real_alpha_mass = nansum(real_pickup_grid * α_grid)
report(gen_alpha_mass / real_alpha_mass)                           # closer to 1 = generator matches real fairness profile
```

**Option B — α as an auxiliary loss (no architectural change).** Penalize generated trajectories whose pickup distribution under-weights positive-α cells.

```text
# Inside the generator's loss
gen_pickup_soft = differentiable_pickup_grid(generator_output)     # (gx, gy, T)
fairness_term   = − nansum(gen_pickup_soft * α_grid)               # negate so minimizing loss raises α-mass
total_loss      = adversarial_loss + λ * fairness_term
```

**Option C — α as a conditioning input (architectural change).** Broadcast α as an extra input channel into the generator's spatial features.

```text
# At generator input assembly
α_channel = where(isnan(α_grid), 0, α_grid)                        # explicit replacement; required for tensor input
generator_input = concat(spatial_features, α_channel, axis=channel_dim)
```

**Active-mask and broadcast notes.** Option C is the one place this how-to recommends replacing NaN with zero, because tensor inputs cannot carry NaN; the replacement is a conditioning artifact, not a semantic claim about α at inactive cells. Options A and B preserve NaN via `nansum`. The 48 × 90 grid alignment between your generator's spatial output and α_grid is a precondition for all three options; mismatched grids require resampling and break the per-cell semantics.

**Relevant pitfalls:** §3 items 1, 2, 6, 7.
