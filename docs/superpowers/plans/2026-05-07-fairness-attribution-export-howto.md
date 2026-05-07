# Fairness-Attribution Export How-To Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md` — a standalone how-to that researchers receiving a FAMAIL fairness-attribution export use to apply the data to GAN, GAIL, or generic offline-RL training. Update the per-export `_README_TEMPLATE` to link to it so recipients of an export directory find the how-to in one click.

**Architecture:** Single markdown artifact authored section-by-section using a shared preamble + recipe library + appendices structure (Approach C from the spec). Cohesion is enforced through (a) a pre-flight that pulls the canonical anchor values once (F-metric values from the manuel-handoff metadata, config constants, sign-convention phrasing) so subsequent sections cite them verbatim and (b) periodic consistency audits between section groups that grep the project for current state and reconcile any drift. The doc is pseudocode-only and prescriptive; it does not re-derive math or opine on metric choice. A one-line side change to the export tool's README template wires up discoverability.

**Tech Stack:** GitHub-flavored Markdown; relative-path links rooted at `famail_temporal/exports/`; pseudocode rendered in `text` fences; no framework imports. Audit steps use `grep`, `wc -w`, and direct file reads — no test runner.

---

## Reference materials (read-only)

The drafter consults these throughout. None are modified by this plan except where explicitly listed in **File structure** below.

| Path | Used for |
|---|---|
| `docs/superpowers/specs/2026-05-07-fairness-attribution-export-howto-design.md` | The approved spec — single source of truth for section structure, length budget, voice, and acceptance criteria |
| `famail_temporal/evaluation/export_fairness_attributions.py` | Tool behavior; `_README_TEMPLATE` lives here (line 276); side change target |
| `famail_temporal/exports/2026-04-27T23-21-57_manuel-handoff/metadata.json` | Canonical example F values cited in §3 pitfall #3 (`overall_F_spatial`, `overall_F_causal`, `n_days`) |
| `famail_temporal/exports/2026-04-27T23-21-57_manuel-handoff/README.md` | Per-export README; the how-to deliberately overlaps with this and must stay sign-convention coherent |
| `famail_temporal/config.py` | `ACTIVE_SUPPLY_THRESHOLD`, `DEMAND_FLOOR`, `GRID_DIMS`, `T`, `N_TIME_BUCKETS`, `TIME_BLOCKS`, `DEMOGRAPHIC_FEATURES` — values cited verbatim |
| `famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md` | Sign-convention reference; §3 framing for the GAIL recipe ("agents averaging F/N…") |
| `famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md` | Demographic-projection R² wording; cited as the deeper-reading pointer |
| `famail_temporal/docs/RESEARCHER_HANDOFF.md` | Project-orientation pointer for newcomers; how-to opens by deferring to it |
| `famail_temporal/docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md` | Export design rationale; §8 (per-driver out of scope) and §2 (day broadcasting) cited from the pitfalls catalogue |
| `famail_temporal/fairness/spatial.py` | Confirms `per_cell_fairness_attribution_spatial` exists with the signature implied by the spec |
| `famail_temporal/fairness/causal.py` | Confirms `per_cell_fairness_attribution_causal` exists with the signature implied by the spec |

---

## File structure

| Path | Status | Responsibility |
|---|---|---|
| `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md` | **Create** | The how-to document itself |
| `famail_temporal/evaluation/export_fairness_attributions.py` | **Modify** | Add a one-line link to the how-to inside `_README_TEMPLATE` (the template starts at line 276 and renders into every per-export `README.md`) |

The plan reads (does not modify) the spec at `docs/superpowers/specs/2026-05-07-fairness-attribution-export-howto-design.md` to recover section-by-section requirements.

---

## Pre-flight: lock anchor values before any section is drafted

These tasks produce no commits — their outputs are inlined into the plan so a fresh executor can pick up without re-deriving them. Every subsequent section MUST cite these anchor values verbatim; if any value here drifts from the project, the consistency audits (Task 8, Task 12, Task 16) will catch it.

### Task 0a: Lock the canonical example F values from the manuel-handoff metadata

**Files:** No file changes. Output is the table below; drafter consults it while writing §3 pitfall #3 and the §1 TL;DR.

- [ ] **Step 1: Read the manuel-handoff metadata**

```bash
cat famail_temporal/exports/2026-04-27T23-21-57_manuel-handoff/metadata.json
```

- [ ] **Step 2: Confirm the locked anchor values match the metadata**

```text
Anchor                                      Value (manuel-handoff export)
------                                      -----------------------------
overall_F_spatial                           0.082156
overall_F_causal                            0.805234
n_days                                      5
n_active_cells (sum across blocks)          ≈34,500 across 24 blocks (≈1,440/block mean)
schema_version                              1.0.0
sign_convention                             positive_is_fair
```

The how-to cites the rounded forms `F_spatial ≈ 0.08` and `F_causal ≈ 0.80` in §3 pitfall #3. Use those rounded forms; do NOT cite full precision.

### Task 0b: Lock the canonical config values

**Files:** No file changes. Output is the table below.

- [ ] **Step 1: Read the relevant constants from config.py**

```bash
grep -nE "^(GRID_DIMS|T|N_TIME_BUCKETS|DEMAND_FLOOR|SUPPLY_FLOOR|ACTIVE_SUPPLY_THRESHOLD|DEMOGRAPHIC_FEATURES) " famail_temporal/config.py
```

- [ ] **Step 2: Confirm the locked config values**

```text
Constant                       Value
--------                       -----
GRID_DIMS                      (48, 90)
T                              24
N_TIME_BUCKETS                 288
DEMAND_FLOOR                   0.5
SUPPLY_FLOOR                   0.1
ACTIVE_SUPPLY_THRESHOLD        0.5
DEMOGRAPHIC_FEATURES           ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita"]
```

If any value above does not match `config.py` at execution time, **STOP** and reconcile before drafting. The how-to must not lie about config.

### Task 0c: Lock the canonical phrasing patterns

**Files:** No file changes. Output is the conventions below.

- [ ] **Step 1: Confirm phrasing conventions**

```text
Concept                                Canonical phrasing in this how-to
-------                                ---------------------------------
Per-cell attribution                    "α" (lowercase Greek alpha) or "per-cell α"
                                        Never "alpha_i" or "attribution_i" in prose
Sign convention                         "positive = more fair" or "positive_is_fair"
                                        Never "higher = better" alone (ambiguous)
Granularity of attribution              "per-(cell, hour-block)"
                                        Never "per-cell-per-hour" (ambiguous)
Inactive cells                          "inactive cells carry NaN"
                                        Never "missing" or "masked"
Two metrics                             "F_spatial" and "F_causal" (underscored, not subscripted)
                                        Never "F-spatial" or "F^causal"
Three artifacts                         "the dense .pkl", "the long .pkl", "the tuples .pkl"
                                        Never reference an "export file" generically — always pick one of the three
```

- [ ] **Step 2: Confirm the audience-line phrasing**

The first line after the title in §front-matter is exactly: `Assumes you have read **`docs/RESEARCHER_HANDOFF.md`**. If you have not, start there.`

This phrasing is locked because it is the doc's primary audience-gate; consistency audits will grep for it.

---

## Section-by-section drafting

Tasks 1–7 produce one section each. Each task ends with a commit. Tasks 8 and 12 are consistency audits that produce no new content but may produce edits.

### Task 1: Front matter (title, audience line, TL;DR, "What you have")

**Files:**
- Create: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Create the file with the locked front-matter content**

```markdown
# Using FAMAIL Fairness-Attribution Exports

Assumes you have read **`docs/RESEARCHER_HANDOFF.md`**. If you have not, start there.

## TL;DR

This document tells you how to load a FAMAIL fairness-attribution export and feed its per-cell fairness signal into a GAIL, GAN, or generic offline-RL training loop. It covers loading patterns, the sign convention you must respect, the axis semantics that have a broadcast trap inside them, three self-contained training-method recipes, a numbered pitfalls catalogue, and a sanity-check checklist.

It does **not** re-derive the fairness math (that lives in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md)), it does **not** prescribe a framework, and it does **not** opine on which metric to use, how to normalize, or how to weight your fairness term against your other losses. It shows you how to apply each option correctly; the choice is yours.

## What you have

Each export directory at `famail_temporal/exports/<timestamp>_<name>/` contains:

- **Three `.pkl` artifacts** — `fairness_attribution_dense.pkl` (block-level tensors for fast lookup), `fairness_attribution_long.pkl` (pandas DataFrame for filtering), `fairness_attribution_tuples.pkl` (dependency-free row iteration). Algebraically equivalent; pick by convenience.
- **`metadata.json`** — provenance sidecar (git SHA, source-data SHA, config snapshot, overall F values, active-cell counts per block).
- **`README.md`** — the auto-generated reference card for that specific export. Carries the export's actual F values and `n_days`. This how-to is the prescriptive companion to that reference card.
```

- [ ] **Step 2: Verify the front matter renders cleanly**

```bash
wc -w famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: between 200 and 300 words.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): scaffold fairness-attribution how-to with front matter"
```

### Task 2: §1.1 — Loading the export

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md` (append §1 header + §1.1)

- [ ] **Step 1: Append the §1 banner and §1.1 content**

```markdown

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
```

- [ ] **Step 2: Verify the section reads well**

```bash
grep -nE "^### §1\.1" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: exactly one match.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §1.1 loading instructions"
```

### Task 3: §1.2 — Sign convention and scale

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §1.2 content**

```markdown

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
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §1.2 sign convention and scale"
```

### Task 4: §1.3 — Axis semantics: the broadcast trap

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §1.3 content**

```markdown

### §1.3 Axis semantics: the broadcast trap

The export's row schema looks four-dimensional: `(x_grid, y_grid, time_bucket, day)`. It is not. Two of those axes are **broadcast duplicates**, not independent samples:

- **`time_bucket` within an hour-block.** Fairness is computed at `(x, y, time_block)` granularity (24 hourly blocks). Each block contains 12 five-minute `time_bucket` values. The same per-block α is duplicated across all 12 buckets in the block.
- **`day`.** Fairness is computed pooled across the dataset's `n_days` days, not per-day. The same pooled α is duplicated across every value of the `day` index.

Three concrete consequences:

- **IID sampling along broadcast axes inflates effective sample size.** Drawing N rows uniformly from `df` and treating them as independent observations gives you up to `12 × n_days =` 60 duplicates per (cell, block) in the manuel-handoff export. Variance estimates and standard errors computed against that count are wrong by an order of magnitude or more.
- **"Per-bucket" or "per-day" features are pass-throughs.** Any feature you compute by picking a single (bucket, day) value is identical to its sibling values in the same block; there is no per-bucket or per-day signal to extract.
- **Aggregations along broadcast axes are pass-throughs of the per-block / pooled value.** `mean` and `std` over a broadcast axis return the per-block / pooled value with zero spread.

If your training loop needs a per-state reward at finer granularity than `(cell, hour-block)`, you are looking up a duplicated value, not a finer signal. That is fine — it just means your model is being trained on the same target that the audit measured. Decisions to break the broadcast (e.g., per-day fairness) require recomputing the audit, which is out of scope for this export.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §1.3 axis semantics broadcast trap"
```

### Task 5: §1.4 — Active vs inactive cells

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §1.4 content**

```markdown

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
```

- [ ] **Step 2: Confirm the threshold value cited matches config**

```bash
grep -n "ACTIVE_SUPPLY_THRESHOLD" famail_temporal/config.py
```
Expected: shows `ACTIVE_SUPPLY_THRESHOLD = 0.5`. If the value has changed since pre-flight, update the section.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §1.4 active vs inactive cells"
```

### Task 6: §1.5 — Two metrics, briefly

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §1.5 content**

```markdown

### §1.5 Two metrics, briefly

Each export carries two attribution columns; they measure different things and are not interchangeable.

- **`spatial_fairness_attribution`** decomposes `F_spatial`, a Gini-based measure of equity in service exposure across active `(cell, hour-block)` units. Its per-cell α captures how a cell's service-rate ratios (DSR = pickup/supply, ASR = dropoff/supply) compare to the rest of the active set.
- **`causal_fairness_attribution`** decomposes `F_causal = 1 − r²_demo`, where `r²_demo` is the share of the demand-adjusted residual variance explained by neighborhood demographics. Its per-cell α captures whether the cell's residual service rate aligns with neighborhood wealth (negative α) or is uncorrelated with it (positive α).

Full derivation in [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md). Causal-specific methodology in [`../docs/F_CAUSAL_METHODOLOGY_NOTES.md`](../docs/F_CAUSAL_METHODOLOGY_NOTES.md).

**This document does not prescribe which metric to use.** Both are exported because they measure different things; the choice depends on what your model is trying to optimize and is yours to make.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §1.5 two metrics overview"
```

### Task 7: §2 banner

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append the §2 banner**

```markdown

---

## §2. Recipes

Each recipe is self-contained; pick the one that matches your training method. All three assume you have read §1 (sign convention, broadcast trap, NaN handling) and use the loading patterns from §1.1.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): scaffold §2 recipes section banner"
```

### Task 8: Consistency audit — preamble against project

**Files:** None modified except by remediation if drift is found.

This audit fires after §1 is complete and before any recipe is drafted, because every recipe depends on shared-preamble facts.

- [ ] **Step 1: Re-grep config values cited in §1.4**

```bash
grep -n "ACTIVE_SUPPLY_THRESHOLD\|DEMAND_FLOOR\|GRID_DIMS\|N_TIME_BUCKETS" famail_temporal/config.py
```

Compare each value against what §1 cites. If `ACTIVE_SUPPLY_THRESHOLD = 0.5` no longer matches, update §1.4. If `GRID_DIMS = (48, 90)` no longer matches, update §1.3. The how-to does not invent values — it quotes them.

- [ ] **Step 2: Re-grep the spec to confirm §1 covers every promised subsection**

```bash
grep -nE "^### §1\." famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: §1.1, §1.2, §1.3, §1.4, §1.5 — exactly five subsection headings.

- [ ] **Step 3: Confirm cross-doc references resolve**

```bash
test -f famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md && echo OK
test -f famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md && echo OK
test -f famail_temporal/docs/RESEARCHER_HANDOFF.md && echo OK
```
Expected: three lines of `OK`. Any missing file means a relative-path link in the how-to is broken.

- [ ] **Step 4: Confirm phrasing conventions from Task 0c are upheld**

```bash
grep -nE "alpha_i|attribution_i|F-spatial|F\^causal" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: zero matches. Any match is a phrasing-convention violation; fix it.

- [ ] **Step 5: If any of the four steps above produced edits, commit them**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): preamble consistency audit fixes"
```

If no edits, skip the commit and move on.

### Task 9: §2.1 — Recipe: GAIL / imitation learning reward shaping

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §2.1 content**

```markdown

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
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §2.1 GAIL reward-shaping recipe"
```

### Task 10: §2.2 — Recipe: GAN training

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §2.2 content**

```markdown

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
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §2.2 GAN-training recipes (A/B/C)"
```

### Task 11: §2.3 — Recipe: Generic offline RL

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §2.3 content**

```markdown

### §2.3 Recipe: Generic offline RL (Q-learning, CQL, BCQ, IQL)

**Where α enters:** as a per-state reward bonus on top of whatever extrinsic reward your offline-RL setup uses. Mechanically identical to the GAIL recipe (§2.1) minus the imitation-discriminator structure.

**Pseudocode for reward augmentation:**

```text
# Pre-load once before training
load fairness_attribution_dense.pkl as dense
α_grid = dense["spatial"]                    # or dense["causal"]
mask   = dense["active_mask"]
λ      = a scalar weight you choose

# When constructing the offline replay buffer (or relabeling its rewards):
for transition (s, a, r, s') in dataset:
    if mask[s.cell_x, s.cell_y, s.time_block]:
        r_augmented = r + λ * α_grid[s.cell_x, s.cell_y, s.time_block]
    else:
        r_augmented = r + OFF_SUPPORT_PENALTY
    store (s, a, r_augmented, s') in replay buffer
```

**Active-mask and broadcast notes.** The key precondition is **state-granularity alignment**. The dataset is per-(cell, hour-block); if your RL state is finer than that — for example, per-(cell, 5-minute-bucket) — then per §1.3 you are looking up the same α 12 times across the buckets in the block. That is correct behavior and matches the audit's measurement granularity, not a bug. If your RL state is coarser (e.g., per-cell across all blocks), you must aggregate α across blocks; this aggregation is your design choice and is not prescribed here.

**Relevant pitfalls:** §3 items 1, 2, 4.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §2.3 generic offline-RL recipe"
```

### Task 12: Consistency audit — recipes against project

**Files:** None modified except by remediation if drift is found.

This audit fires after §2 is complete and before appendices are drafted. It checks both internal coherence (across the three recipes) and external coherence (against `famail_temporal/`).

- [ ] **Step 1: Confirm the three recipes share structure**

```bash
grep -nE "^### §2\." famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -nE "^\*\*Where α enters" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -nE "^\*\*Active-mask and broadcast notes" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -nE "^\*\*Relevant pitfalls" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: three matches in each command (one per recipe). If any line has fewer than three, a recipe is missing the corresponding sub-element.

- [ ] **Step 2: Confirm pseudocode references match the export tool's actual output**

```bash
grep -nE 'dense\["spatial"\]|dense\["causal"\]|dense\["active_mask"\]|dense\["metadata"\]' famail_temporal/evaluation/export_fairness_attributions.py
```
Expected: matches showing the dense payload uses keys `spatial`, `causal`, `active_mask`, `D`, `S`, `Y`, `metadata` (see [`export_fairness_attributions.py:199-208`](../../famail_temporal/evaluation/export_fairness_attributions.py#L199-L208)). If any key in the recipes does not appear there, fix the recipe.

- [ ] **Step 3: Confirm referenced functions exist**

```bash
grep -nE "def per_cell_fairness_attribution_(spatial|causal)" famail_temporal/fairness/spatial.py famail_temporal/fairness/causal.py
```
Expected: two matches. The how-to does not name these functions explicitly in pseudocode, but [`FAIRNESS_DECOMPOSITION_FORMULATION.md`](../../famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md) §4 names them, and the how-to cites that doc; an absent function would invalidate the citation.

- [ ] **Step 4: Confirm sign-convention coherence**

```bash
grep -nE "positive.*=.*more fair|positive.*=.*above-baseline|positive_is_fair" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -nE "higher = better|negative = fair" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: the first command produces matches throughout the doc; the second produces zero matches. The latter would indicate a sign-flip relative to the project's frozen convention.

- [ ] **Step 5: Confirm grid-dimension consistency**

```bash
grep -n "48 × 90\|48x90\|48, 90" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -n "GRID_DIMS" famail_temporal/config.py
```
Expected: the dimensions cited in the how-to match `GRID_DIMS` in config.

- [ ] **Step 6: If any step above produced edits, commit them**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): recipes consistency audit fixes"
```

### Task 13: §3 — Pitfalls catalogue

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §3 content**

```markdown

---

## §3. Pitfalls catalogue

Numbered for reference from the recipes.

1. **Treating broadcast axes as independent samples.** Per-bucket and per-day axes carry duplicated values, not independent observations. IID sampling along them inflates effective sample size by up to `12 × n_days` per (cell, block) and corrupts variance estimates. See §1.3.

2. **Clamping per-cell α to [0, 1].** Per-cell α is signed and unbounded; only the overall metric F is in [0, 1]. Clamping silently discards the negative-fair signal and turns a signed reward into a one-sided one. If your loss requires bounded scalars, normalize on your side — pick a transform that respects the sign. See §1.2.

3. **Mistaking the F_spatial / F_causal magnitude imbalance for a bug.** A representative export (the manuel-handoff snapshot) shows `F_spatial ≈ 0.08` and `F_causal ≈ 0.80`. The two metrics measure different things on different scales — Gini-based exposure equity vs demographic explanatory power of the demand-adjusted residual — and are not expected to agree. An order-of-magnitude gap is normal.

4. **Treating attribution as per-trajectory.** Attribution is per-(cell, hour-block); two trajectories whose pickups land in the same cell-block share the same α. Per-driver fairness attribution is explicitly out of scope (see [`../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`](../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md) §8). If you build a model that conditions on driver identity, α is still the per-cell quantity — not a per-driver one.

5. **Uniform sampling over the long format without stratifying on `is_active`.** The long-format DataFrame has roughly `48 × 90 × 288 × n_days` rows, and only ~15% of them are active in a typical export. Uniform sampling trains on a lot of NaN. Filter on `is_active` first, or use the dense format and apply a mask.

6. **Treating `demand_D` and `supply_S` as per-bucket counts.** They are mean-hourly rates at the block level, not raw counts at the bucket level. Downstream features that assume bucket-level counts are off by a factor of 12 (or more, depending on aggregation).

7. **Combining attributions across exports without re-checking `n_days` and `famail_git_sha`.** Attribution magnitudes scale with the active-set size and depend on the source-data SHA. Cross-export aggregation requires explicit reconciliation against both `metadata["famail_git_sha"]` and `metadata["source_data_processing_metadata"]["git_sha"]`; merging exports from different commits silently mixes fairness audits computed under different rules.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §3 pitfalls catalogue (7 items)"
```

### Task 14: §4 — Sanity-check checklist

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §4 content**

```markdown

---

## §4. Sanity-check checklist

Run these after loading the export. Each invariant takes one line; if any fail, your load is corrupt and your training run will be too.

```text
# Sum-to-F invariants (the load-bearing math anchor)
nansum(spatial_attribution)  ≈  metadata["overall_F_spatial"]    # tolerance: 1e-5
nansum(causal_attribution)   ≈  metadata["overall_F_causal"]     # tolerance: 1e-5

# NaN-position invariants
isnan(spatial_attribution)   ==  ~active_mask                    # element-wise
isnan(causal_attribution)    ==  ~active_mask                    # element-wise

# Broadcast-equality invariants (sanity-check the broadcast trap)
spatial[x, y, b1]            ==  spatial[x, y, b2]               # for any b1, b2 in same hour-block
spatial[..., d1]             ==  spatial[..., d2]                # for any day indices d1, d2

# Active-count invariants
metadata["n_active_cells_per_block"][t]  ==  active_mask[..., t].sum()  # per block t

# Cross-format invariants (same data, three views)
dense values  ==  long DataFrame values  ==  tuples row values   # for any chosen (x, y, time_bucket, day)
```

If `nansum(spatial_attribution)` differs from `metadata["overall_F_spatial"]` by more than `1e-5`, the loaded array is not the export tool's output — most likely a stale file, a partial download, or a downstream step that mutated the array in place. Stop and re-load before training.
```

- [ ] **Step 2: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §4 sanity-check checklist"
```

### Task 15: §5 — Pointers

**Files:**
- Modify: `famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`

- [ ] **Step 1: Append §5 content**

```markdown

---

## §5. Pointers

For depth on what this how-to summarizes:

- **Math (1/N decomposition, Gini, demographic R²):** [`../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`](../docs/FAIRNESS_DECOMPOSITION_FORMULATION.md)
- **F_causal methodology, DEMAND_FLOOR rationale, two-R² diagnostic:** [`../docs/F_CAUSAL_METHODOLOGY_NOTES.md`](../docs/F_CAUSAL_METHODOLOGY_NOTES.md)
- **Project orientation (the right entry point if you hit this doc with no FAMAIL context):** [`../docs/RESEARCHER_HANDOFF.md`](../docs/RESEARCHER_HANDOFF.md)
- **Export tool design rationale (per-driver scope, day broadcasting, format choice):** [`../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`](../docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md)
- **Per-export reference card with that export's actual F values:** the `README.md` inside the export directory you received.
```

- [ ] **Step 2: Verify all linked files exist**

```bash
test -f famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md && echo OK
test -f famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md && echo OK
test -f famail_temporal/docs/RESEARCHER_HANDOFF.md && echo OK
test -f famail_temporal/docs/FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md && echo OK
```
Expected: four `OK` lines. Any missing file is a broken link.

- [ ] **Step 3: Commit**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): write §5 pointers"
```

### Task 16: Final consistency audit — full doc against project

**Files:** None modified except by remediation if drift is found.

A full pass that re-runs every consistency check from Tasks 8 and 12 plus several whole-document checks. This is the last opportunity to catch drift before the side change wires the doc into per-export READMEs.

- [ ] **Step 1: Re-run preamble audit checks (Task 8 steps 1, 3, 4)**

```bash
grep -n "ACTIVE_SUPPLY_THRESHOLD\|DEMAND_FLOOR\|GRID_DIMS\|N_TIME_BUCKETS" famail_temporal/config.py
test -f famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md && echo OK
test -f famail_temporal/docs/F_CAUSAL_METHODOLOGY_NOTES.md && echo OK
test -f famail_temporal/docs/RESEARCHER_HANDOFF.md && echo OK
grep -nE "alpha_i|attribution_i|F-spatial|F\^causal" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: config values match, three `OK`, zero phrasing violations.

- [ ] **Step 2: Re-run recipes audit checks (Task 12 steps 2, 4, 5)**

```bash
grep -nE 'dense\["spatial"\]|dense\["causal"\]|dense\["active_mask"\]|dense\["metadata"\]' famail_temporal/evaluation/export_fairness_attributions.py
grep -nE "positive.*=.*more fair|positive.*=.*above-baseline|positive_is_fair" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -nE "higher = better|negative = fair" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
grep -n "48 × 90\|48x90\|48, 90" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: dense keys match, sign-convention phrasing present, anti-phrasing absent, grid dims match.

- [ ] **Step 3: Confirm the manuel-handoff F values cited in §3 still match metadata**

```bash
grep -nE '"overall_F_spatial"|"overall_F_causal"|"n_days"' famail_temporal/exports/2026-04-27T23-21-57_manuel-handoff/metadata.json
```
Expected: `overall_F_spatial ≈ 0.0822`, `overall_F_causal ≈ 0.8052`, `n_days = 5`. The how-to's §3 pitfall #3 cites `F_spatial ≈ 0.08` and `F_causal ≈ 0.80`. If the metadata file no longer rounds to those values, either re-pin the example to a current export or update the rounded form.

- [ ] **Step 4: Confirm acceptance criteria from the spec**

```bash
grep -cE "^### §1\." famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md   # expected: 5
grep -cE "^### §2\." famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md   # expected: 3
grep -cE "^[0-9]+\. \*\*" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md   # ≥ 7 numbered pitfalls
wc -w famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: 5, 3, ≥7, and word count between 3,000 and 4,500 (per spec acceptance criterion 7). If word count exceeds 4,500, prune verbose passages — particularly in the recipes — until it lands. If under 3,000, the doc is probably skipping content; re-check each acceptance criterion.

- [ ] **Step 5: Confirm all four upstream-doc pointers in §5 resolve**

```bash
grep -nE "FAIRNESS_DECOMPOSITION_FORMULATION|F_CAUSAL_METHODOLOGY_NOTES|RESEARCHER_HANDOFF|FAIRNESS_ATTRIBUTION_EXPORT_DESIGN" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: at least one match for each of the four.

- [ ] **Step 6: Confirm the locked phrasing patterns from Task 0c are upheld throughout**

```bash
# Anti-phrasings that must NOT appear:
grep -nE "alpha_i|attribution_i|F-spatial|F\^causal|per-cell-per-hour" famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
```
Expected: zero matches.

- [ ] **Step 7: If any step produced edits, commit them**

```bash
git add famail_temporal/exports/USING_FAIRNESS_ATTRIBUTION_EXPORTS.md
git commit -m "docs(exports): final consistency audit fixes"
```

### Task 17: Side change — wire the per-export README to point at the how-to

**Files:**
- Modify: `famail_temporal/evaluation/export_fairness_attributions.py` (the `_README_TEMPLATE` string starting at line 276)

This is the only mechanism by which a recipient with just an export directory finds the how-to. Without it the doc is effectively orphaned.

- [ ] **Step 1: Read the template's current Contact section**

```bash
sed -n '385,400p' famail_temporal/evaluation/export_fairness_attributions.py
```

The current Contact section (line 390 onward) reads:

```text
## Contact

Methodology questions:
`docs/F_CAUSAL_METHODOLOGY_NOTES.md` and
`docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`.
```

- [ ] **Step 2: Replace it with a version that links to the how-to**

In `famail_temporal/evaluation/export_fairness_attributions.py`, change the Contact section inside `_README_TEMPLATE` (around lines 390–394) from:

```text
## Contact

Methodology questions:
`docs/F_CAUSAL_METHODOLOGY_NOTES.md` and
`docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`.
```

to:

```text
## How to use this export

For a prescriptive how-to on plugging this export into a GAN, GAIL, or
generic offline-RL training loop — including loading patterns, the
sign convention, the broadcast trap, pitfalls, and sanity checks —
see [`../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md`](../USING_FAIRNESS_ATTRIBUTION_EXPORTS.md).

## Contact

Methodology questions:
`docs/F_CAUSAL_METHODOLOGY_NOTES.md` and
`docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`.
```

- [ ] **Step 3: Render a fresh README via a smoke-export and verify the link is present**

```bash
cd /home/robert/FAMAIL && python -m famail_temporal.evaluation.export_fairness_attributions --name plan-smoke --max-trajectories 200 --max-drivers 5 --output-root /tmp/famail-howto-smoke
```

Expected: command succeeds and prints an `[export] README` path.

```bash
grep -n "USING_FAIRNESS_ATTRIBUTION_EXPORTS.md" /tmp/famail-howto-smoke/*/README.md
```

Expected: at least one match. If zero matches, the template edit did not land.

- [ ] **Step 4: Clean up the smoke-export directory**

```bash
rm -rf /tmp/famail-howto-smoke
```

- [ ] **Step 5: Commit**

```bash
git add famail_temporal/evaluation/export_fairness_attributions.py
git commit -m "feat(export): link per-export README to fairness-attribution how-to"
```

### Task 18: Run the existing export-tool tests to confirm the template edit did not break them

**Files:** None modified.

- [ ] **Step 1: Locate and run the export-tool tests**

```bash
grep -rln "export_fairness_attributions\|write_readme" famail_temporal/tests/
```

If a test file is found, run it:

```bash
cd /home/robert/FAMAIL && pytest famail_temporal/tests/<found-test-file> -v
```

Expected: all tests pass. If a `write_readme` test asserts on the absence of the new section, update the test to expect it. If no test file references the export tool's README rendering, that is also acceptable — the smoke-export in Task 17 step 3 is the primary verification.

- [ ] **Step 2: If a test was updated, commit it**

```bash
git add famail_temporal/tests/<file>
git commit -m "test(export): update README assertions for how-to link"
```

If no test changes, skip the commit.

---

## Self-review

Run the full plan against the spec one more time:

- **Spec coverage.** Every section in the spec maps to a task: front matter (Task 1), §1.1–§1.5 (Tasks 2–6), §2 banner (Task 7), §2.1–§2.3 (Tasks 9–11), §3 (Task 13), §4 (Task 14), §5 (Task 15), the side change (Task 17). Periodic consistency audits live at Tasks 8, 12, and 16 — three checkpoints across the three phases (after preamble, after recipes, after appendices) per the user's requirement that the document be periodically verified against the famail_temporal project.
- **Placeholder scan.** Every step contains the actual content the executor will write or run. No "TBD," no "fill in the recipe here," no "similar to Task N." Pseudocode blocks and audit commands are concrete.
- **Type/name consistency.** Pseudocode uses `dense["spatial"]`, `dense["causal"]`, `dense["active_mask"]`, `dense["metadata"]` consistently across §1.1, §2.1, §2.2, §2.3 — and these match the actual payload keys in [`export_fairness_attributions.py:199-208`](../../famail_temporal/evaluation/export_fairness_attributions.py#L199-L208). Function names cited (`per_cell_fairness_attribution_spatial`, `per_cell_fairness_attribution_causal`) match the canonical names in [`FAIRNESS_DECOMPOSITION_FORMULATION.md`](../../famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md) §4.
- **Audit cadence honors the user's request.** Three audit tasks (8, 12, 16) plus inline grep-against-config checks at Tasks 5 and 17. Each audit re-reads the relevant project files at execution time, so drift introduced between drafting and execution is caught — the doc is verified against the project's *current* state, not a snapshot.

---

## Out of scope (deferred to a future plan if pursued)

- Adding fairness-attribution unit tests beyond what already exists in `famail_temporal/tests/` for the export tool.
- Writing per-day or per-driver fairness attribution recipes (excluded by `FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md` §2 and §8).
- Adding consumer-side normalization helpers as a Python module.
- Translating the pseudocode into PyTorch / JAX / TensorFlow.
