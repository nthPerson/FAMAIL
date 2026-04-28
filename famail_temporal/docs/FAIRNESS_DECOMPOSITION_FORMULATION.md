# Fairness Decomposition Formulation

**Purpose.** Documents how we attribute the project's fairness metrics
(`F_spatial`, `F_causal`) to individual `(x, y, time-block)` cells. The
per-cell attribution is used in two places:

1. **The trajectory-modification algorithm** — to identify cells whose
   trajectories most need modification to improve overall fairness.
2. **The Fairness-Attribution Export Tool** — to produce a per-cell
   dataset for downstream consumers (Manuel's GAN/GAIL agent, this
   project's baseline GAN, paper supplementary material).

**Single source of truth.** One canonical decomposition per metric.
Everywhere this project says "fairness attribution," it means exactly
this. Readers should not have to learn two parallel definitions.

---

## TL;DR

For each active cell `i` and each fairness metric `F ∈ {F_spatial, F_causal}`:

```
attribution_i = (1 / N_active) − unfairness_contribution_i
Σᵢ attribution_i = F
```

where `unfairness_contribution_i` is the cell's contribution to the
underlying unfairness quantity (`Gini` for spatial, `r²_demo` for
causal), and `N_active` is the count of active cells (cells in the
fairness audit's active set).

**Sign convention:**
- `attribution_i > 0`: cell contributes *more than its uniform share*
  to fairness → cell is helping the system be fairer.
- `attribution_i ≈ 0`: cell is roughly neutral.
- `attribution_i < 0`: cell contributes less than baseline → cell is
  dragging fairness down.

**Sum invariant:** `Σᵢ attribution_i = F` for both metrics. Reading:
"the sum of all per-cell fairness contributions equals the overall
fairness metric, which is in [0, 1]." Direct alignment with the
published metrics — no `(1 − F)` complement to track.

---

## 1. Why this formulation

### The decomposition problem

Both fairness metrics are scalars in `[0, 1]` with the convention
"higher = fairer." A natural question is: *which cells contribute more
to fairness, and which contribute less?* That is, can we distribute
the global F across N active cells in a way that:

- Each cell gets a signed value (positive or negative).
- The values sum to F (matching the published metric).
- The interpretation aligns with the metric's overall sign convention
  (positive value = good for fairness).

The challenge is that both `F_spatial` and `F_causal` arise from
formulations whose **natural per-cell decomposition lands on the
unfairness side** (`1 − F`), not on the fairness side (`F`).

### F_spatial: structurally `1 − Gini`

```
F_spatial = 1 − 0.5 · (Gini(DSR) + Gini(ASR))
Gini(x)  = Σᵢ contribᵢ(x)
contribᵢ = Σⱼ |xᵢ − xⱼ| / (2 N² · mean(x))
```

The Gini coefficient *is* a sum of per-cell contributions — it's
literally how Gini is computed. Each `contribᵢ` is non-negative
(absolute values) and the sum equals Gini, which equals `1 − F_spatial`.

Decomposing `F_spatial` directly requires inserting a baseline. The
simplest defensible choice is `1/N`:

```
αᵢ_spatial = (1/N) − 0.5·(gini_dsrᵢ + gini_asrᵢ)
Σᵢ αᵢ_spatial = 1 − 0.5·(Gini(DSR) + Gini(ASR)) = F_spatial  ✓
```

The `1/N` baseline answers: *"if every cell contributed equally to
F_spatial, this would be its share."* Any cell's deviation from that
baseline is its signed contribution above or below uniform.

### F_causal: structurally `R'(I−H)R / R'MR`

```
F_causal = R'(I−H)R / R'MR
1 − F_causal = (R'MR − R'(I−H)R) / R'MR  =  r²_demo
```

`r²_demo` decomposes naturally per-cell:

```
r²_demo_contribᵢ = ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR
Σᵢ r²_demo_contribᵢ = r²_demo = 1 − F_causal
```

Each per-cell value is the difference of two squared residuals at
cell `i`: centered residual squared minus post-demographic-fit residual
squared. **This decomposition is signed by construction**: a cell
where demographic regression *worsens* the fit (anti-correlation)
contributes negatively, even though the sum is always in `[0, 1]`.

Decomposing `F_causal` directly with the same `1/N` baseline:

```
αᵢ_causal = (1/N) − r²_demo_contribᵢ
          = (1/N) − ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR
Σᵢ αᵢ_causal = 1 − r²_demo = F_causal  ✓
```

### Why a uniform 1/N baseline

The baseline is a deliberate, defensible modeling choice — it asks
*"in a perfectly fair system, what would each cell contribute?"*
Three observations:

1. **In the perfectly-fair limit** (`Gini = 0` or `r²_demo = 0`),
   every cell's `contribᵢ = 0`, so every `αᵢ = 1/N` and `Σ αᵢ = 1 = F`.
   Each cell contributes uniformly. ✓
2. **In the perfectly-unfair limit** (`Gini = 1` or `r²_demo = 1`),
   the outlier cells absorb `~1` of the unfairness. Their `αᵢ ≈ 1/N − 1
   ≈ −1`; balanced cells stay near `1/N`. `Σ αᵢ = 0 = F`. ✓
3. **Information-theoretically**, the uniform baseline is the
   minimum-assumption prior: no auxiliary signal (demand, supply,
   demographics) is injected into the baseline itself. Any deviation
   from `1/N` is therefore attributable to the metric's underlying
   `contribᵢ`, not to a chosen weighting scheme.

### What this is mathematically equivalent to

The `1/N`-shifted decomposition carries **identical information** to
the underlying `(1 − F)` decomposition; it is simply re-anchored:

```
unfairness_contribᵢ = (1/N) − αᵢ
αᵢ                  = (1/N) − unfairness_contribᵢ
```

Per-cell rankings, pairwise comparisons, and signed differences are
all preserved. What changes is the **sum invariant** (sum-to-F instead
of sum-to-(1−F)) and the **alignment with the published metric's
"higher = fairer" convention** (positive = fair, negative = unfair).

---

## 2. Worked examples

### F_spatial

**Perfectly fair system** (`Gini(DSR) = Gini(ASR) = 0` ⇒ `F_spatial = 1`):

```
gini_dsrᵢ = gini_asrᵢ = 0 for all i
αᵢ_spatial = 1/N
Σ αᵢ = N · (1/N) = 1 = F_spatial  ✓
```

Every cell contributes its uniform share. There is no fairness
information that distinguishes cells.

**Perfectly unfair system** (`Gini = 1` ⇒ `F_spatial = 0`):

One cell carries almost all the Gini mass; others carry ~0.

```
For the outlier cell:    αᵢ ≈ 1/N − 1 ≈ −1
For the balanced cells:  αᵢ ≈ 1/N
Σ αᵢ ≈ −1 + (N−1)·(1/N) ≈ 0 = F_spatial  ✓
```

The outlier is dragging the metric to zero. Other cells are
contributing their fair share, but it's not enough to recover.

**Realistic system** (`F_spatial ≈ 0.6`):

Most cells have `αᵢ ≈ 1/N`, with a long tail of cells with
substantial negative `αᵢ` (cells that contribute disproportionately
to the Gini). Manuel's GAN can use these directly: a high-magnitude
negative `αᵢ` is a "this cell needs help" signal.

### F_causal

**Perfectly fair system** (`r²_demo = 0` ⇒ `F_causal = 1`):

Demographics explain none of the residual variance.

```
For all i:  ((MR)ᵢ² − ((I−H)R)ᵢ²) = 0
αᵢ_causal = 1/N
Σ αᵢ = 1 = F_causal  ✓
```

**Perfectly unfair system** (`r²_demo = 1` ⇒ `F_causal = 0`):

All variance is demographic-explained.

```
Σᵢ ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR = 1
average per-cell shift: 1/N − 1/N = 0
Σ αᵢ = 0 = F_causal  ✓
```

Cells where demographics are most explanatory have negative `αᵢ`;
cells *anti-explained* by demographics have `αᵢ > 1/N` (they are
evidence against demographic discrimination).

---

## 3. Sign convention and consumer interpretation

For both metrics, in the `αᵢ` representation:

| `αᵢ` | Cell semantics | Consumer interpretation |
|---|---|---|
| `> 1/N` | Cell contributes *more than* its uniform share to fairness | Cell is doing well; for the F_causal case, may even be *evidence against* demographic discrimination |
| `≈ 1/N` | Neutral; cell carries roughly its uniform share | Cell is neither helping nor hurting beyond baseline |
| `0 < αᵢ < 1/N` | Cell contributes *less than* its uniform share but still positively | Cell is mildly underperforming the baseline |
| `≈ 0` | Cell contributes nothing to fairness | Cell is at the negative-fair / anti-fair boundary |
| `< 0` | Cell *drags fairness down* | Cell is unfair; high priority for intervention |

**For Manuel's GAN reward shaping:** use `αᵢ` directly as a per-cell
reward. The agent is trained to prefer cells with positive `αᵢ` and
avoid cells with negative `αᵢ`. The `1/N` baseline is meaningful: an
agent that randomly visits any cell averages a reward of `F/N`; only
agents that preferentially visit positive-`αᵢ` cells beat that.

**For the trajectory-modification algorithm:** rank trajectories by
their pickup cell's `αᵢ` *ascending* — the most negative `αᵢ` (most
unfair contribution) gets the highest priority for modification.
`select_top_k` chooses trajectories with `αᵢ < 0` (cells that are
strictly dragging fairness below baseline).

---

## 4. Function reference

The canonical implementations live in:

| Module | Function | Sums to | Notes |
|---|---|---|---|
| `famail_temporal/fairness/spatial.py` | `per_cell_fairness_attribution_spatial(pickup_N, dropoff_N, active_N)` | `F_spatial` | Internally calls `per_unit_gini_decomposition` for DSR/ASR; combines with 1/N shift. Returns 1-D N-vector. |
| `famail_temporal/fairness/causal.py` | `per_cell_fairness_attribution_causal(R, X_demo, XtX_inv)` | `F_causal` | Compact FWL form; works at any N. Returns 1-D N-vector. |

Both functions:
- Return a 1-D tensor of length `N_active`.
- The sum of the returned tensor equals the overall metric (within `EPS`).
- Are differentiable through `R` (causal) or `pickup_N` (spatial) for
  the trajectory-modification algorithm's gradient computation.

---

## 5. Consumers and downstream invariants

### 5.1 Trajectory-modification algorithm

`famail_temporal/algorithm/attribution.py::compute_per_unit_attribution`
loads `αᵢ_spatial` and `αᵢ_causal` from the bundle and returns them.
Downstream:

- `rank_trajectories(trajectories, attribution, unit_map)` ranks
  ascending — most-negative `αᵢ` first.
- `select_top_k(scored, k)` picks the first `k` trajectories with
  `αᵢ < 0`.

This inverts the prior code's "rank by unfairness contribution
descending"; the math is equivalent. Tests pin both the sum invariant
(`sum(αᵢ) ≈ F`) and the ranking direction (negative-most-first).

### 5.2 Evaluation / grid output

`famail_temporal/evaluation/grid.py` builds the `(48, 90, T, 4)`
fairness-aware grid. Channels 0 and 1 are now `αᵢ_spatial` and
`αᵢ_causal` (sums-to-F) rather than the prior `1 − F` decompositions.

### 5.3 Fairness-Attribution Export Tool

`famail_temporal/evaluation/export_fairness_attributions.py` (forthcoming)
emits per-`(x, y, time_bucket, day)` rows containing both `αᵢ_spatial`
and `αᵢ_causal` plus context columns. See
[`FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md`](FAIRNESS_ATTRIBUTION_EXPORT_DESIGN.md).

---

## 6. Decision audit trail

| Decision | Choice | Rationale |
|---|---|---|
| Decomposition target | `F` (sum to F), not `1 − F` | Aligns with published metrics; researcher expectation. |
| Baseline for the `1/N` shift | Uniform across active cells | Minimum-assumption prior; no auxiliary signal injected. Defensible without modeling claims. |
| Sign of `αᵢ` | Positive = above-baseline fair contribution | Aligns with overall metric convention ("higher = fairer"). |
| Single function per metric (no `unfairness_attribution_*` companion) | Simplicity / readability | One decomposition per metric; consumers always mean the same thing. |
| Internal vs export consistency | Same function used in modifier and in export | Eliminates "two attribution methods" ambiguity. |

---

## 7. Relationship to prior code

The previous codebase exposed `per_unit_attribution_from_compact`
(sum to `1 − F_causal`), `per_unit_attribution_signed_from_compact`
(magnitude × sign(HR), different decomposition), and
`compute_spatial_attribution` (sum to `1 − F_spatial`). All have been
removed in favor of the two canonical functions above. The
relationship between old and new:

```
old: per_unit_attribution_from_compact(R, X, XtX⁻¹) → sums to 1 − F_causal
new: per_cell_fairness_attribution_causal(R, X, XtX⁻¹) → sums to F_causal

new == 1/N − old   (per element)
old == 1/N − new   (per element)
```

The signed-by-HR variant is dropped entirely; the directional
information it carried is not load-bearing for either the modifier or
the export, and keeping it would have doubled the public function
surface.

---

## 8. Implementation notes (for future maintainers)

- `1/N` is computed at runtime using `len(R)` (causal) or
  `len(pickup_N)` (spatial). This is the count of *active cells* (the
  active-mask `True` count); inactive cells are not part of the
  decomposition and do not enter the sum.
- The sum-to-F invariant is enforced by `EPS`-tolerant tests at small
  N and at production N; deviations beyond `1e-5` indicate a bug.
- For very small `N` (e.g., test fixtures with `N < 10`), the `1/N`
  shift may be a sizeable fraction of typical `αᵢ` magnitudes. Tests
  account for this by checking signs and sum invariant rather than
  magnitude bounds.

---

## Change log

- **2026-04-24** — Initial version. Locks the 1/N-shifted formulation
  as the project's canonical fairness attribution. Replaces the prior
  `(1 − F)` decomposition functions throughout the codebase.
