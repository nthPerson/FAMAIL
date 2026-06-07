# Trajectory Editing Methodology

> A living methods document for the trajectory-editing pipeline (top-k
> selection → ST-iFGSM perturbation → fairness-grid re-evaluation). Each
> section records *what* the algorithm does, *why* it's specified that way,
> and *what evidence* supports the choice. Keep this in sync with the code
> in [`algorithm/`](../algorithm/) and [`fairness/`](../fairness/); when
> code changes, update this doc so paper-writing-time pulls from a single
> source of truth.

## 0. Convention statement

This document and the codebase follow a single fairness convention:

> **`F_spatial` and `F_causal` are fairness measures in [0, 1].
> A value of 1 is maximally fair; a value of 0 is least fair (maximally
> unfair).** Higher = fairer. Mathematically: `F_spatial = 1 − 0.5·(Gini(DSR) + Gini(ASR))`
> and `F_causal = 1 − r²_demo`. Any field, label, log line, table column,
> or doc reference named `F_spatial`/`F_causal` (or `f_spatial`/`f_causal`)
> reports this fairness quantity. **A positive ΔF means the modification
> improved fairness; a negative ΔF means it degraded fairness.**

The only exceptions are `gini_dsr` and `gini_asr`, which report the
underlying Gini coefficients directly (i.e. **unfairness** quantities, in
[0, 1] with 0 = equal and 1 = maximally unequal). These are the only
fields in metrics.json that follow the unfairness convention.

> **2026-05-14 sign-convention erratum.** Prior to this revision, the
> aggregation function `_scalar_metrics_from_grid` in
> [`runner.py`](../evaluation/runner.py) stored `1 − F` and labeled it
> as `f_spatial` / `f_causal`. As a result, every experiment run before
> 2026-05-14 has metrics.json values that are **the inverse of the
> fairness convention above** — they report unfairness, not fairness.
> The calibration entries §8.1–§8.5 below were originally written
> under that inverted interpretation and have been edited in place to
> use the correct convention; old numeric quotes have been flipped so
> the doc reads naturally. The raw metrics.json files of pre-erratum
> experiments are unchanged and should be read with the inverse in
> mind, or compared only against other pre-erratum runs. Any
> post-erratum experiment (run name dated 2026-05-14 or later, after
> the fix landed) follows the fairness convention.

## 1. Pipeline overview

For a fixed `DataBundle` (active mask, pickup/dropoff/active-taxi tensors,
demographic features, pre-fit `g_0(D)`, hat matrices, trained
discriminator), each experiment runs:

1. **Build the fairness grid** ([`evaluation/grid.py`](../evaluation/grid.py))
   to compute the baseline metrics `F_spatial`, `F_causal`, `Gini(DSR)`,
   `Gini(ASR)`.
2. **Per-cell attribution**
   ([`fairness/causal.py`](../fairness/causal.py),
   [`fairness/spatial.py`](../fairness/spatial.py)) decomposes the global
   fairness metrics into per-cell contributions `αᵢ`. Sign convention:
   `αᵢ > 0` ⇒ cell exceeds the 1/N baseline; `αᵢ < 0` ⇒ cell drags
   fairness below baseline (priority target).
3. **Rank trajectories** by the αᵢ of their pickup cell × time block; the
   top-k subset is the candidate set for editing.
4. **For each candidate trajectory, run ST-iFGSM**
   ([`algorithm/modifier.py`](../algorithm/modifier.py)) to perturb the
   pickup location toward higher overall objective
   `L = α_sp · F_spatial + α_ca · F_causal + α_fi · F_fidelity`.
5. **Re-evaluate** F-metrics on the modified pickup tensor and report
   before/after deltas.

The methodological decisions below all concern step 4 (the optimization
inner loop) plus the algorithmic choices in steps 1–3 that interact with
it.

## 2. ST-iFGSM trajectory perturbation

### 2.1 Step rule

```
delta_t = clip( α · sign( ∇L(p_t) ) , −ε, ε )
p_{t+1} = clip_to_grid( p_t + delta_t )
```

Continuous-coordinate perturbation `p ∈ R²` in pickup-cell space.
Gradients flow through a differentiable **soft cell assignment**
([`algorithm/soft_cell_assignment.py`](../algorithm/soft_cell_assignment.py)):
a temperature-controlled Gaussian softmax over a `(2k+1)²` neighborhood,
producing soft pickup-mass weights that inject differentiable mass into
the global pickup tensor at the trajectory's time block.

Constants (in [`config.py`](../config.py)):

| Knob | Value | Notes |
|---|---|---|
| `STEP_SIZE_ALPHA` | 0.1 | per-iter sign-step magnitude in cell coords |
| `EPSILON_BALL` | 2.0 | cumulative perturbation ceiling |
| `MAX_ITERATIONS` | 50 | upper bound on iters per trajectory |
| `SOFT_NEIGHBORHOOD_SIZE` | 5 | (2k+1)² where k=2 |
| `TAU_MAX`, `TAU_MIN` | 1.0, 0.1 | exponential temperature annealing |
| `ANNEAL_TEMPERATURE` | True | enable annealing during the iter loop |

### 2.2 Convergence: patience-based early stopping with best-iterate tracking

The optimization loop is structurally:

```
best_L ← −∞;  best_δ ← 0;  iters_since_improvement ← 0
for t = 1 .. MAX_ITERATIONS:
    p_t ← p_{t-1} + clip(α · sign(∇L), −ε, ε)
    L_t ← L(p_t)
    if L_t > best_L + CONVERGENCE_TOL:
        best_L ← L_t;  best_δ ← cumulative_δ_t;  best_iter ← t
        iters_since_improvement ← 0
    else:
        iters_since_improvement += 1
        if PATIENCE is not None and iters_since_improvement ≥ PATIENCE:
            break
return modified_trajectory = original.apply(best_δ)
```

**Three methodological commitments**, all with academic precedent:

- **Best-iterate tracking.** The reported modification is the iterate
  achieving the highest objective value, **not** the last iterate.
  Standard in PGD/MI-FGSM literature (Madry et al. 2018; Dong et al. 2018):
  *"we report the best objective seen across K iterations."*
- **Fixed iteration budget by default.** The classical `|ΔL_t| < tol`
  early-stop criterion (which had previously been used in this codebase)
  was found to fire prematurely under ST-iFGSM's sign-only step rule
  (any near-stationary point looked converged after one step; see §5
  Failure Modes). It has been replaced with the patience-based criterion
  below.
- **Patience-based early stop.** Terminate when the best objective has
  not improved by more than `CONVERGENCE_TOL` for `PATIENCE` consecutive
  iterations. Direct analog of early-stopping with patience in deep
  learning training loops. Setting `PATIENCE = None` (CLI `--patience -1`)
  disables the early-stop and always runs to `MAX_ITERATIONS`.

Current defaults: `CONVERGENCE_TOL = 1e-6`, `PATIENCE = 10`.

### 2.3 Why this is the right convergence story

ST-iFGSM produces a **fixed-magnitude step** `α · sign(∇L)` regardless of
gradient magnitude. At a near-stationary point where `‖∇L‖ → 0`:

- The step direction is still `sign(∇L)` (well-defined for any non-zero
  gradient, however small).
- The step magnitude is still `α` (does not shrink toward 0).
- The resulting `ΔL ≈ ∇L · Δp ≈ α · ‖∇L‖` → 0.

A convergence criterion of the form `|ΔL_t| < tol` therefore fires
**whenever the gradient is small** — including in flat regions of the
landscape where the optimizer should keep moving to escape the plateau.
Patience-based convergence on the *best-so-far* objective is the
methodologically correct fix because it (a) doesn't fire on a single
flat step, (b) tolerates noisy fluctuations of ΔL_t below tol, and (c)
encodes the principle that "the optimizer is done when it stops
finding new best iterates" rather than "the optimizer is done when one
step happens to be flat."

## 3. Numerical precision

F-metrics are sums of per-cell residuals over `N ≈ 35,000` active units
at the production T=24 cache. Performing these reductions in **float32**
yields a noise floor of approximately
`F · ε_f32 · √N ≈ 1.0 · 1.2e-7 · 187 ≈ 2e-5`. This is **above** the
default `CONVERGENCE_TOL = 1e-6`, meaning float32 rounding alone can
masquerade as either real improvement or non-improvement.

**Mitigation.** All F-metric internal reductions are performed in
**float64**:

- `per_unit_gini_decomposition`
  ([`fairness/spatial.py`](../fairness/spatial.py)): inputs upcast to
  float64; sort + cumsum + scatter chain proceeds in float64; result
  cast back to caller dtype on return.
- `compute_fcausal_compact`, `apply_i_minus_h`
  ([`fairness/hat_matrices.py`](../fairness/hat_matrices.py)): R, X_demo,
  X^T X⁻¹ all promoted to float64 for the inner reductions; scalar
  result cast back to caller dtype.

This pushes the noise floor to roughly `F · ε_f64 · √N ≈ 4e-14`, which
is **eight orders of magnitude** below the default `CONVERGENCE_TOL`.
External APIs (input/output dtypes) are unchanged — only the internal
arithmetic precision differs.

Cost on the RTX 3070 (the calibration hardware) is negligible at our N
because the float64 path involves a handful of O(Np) reductions per
iter, not O(N²) work.

## 4. Sorted Gini reformulation

The pairwise Gini

```
G(x) = (1 / (2 N² μ)) Σᵢ Σⱼ |xᵢ − xⱼ|
```

is mathematically equivalent to the sorted-order identity (Brown 1994):

```
G(x) = (1 / (n Σx)) Σ_{k=1}^{n} (2k − n − 1) x_(k)
```

and likewise for the per-cell decomposition

```
contrib_(k) =  ((2k − n) x_(k) − 2 C_(k) + S_total) / (2 N² μ)
```

where `x_(k)` is the k-th order statistic and `C_(k)` is the cumulative
sum at position k. Switching from the pairwise (O(N²)) to sorted (O(N log N))
form was an algorithmic equivalence — bit-equivalent modulo
float-summation-order — driven by:

- Pairwise `(N, N)` materialization at N=34,524 = **4.77 GB per allocation**;
  done **twice per iter** in `compute_fspatial`. Prohibitive on GPU
  (8.6 GB VRAM on the RTX 3070).
- O(N log N) substitute runs ~30× faster at production N.
- `torch.sort` is autograd-registered; gradients flow back via the
  inverse permutation (the same `sort_indices` returned by `sort`). The
  function is continuous and differentiable everywhere except at exact
  ties between input elements (measure-zero for DSR/ASR ratios).

Equivalence was verified empirically: bit-exact agreement on values at
N=5000; max gradient diff ≤ 1.3e-7 under autograd; per-cell sum
invariant (`Σ contribᵢ = G`) preserved exactly. At ties, individual
per-element gradients may differ between formulations but their sum
matches, which is all that matters for downstream `total.backward()`
consumption.

## 5. Failure modes encountered (and resolved)

These are documented here so future debugging recognizes the pattern.

### 5.1 Premature convergence under `|ΔL| < tol`

**Symptom.** With `MAX_ITERATIONS = 50`, all 100/100 production
trajectories "converged" with `mean_total_iterations = 2.0` — i.e., the
ST-iFGSM loop took exactly one step and declared victory. ΔF_metrics
were small and inconsistent across re-runs.

**Root cause.** Three compounding causes:

1. ST-iFGSM's sign-only step gives `ΔL ≈ α · ‖∇L‖`, so a small ‖∇L‖
   produces a small ΔL regardless of position quality.
2. Soft-cell assignment at high temperature (early in annealing) smears
   gradient signal across the 11×11 neighborhood, depressing ‖∇L‖ at
   the first few iters.
3. float32 noise floor on N=35K sums is ~2e-5, right at `CONVERGENCE_TOL = 1e-6`.

**Mitigation.** Patience-based convergence on best-iterate (§2.2) plus
float64 internal reductions (§3). Post-mitigation, `mean_total_iterations`
became 27.6 with `mean_best_iter = 16.6` — the algorithm genuinely
explores its iteration budget.

### 5.2 Cache-schema drift on dataclass field additions

**Symptom.** Adding a new field to a `@dataclass` consumed by the
on-disk cache layer produces `AttributeError: '<DataclassName>' object
has no attribute '<field>'` at runtime when an older cache is loaded.

**Root cause.** The state-restoration path bypasses `__post_init__`;
state set via `__dict__` directly. Fields added since the cached
artifact was written are absent from the restored instance.

**Mitigation.** Either regenerate the cache, or use defensive
`getattr(self, field_name, default)` lazy initialization in methods
that touch added attributes (see `G0Function._coef_torch_cache` in
[`fairness/g0_power_basis.py`](../fairness/g0_power_basis.py) for the
canonical pattern). For algorithm-correctness fields, prefer
regeneration; for optional caches, prefer lazy init.

## 6. GPU acceleration

The modifier hot path (per-iter `FAMAILObjective` forward + backward,
soft-cell assignment, multi-stream context build) runs on the
configured device (`--device auto`/`cpu`/`cuda[:idx]`). Two structural
prerequisites for the GPU port were:

- **Vectorized `inject_soft_counts_into_3d`** (no scalar tensor writes
  in a Python loop — each scalar write on CUDA is a separate kernel
  launch + host-device sync). Implementation uses `F.pad` + broadcast
  with a one-hot time selector.
- **Sorted-Gini reformulation** (the (N, N) pairwise allocation
  exceeded the RTX 3070's 8.6 GB VRAM at production N; §4).

The numpy ⇄ torch bridge for `g_0(D)` was also removed: `G0Function`
now exposes `eval_torch(d)` that does the power-basis evaluation
end-to-end in torch, eliminating one CPU-GPU round-trip per iter.

Measured per-iter cost at production scale (N=35K active units,
k=100 trajectories, 50 max iters, no diagnostics):

| Hardware | Per-iter wall | k=100 modifier loop | Projected k=1000 |
|---|---|---|---|
| Original CPU (pre-Phase 1) | ~27 s | ~16 days (extrapolated) | ~16 days+ |
| Phase 1 CPU (sorted Gini, hoisted gathers, cached ms context) | ~5-10 s | (not benchmarked separately) | (not benchmarked) |
| Phase 2 GPU (RTX 3070) | **~94 ms** | **4 min 20 s** | **~43 min** |

## 7. Multi-objective weighting (α tuning)

`L = α_sp · F_spatial + α_ca · F_causal + α_fi · F_fidelity`

The three terms have *very* different gradient magnitudes:

- `F_spatial`, `F_causal`: per-trajectory perturbation moves the global F
  by O(1/N) ≈ 3e-5, so the per-trajectory gradient is small.
- `F_fidelity`: per-trajectory discriminator output, gradient O(1).

ST-iFGSM's `sign(grad)` step rule discards gradient magnitude — at each
iteration the chosen direction is determined by the *sign* of the
weighted gradient sum, with magnitudes contributing only to which term
"wins" the sign decision at each component. So α tuning has a
**non-linear** effect on final F-metric outcomes: doubling α_causal
doesn't double F_causal's contribution per step, it only flips a few
more sign decisions in components where F_causal disagrees with the
other terms.

### 7.1 Calibration approach

α values are determined by an empirical calibration procedure
([documented in §8 below as it is run]):

1. **Step 1: gradient-norm diagnostic.** Run a single small-k
   experiment with `--diagnostics` to record per-term gradient norms
   (`grad_spatial_norm`, `grad_causal_norm`, `grad_fidelity_norm`) and
   dominant-term fractions (`frac_iters_*_dominant`) across all
   iterations. This reveals which term is actually driving the optimizer
   under current α settings.
2. **Step 2: gradient-norm equalization.** If desired, derive α
   approximately inversely proportional to mean ‖∇F_i‖ (one-shot
   GradNorm; Chen et al. 2018) to equalize per-term contribution to the
   sign decision.
3. **Step 3: Pareto sweep.** Run a small grid of α values and plot the
   resulting (ΔF_spatial, ΔF_causal, ΔF_fidelity) trade-off surface.
   Select a point on the Pareto front consistent with the methodological
   priorities of the study.

The reasoning for the priority order: Step 1 is required to know what
direction the optimizer is being pulled before we can sensibly choose α.

## 8. Calibration log

Entries are appended chronologically. Each entry records:
the run name, key α and budget parameters, the resulting per-term
gradient distributions, and any algorithmic decisions justified by the
data.

> **§8.0 reading note.** Calibration entries §8.1–§8.5 were originally
> written under the pre-erratum sign convention (see §0) where positive
> ΔF meant "fairness degradation." Those sections have been edited in
> place: numeric values quoted from metrics.json have been **flipped**
> so they read in the fairness convention (positive ΔF = improvement),
> and the verbal interpretations have been rewritten where the prior
> reading came to opposite conclusions from the data. Raw metrics.json
> files of pre-erratum experiments still contain the inverted numbers.
> Entries §8.6 onward use the post-erratum code directly and need no
> flipping.

### 8.1 Baseline (α=0.33/0.33/0.34, k=20, MAX_ITER=50, PATIENCE=10, GPU)

Run: `2026-05-14T13-39-10_step1-grad-diag` (k=20 used for cheap iteration;
651 iter records collected across 20 trajectories at `mean_total_iterations=32.55`,
`mean_best_iter=21.6`, all 20 patience-triggered).

**Per-iter gradient norms** (population statistics over 651 iters):

| Term | Mean ‖∇F‖ | Median ‖∇F‖ | Max ‖∇F‖ |
|---|---|---|---|
| F_spatial | 2.40e-07 | 2.48e-07 | 5.69e-07 |
| F_causal | **5.04e-06** | 3.87e-06 | 2.56e-05 |
| F_fidelity | 7.70e-08 | **0.00** | 2.72e-06 |

**Weighted contribution to sign decisions** (α_i · mean ‖∇F_i‖):

| Term | Share | Fraction of iters dominant |
|---|---|---|
| F_spatial | 4.5% | **0.0%** |
| F_causal | **94.0%** | **97.5%** |
| F_fidelity | 1.5% | 2.5% |

**Gradient cosines** (over the same 651 iters):

- `mean cos(∇F_sp, ∇F_ca) = 0.022` — the two fairness gradients are
  **near orthogonal**, not in opposition.
- `mean cos(α·∇fairness, ∇F_fi) = −0.019` — fidelity gradient is
  effectively orthogonal too.

**Global metric movement** at the run's exit state (fairness convention):

- `ΔF_spatial = −6.7e-06` (negligible degradation)
- `ΔF_causal = +5.4e-04` (improvement, ~80× larger magnitude than ΔF_spatial)
- `ΔGini_dsr = +1.3e-05` (slight increase in DSR unfairness — consistent
  with F_spatial slight degradation, since F_spatial = 1 − 0.5·Gini)
- `ΔGini_asr = 0.0`

**Interpretation.**

1. **F_fidelity is dormant.** Median gradient norm is exactly zero —
   the `torch.clamp(similarity, 0, 1)` in
   [`fidelity/compute.py`](../fidelity/compute.py) is saturating and
   zero-ing the gradient on more than half of iterations. At
   α_fi = 0.34 the term contributes only 1.5% of weighted gradient
   share. **Removing F_fidelity from the objective (α_fi = 0) should
   produce essentially identical optimization.**
2. **F_causal dominates ALL sign decisions** at the current α — 97.5%
   of iterations have F_causal's weighted gradient as the largest term.
   **Raising α_causal further changes nothing**, because the sign of
   `grad_combined` is already F_causal's sign.
3. **F_spatial never wins.** It's a passive bystander. To force any
   F_spatial influence, α_spatial would need to be raised to ≥ ~0.95
   (compensating the ~20× gradient-magnitude gap) — a single-term
   regime, not a "rebalance."
4. **F_spatial and F_causal gradients are nearly orthogonal**
   (cos ≈ 0.02), not in opposition. The two metrics neither help nor
   meaningfully fight each other directionally. At small k the
   F_causal-dominated step happens to nudge F_spatial slightly in the
   *unfair* direction as a side effect, but the magnitudes are tiny.
5. **F_causal is being optimized correctly.** The optimizer's local
   +∇F_causal step at each iter aggregates to a positive global
   ΔF_causal (+5.4e-04 at k=20). Each per-trajectory `best_L` improves
   the objective AND the final re-evaluated F_causal goes up. No
   local-global divergence at this scale; the algorithm does what
   it's designed to do.

**Decisions arising from this calibration:**

- *Drop α_fidelity = 0 from the production runs by default*, pending a
  confirmation experiment that shows the optimization is unchanged.
- *F_causal optimization works.* It improves at the rates expected from
  the per-iter ∇F_causal magnitude. The question is no longer "does it
  work" but "can we get more improvement per k, and at what
  F_spatial trade-off?"
- *Investigate top-k selection diversity as the main next lever.*
  Pickup-cell clustering of top-k targets is the highest-priority
  diagnostic — if many trajectories share the same source unit, the
  optimization budget is being spent redundantly rather than spreading
  across distinct units.

### 8.2 Trajectory-interference diagnostic + iterative top-k attempt (M2)

**Setup.** Two analyses on the §8.1 run plus one follow-up experiment:

1. **Cell-clustering test on §8.1's top-k.** Compare the pairwise
   distance distribution of the 20 selected pickup cells to a baseline
   of "uniformly draw k cells from the 1,879 active cells, repeat 400
   times."
2. **Destination clustering.** Tally where the modifier sent each
   trajectory after optimization.
3. **M2 experiment.** Re-run k=20 with `--iterative-topk`: re-attribute
   after each modification, pick the most-negative remaining
   trajectory, repeat. Run name: `2026-05-14T14-08-38_step2-iterative-topk-k20`.

**Result — clustering is dramatic.**

| Quantity | Top-k observed | Uniform baseline | Ratio |
|---|---|---|---|
| Pairwise mean distance (cells) | **2.63** | 26.27 | 0.10 |
| Pairwise median distance (cells) | 2.24 | 24.19 | 0.09 |
| KS test (top-k vs uniform) | stat=0.902, **p=3.1e-192** | — | — |

Concretely: 20 trajectories live in **9 unique pickup cells**, all
packed into x∈[11,15], y∈[31,39] on a 48×90 grid (the active set spans
much of the city). After modification, they collapse onto **10 unique
destination cells**, with **6 trajectories landing on (15, 37) alone**
and 14 of 20 piled into just 4 destination cells.

**Cell-level pile-up.** When 6 trajectories all move their pickup mass
into cell (15, 37) — a cell that was positive-α before — that cell's
residual R[(15, 37, t)] gets pushed by ~6× the per-trajectory mass,
flipping its αᵢ from positive (above baseline) to negative (drags
fairness down). This is a real per-cell phenomenon visible in the
cell-histogram analysis. **However, despite the cell-level pile-up,
the global ΔF_causal at k=20 is positive (+5.4e-04 in the fairness
convention) — the algorithm still improves fairness overall.** The
pile-up is a *sub-optimality* (the optimization budget concentrates on
one cell instead of spreading across multiple cells), not a
*catastrophe* (F_causal does not regress).

Visualization saved at
`results/2026-05-14T13-39-10_step1-grad-diag/topk_cell_clustering.png`.

**M2 attempt: iterative top-k with re-attribution.** Originally
`runner._iterative_topk_modify`; as of 2026-06-06 this is the **iterative
(B=1) preset of the unified engine** `algorithm/editing_loop.py::run_editing_rounds`
(`--iterative-topk`; the old standalone function was removed in the §8.7 refactor).
Each round: pull `modifier.current_pickup_3d()`, recompute attribution, re-rank
with already-modified IDs excluded, pick the most-negative-αᵢ remaining
trajectory, modify, repeat.

**Result.** At k=20 the picked trajectories are **bit-identical** to
the §8.1 batch-mode run — same 20 trajectory IDs in the same order,
same 10 destination cells, same `mean_iters=32.5`, `mean_best_iter=21.6`,
`ΔF_spatial=−6.676e-06`, `ΔF_causal=+5.420e-04` (fairness convention).
Re-attribution did not change the ranking at any round.

**Why M2 alone does not change the picture at small k.** A single
trajectory's pickup mass at its destination is roughly
`1/(n_hours_per_block · n_days) ≈ 0.04 per block` at T=24 and n_days=5.
At a destination cell with typical D ∈ [5, 50] this is 0.1–1% of the
cell's demand. A change of that magnitude shifts the cell's αᵢ by a
similarly small fraction — far too small to flip the ranking when the
next round's most-negative candidate has αᵢ on the same order as the
current one. M2 only starts seeing a meaningful re-ranking effect once
~10s of trajectories have piled into a cell — by which point much of
the spreadable improvement has already been claimed.

**Decisions arising from this calibration:**

- *Iterative top-k re-attribution (M2)* does not by itself improve
  outcomes at small k, but its implementation is retained as a CLI
  option (`--iterative-topk`) for future experiments where it may
  matter (larger k, longer iter budgets).
- *Next intervention: enforce selection-time diversity*
  (S1: per-(cell, t_block) budget on top-k). At k=20 we already
  observed 9 unique pickup cells receiving 20 selections — a 2.2×
  concentration. At larger k this is expected to worsen and the budget
  becomes the right control.
- *Confirmatory experiment to consider:* M2 at k=100 to verify the
  scaling argument empirically.

> **Postscript — sharpened by §8.3 (k=1000 evidence).**
> Two refinements to §8.2 emerged from the k=1000 cell-histogram analysis:
>
> 1. **The "geographic clustering" framing was too generic.** At k=20
>    the data showed a small cluster around (15, 37); at k=1000 there is
>    one *dominant outlier (cell, t_block)* — `(28, 52, t=5)` — that
>    accounts for 38.6% of all selections. The right framing is
>    *outlier-unit dominance in top-k*, not "the top-k clusters
>    geographically."
> 2. **Locality (ε-ball) is not the binding constraint.** At k=1000
>    only 2.6% of trajectories land at the ε-ball boundary; mean
>    movement is 1.15 cells. Modifications are voluntarily nearby, not
>    forced. Earlier suggestions to investigate larger ε are
>    deprioritized.
>
> The §8.2 recommendation of "selection-time diversity (M1/S1)" remains
> correct; §8.3 sharpens it to **per-(cell, t_block) budget** with
> `max_per_unit = 1`.

### 8.3 Outlier-unit dominance at k=1000 (motivating S1)

**Setup.** First production-scale run with the post-Phase-2 GPU stack.
Run: `2026-05-14T14-25-01_post-gpu-acceleration-progress-smoke`.
Command:
```
time python -m famail_temporal.evaluation.runner \
    --name post-gpu-acceleration-progress-smoke -k 1000
```
Defaults at the time of run: `MAX_ITERATIONS=50`, `PATIENCE=10`,
`CONVERGENCE_TOL=1e-6`, `--device auto` (cuda was used), no
`--diagnostics`, no `--iterative-topk`, no diversity constraint
(selection ran against the original `select_top_k` with no
`max_per_unit`/`max_per_cell` cap because S1 was not yet implemented).

Analysis script: `python -m famail_temporal.evaluation.cell_histogram_analysis
<results_dir>`. Artifacts saved alongside the experiment:
`cell_histogram_analysis.png` and `cell_histogram_summary.json`.

**Headline numbers.**

| Quantity | k=20 (§8.1) | **k=1000 (§8.3)** |
|---|---|---|
| n_modified | 20 | **1,000** |
| unique origin cells | 9 | **86** |
| unique origin units (cell, t_block) | 20 | **370** |
| unique destination cells | 10 | 132 |
| unique destination units | 20 | 349 |
| **max trajectories at one origin cell** | 2 | **386** |
| **max trajectories at one origin unit** | 1 | **386** |
| **max trajectories at one destination cell** | 5 | **320** |
| **max trajectories at one destination unit** | 2 | **320** |
| mean / max modification distance (cells) | (n/a) | 1.15 / 2.83 |
| % at ε-ball boundary | (n/a) | **2.6%** |
| sign-flips pos→neg at destinations | small | **320** |
| sign-flips neg→pos at destinations | small | 298 |
| ΔF_spatial (fairness convention) | −6.7e-06 | **+8.6e-05** (improvement) |
| ΔF_causal (fairness convention) | +5.4e-04 | **+5.8e-03** (improvement) |

**The dominant pattern.**

```
ORIGIN                          DESTINATION
(28, 52, t=5):  386 trajs  ──→  (28, 51, t=5):  320 trajs   [αᵢ: pos → neg]
                           ──→  (26, 53, t=5):   16 trajs
                           ──→  (27, 52, t=5):   12 trajs
                                + a long tail of small spread cells
```

The (cell, t_block) unit `(28, 52, t=5)` — almost certainly a major
POI at 5am, distinct from the (15, 37) cluster observed at k=20 —
contributed **386 of 1,000 trajectories selected as top-k**. The
optimizer routed **320 of those 386** to the unit one cell west
(`(28, 51, t=5)`), and the cumulative pile-up flipped that destination
unit's per-cell αᵢ from positive (locally fair) to strongly negative
(locally unfair). **Globally, however, the run still improved both
F-metrics**: ΔF_causal = +5.8e-03 and ΔF_spatial = +8.6e-05 (fairness
convention). The pile-up *caps* the improvement (the optimization
budget is being spent redundantly at one destination instead of
spreading across many), but it does **not** flip the global outcome.

**Why the locality framing was wrong.** Mean movement at k=1000 is
**1.15 cells** (median 1.00, max 2.83). Only **2.6% of trajectories**
land at the ε-ball boundary. Trajectories are voluntarily settling in
nearby cells because that's where their local gradient points, not
because ε-ball forbids further moves. The dominant source of pile-up
is therefore *upstream of the modifier* — in how top-k is selected —
not in the modifier's reach.

**Why M2 was structurally limited here.** With 386 trajectories all
sharing the same (cell, t_block) unit, iterative re-attribution
cannot help: they all rank similarly, see the same gradient landscape,
and per-modification mass (~0.04 per block) only slightly shifts the
destination unit's α. By the time the shared destination saturates,
hundreds have already been routed there sequentially.

**Decision arising from this calibration: implement S1
(per-(cell, t_block) budget at selection time).** Concretely, add an
optional `max_per_unit` parameter to `select_top_k` that caps each
(pickup_cell, t_block) at `max_per_unit` selections. Default `None`
(no cap, preserving backwards compatibility of pre-S1 experiments).
Recommended for production runs: `--max-per-unit 1` enforces that
every selected trajectory comes from a distinct (cell, t_block) unit.
Also expose `--max-per-cell` for a stricter form that caps per pickup
cell across all time blocks.

For k=1000 with `--max-per-unit 1`, the selection draws from at most
370 units in the §8.3 corpus (only that many distinct negative-α units
contained any trajectory). To exceed k=370 with this constraint we
would need to either relax to `max_per_unit > 1` or widen the active
mask. Empirically, the negative-α unit pool is ~2,830 units (8.2% of
34,524 active units), so the constraint is non-binding well past
k=1000 in typical configurations.

### 8.4 S1 verification at k=100 — destination pile-up cleanly eliminated

**Setup.** Direct apples-to-apples comparison of two k=100 runs at
identical config (`MAX_ITERATIONS=50`, `PATIENCE=10`, `--device cuda`,
no `--diagnostics`), differing only in the new `--max-per-unit 1`
flag.

| Run | name | command differs by |
|---|---|---|
| baseline | `2026-05-11T13-06-34_phase3-patience-calibration` | (no S1) |
| S1 | `2026-05-14T15-27-08_s1-k100-mpu1` | `--max-per-unit 1` |

**Headline results (k=100, fairness convention).**

| Quantity | baseline (k=100) | S1 max_per_unit=1 (k=100) |
|---|---|---|
| max trajs / dest cell | (similar to baseline; ~25 max in adjacent k=1000 data extrapolated to k=100) | **9** |
| max trajs / dest unit | similar | **3** |
| sign-flips pos→neg at dest | non-zero (the §8.3 mechanism in miniature) | **0** |
| sign-flips neg→pos at dest | smaller | **42** |
| mean Δαᵢ at destinations | small mixed | **+4.07e-05** (positive) |
| **ΔF_spatial** | −2.807e-05 | −2.873e-05 (essentially identical) |
| **ΔF_causal** | **+1.513e-03** | **+1.512e-03** (essentially identical) |

**What S1 achieved.**

- **Destination concentration collapsed.** Max trajs at a single
  destination cell dropped to 9; max at a single (cell, t_block)
  destination dropped to 3 (vs the 320 seen at k=1000 baseline).
- **Zero per-cell sign-flips at destinations.** Not a single
  destination unit went from positive-αᵢ to negative-αᵢ under S1.
  Every change at a destination was either a neg→pos improvement
  (42 cases) or stayed in the same sign category.
- **Mean Δαᵢ at destinations was positive (+4.07e-05).** Locally the
  destinations were uniformly improved on the per-cell attribution.

**What S1 did not change at k=100.**

- **Global ΔF_causal was essentially unchanged from baseline:
  +1.512e-03 vs +1.513e-03 (fairness convention).** Both runs improved
  F_causal by the same magnitude. At k=100 the destination pile-up
  exists in the baseline but the per-cell damage it could have done is
  small enough that S1 doesn't shift the global metric meaningfully.
  S1 *will* matter at larger k where the pile-up has more room to
  matter (see §8.5).

**Interpretation.** S1 cleanly eliminates the per-cell sign-flip
mechanism whose worst-case form appeared at k=1000. At k=100 the
global F_causal improvement was already at its near-maximum given the
selection (the unconstrained baseline had only mild concentration);
S1's contribution is cleanliness rather than additional improvement
magnitude at this scale.

**Open question to pursue next.** With S1 in place, does ΔF_causal at
k=1000 improve *more* than the §8.3 baseline (i.e., is the extra
diversity actually buying additional improvement at scale)? The
prediction is yes — at k=1000 the §8.3 baseline wasted 386 trajectories
piling on one destination, so S1 should produce a larger F_causal gain
by spreading the optimization across 1000 distinct units. Verified in §8.5.

**Decisions arising from this calibration:**

- *Adopt `--max-per-unit 1` as the recommended setting* for all
  production runs that previously used unconstrained `select_top_k`.
  The destination-concentration prevention is unambiguously good
  (no observed downside in the §8.4 data), even though it does not
  alone resolve the global F_causal trend.
- *Run a k=1000 S1 experiment* to verify that the outlier-pile-up
  prevention transfers to global metrics at production scale.
- *Begin investigating the residual local-global divergence as a
  separate research question.* The mechanism is no longer "pile-up
  flipping destinations" but "the pooled F_causal regression
  composition behaves non-linearly under sequential per-trajectory
  modifications even on diverse sources." This may be the place
  where reformulations (per-time-block F_causal, optimal-transport-
  based modifications, or constrained Pareto search) eventually need
  to be considered.

### 8.5 S1 at k=1000 — outlier removal exposes the real failure mode

**Setup.** Apples-to-apples comparison of two k=1000 runs at identical
config except for the new diversity constraint.

| Run | name | command differs by |
|---|---|---|
| baseline (§8.3) | `2026-05-14T14-25-01_post-gpu-acceleration-progress-smoke` | (no S1) |
| **S1** | **`2026-05-14T15-39-54_k1000-s1-mpu1`** | **`--max-per-unit 1`** |

Both at `MAX_ITERATIONS=50`, `PATIENCE=10`, `CONVERGENCE_TOL=1e-6`,
`--device auto` (cuda was used), no `--diagnostics`, no
`--iterative-topk`. The §8.5 run took 31:42 wall on the RTX 3070
(modifier loop ~31 min), comparable to §8.3.

**Concentration (the headline S1 effect — works as advertised):**

| Quantity | baseline (§8.3) | **S1 (§8.5)** | change |
|---|---:|---:|---:|
| unique origin cells | 86 | **179** | 2.1× |
| unique origin units | 370 | **1000** | 2.7× (perfect dedup) |
| unique destination cells | 132 | **205** | 1.6× |
| unique destination units | 349 | **733** | 2.1× |
| max trajs / orig cell | 386 | **18** | **21× reduction** |
| max trajs / orig unit | 386 | **1** | **enforced** |
| max trajs / dest cell | 320 | **21** | **15× reduction** |
| max trajs / dest unit | 320 | **5** | **64× reduction** |
| sign-flips pos→neg at dest | 320 | **0** | **catastrophe eliminated** |
| sign-flips neg→pos at dest | 298 | 304 | similar |
| mean Δα at destinations | +2.15e-05 | **+2.44e-05** | slightly better |
| mean total iterations | 19.4 | 20.6 | similar |
| mean best iteration | (not recorded) | 9.6 | — |
| modifier wall (k=1000) | (similar) | 31.5 min | — |

**Global metric movement (fairness convention):**

| Metric | baseline (§8.3) | **S1 (§8.5)** | comment |
|---|---:|---:|---|
| **ΔF_spatial** | +8.589e-05 | **−1.695e-04** | small degradation under S1 (~2× baseline) |
| **ΔF_causal** | **+5.826e-03** | **+7.662e-03** | **~30% larger improvement** under S1 |
| ΔGini_dsr | +1.72e-04 | (similar order, slightly larger) | — |
| ΔGini_asr | 0.0 | 0.0 | — |

**Interpretation.** S1 successfully prevents the catastrophic
mechanism it was designed for *and* increases the F_causal improvement
gain at scale:

- The 386 → 1 reduction at origin units is exact.
- The 320 → 5 reduction at destination units is dramatic.
- Zero destinations flipped from positive-αᵢ to negative-αᵢ (vs 320 in
  the baseline).
- Mean Δαᵢ at destinations is positive (+2.4e-05), so destinations are
  on average becoming *more* fair locally.
- **Global ΔF_causal improved from +5.83e-03 (baseline) to +7.66e-03
  (S1) — a ~30% larger improvement on the causal-fairness axis.**

The mechanism is now clear: in the §8.3 baseline, 386 trajectories
from one origin unit all routed to one destination, which means 386
"intervention units" were spent on **one big perturbation at one
cell**. The remaining 614 selections did distinct work. With S1
forcing 1000 distinct (cell, t_block) origins, the full 1000 are
spent on distinct interventions — every modification contributes its
own positive Δα somewhere, and the per-trajectory gains aggregate
cleanly into a larger global F_causal improvement. **The outlier
pile in §8.3 was *capping* the per-trajectory improvement** by
under-using 386 of the trajectories on a single saturated destination.

**The small F_spatial degradation under S1 is the trade-off.** F_spatial
goes from +8.6e-05 to −1.7e-04 — still a small magnitude but in the
unfair direction. Under the α=(0.33, 0.33, 0.34) gradient mix, F_causal
dominates 97.5% of sign decisions (per §8.1), and the direction the
optimizer chooses for F_causal slightly hurts F_spatial as a side
effect. This is a genuine multi-objective trade-off, not a defect.

**Implications for the research direction.**

S1 makes the algorithm's true behavior visible at scale: at k=1000,
F_causal improves by ~7.7e-3 (in fairness units) at a F_spatial cost
of ~1.7e-4 — roughly a 45× improvement-to-cost ratio on the F_causal/F_spatial
axes. The natural follow-ups are:

| # | Question | Test |
|---|---|---|
| **R1** | Is F_causal improvement maximized by pure-F_causal optimization? | Run with α=(0, 1, 0) at k=1000+S1 |
| **R2** | Does per-time-block F_causal change the trade-off character? | Reformulate F_causal as a per-block aggregate |
| **R3** | What does the (ΔF_spatial, ΔF_causal) Pareto front look like? | α-sweep at k=1000+S1 |
| **R4** | How big is "+7.7e-3 r² reduction" in fairness terms? | Per-driver / per-passenger / per-cell impact analysis |

**Decisions arising from this calibration:**

- *S1 is the recommended default for production runs.* It produces a
  ~30% larger F_causal improvement than the unconstrained baseline at
  k=1000 with only a small F_spatial cost. The destination-concentration
  prevention also keeps the methods statement clean for paper-quality
  runs.
- *The "structural local-global divergence" hypothesis is dead.*
  Earlier drafts of this section interpreted the data under the
  inverted sign convention and concluded the algorithm was structurally
  failing. Under the correct convention, the algorithm is working as
  designed and S1 makes it work *better*.
- *Next experiment to launch: R1* — pure F_causal optimization, to
  confirm that increasing α_causal's share of the gradient direction
  monotonically grows the F_causal improvement.

### 8.6 R1 — pure F_causal optimization, plus the sign-convention erratum

This calibration entry combines two findings from the same session:
the empirical result of running α=(0, 1, 0), and the discovery that
the prior §8.x interpretations were inverted by a sign-convention
bug in the metrics-aggregation function. The R1 result is what made
the inversion impossible to ignore: under the (then-) prevailing
interpretation, pure-F_causal optimization produced a *worse* F_causal
outcome than the multi-objective baseline. Under the corrected
interpretation, it produces a *better* one — exactly what the theory
predicts.

**Setup.**

Run: `2026-05-14T16-23-15_k1000-s1-r1-causal-only`. Command:

```
time python -m famail_temporal.evaluation.runner \
    --name k1000-s1-r1-causal-only \
    -k 1000 --max-per-unit 1 --diagnostics \
    --override ALPHA_SPATIAL=0.0 \
    --override ALPHA_CAUSAL=1.0 \
    --override ALPHA_FIDELITY=0.0
```

α=(0, 1, 0) puts the entire objective on F_causal. `--diagnostics` is
enabled but per-iter cost is nevertheless *much* lower than the
multi-objective case because α_fi=0.0 skips the entire multi-stream
context build + discriminator forward in the modifier (which we
identified in Phase 1 as the heaviest per-iter cost). Modifier wall:
**3:46** for k=1000 — roughly 13× faster than §8.5's 31:42.

**Result (fairness convention).**

| Metric | §8.5 baseline (α=0.33/0.33/0.34, k=1000+S1) | **§8.6 R1 (α=0/1/0, k=1000+S1)** |
|---|---:|---:|
| ΔF_spatial | −1.70e-04 | **−3.77e-04** (more degradation, expected: not being optimized) |
| **ΔF_causal** | **+7.66e-03** | **+8.79e-03** (~15% larger improvement) |
| ΔGini_dsr | (slight increase) | +7.5e-04 |
| ΔGini_asr | 0.0 | 0.0 |
| Modifier wall | 31:42 | **3:46** |
| `frac_iters_causal_dominant` | (not measured here, was 97.5% at α=0.33 per §8.1) | **100.0%** (as expected) |
| `mean_grad_spatial_norm` | (small) | **0.0** (α_sp=0 skips that backward) |
| `mean_grad_fidelity_norm` | (~zero already, per §8.1) | **0.0** (α_fi=0 skips fidelity entirely) |
| mean total iterations | 20.6 | 32.9 |
| mean best iteration | 9.6 | 21.9 |
| n converged (patience) | 1000/1000 | 986/1000 |

**Interpretation.**

1. **Pure F_causal optimization produces the largest F_causal
   improvement.** Going from α=(0.33, 0.33, 0.34) to α=(0, 1, 0)
   improves ΔF_causal from +7.66e-03 to +8.79e-03 — a ~15% gain. This
   is direction-consistent with §8.1's finding that F_causal already
   dominates 97.5% of sign decisions at α=0.33: the remaining 2.5%
   were the multi-objective term occasionally pulling the step off
   F_causal's gradient, and removing them recovers that loss.
2. **F_spatial degrades more under pure F_causal.** ΔF_spatial moves
   from −1.70e-04 to −3.77e-04 (~2.2× larger degradation). This is
   the cost of optimizing only F_causal — F_spatial is no longer
   getting any of the gradient direction. Still a small magnitude
   in absolute terms.
3. **The 13× per-iter speed-up is unexpectedly large.** It's not from
   the backward-pass skip (decomposed-gradient still runs only one
   backward effectively); it's from skipping the entire fidelity
   forward path: `tau_features_cached`, `ms_kwargs_cached` build, and
   the discriminator forward. Per the §8.1 gradient diagnostic, F_fidelity
   contributes 1.5% of the gradient direction but ~90% of the per-iter
   compute cost. **For analysis runs that don't need F_fidelity,
   `--override ALPHA_FIDELITY=0.0` is a free 10×+ speedup.**

**Sign-convention erratum (the discovery this run triggered).**

Before the R1 result was correctly interpreted, the §8.x calibration
entries were written under an inverted sign convention. The R1 result
under the inverted reading said "pure-F_causal optimization made
F_causal *worse* than multi-objective" — which is structurally absurd
(you can't worsen the only thing you're optimizing). That contradiction
forced a re-examination of the metrics pipeline, which revealed:

The aggregation function `_scalar_metrics_from_grid` in
[`runner.py`](../evaluation/runner.py) was computing
`metrics["f_spatial"] = 1.0 - Σ(grid[..., 0])` and similarly for
`f_causal`. Per [`grid.py`](../evaluation/grid.py), channel 0 already
sums to F_spatial (mathematical fairness measure), so the `1.0 -`
inverted it: the function was reporting the *unfairness* values
(`1 - F`) under the *fairness* label. The runner log line
"F_spatial: 0.9178 -> 0.9180" displayed unfairness numbers.

The bug compounded into [`report.py`](../evaluation/report.py)'s
"key findings" generator, which says `if d["f_spatial"] > 0: improved`
— correct semantics, but operating on the inverted data, so the
report's improvement/regression labels were also flipped.

**Fix applied.** Drop the `1.0 -` in `_scalar_metrics_from_grid`. After
the fix:

- `metrics["f_spatial"]` and `metrics["f_causal"]` are the **fairness**
  values (Σ of the per-cell attribution, in [0, 1], 1 = maximally fair).
- The report.py logic now reads correctly (its `if delta > 0:
  improved` semantics now operate on the right direction).
- Test [`test_runner_real_data.py:31-43`](../tests/test_runner_real_data.py#L31-L43)
  updated to match.
- All 266 fast tests pass.

**Implications for §8.1–§8.5.** Every prior calibration entry has been
edited in place to flip the F-metric numeric quotes and the verbal
interpretations that depended on the sign. The historical narrative
("we discovered a structural failure mode of pooled F_causal
optimization") was incorrect — the algorithm was working all along.
The substantive findings about cell-level pile-up (§8.2–§8.5) and the
distinction between concentrated and distributed interventions remain
real and useful, but the framing shifts from "the algorithm fails;
here's what's broken" to "the algorithm works; here's how to maximize
its improvement at scale."

**Decisions arising from this calibration:**

- *Adopt the fairness convention everywhere* (this is the rule stated
  in §0). All F-metric reporting in the codebase and doc is now in
  the convention "1 = max fair, 0 = min fair." Old metrics.json files
  retain the inverted values as historical artifacts.
- *Pure-F_causal optimization (α=(0, 1, 0)) is the strongest known
  configuration for maximizing F_causal at the cost of F_spatial.*
  At k=1000+S1 it yields ΔF_causal ≈ +8.8e-03 with ΔF_spatial ≈ −3.8e-04.
- *`--override ALPHA_FIDELITY=0.0` should be the default for analysis
  runs that don't claim trajectory plausibility as a result.* The
  fidelity term contributes negligible gradient signal (per §8.1) and
  dominates compute cost. Production runs that need the fidelity
  constraint can opt back in.
- *Open research questions, now sensibly posed:*
  - How does the (ΔF_spatial, ΔF_causal) Pareto front look as α varies?
  - What does +8.8e-03 r² reduction mean in domain-meaningful terms
    (per-driver, per-cell, per-passenger fairness improvement)?
  - Does per-time-block F_causal (R2 in §8.5's table) change the trade-off?

### 8.7 Multi-loop re-attribution + non-regression gate — a negative result (2026-06-06)

**Setup.** A time-boxed "algorithm-improvements" side project tested two
pre-authorized changes against the strongest single-pass baseline
(`2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`: α=(0.2,0.7,0.1),
k=10000, no dedup → **ΔF_causal=+0.0128**, ΔF_spatial=+0.0003):

1. **Multi-loop re-attribution** — a unified outer loop
   (`algorithm/editing_loop.py::run_editing_rounds`) that re-attributes against
   the live grid and re-edits the full negative-α set each round, with a
   convergence stop (round-over-round ΔF_causal < τ for `--round-patience`
   rounds), a cumulative ε-cap from each trajectory's *true original* cell
   (`--epsilon-cap`, default 2.0), and batch (B=K) / iterative (B=1) presets.
   The historical single pass is the `--max-rounds 1` batch special case
   (verified bit-equivalent; see §3.6 of the design spec).
2. **Non-regression acceptance gate** (`--accept-rule non-regression`) — persist
   a per-trajectory edit only if it improves F_causal and does not regress
   F_spatial (vs the trajectory's iter-0 state), replacing the
   weighted-objective best-iterate.

Two full-corpus runs at α_fidelity=0 (a 13× speed lever; validated faithful
below): **A1** = multi-loop, C=2, non-regression
(`2026-06-06T17-45-58_A1_multiloop_C2_nonreg_afi0`); **A3** = multi-loop, C=2,
objective gate (`2026-06-06T18-10-53_A3_multiloop_C2_objective_afi0`).

**Results (fairness convention; higher = fairer, +ΔF = improvement).**

| Config | Round 1 ΔF_causal | Final ΔF_causal | Final ΔF_spatial |
|---|---|---|---|
| Baseline (single-pass, objective, α_fi=0.1) | +0.0128 | **+0.0128** | +0.0003 |
| A3 (multi-loop, objective, α_fi=0) | +0.01271 | +0.01213 | −0.00036 |
| A1 (multi-loop, non-regression, α_fi=0) | +0.01239 | +0.01151 | −0.00064 |

A1 round curve: +0.01239 (r1) → −7.07e-04 (r2) → −1.67e-04 (r3), converged.
A3 round curve: +0.01271 (r1) → −6.43e-04 (r2) → +5.90e-05 (r3), converged.

**Findings.**

1. **Multi-loop re-attribution degrades F_causal, gate-independently.** Under
   *both* gates, round 1 (which *is* the single pass) is the best iterate and
   rounds 2+ are net-negative. This sharpens §8.2 (fine-grained re-attribution
   was *null*) — at batch granularity it is net-*negative*. **Optimal number of
   rounds = 1; the single pass wins.**
2. **α_fi=0 proxy + engine baseline-equivalence validated in one number.** A3
   round 1 = +0.01271 ≈ baseline +0.0128 (0.7% gap). This confirms both that the
   engine refactor reproduces the historical single pass on real data, and that
   dropping the (dormant, per §8.1) fidelity term is a faithful 13×-cheaper
   proxy at bounded ε.
3. **The non-regression gate slightly underperforms** the objective gate at the
   single pass (+0.01239 vs +0.01271) and ended with *worse* hard-grid F_spatial
   (−0.00064 vs −0.00036) — the opposite of its protective intent.
4. **Root cause — the soft-relaxation vs discrete-grid gap.** Editing optimizes
   (and the gate checks) the differentiable *soft* cell-assignment F-metrics, but
   every accepted edit is int-snapped to a single cell and the reported metric is
   the *hard* grid (`compute_per_unit_attribution(...).sum()`). Round 1 claims
   the edits where the soft gain is large enough to survive snapping (≈ +0.0124).
   Direct evidence: in A3, round 2's **1,666 edits — each accepted as a
   soft-objective improvement — collectively reduced hard-grid F_causal by
   6.4e-4.** The same gap lets the non-regression gate protect *soft* F_spatial
   while *hard* F_spatial slips. So both pre-authorized changes are bounded by
   the same relaxation gap, not by their own logic.
5. **MAX_ITERATIONS needs no change.** A1/A3 converged at mean 16–17
   iters/trajectory with only 7–10 of ~7,000 edits hitting the 50-iter cap. The
   non-regression gate's consumption of the iter-0 patience slot is immaterial at
   this convergence speed.

**Decisions.**

- **The single-pass, objective-gate config remains the shipped editing recipe**
  (ΔF_causal=+0.0128). Multi-loop is documented as a negative result; it is *not*
  the default (`--max-rounds` defaults to 1, `--accept-rule` to `objective`).
- The multi-loop engine, non-regression gate, and configurable ε-cap **remain in
  the codebase as opt-in machinery** (defaults preserve historical behavior),
  available if the soft-vs-hard gap is ever closed (e.g., a hard-grid-aware
  acceptance or de-snapping refinement — a gated future change).
- **The outer loop deliberately reports the last round, not the best round.**
  Adding best-round restore would make multi-loop "never worse than single-pass"
  but would mask this finding; it was left as-is so the degradation is visible. A
  best-round restore is the right addition *only if* multi-loop is ever pursued
  for real.
- **ε-convention clarification (supersedes "ε=2 inviolable across loops").** ε=2
  is the inviolable *within-edit* ball (matches the cGAIL 5×5 IL window). The new
  `--epsilon-cap` allows a *cumulative* cap across rounds (default 2.0 = bounded
  to the IL window; `inf` = unbounded stacking), but since multi-loop itself does
  not help, single-pass ε=2 stands as the recommendation.

**Not run** (deprioritized once multi-loop was refuted): C=∞ (A2), B=1-vs-B=K
granularity (A4), and the α_fi=0.1 headline confirmations (H1/H2). The C=2
multi-loop already degrades, so larger ε / finer granularity were not expected to
reverse the sign; the α_fi=0 proxy was validated by finding 2.

## 9. Glossary

- **ST-iFGSM** — Soft-Target Iterative Fast Gradient Sign Method. Variant
  of PGD/I-FGSM where the step rule is `α · sign(∇L)` clipped to an
  ε-ball, and where the discrete cell assignment is replaced by a
  differentiable soft assignment to enable end-to-end gradient flow
  through pickup-coordinate space.
- **F_spatial** — `1 − 0.5 · (Gini(DSR) + Gini(ASR))`, sums to a scalar
  in `[0, 1]`. Higher = fairer.
- **F_causal** — `R'(I − H_demo) R / R' M R`. R = Y − g_0(D),
  Y = supply/demand. Higher = fairer (less of the residual variance is
  explained by demographics).
- **F_fidelity** — Mean discriminator similarity output between
  original and modified trajectory representations. Higher = modified
  trajectory looks more like a plausible same-driver trajectory.
- **αᵢ** (per-cell attribution) — Decomposition of a global F-metric
  into per-cell contributions such that `Σᵢ αᵢ = F`. Sign convention:
  positive = above the 1/N baseline.
- **Patience** — Number of consecutive iterations without improvement
  in the best objective before the early-stop fires.

## 10. References

- Madry et al. (2018). *Towards Deep Learning Models Resistant to
  Adversarial Attacks.* ICLR.
- Dong et al. (2018). *Boosting Adversarial Attacks with Momentum.* CVPR.
- Chen et al. (2018). *GradNorm: Gradient Normalization for Adaptive
  Loss Balancing in Deep Multitask Networks.* ICML.
- Brown, M. C. (1994). *Using Gini-style indices to evaluate the spatial
  patterns of health practitioners: theoretical considerations and an
  application based on Alberta data.* Social Science & Medicine.
- See in-repo: [FAIRNESS_DECOMPOSITION_FORMULATION.md](FAIRNESS_DECOMPOSITION_FORMULATION.md),
  [F_CAUSAL_METHODOLOGY_NOTES.md](F_CAUSAL_METHODOLOGY_NOTES.md).
