# External fairness metrics (before→after edit) — design

**Date:** 2026-07-02
**Author:** Robert (+ Claude)
**Status:** Approved design → ready for implementation plan
**Context:** Meeting 41 (2026-07-02) P0 action — "the big one." See [[meeting41-plan]].

---

## 1. Motivation

FAMAIL's trajectory editor optimizes an objective containing `F_causal` (a demand-adjusted
double-regression fairness proxy). Reporting a fairness improvement *only* on the metric we
optimized is circular — "you can perform gradient ascent, you're going to optimize for it."
Dr. Zhang's directive: demonstrate the fairness gain on **established fairness metrics that are
NOT in the objective**, computed **before-edit → after-edit**, so the claim is not self-certifying.

This spec covers computing four such metrics over the edited datasets. It is **pure evaluation**:
it consumes the editor's outputs and touches no editing code or intermediate calculations, so the
algorithm-change protocol does not apply.

## 2. Goals / non-goals

**Goals**
- Compute **supply/demand ratio**, **demographic parity**, **disparate impact**, and the
  **Theil index**, before vs after edit, with bootstrap confidence intervals.
- Run across four edited datasets: **Shenzhen PRIMARY `{housing,comp,migrant}`**, its two
  sensitivity feature-sets (`{housing,gdp,comp}`, `{housing,comp,migrant,logpopdensity}`), and
  **SF sf12**.
- Group under **two strategies** (district-extremes and median-split) × **three equity axes**
  (housing, comp, migrant) as a built-in robustness check.
- Emit machine-readable JSON, paper-ready markdown tables, and before→after figures with error bars.

**Non-goals**
- No change to the editor, the objective, or `F_causal`/`F_spatial`.
- No Theil-on-DSR/ASR variant that mirrors `F_spatial`'s Gini — that is the separate P1 action A6.
- No new second dataset; SF sf12 is the existing external-validity set.
- Not a statistical significance test beyond bootstrap CIs.

## 3. The outcome variable

All four metrics operate on the continuous per-active-unit **service ratio**

```
Y = supply / demand = active_taxis_N / max(pickup_N, DEMAND_FLOOR)        (DEMAND_FLOOR = 0.5)
```

Higher `Y` = better served. This is the F_causal-aligned convention already used by
`fairness/causal.py` and `baselines/district_metrics.py`. `supply := active_taxis_3d`,
`demand := pickup_3d` (pickups). Editing changes **demand** (relocated pickups) only; **supply**
and the active-unit mask are unchanged.

## 4. Metric definitions

Given `Y_N` (per active unit) and a binary grouping into disadvantaged **D** and advantaged **A**
unit sets:

| Metric | Definition | Fair value | Improvement |
|---|---|---|---|
| Supply/demand ratio | levels `mean(Y|D)`, `mean(Y|A)` (+ their gap) | equal | gap → 0 |
| Demographic parity (`DP`) | `mean(Y|A) − mean(Y|D)` (signed) | `0` | `|DP|` ↓ |
| Disparate impact (`DI`) | `mean(Y|D) / mean(Y|A)` (0.8 rule) | `1.0` | `DI` → 1 |
| Theil index (`T`) | between-district decomposition of `Y` inequality | `0` | `T` ↓ |

**Theil index (between-district), on `Y`:**
```
ȳ      = mean(Y over all active units)
ȳ_g    = mean(Y over units in district g)
T_between = Σ_g (N_g / N) · (ȳ_g / ȳ) · ln(ȳ_g / ȳ)
```
Uses all districts directly → **grouping-independent**, one value per dataset. Zero-service units
contribute 0 (limit `y·ln y → 0`), so `Y=0` units are safe. Computed on `Y` (not DSR) so all four
metrics share one interpretable outcome — a deliberate scope decision (§2).

**Disadvantaged pole per axis (labeling choice, documented, flippable):**
low housing price, low compensation, high migrant ratio. Raw group means are always reported, so
the direction of any disparity is visible regardless of the label.

## 5. Grouping strategies

For each equity axis ∈ {housing, comp, migrant}, produce a per-unit group label two ways:

- **District-extremes** — rank the distinct demographic regions by the axis value; **D** = bottom
  third, **A** = top third (Shenzhen: 3-vs-3 of 10 districts, matching existing `compute_di`;
  SF: bottom/top third of ACS tracts). Middle third excluded. (For migrant, "disadvantaged =
  high" flips the ranking direction.)
- **Median split** — **D**/**A** = active units below/above the axis median across all units. No
  exclusions.

Result: `DP`, `DI`, and supply/demand-ratio each yield **3 axes × 2 groupings = 6** numbers per
dataset; Theil yields **1**. Directional agreement across the two groupings is a robustness signal.

**Grouping axes are always the 3 equity axes, independent of the edit's objective feature set.** So
demographic values are rebuilt directly from the demographics source (§6), not un-z-scored from the
edit's `hat_matrices` (which reflect whichever feature set was optimized). This keeps grouping
identical across all four datasets and makes the "not the optimized quantity" property explicit.

## 6. Data flow

Pure functions, N-vector in → scalar out (mirroring the grid-unaware `fairness/` module design).

**Before-edit (per dataset):** `DataBundle.load()` for the matching city/config →
- `Y_before` from `bundle.pickup_3d`, `bundle.active_taxis_3d`, `bundle.mask_3d`;
- per-unit raw values for housing/comp/migrant, rebuilt from the demographics source
  (`cell_demographics.pkl` → `data/demographics.enrich_demographics` → select the 3 equity
  columns → index by `mask_3d`). SF uses its ACS-filled equivalent (same 3 feature names).
- **Region id per active unit derived from the demographic values themselves** — cells sharing an
  identical (housing, comp, migrant) profile form a region (recovers Shenzhen's 10 districts and
  SF's ACS tracts). This is **city-agnostic**: it does NOT use `district_metrics.district_of_active_units`,
  which is Shenzhen-only (SF's `grid_to_district_mapping.pkl` carries only `valid_mask`, no district
  ids). Regions are used by Theil; the per-axis district-extremes grouping ranks regions by that axis.

**Planning refinement (2026-07-02):** `DataBundle.load()` takes no city argument — city is selected
by the `FAMAIL_CITY` env var at process launch, so SF runs as `FAMAIL_CITY=sf12 python -m ...`. And
because SF lacks a district-id file, grouping/regions are derived from demographic values (above)
rather than the Shenzhen-only district helpers. The Shenzhen `compute_di`/`district_of_active_units`
path is retained only as a cross-check test for the migrant axis.

**After-edit:** reconstruct `after_pickup_3d` by relocating each edited pickup's per-event mass
(`baselines/datasets.pickup_mass`) from its original to modified cell, read from `histories.pkl`
(`.original` / `.modified`), consistent with the modifier and `build_filtered_pickup_3d`; floor at
`DEMAND_FLOOR`. Supply and mask unchanged → `Y_after`. (Reuse the reconstruction that
`baselines/run_metric_hardening.py` already performs; prefer sharing its helper over duplicating.)

**Datasets & edit dirs:** the four edited results dirs are passed/looked-up per run. Grouping axes
stay {housing, comp, migrant} for all four.

## 7. Uncertainty — paired unit-level bootstrap

Resample active-unit indices with replacement (`B = 1000`, fixed seed), recompute `Y_before`,
`Y_after`, and every metric on the **same** resampled indices → 95% percentile CIs on the levels
**and on the Δ**. The paired design yields a CI on "did it improve," controlling for the shared
unit sample.

**Documented caveat:** units are not iid (spatial correlation; district-constant demographics), so
these are first-order CIs. A clean **driver-level** bootstrap is *not* available: the demand grid is
an independent mean-hourly counts artifact, not a per-driver sum
([datasets.py:89-93](../../../famail_temporal/baselines/datasets.py)), and supply is a fixed
environmental grid — so drivers cannot be resampled into a demand grid without rebuilding the
pipeline. Unit-level is the honest, clean choice given the data construction. SF sf12 CIs will be
wide (small sample) — expected and disclosed.

Bootstrap edge cases: a replicate that empties a group (possible for tiny SF groups) yields `NaN`
for that replicate/metric and is dropped from the percentile computation, with the drop count logged.

## 8. Architecture

New, self-contained; reuses `baselines/district_metrics.py`; leaves `run_metric_hardening.py` intact.

**`famail_temporal/baselines/external_fairness.py`** — pure, grid-unaware:
- `supply_demand_ratio(Y, groups) -> {mean_D, mean_A, gap}`
- `demographic_parity(Y, groups) -> float`  (signed `DP`)
- `disparate_impact(Y, groups) -> float`  (`DI`)
- `theil_index(Y, districts) -> float`  (`T_between`)
- `district_extremes(region_value, region_of_unit, frac=1/3, disadvantaged_high) -> groups`
- `median_split(unit_value, disadvantaged_high) -> groups`
- `paired_bootstrap(Y_before, Y_after, specs, B=1000, seed) -> {name: {before_ci, after_ci, delta_ci}}`

`groups` is a per-unit int array: `0=A`, `1=D`, `-1=excluded`. Each metric fn takes `(Y, labels)`
and returns a float. `paired_bootstrap`'s `specs` is a list of `(name, metric_fn, labels)` — each
metric carries **its own** per-unit label array (group labels for `DP`/`DI`/supply-demand, district
labels for Theil). One resampled index vector per replicate is applied to `Y_before`, `Y_after`, and
every metric's `labels` array in lockstep, so all metrics share the same paired resample.

**`famail_temporal/baselines/run_external_fairness.py`** — CLI orchestrator:
`python -m famail_temporal.baselines.run_external_fairness --edit-dir <dir> --dataset <shz|sf> [--out-dir ...] [--seed 0] [--bootstrap 1000]`
Loads bundle → builds `Y_before`/`Y_after`, districts, per-axis demographics → computes all
metrics × axes × groupings + bootstrap → writes outputs (§9). Follows the repo CLI convention
(`main(argv=None) -> int`, argparse `prog=`, `raise SystemExit(main())`).

## 9. Outputs

Written to `famail_temporal/baselines/external_fairness/results/<ts>_<dataset>/`:
- **`external_fairness.json`** — every metric × axis × grouping, `before` / `after` / `delta` each
  with point estimate + 95% CI; plus run metadata (edit-dir, dataset, git sha, seed, B).
- **Markdown before/after/Δ table** per dataset (mirror `evaluation/report.py::_fairness_table`),
  and one **combined cross-dataset comparison table**.
- **Figures** (matplotlib) — per-metric before→after with CI error bars (forest / grouped bar),
  matching the "every core result has a figure with error bars" bar from Meeting 41.

## 10. Testing (TDD)

Unit tests on synthetic N-vectors with known answers:
- Perfect parity (`Y` equal across groups) → `DP=0`, `DI=1`, `T_between=0`.
- Hand-computed skewed case for each metric.
- Monotonicity: relocating service toward **D** improves (moves toward fair) every metric.
- Grouping helpers: correct D/A/excluded assignment; disadvantaged-pole flip works per axis.
- Theil: zero-service units contribute 0; invariance to scaling `Y` by a positive constant.
- Bootstrap: determinism under fixed seed; CI brackets the point estimate; empty-group replicate
  is dropped and counted.
Integration smoke: run on one Shenzhen edit dir end-to-end; assert JSON schema + table/figure files
exist and Δ signs are finite.

## 11. Risks / open items (resolve during planning)

- **SF adaptation (RESOLVED during planning):** SF has no district-id file, so grouping/regions are
  derived from demographic values (city-agnostic). The demographics-rebuild chain works on SF
  (same 3 feature names, ACS-filled). SF is run via `FAMAIL_CITY=sf12`; no code branch needed beyond
  reading `config.CITY` for output labeling.
- **After-grid reconstruction:** confirm the exact mechanism `run_metric_hardening` uses and share
  it rather than reimplement; verify relocated-mass flooring matches the modifier.
- **District-extremes on SF:** number of ACS tracts determines the "third" cut; parameterize.
- **Theil interpretability:** report alongside a note that it is between-district inequality of the
  supply/demand ratio (distinct from the group-comparison metrics).

## 12. References

- Meeting 41 plan: [[meeting41-plan]] · Fairness theory: `PAPER/argument/03_fairness_theory.md`
- Reused code: `baselines/district_metrics.py` (`compute_di`, `district_of_active_units`),
  `baselines/run_metric_hardening.py` (before/after harness + table), `baselines/datasets.py`
  (`pickup_mass`, `build_filtered_pickup_3d`), `baselines/localized_metrics.py`
  (`edited_units_from_histories`), `data/loader.py` (`DataBundle.load`),
  `evaluation/report.py` (`_fairness_table`).
