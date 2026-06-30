# F_causal demographic-feature robustness: 3-feature vs 4-feature (two SENSITIVITY sets)

> **Scope note (2026-06-29).** This table compares the two **sensitivity** sets (the original 3-feature SES/income
> set and the density-augmented 4-feature set). The paper's **PRIMARY** metric is now a third set,
> **{housing, comp, migrant}** (see `../README.md` and `../../by_feature_set/housing-comp-migrant/`). The corrected,
> non-overclaiming framing lives in `../README.md`; this file is retained for the 3↔4 side-by-side. The full **3-way**
> comparison including the PRIMARY set is in **`comparison_across_sets.md`** (the canonical cross-set table). Some original wording below is
> tightened inline per the adversarial review (density attribution, most-fair, seed-mean transcription, null rows).

**Question (PI):** is the F_causal demographic-variable choice defensible, and does the argument survive a
different/richer feature set?

**Answer:** The two-pillar argument reproduces under both the 3-feature and 4-feature lenses; only the absolute
F_causal scale shifts (F_causal is feature-set-specific). The 4-feature set
{AvgHousingPricePerSqM, CompPerCapita, **MigrantRatio**, **LogPopDensity**} **Pareto-improves on / is co-dominant
with** the 3-feature set on the (lower F_causal, lower VIF) frontier — *not* a strict domination (an alternative
{housing, GDP, comp, logpop} ties its F_causal at lower VIF). Its lower before-edit F_causal is driven **~90% by the
LogPopDensity (demand-density) axis**, not by population structure (see `../README.md`). Editor targeting is preserved
within the housing-retaining family (top-2293 Jaccard 0.93). **Every *directional* conclusion holds; the null rows
(L2 vanilla transfer, model-level variance) reproduce as nulls.**

## Side-by-side (cleaned data)

| result | 3-feature | 4-feature | conclusion |
|---|---|---|---|
| **Editor** F_causal before → after (causal-emphasis) | 0.8069 → 0.8193 (Δ **+0.0124**), 2293 edits | 0.7253 → 0.7409 (Δ **+0.0156**), 2442 edits | editor improves causal fairness (more so under the richer metric) |
| **L1** edited = fairest *faithful* source? (seed MEANS) | edited **0.8193** > raw 0.8069 ≈ bc 0.8064 (gan 0.8153 disqualified) | edited **0.7409** > raw 0.7253 ≈ bc 0.7252 (gan 0.7369 disqualified) | ✅ holds; bc ≈ raw (tied); gate passed both. raw/edited are deterministic (std 0) |
| **L2** vanilla driver-conditioned BC transfer (edited − raw) | −0.0009 (p = 0.44, n.s.) | −0.0010 (p = 0.31, n.s.) | ✅ vanilla BC averages it away |
| **weighted-BC** edited_w30 Δ vs raw | **+0.0260** (t-CI [+0.024,+0.028]) | **+0.0274** (t-CI [+0.024,+0.031]) | ✅ upweighting recovers it; CIs exclude 0, monotone dose-response, 6/6 seeds (p = 0.031 is the n=6 floor, not a magnitude) |
| weighted-BC most_fair_w30 (select already-fair) | +0.0012 (n.s., p = 0.56, mixed sign) | +0.0022 (p = 0.031 = n=6 all-same-sign floor) | **metric-dependent, NOT robust**: n.s. under 3-feature, tiny all-same-sign under 4-feature; ~12× smaller than editing either way |
| weighted-BC random_w30 (placebo) | −0.0004 (n.s.) | +0.0013 (n.s.) | ✅ placebo null |
| **variance** model-level (b0 vs FAMAIL) | 0.8106 / 0.8102 (Δ −0.0004, null) | 0.7290 / 0.7296 (Δ +0.0006, null) | ✅ model-level within noise |

## What changed, and what didn't
- **The absolute F_causal scale dropped** (~0.81 → ~0.73 baseline): the 4-feature metric residualizes against more
  between-district variance. Crucially this drop is **~90% LogPopDensity** (a demand-density / geography axis), **not**
  population structure (MigrantRatio's incremental contribution is ~0) — so do **not** read the lower baseline as
  "more *demographic* inequity captured." F_causal is a *feature-set-specific* measure, reported as such; the PRIMARY
  paper metric (`../README.md`) uses the purely-demographic {housing, comp, migrant} set, whose before-edit F_causal
  (0.799) is *higher* than this density set's.
- **Every directional conclusion holds.** L1 (edited fairest faithful), L2 (vanilla negative transfer), weighted-BC
  (dose-responsive recovery, CIs excluding 0), and the model-level null all reproduce — consistent with targeting
  stability *within the housing-retaining family* (Jaccard 0.93). The L2 and model-level rows are **nulls that
  reproduce as nulls**, not positive conclusions.
- **Most-fair (SELECT) is metric-dependent, not a robust tier.** It is +0.0012 (n.s., mixed sign) under 3-feature and
  +0.0022 under 4-feature — but the 4-feature "significance" is only the n=6 all-same-sign Wilcoxon floor (0.03125),
  which **fails to replicate** under the 3-feature metric. So no robust "gradient" can be claimed: the defensible
  ordering is **EDIT (+0.027, robust in both) ≫ select (weak, metric-dependent) > random (null in both)**. Selection
  captures at most ~1/12 of the editing gain.

## Defensibility for the paper
- The feature choice is backed by a committed, reproducible analysis (marginal-contribution table, VIF/corr matrix,
  set search, (F_causal, VIF) Pareto) rather than an undocumented dashboard session.
- Recommended framing: report F_causal as a feature-set-specific, **associational** measure on **10 district-level**
  profiles (ecological-resolution caveat); lead the headline with the PRIMARY {housing, comp, migrant} set; include
  this sensitivity table + the selection analysis as an appendix; and scope the robustness claim to the
  **housing-retaining family** (Jaccard ≥ 0.92), disclosing that the full sweep is formally FRAGILE because dropping
  housing collapses targeting. Do **not** claim strict dominance over the 3-feature set (the sets are co-dominant on
  the (F_causal, VIF) frontier), and do **not** attribute the scale drop to population structure (it is ~90% density).

## Data sources
- 3-feature: `results/2026-06-26T12-32-59_..._cleaned/`, `results/{level1_table_v2/cleaned_5seed,
  weighted_bc_sweep/cleaned_6seed, level2_table/cleaned_5seed, variance_suite/cleaned_5seed}/`.
- 4-feature: `results/2026-06-28T11-46-12_..._cleaned_4feat/`, `results/{level1_table_v2/cleaned_4feat_5seed,
  weighted_bc_sweep/cleaned_4feat_6seed, level2_table/cleaned_4feat_5seed, variance_suite/cleaned_4feat_5seed}/`.
- Selection analysis: `famail_temporal/analysis/fcausal_feature_sensitivity.py`,
  `results/analysis/fcausal_feature_sensitivity/`. See also `results/RESULTS_INDEX.md`.
