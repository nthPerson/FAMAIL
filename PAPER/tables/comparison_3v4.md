# F_causal demographic-feature robustness: 3-feature vs 4-feature

**Question (PI):** is the F_causal demographic-variable choice defensible / optimal, and does the argument survive a better-justified feature set?

**Answer:** A feature-sensitivity analysis (`famail_temporal/analysis/fcausal_feature_sensitivity.py`) showed the
original **3-feature** set {AvgHousingPricePerSqM, GDPperCapita, CompPerCapita} is *dominated* by a **4-feature**
set {AvgHousingPricePerSqM, CompPerCapita, **MigrantRatio**, **LogPopDensity**} that spans four distinct,
low-co-linearity axes (SES / income / population-structure / density), captures more demographic-driven demand
inequity (lower before-edit F_causal) at max VIF 4.51 (< the <10 policy), and **preserves editor targeting**
(top-2293 most-unfair-cell Jaccard 0.93). We adopted the 4-feature set (config commit `7fb1fb2`) and re-ran the
full pipeline. **Every conclusion holds.**

## Side-by-side (cleaned data)

| result | 3-feature | 4-feature | conclusion |
|---|---|---|---|
| **Editor** F_causal before → after (causal-emphasis) | 0.8069 → 0.8193 (Δ **+0.0124**), 2293 edits | 0.7253 → 0.7409 (Δ **+0.0156**), 2442 edits | editor improves causal fairness (more so under the richer metric) |
| **L1** edited = fairest *faithful* source? | edited **0.8193** > raw 0.8069 > bc 0.8045 (gan 0.8152 disqualified) | edited **0.7409** > raw 0.7253 > bc 0.7223 (gan 0.7385 disqualified) | ✅ holds; gate passed both |
| **L2** vanilla driver-conditioned BC transfer (edited − raw) | −0.0009 (p = 0.44, n.s.) | −0.0010 (p = 0.31, n.s.) | ✅ vanilla BC averages it away |
| **weighted-BC** edited_w30 Δ vs raw | **+0.0260** (p = 0.031) | **+0.0274** (p = 0.031) | ✅ upweighting recovers it, significant + dose-responsive |
| weighted-BC most_fair_w30 (select already-fair) | +0.0012 (n.s.) | +0.0022 (p = 0.031) | weakly positive under richer metric, still **~12× smaller than editing** |
| weighted-BC random_w30 (placebo) | −0.0004 (n.s.) | +0.0013 (n.s.) | ✅ placebo null |
| **variance** model-level (b0 vs FAMAIL) | 0.8106 / 0.8102 (Δ −0.0004, null) | 0.7290 / 0.7296 (Δ +0.0006, null) | ✅ model-level within noise |

## What changed, and what didn't
- **The absolute F_causal scale dropped** (~0.81 → ~0.73 baseline): by design — the 4-feature metric captures
  demographic-driven inequity (density + population structure) that the 3-feature set missed. F_causal is a
  *feature-set-specific* measure, reported as such.
- **No qualitative conclusion changed.** L1 (edited fairest faithful), L2 (vanilla negative transfer),
  weighted-BC (significant dose-responsive recovery), and the model-level null all reproduce — consistent with the
  predicted targeting stability (Jaccard 0.93: the editor flags essentially the same trajectories either way).
- **One honest nuance (most-fair):** under the richer metric, upweighting the *already-fairest* trajectories is no
  longer exactly zero — it is weakly but consistently positive (most_fair_w30 +0.0022, all-6-seeds), while the
  random placebo stays null. The clean ordering is therefore **EDIT (+0.027) ≫ SELECT (+0.002) > RANDOM (null)**:
  selecting fair data helps a little, *editing* helps ~12× more. This is arguably a stronger story than
  "selection does nothing," because it shows the gradient.

## Defensibility for the paper
- The feature choice is now backed by a committed, reproducible analysis (marginal-contribution table, VIF/corr
  matrix, set search, Pareto) rather than an undocumented dashboard session.
- Recommended framing: report F_causal as a feature-set-specific measure; justify the four axes; include the
  sensitivity/dominance table as an appendix; note that the editor targeting (hence all conclusions) is robust to
  the feature choice (Jaccard ≥ 0.92 across all housing-retaining sets).

## Data sources
- 3-feature: `results/2026-06-26T12-32-59_..._cleaned/`, `results/{level1_table_v2/cleaned_5seed,
  weighted_bc_sweep/cleaned_6seed, level2_table/cleaned_5seed, variance_suite/cleaned_5seed}/`.
- 4-feature: `results/2026-06-28T11-46-12_..._cleaned_4feat/`, `results/{level1_table_v2/cleaned_4feat_5seed,
  weighted_bc_sweep/cleaned_4feat_6seed, level2_table/cleaned_4feat_5seed, variance_suite/cleaned_4feat_5seed}/`.
- Selection analysis: `famail_temporal/analysis/fcausal_feature_sensitivity.py`,
  `results/analysis/fcausal_feature_sensitivity/`. See also `results/RESULTS_INDEX.md`.
