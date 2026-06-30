# F_causal robustness across all three demographic feature sets (3-way)

**Question.** The two-pillar argument is measured with F_causal, which is feature-set-specific. Does it reproduce
across demographic feature sets, and how much does only the *scale* (vs the *conclusions*) move?

**Answer.** Every directional conclusion reproduces under all three sets; only the absolute F_causal scale shifts.
The **PRIMARY** {housing, comp, migrant} set gives the cleanest result (SELECT is genuinely null under it). All values
below are **seed MEANS** (not seed-0); `p = 0.03125` is the n = 6 Wilcoxon floor (= all-6-seeds-same-sign), so the
weighted-BC evidence is carried by the **t-CIs + monotone dose-response + 6/6 sign-consistency + null placebo**, not by
an uncorrected p.

## Side-by-side (cleaned data)

| result | ★ {housing,comp,migrant} PRIMARY | {housing,gdp,comp} | {housing,comp,migrant,logpop} | conclusion |
|---|---|---|---|---|
| before-edit F_causal (scale) | 0.7988 | 0.8069 | 0.7253 | feature-set-specific scale; PRIMARY is **not** the lowest (not unfairness-maximizing) |
| **Editor** F_causal before→after | 0.7988 → **0.8132** (Δ **+0.0144**) | 0.8069 → 0.8193 (Δ +0.0124) | 0.7253 → 0.7409 (Δ +0.0156) | editor improves causal fairness in all |
| **L1** edited = fairest faithful? (means) | edited **0.8132** > raw 0.7988 ≈ bc 0.7980 (gan 0.8089 disq.) | edited 0.8193 > raw 0.8069 ≈ bc 0.8064 (gan 0.8153 disq.) | edited 0.7409 > raw 0.7253 ≈ bc 0.7252 (gan 0.7369 disq.) | ✅ holds in all; bc ≈ raw; gate passed |
| **L2** vanilla BC transfer (edited−raw, n=5) | −0.0012 (p = 0.44, n.s.) | −0.0009 (p = 0.44, n.s.) | −0.0010 (p = 0.31, n.s.) | ✅ vanilla BC averages it away (null) |
| **weighted-BC** edited_w30 Δ vs raw | **+0.0311** (t-CI [+0.027,+0.035]) | +0.0260 (t-CI [+0.024,+0.028]) | +0.0274 (t-CI [+0.024,+0.031]) | ✅ upweighting recovers it; CIs exclude 0, dose-responsive, 6/6 |
| **most_fair_w30** (SELECT) | **+0.0004 (n.s., p = 1.0, mixed)** | +0.0012 (n.s., p = 0.56, mixed) | +0.0022 (p = 0.031 = all-same-sign floor) | **weak / metric-dependent; null under PRIMARY** |
| **random_w30** (placebo) | −0.0009 (n.s.) | −0.0004 (n.s.) | +0.0013 (n.s.) | ✅ placebo null on F_causal |
| edit ÷ select ratio @ w30 | **~70×** | ~22× | ~12× | editing dominates selection; sharpest under PRIMARY |
| **variance** model-level (b0 vs FAMAIL, n=5) | −0.0011 ± 0.0032 (null) | −0.0004 ± 0.0014 (null) | +0.0006 ± 0.0010 (null) | ✅ model-level within noise |

## What changed, and what didn't

- **Only the scale shifts.** Before-edit F_causal is 0.799 / 0.807 / 0.725 across the three sets — F_causal is a
  feature-set-specific measure, reported as such. The **PRIMARY 0.799 is not the lowest**, so the headline metric is
  not the one that maximizes apparent baseline unfairness.
- **Every directional conclusion reproduces:** L1 (edited fairest faithful), L2 (vanilla negative transfer; null),
  weighted-BC (dose-responsive recovery, CIs exclude 0), and the model-level null — all three sets agree. This is
  consistent with editor-targeting stability within the housing-retaining family (top-cell Jaccard ≥ 0.92). See
  `fig_feature_robustness.png` (the 3-way dumbbell).
- **SELECT is weak everywhere and cleanly null under PRIMARY.** most_fair_w30 is n.s. with mixed signs under both
  {housing,comp,migrant} (p = 1.0) and {housing,gdp,comp} (p = 0.56); under the density set its "p = 0.031" is only
  the n = 6 all-same-sign floor on a +0.0022 effect. So there is **no robust SELECT tier** — the defensible ordering
  is **EDIT (+0.026…+0.031, robust in all three) ≫ select (weak / null) > random (null)**. Under the PRIMARY metric
  this is unambiguous (edit ~70× select), which is why the PRIMARY set is also the cleanest to present.

## Data sources
- PRIMARY: `results/2026-06-29T12-06-55_…_cleaned_hcm/` + `results/{level1_table_v2/cleaned_hcm_5seed,
  weighted_bc_sweep/cleaned_hcm_6seed, level2_table/cleaned_hcm_5seed, variance_suite/cleaned_hcm_5seed}/`.
- {housing,gdp,comp}: `…_cleaned/` + `…/cleaned_5seed` & `…/cleaned_6seed`.
- {housing,comp,migrant,logpop}: `…_cleaned_4feat/` + `…/cleaned_4feat_5seed` & `…/cleaned_4feat_6seed`.
- The focused density-vs-original 2-way sub-analysis (with the LogPopDensity attribution + co-dominance discussion)
  is `comparison_3v4.md`. Selection analysis: `fcausal_feature_selection.md` / `fcausal_feature_sensitivity.md`.
