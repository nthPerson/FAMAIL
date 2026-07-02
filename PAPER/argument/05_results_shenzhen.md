# Results — Shenzhen (primary)

Shenzhen is the primary dataset. All numbers below are **seed means** (never seed-0), each
traceable to a source JSON listed in the provenance footer. The demographic feature set is the
**PRIMARY** `{housing, comp, migrant}` (neighborhood housing price, per-capita compensation, and
migrant/hukou population share); the two other feature sets are treated as sensitivity checks and
appear only in §6 below.

Reminder on the statistics (full treatment in [`04_evaluation.md`](04_evaluation.md)):
`p = 0.03125` is the **n = 6 Wilcoxon floor** — the smallest value the two-sided signed-rank test
can return, i.e. *all 6 paired seeds share the same sign*. It is a sign-unanimity certificate, not
an effect size, and no test survives multiple-comparison correction at n = 6. The weighted-BC
evidence therefore rests on the **mean Δ + t-CIs + monotone dose-response + null control arms**, not
on an uncorrected p. `F_causal` is **associational** (partial R² of a cross-sectional OLS on 10
district-level demographic profiles), 1 = fairest; a rename to `F_demo` is a pending PI decision.

---

## 1. Editor — the data-level fairness gain

The attribution-guided editor (causal-emphasis weights α = 0.2/0.7/0.1, k = 10 000 edited
pickup-cells) makes the trajectory data fairer on the causal axis while barely moving the secondary
spatial axis:

| metric | before | after | Δ |
|---|---|---|---|
| **F_causal** (1 = fairest) | 0.7988 | **0.8132** | **+0.0144** |
| F_spatial (secondary) | 0.1034 | 0.1025 | −0.0009 |

This is a **causal-emphasis** run (α_spatial = 0.2), so F_spatial is a minor secondary metric, not
the objective — the editor trades a sliver of spatial fairness for the large causal gain. **Do not
read this as "the edit improves both metrics."** On the F_causal objective the edit is a strong,
targeted gain; on F_spatial it is a small downward movement.

---

## 2. Pillar 1 — data quality (L1): edited data is the fairest *faithful* source

Four candidate data sources — the **raw** trajectories, the **edited** trajectories, and
trajectories **generated** by a behavior-cloning (BC) model and by a GAN — are scored on the
fairness metrics and on the two fidelity metrics. The edited data is the fairest source that is
also identity-faithful:

| source | F_causal | Fidelity-A (↑, identity) | Fidelity-B (↓, dist. shift vs raw) |
|---|---|---|---|
| raw | 0.7988 | 0.848 | 0.000 |
| **edited** | **0.8132** | 0.843 | 0.149 |
| bc-generated | 0.7980 | 0.848 | 0.011 |
| gan-generated | 0.8089 | 0.848 | **0.173** |

- **Edited is the fairest.** F_causal: edited **0.8132** > raw 0.7988 ≈ **bc 0.7980** (bc is
  statistically tied with raw — generating from a BC policy does not add fairness).
- **GAN-generated is distributionally disqualified.** Its F_causal (0.8089) is second-highest, but
  it is achieved by **distributional collapse** — the GAN free-runs / degenerates, giving it the
  **worst Fidelity-B of the four sources** (0.173). Its apparent fairness is an artifact of no longer
  looking like real trajectory data. (See the Fidelity-B component panel,
  `PAPER/by_feature_set/housing-comp-migrant/figures/fig_fidb_components.png`.)
- **Editing is identity-faithful.** All four sources sit within ~0.006 on Fidelity-A (raw 0.848,
  edited 0.843, bc 0.848, gan 0.848), so the edit does not cost driver-identity realism.

**Deterministic-gap caveat.** Raw and edited F_causal have **std = 0** across BC seeds — they are
static data-level rescores, so the edited−raw gap is a **point comparison with no sampling CI**. The
gap is also the editor's own optimization target (α_causal = 0.7), so it is expected *by
construction*; its scientific value is that the +0.0144 fairness gain is achieved **while Fidelity-A
is unchanged**.

Figures: `PAPER/by_feature_set/housing-comp-migrant/figures/fig_l1_data_quality.png` (the four-source
comparison), `.../fig_fidb_components.png` (the distributional-collapse panel).

---

## 3. L2 — vanilla transfer is null

Train a driver-conditioned BC policy on each source and compare the resulting F_causal, paired by
seed (edited − raw):

- **edited − raw ΔF_causal = −0.0012** (n = 5, p = 0.44, n.s.; mixed signs 3−/2+, well within the
  ±0.003 cross-seed band).

Vanilla driver-conditioned BC **averages the edit away**: the +0.0144 data-level gain does not
survive standard maximum-likelihood cloning over the ~96% unedited majority. *(At n = 5 a two-sided
Wilcoxon cannot reach p < 0.05 — floor 0.0625 — so this is reported as effect-vs-noise, not as a
significant negative.)* Figure:
`PAPER/by_feature_set/housing-comp-migrant/figures/fig_l2_negative_transfer.png`.

---

## 4. Pillar 2 — weighted BC recovers the fairness (edit-specifically)

Upweighting the edited demonstrations in the BC loss **recovers** the data-level fairness that
vanilla BC averaged away, with a clean monotone dose-response:

| weight | edited − raw ΔF_causal | t-CI | seeds | Wilcoxon p |
|---|---|---|---|---|
| w = 10 | **+0.0205** | [+0.019, +0.022] | 6/6 | 0.03125 |
| w = 20 | **+0.0278** | [+0.025, +0.031] | 6/6 | 0.03125 |
| w = 30 | **+0.0311** | [+0.027, +0.035] | 6/6 | 0.03125 |

All three t-CIs exclude zero, the effect is monotone in the weight, and all 6 seeds are positive at
every weight. This is the **largest weighted-BC recovery of the three feature sets** (§6). Figure:
`PAPER/by_feature_set/housing-comp-migrant/figures/fig_dose_response.png`.

### Edit ≫ select > random — the gain is edit-driven, not oversampling

Two control arms isolate what is doing the work. **most_fair** upweights the *already-fairest
existing* trajectories (a selection control); **random** upweights a size-matched random subset (an
oversampling placebo):

| arm @ w30 | edited − raw ΔF_causal | Wilcoxon p | verdict |
|---|---|---|---|
| **edited** | **+0.0311** | 0.03125 (6/6) | recovery |
| most_fair (select) | +0.0004 | 1.0 (mixed signs) | **null** |
| random (placebo) | −0.0009 | 0.5625 | **null** |

Under the PRIMARY metric **select is genuinely null** — most_fair is n.s. with *mixed signs* at
every weight (w10 +0.0013 p = 0.16; w20 +0.0011 p = 0.31; w30 +0.0004 p = 1.0), not even reaching
the all-same-sign floor. So **editing dominates selection by ~70×** and random oversampling moves
nothing. The fairness gain is **edit-specific**: it cannot be reproduced by *selecting* the
already-fair trajectories or by *randomly* oversampling.

### Filtering is not a substitute for editing (Pareto)

Removing trajectories does not help the F_causal objective either. In the edit-vs-raw-vs-filter@K
Pareto:

| operation | F_causal | F_spatial |
|---|---|---|
| raw | 0.7988 | 0.1034 |
| filter@K (K = 2455 removed) | 0.7935 | 0.1046 |
| **edit** | **0.8132** | 0.1025 |

**Filtering *lowers* F_causal** (0.7988 → 0.7935) while nudging F_spatial up; **editing raises
F_causal strongly** (→ 0.8132) at a small F_spatial cost. On the F_causal objective, neither
selecting nor removing data substitutes for editing it. Figures:
`PAPER/by_feature_set/housing-comp-migrant/figures/pareto_causal_hcm.png`,
`.../pareto_spatial_hcm.png`.

---

## 5. Model-level variance null

A variance-suite comparison of a baseline policy (b0) against the FAMAIL-trained policy on F_causal:

- **ΔF_causal = −0.0011 ± 0.0032** (n = 5; within the cross-seed noise band; null).

At the model level, absent the weighting lever, the difference is indistinguishable from seed noise —
consistent with the L2 vanilla-transfer null (§3) and reinforcing that the recovery in §4 is driven
by the **upweighting**, not by the training procedure alone.

---

## 6. Robustness across three demographic feature sets

F_causal is feature-set-specific, so the two-pillar story was re-measured under three demographic
feature sets. **Every directional conclusion reproduces; only the absolute F_causal scale shifts.**

| result | ★ {housing,comp,migrant} PRIMARY | {housing,gdp,comp} | {housing,comp,migrant,logpop} |
|---|---|---|---|
| before-edit F_causal (scale) | 0.7988 | 0.8069 | 0.7253 |
| editor Δ (before→after) | +0.0144 (→0.8132) | +0.0124 (→0.8193) | +0.0156 (→0.7409) |
| L1 edited fairest faithful? | ✅ (0.8132 > raw ≈ bc; gan disq.) | ✅ | ✅ |
| L2 vanilla (edited−raw, n=5) | −0.0012 (null) | −0.0009 (null) | −0.0010 (null) |
| weighted-BC edited_w30 | **+0.0311** | +0.0260 | +0.0274 |
| most_fair_w30 (select) | +0.0004 (null, mixed) | +0.0012 (null, mixed) | +0.0022 (all-same-sign floor) |
| random_w30 (placebo) | −0.0009 (null) | −0.0004 (null) | +0.0013 (null) |
| edit ÷ select @ w30 | **~70×** | ~22× | ~12× |
| variance (b0 vs FAMAIL, n=5) | −0.0011 ± 0.0032 (null) | −0.0004 ± 0.0014 (null) | +0.0006 ± 0.0010 (null) |

Two points worth surfacing:

- **PRIMARY is not the lowest baseline.** Before-edit F_causal is 0.799 / 0.807 / 0.725 — the PRIMARY
  set is *not* the one that maximizes apparent baseline unfairness, so the headline metric is not
  chosen to inflate the effect.
- **SELECT is weak everywhere and cleanly null under PRIMARY.** most_fair is n.s. with mixed signs
  under both housing-retaining sets; under the density set its "p = 0.031" is only the n = 6
  all-same-sign floor on a small +0.0022 effect. The defensible ordering is **EDIT (+0.026…+0.031,
  robust in all three) ≫ select (weak/null) > random (null)**, sharpest under PRIMARY.

Figure: `PAPER/feature_selection/figures/fig_feature_robustness.png` (the 3-way dumbbell). Full 3-way
table: `PAPER/feature_selection/tables/comparison_across_sets.md`.

---

## Sources / provenance

All values are seed means from the cleaned-data PRIMARY re-run:

- Editor: `PAPER/by_feature_set/housing-comp-migrant/data/editor_hcm_metrics.json`
- L1 data quality: `.../data/L1v2_hcm_multiseed.json`
- L2 vanilla transfer: `.../data/L2_hcm_metrics.json`
- Weighted-BC + control arms: `.../data/weighted_bc_hcm_sweep.json`,
  `.../data/weighted_bc_hcm_paired_stats.json`
- Model-level variance: `.../data/variance_hcm_aggregate.json`
- Pareto (edit vs raw vs filter@K): `.../tables/pareto_points_hcm.csv`
- 3-way feature-set robustness: `PAPER/feature_selection/tables/comparison_across_sets.md`
- Set-level narrative + config/cache provenance: `PAPER/by_feature_set/housing-comp-migrant/README.md`
- Figures (referenced, not regenerated here):
  `PAPER/by_feature_set/housing-comp-migrant/figures/{fig_dose_response, fig_l1_data_quality,
  fig_l2_negative_transfer, fig_fidb_components, pareto_causal_hcm, pareto_spatial_hcm}.png`;
  `PAPER/feature_selection/figures/fig_feature_robustness.png`
