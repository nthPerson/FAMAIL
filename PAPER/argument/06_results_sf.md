# Results — San Francisco (external validity)

San Francisco (SF Cabspotting, `sf12` density-matched subsample) is the **external-validity**
dataset. The purpose is not a magnitude contest with Shenzhen — F_causal is a **city-specific,
associational** R² measured against **ACS demographic proxies**, so absolute levels are not
cross-city commensurable. The purpose is to test whether **every directional conclusion of the
two-pillar argument reproduces on an independent city with no algorithm, metric, or fidelity
change.** It does. All values below are seed means (or the deterministic data-level rescore where
noted), traceable to the source JSONs/CSVs in the provenance footer.

Statistical reminder (details in [`04_evaluation.md`](04_evaluation.md)): `p = 0.03125` is the n = 6
Wilcoxon floor (all 6 seeds share a sign), a sign-unanimity certificate rather than an effect size.

---

## 1. The dual claim — fairer *and* realistic, with no algorithm change

The identical editor (causal-emphasis α = 0.2/0.7/0.1, fidelity ON, `-k 2000` → 1371 trajectories
edited, 1341 converged, mean 25.3 iterations) makes the SF data fairer on the causal axis while
preserving driver realism:

| metric | before | after | Δ |
|---|---|---|---|
| **F_causal** (1 = fairest) | 0.8752 | **0.8891** | **+0.0139** |
| F_spatial (secondary) | 0.1846 | 0.1817 | −0.0030 |
| **F_fidelity** (realism) | — | **0.968** | edit-induced Δ ≈ **−1.5e-5** |

- **F_fidelity = 0.968** is the mean discriminator P[same driver | original, edited] over the 1371
  edits (min 0.922, median 0.979); the **edit itself barely moves it** (mean drop 1.5e-5) — edited SF
  trajectories are still recognized as the same driver. As on Shenzhen, this is a *causal-emphasis*
  run, so F_spatial moves down slightly (do not claim "improves both metrics").
- **Fidelity is inert as a gradient.** A matched run with fidelity OFF at the same `-k 2000` gives
  ΔF_causal **+0.01392** vs **+0.01394** with fidelity ON — a 2e-5 difference. Turning fidelity on
  costs **zero** fairness; the fidelity gradient w.r.t. the edited pickup cell is ~0 (2.6e-11). This
  matches Shenzhen's `fidelity-grad ≈ 0` property (§7.3 of the SF findings; see also
  [`03_fairness_theory.md`](03_fairness_theory.md)).
- **Two ΔF_causal figures, both correct.** The **+0.0199** figure is the *subsample-selection* metric
  (causal-emphasis over the entire unfair pool, ~762 highest-attribution trajectories, fidelity off) —
  the metric that chose sf12 over sf50. The **+0.0139** figure is the *dual-claim headline* (`-k 2000`
  → 1371 edits, fidelity on). Different edit subsets (the dual-claim run edits more, lower-impact
  trajectories, lowering the per-trajectory-averaged gain), not a regression.
- The F_fidelity discriminator was retrained for SF and reached **val-AUC 0.998**.

Figure: `PAPER/second-dataset/figures/sf_supply_demand.png` (the sf12 supply/demand regime).

---

## 2. Pillar 1 — data quality (L1): edited is the fairest *faithful* source

The Fidelity-A validation gate PASSED (real-anchored: matched real-driver pairs 0.958 vs mismatched
0.034, margin 0.20), so Fidelity-A is trusted on sf12 despite the 12-identity concern. Four data
sources scored on the fairness + fidelity axes:

| source | F_causal | Fidelity-A (↑, identity) | Fidelity-B (↓, dist. shift vs raw) |
|---|---|---|---|
| raw | 0.8752 | 0.958 | 0.0000 |
| **edited** | **0.8891** | 0.958 | 0.1058 |
| bc-generated (MLE) | 0.8789 | 0.958 | 0.0100 |
| gan-generated (WGAN-GP) | 0.8794 | 0.958 | 0.0269 |

**Edited is the fairest source** (F_causal 0.8891 > raw 0.8752 ≈ bc 0.8789 ≈ gan 0.8794) while
**identity-faithful** (Fidelity-A 0.958 = raw). Edited's Fidelity-B (0.106) is the highest of the
non-raw sources — expected, since the edit deliberately relocates pickups within the ε = 2-cell ball;
it is a modest divergence. **Pillar 1 reproduces Shenzhen.** As on Shenzhen, raw and edited F_causal
are deterministic data-level rescores (no sampling CI), and the edited−raw gap is the editor's own
objective — its value is being achieved at unchanged Fidelity-A.

One qualitative difference: on SF the GAN-generated source is **not** distributionally disqualified
(its Fidelity-B is a healthy 0.0269, not the ~0.32 collapse seen on Shenzhen). Pillar 1 does not
depend on disqualifying the GAN — edited still wins outright — so if anything the claim is *cleaner*
here. Details in §7 below.

---

## 3. L2 — vanilla transfer is null

Driver-conditioned BC trained on each source, F_causal re-scored, paired by seed (5 seeds):

- **edited − raw ΔF_causal = +0.0004 ± 0.0033** (n = 5, Wilcoxon p = 0.81, null).

Vanilla BC averages the edit away — exactly Shenzhen's L2 null. This is the null that Pillar 2
overcomes. *(At n = 5 the two-sided Wilcoxon cannot reach p < 0.05, floor 0.0625; reported as
effect-vs-noise.)*

---

## 4. Pillar 2 — weighted BC recovers the fairness, with *both* controls negative

Upweighting the edited demonstrations in BC (6 seeds), paired ΔF_causal vs raw:

| arm | Δ vs raw (w10 / w20 / w30) | verdict |
|---|---|---|
| **edited** | **+0.0296 / +0.0348 / +0.0387** (6/6, p = 0.03125) | **recovery**, monotone dose-response |
| random placebo | −0.0071 / — / −0.0095 (p = 0.03125) | **negative** — oversampling a random subset *hurts* |
| most-fair select | −0.0117 / −0.0068 / −0.0027 | **negative** — upweighting the already-fair *hurts* |

Importance-weighting **recovers** the fairness (monotone +0.0296 → +0.0387, all 6 seeds, Fidelity-A
unchanged at ~0.958). The result is **sharper than Shenzhen**: there, both control arms were ~null;
on SF **both controls are negative**. This is not a cross-city magnitude claim — it is a difference in
the *sign structure* of the controls. The interpretation (SF findings §7.2): SF's edited slice is
~5× denser (§6 below), so at high weight the random-placebo and most-fair-select arms concentrate BC
capacity on non-fairness-improving subsets and actively degrade F_causal, whereas on Shenzhen's
thinner slice they merely failed to help. Either way the qualitative claim holds and is *cleaner* on
SF: **the gain is edit-specific — not oversampling, not selection.** Figure: the dose-response
recovery is the same money-figure family as Shenzhen's
`PAPER/by_feature_set/housing-comp-migrant/figures/fig_dose_response.png`.

---

## 5. Model-level variance null

Paired b0 (raw-corpus BC) vs FAMAIL (edited-corpus BC), MLE-only, 5 seeds:

- **ΔF_causal = −0.0005 ± 0.0043** (null) — mirroring Shenzhen's −0.0011 ± 0.0032.

The vanilla MLE generator does not transmit the edit at the model level, the model-level companion to
the L2 null. (The disparate-impact metric is N/A for SF — it is a Shenzhen hukou-district ratio and SF
has no administrative-district abstraction.)

---

## 6. Head-to-head — every directional conclusion reproduces

| result | Shenzhen (PRIMARY) | SF (sf12) | agree? |
|---|---|---|---|
| Editor ΔF_causal (causal-emphasis) | +0.0144 (→0.8132) | +0.0139 (→0.8891; +0.0199 full pool) | ✓ on par |
| Pillar 1: edited fairest faithful | edited 0.8132 fairest; Fid-A ≈ raw | edited 0.8891 fairest; Fid-A 0.958 = raw | ✓ |
| L2 vanilla transfer (edited−raw) | −0.0012 (null) | +0.0004 (null) | ✓ null |
| Pillar 2 weighted-BC (w10/20/30) | +0.0205 / +0.0278 / +0.0311 | +0.0296 / +0.0348 / +0.0387 | ✓ (SF sharper) |
| random placebo control | ~null (−0.0009 @ w30) | negative (−0.0071 / −0.0095) | ✓ (SF sharper) |
| most-fair select control | ~null (+0.0004 @ w30) | negative (−0.0117 / −0.0068 / −0.0027) | ✓ (SF sharper) |
| model-level variance | −0.0011 ± 0.0032 (null) | −0.0005 ± 0.0043 (null) | ✓ |
| GAN Fidelity-B (vs raw) | ~0.32 (collapsed → disqualified) | 0.0269 (did not collapse) | ✗ diverges (§7) |
| discriminator val-AUC | 0.982 | 0.998 | — (12 identities easier) |
| Fidelity-A level | ~0.84–0.85 | 0.958 | — (12 identities more separable) |
| edited fraction of corpus | ~2.6% | ~12.6% (1371/10,887) | — |

**Same:** every *directional* conclusion — edited = fairest faithful source; vanilla BC / variance
does not transfer it; weighted BC recovers it edit-specifically; F_fidelity is a profile-dominated
identity-preservation metric. **Differs in magnitude:** SF sits at a higher absolute F_causal baseline
(0.875 vs 0.799) and higher Fidelity-A, and its Pillar-2 recovery + negative controls are *sharper*.
**Differs qualitatively:** the GAN did not collapse (§7). Because the metric is city-specific and
associational, SF is reported as *reproducing / on par with* Shenzhen, not as beating it.

---

## 7. The GAN-did-not-collapse divergence (honest, not load-bearing)

On **Shenzhen**, the WGAN-GP-generated source was disqualified from the faithful-sources comparison
because its Fidelity-B (JS divergence of trajectory-statistic distributions vs raw) collapsed to
~0.32 — the adversarial generator free-runs/degenerates (length and coverage collapse). That collapse
was used in the Shenzhen story as evidence that *generative* data can silently degrade.

On **SF, the GAN did not collapse** — its Fidelity-B is **0.0269**, comparable to BC's 0.0100 and far
below any collapse threshold. So on SF, gan-gen is a faithful source too and is not disqualified.

- **Pillar 1 still holds** — edited (0.8891) is the fairest source regardless; the claim does not
  depend on disqualifying the GAN. Pillar 1 is, if anything, *cleaner* on SF (all three non-raw
  sources faithful, edited still wins).
- **The Shenzhen "GAN collapse" cautionary sub-narrative does NOT transfer to SF** and should not be
  claimed for the second dataset. This is an honest, reportable difference.
- **Likely cause (hypothesis, not verified):** SF's much smaller vocabulary (963 grid-cell tokens vs
  Shenzhen's 4323, ~4.5× smaller) and corpus (~10.9k vs ~95–105k trajectories, ~9× smaller) make the
  WGAN-GP dynamics more stable, so the mode/length collapse that plagued the large-vocab Shenzhen
  setup does not arise.
- **Not load-bearing:** the two-pillar argument rests on edited-vs-raw (data quality) and the
  weighted-BC recovery-vs-controls, neither of which depends on the GAN collapsing. The GAN arm is a
  *supporting* baseline; its different behavior on SF is a dataset-characterization note, not a threat
  to the claims.

---

## Sources / provenance

All SF values are from the merged second-dataset bundle (seed means, or deterministic data-level
rescore where noted):

- Dual claim: `PAPER/second-dataset/tables/dual_claim_sf12.csv`,
  `PAPER/second-dataset/data/sf12_dual_metrics.json`; fidelity-off control:
  `.../data/sf12_fairoff_k2000_metrics.json`; fidelity sensitivity:
  `.../tables/fidelity_sensitivity.csv`; subsample selection: `.../tables/subsample_selection.csv`.
- Discriminator: `.../data/sf12_discriminator_training.json` (best val-AUC 0.998).
- Pillar 1 (L1): `.../tables/eval_l1_data_quality.csv`, `.../data/eval_l1v2_sf12_metrics.json`.
- L2 vanilla: `.../tables/eval_l2_transfer.csv`, `.../data/eval_l2_sf12_paired_stats.json`.
- Weighted-BC + controls: `.../tables/eval_weighted_bc_recovery.csv`,
  `.../data/eval_weighted_bc_sf12_paired_stats.json`.
- Model-level variance: `.../tables/eval_variance_model_level.csv`,
  `.../data/eval_variance_sf12_aggregate.json`.
- Full synthesis (§5 dual claim, §6 two-pillar eval, §7 head-to-head): `PAPER/second-dataset/FINDINGS.md`.
- Figure (referenced, not regenerated): `PAPER/second-dataset/figures/sf_supply_demand.png`.
