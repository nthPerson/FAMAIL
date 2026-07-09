# Limitations & open questions

Stated candidly. None of these overturn the two-pillar argument, but each bounds what may be claimed.

## Metric & statistical limitations

1. **F_causal is associational, not causal.** It is the partial R² of a cross-sectional OLS of the
   demand-adjusted service residual on observational demographics — no identification, no
   counterfactual. On Shenzhen the demographics resolve to **10 district-level profiles**, so the
   regression has few degrees of freedom and an **ecological-fallacy** exposure (a district-level
   association is not an individual-level effect). Interpret magnitudes as a fairness *audit* signal,
   not a causal estimate. **Demand-adjustment further assumes demand is exogenous, but recorded demand
   is itself suppressed by historical under-supply** — latent demand is censored where service was
   historically thin, so conditioning on it can *under-detect* real inequity (the feedback-loop pathology
   of Ensign et al. 2018 and Lum & Isaac 2016). This demand endogeneity is the same phenomenon as the
   editor's leveling-down behavior; see
   [`../objective-motivation/LEVELING_DOWN.md`](../objective-motivation/LEVELING_DOWN.md).

2. **Small-n significance floors, no multiple-comparison survival.** The weighted-BC evidence uses 6
   paired seeds, where the two-sided Wilcoxon floors at **`p = 0.03125`** (sign-unanimity, not effect
   size), and no test survives Bonferroni at this n; the L2 / variance nulls use 5 seeds, where the
   test cannot even reach p < 0.05 (floor 0.0625). The argument therefore rests on **effect
   direction + magnitude + t-CI separation + monotone dose-response + the control arms**, not on any
   uncorrected p-value.

3. **The L1 data-level gap is deterministic.** Raw and edited F_causal are static rescores with
   std = 0, so the edited − raw L1 gap is a point comparison with **no sampling CI**, and it is the
   editor's own optimization target — its value is being achieved at unchanged Fidelity-A, not that it
   is "significant."

4. **Fidelity is profile-dominated.** F_fidelity / Fidelity-A certify **driver-identity preservation**
   under an edit, **not** fine-grained trajectory-**shape** realism: the discriminator can score
   largely from the driver-profile stream, so its gradient w.r.t. the edited pickup cell is ~0 (true
   on **both** cities — Shenzhen 4.7e-6, SF 2.6e-11). A seeking-sensitive discriminator (drop the
   profile stream, or add same-driver-corrupted-seeking hard negatives) is a deferred option that
   would require re-running Shenzhen the same way for parity.

## Dataset & external-validity limitations

5. **The "GAN collapse" sub-claim does not transfer to SF.** On Shenzhen the GAN-generated source was
   disqualified because it collapsed distributionally (Fidelity-B ~0.32); on SF the **GAN did not
   collapse** (Fidelity-B 0.027), so the cautionary "generation silently degrades" sub-narrative is
   Shenzhen-specific and must not be claimed for the second dataset. Pillar 1 does not depend on it
   (edited wins outright either way).

6. **Small samples on SF.** The SF result rests on a 12-driver density-matched subsample and 5–6
   seeds. The subsample was necessary (the full 536-taxi fleet saturates F_causal → ~0.982 with no
   editable gradient), but it is small.

7. **SF demographics are ACS proxies.** SF fills the Shenzhen feature *names* with ACS values
   (`migrant` = foreign-born share, an ACS proxy, **not** hukou; `housing` = median home value;
   `comp` = per-capita income). Because F_causal is city-specific and associational, absolute
   baselines are **not cross-city comparable** (SF 0.875 ≠ Shenzhen 0.799) — SF demonstrates that the
   *conclusions* reproduce, not that the magnitudes match. The disparate-impact metric is also N/A for
   SF (it is a Shenzhen hukou-district ratio; SF has no administrative districts).

## Naming

8. **`F_causal → F_demo` is a pending decision.** The name carries an unwarranted causal connotation;
   a rename to `F_demo` (demand-adjusted demographic independence) is favored but is held as a PI-meeting
   agenda item. The metric and every number are unaffected — this doc keeps `F_causal` + the
   associational caveat until the decision is made.

## Open questions

- **What training procedure best realizes the data-level fairness?** Vanilla BC averages the edit
  away; upweighting recovers it. Whether other objectives (fairness-aware losses, other imitation
  algorithms) realize it more efficiently is open.
- **Other downstream model classes on edited data** — e.g. whether a GAN/WGAN trained on the edited
  corpus inherits the fairness, and how weight selection trades off against fidelity at higher weights.

## Credibility — adversarial review

The deliverable underwent **three adversarial-review rounds**, with findings **verified against the
source files (numbers recomputed, not just read)**:

- **REVIEW_A / REVIEW_B** (paper content + dirty-vs-clean cleanup): **29 confirmed findings, 0 of 29
  refuted.** All were framing/labeling overreaches (figure-honesty fixes, cleanup-table captions,
  statistical caveats) and have been applied across the bundle.
- **REVIEW_C** (on the PRIMARY set + the branch code, 2026-06-30): **0 critical, 1 substantive, ~8
  minor.** The one substantive finding — a Pareto bullet that read as "edit improves both metrics"
  when the causal-emphasis editor actually *lowers* F_spatial slightly — was **fixed** (edit-dominance
  is on the F_causal objective only). The branch code verified sound (full suite 700-pass) and all
  PRIMARY headline numbers recomputed as correct seed means (not seed-0).

## Sources / provenance

- Review disposition: `PAPER/reviews/README.md`, `PAPER/reviews/REVIEW_C_primary.md`,
  `PAPER/reviews/REVIEW_confirmed_findings.md`.
- SF caveats: `PAPER/second-dataset/FINDINGS.md` §9; profile-dominance:
  `PAPER/second-dataset/tables/fidelity_sensitivity.csv`. Statistical conventions:
  [`04_evaluation.md`](04_evaluation.md). Per-city numbers:
  [`05_results_shenzhen.md`](05_results_shenzhen.md), [`06_results_sf.md`](06_results_sf.md).
