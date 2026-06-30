# Adversarial review C — PRIMARY {housing,comp,migrant} deliverable + branch code

Run 2026-06-30 (workflow `wf_0079dbf8-204`, 19 agents, 5 review dimensions, adversarial verify). The structured-output
emission hit a harness validation glitch on the `fix` field, so findings were salvaged from the agent transcripts.
**Overall: 0 critical, 1 substantive, ~8 minor.** Branch code verified sound ("correct with only minor comment nits");
PRIMARY numbers verified correct seed MEANS (not seed-0); the two-pillar story holds and is cleanest under PRIMARY.

## Confirmed findings + disposition

| # | sev | finding | disposition |
|---|---|---|---|
| 1 | **substantive** | PRIMARY README Pareto bullet: F_spatial direction inverted — claims "edit improves both" but edit *lowers* F_spatial (0.1034→0.1025) and filter *raises* it (→0.1046). Edit-dominance holds on **F_causal** only (causal-emphasis editor, α_spatial=0.2). | **FIXED** — bullet rewritten: edit dominates on the F_causal objective; F_spatial movements are small and not this run's target. |
| 2 | minor | `comparison_3v4.md` scope note still says the 3-way "will be added when the re-run completes" — but `comparison_across_sets.md` now exists. | **FIXED** — points to the existing 3-way. |
| 3 | minor | PRIMARY README GAN disqualification leans on the L1v2 Fid-B 0.173 (vs edited 0.1485, a modest gap) rather than the stronger L2 distributional-collapse evidence. | **FIXED** — wording cites the collapse, not only 0.173. |
| 4 | minor | density (4feat) README lacks the associational/10-district caveat the PRIMARY + 3feat READMEs carry (top-level has it; 4feat only defers). | **FIXED** — caveat line added. |
| 5 | minor | PRIMARY README Fid-A "raw 0.849" rounds the 0.8485 mean up; reads like a seed-0 value. | **FIXED** — 0.848. |
| 6 | minor | 3-way robustness dumbbell: markers overlap on the two null rows; top-row `above2` label collides with the 2-line title. | **FIXED** — per-set y-jitter + top headroom. |
| 7 | minor | feature_selection "per-cell Spearman ≥ 0.84 (housing-retaining family)" is over-broad — true for the 3 reported sets, but some housing-retaining sets fall below. | **FIXED** — scoped to "the three reported sets". |
| 8 | minor | `shared_cleanup/tables/cleanup_delta_editor.csv` still lacks the 3-feature provenance label its prior fix mandated (covered by the README + .md tables; CSV not cited standalone). | **FIXED** — provenance comment row added. |
| 9 | minor | sink decomposition "share_of_global_shift" labeled "share" but is a multiple (416%), not a ≤100% share (already disclosed as the "no 416% paradox" in prose). | noted; the prose already disambiguates — left as-is with the existing disclosure. |

## Verified clean (no action)
- Branch code (`paper_figures.py` 3-way generalization, `_ci_half_from_std`, `_robustness_numbers` null/deterministic
  flags, `--feat`/`--compare-feat`; `experiment_delta.py` caption/verdict; `dataset_summary.py` denominator; GAN test
  stub; `.gitignore` negation) — no bugs; full suite 700-pass.
- All PRIMARY headline numbers (editor, L1 means, L2, weighted-BC means + CIs, most_fair/random, variance) recomputed
  from source and match; MEAN-not-seed-0 discipline holds; the prior 3feat-bc-0.8045 seed-0 bug is fixed to 0.8064.
- 29 prior findings' fixes confirmed carried into the restructured tree (edit-dominant, most-fair metric-dependent,
  density-drives-scale, co-dominance, housing-retaining/FRAGILE, n=6 floor, deterministic gap, associational caveat,
  removal-rate denominator, redistribution residual, data-driven sink caption).
