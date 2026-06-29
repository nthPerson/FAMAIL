# reviews — adversarial review of the paper bundle

Two independent adversarial reviews of this deliverable, plus the consolidated findings list. Every finding was
verified against the actual source files (numbers recomputed, not just read); **0 of 29 were refuted**.

| file | scope |
|---|---|
| `REVIEW_A_paper_content.md` | paper content: data trustworthiness, statistics, figures, and the demographic-feature / co-linearity vetting |
| `REVIEW_B_dirty_vs_clean.md` | the dirty-vs-clean data-cleanup work (filter correctness, F_spatial decomposition, labeling) |
| `REVIEW_confirmed_findings.md` | all 29 confirmed findings with verdicts, verified numbers, and concrete fixes |

**Disposition.** The data are trustworthy; the cleanup is sound. The findings were framing/labeling overreaches, which
have been applied across this bundle: the figure-honesty fixes (dose-response occlusion, truncated Fidelity-A axis,
"preserves shape" overclaim, the mislabeled/null robustness row) are in `famail_temporal/analysis/paper_figures.py`;
the cleanup-table fixes (data-driven sink caption, removal-rate denominator, unweighted-arm flip) are in
`analysis/{experiment_delta,dataset_summary}.py`; and the statistical / feature-choice caveats (n=6 Wilcoxon floor,
multiple-comparison, n=5 null framing, deterministic L1 gap, edit-dominant-not-edit-specific, density-drives-the-scale,
co-dominance, housing-retaining/FRAGILE scope, 10-district ecological resolution, associational-not-causal) are baked
into the per-set READMEs, `feature_selection/`, and `shared_cleanup/`.

**One item is held as a PI-meeting agenda item, not yet applied:** renaming "F_causal" / "causal fairness" to an
explicitly associational construct — the RA favors **`F_demo`** (demand-adjusted demographic independence). Decision
(2026-06-29): **keep the `F_causal` name + add the associational methods caveat now** (done throughout this bundle);
**raise the `F_demo` rename with Dr. Zhang at the next meeting** before any paper-wide rename. The metric and all
numbers are unaffected either way — it is purely a naming/framing choice.
