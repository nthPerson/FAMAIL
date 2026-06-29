# FAMAIL — paper results bundle

**Purpose.** A single, self-contained, auditable directory holding the **final** results, tables, and figures for
the FAMAIL paper, so nothing has to be reconstructed from the broader (git-ignored) `famail_temporal/results/`
tree. Every artifact names the exact source file it was derived from (provenance).

**Study in one paragraph.** Raw Shenzhen taxi GPS data contained per-driver "stuck-GPS" pickup-sink artifacts; we
detect and filter them (data cleanup, `shared_cleanup/`), then run the FAMAIL trajectory editor (attribution +
ST-iFGSM) to make the data fairer, and test whether the fairness propagates into trained behavior-cloning policies.
We report results under **three demographic feature sets** for the fairness metric F_causal, organized as standalone
directories so each is self-describing and the argument's robustness to the feature choice is explicit.

## Layout

```
PAPER/
  by_feature_set/
    housing-comp-migrant/                 ★ PRIMARY — equity set {housing, comp, migrant}
    housing-gdp-comp/                       original 3-feature set {housing, GDP, comp}      (sensitivity)
    housing-comp-migrant-logpopdensity/     density-augmented 4-feature set                 (sensitivity)
  shared_cleanup/        demographic-INDEPENDENT data-cleanup / F_spatial artifacts (valid for ALL sets)
  feature_selection/     the demographic-feature vetting + cross-set comparison
  reviews/               two adversarial-review reports + the 29 confirmed findings
```

Each `by_feature_set/<combo>/` and the two cross-cutting dirs carry their own `README.md` describing contents and
data-source provenance.

## The three feature sets — why three, and which is PRIMARY

`F_causal = 1 − R²_demo` is a **feature-set-specific** fairness measure (the partial R² of the demand-adjusted
service residual on a chosen set of district-level demographic axes). Its absolute scale therefore depends on the
chosen axes; the editor's *direction* and *targeting* do not (top-cell Jaccard ≥ 0.92 across all housing-retaining
sets). We report three:

| dir | set | before-edit F_causal | role | one-line rationale |
|---|---|---|---|---|
| `housing-comp-migrant/` | {housing, comp, **migrant**} | **0.799** | **PRIMARY** | three equity-salient axes (wealth / income / migrant population structure); most defensible as *demographic* fairness; **higher** before-edit F_causal than the density set, so it is not the unfairness-maximizing lens |
| `housing-gdp-comp/` | {housing, GDP, comp} | 0.807 | sensitivity | the original SES/income set; shows conclusions predate the migrant/density choices |
| `housing-comp-migrant-logpopdensity/` | {housing, comp, migrant, **logpopdensity**} | 0.725 | sensitivity | adds a population-**density** demand control; shows conclusions survive a richer (but less purely-demographic) lens |

**The headline numbers in the paper come from `housing-comp-migrant/`.** The other two are reported as
robustness/sensitivity: the qualitative two-pillar story reproduces under all three; only the absolute F_causal scale
shifts. See `feature_selection/` for the side-by-side and the selection analysis.

> **Status (2026-06-29):** the PRIMARY `housing-comp-migrant/` experiment re-run is **in progress** (editor
> before-edit F_causal 0.799 confirmed; ~16h pipeline). Its `figures/`, `tables/`, `data/` will be populated when the
> run completes. The two sensitivity sets are complete.

## The two-pillar argument (reproduces under all three sets)

- **Pillar 1 (L1 — data quality):** the **edited** dataset is the *fairest faithful* source (higher F_causal than
  raw/BC-gen; GAN-gen disqualified by distributional collapse), while remaining identity-faithful (Fidelity-A
  unchanged). *Caveat (by construction):* F_causal is also the editor's optimization target, so the edited>raw gap is
  expected; its value is that it is achieved without sacrificing identity faithfulness, and the edited−raw gap is a
  **deterministic** data-level quantity (no sampling CI).
- **L2 (vanilla transfer):** driver-conditioned BC trained on edited data does **not** transfer the fairness
  (edited−raw ΔF_causal within the ±0.003 cross-seed band; n.s.) — vanilla BC averages it away.
- **Pillar 2 (weighted BC):** **upweighting** the edited demonstrations **recovers** it — ΔF_causal ≈ +0.02/+0.03 at
  weights 10/20/30, monotone dose-response, all 6 seeds same sign, t-CIs excluding zero. The random-subset **placebo
  is null on F_causal** → the gain is edit-driven, not generic oversampling.
- **Edit ≫ select > random (PI-requested ablation):** editing the unfair trajectories beats upweighting the
  already-fairest ones, which beats the random placebo. Selection captures only a small, **metric-dependent** fraction
  of the gain (see the per-set READMEs); filtering unfair trajectories does not help (Pareto). So the gain is
  **edit-dominant**, not reproducible by selecting or removing data.

## Statistical reporting conventions (read before citing any p-value)

- **n = 6 Wilcoxon floor.** With 6 paired seeds the two-sided exact Wilcoxon signed-rank p **floors at 2/2⁶ =
  0.03125**, attained exactly when all 6 paired differences share a sign. So "p = 0.031" certifies **sign-unanimity
  only** and carries no effect-magnitude information. Effect size is read from the **mean Δ and its t-CI**, not from p.
- **Multiple comparisons.** Each weighted-BC family is 36 paired tests; at n = 6 no test can clear Bonferroni
  (0.05/36 ≈ 0.0014 < 0.03125). We therefore do **not** lean on per-test corrected significance; the evidence is the
  CI separation + monotone dose-response + 6/6 sign-consistency + null placebo, which are structural checks
  independent of an uncorrected p.
- **n = 5 nulls (L2, variance).** A two-sided n = 5 Wilcoxon cannot reach p < 0.05 (floor 0.0625), so we report these
  nulls as **effect size vs the cross-seed noise band**, not as a powered test of zero.
- **F_causal is associational, not causal.** It is the partial R² of a cross-sectional OLS of the demand-adjusted
  residual on observational **district-level** demographics — no instrument, no counterfactual, no unconfoundedness.
  It measures demographic *predictability* of service, not a causal effect. The demographics have only **10 distinct
  district profiles** (an ecological-resolution limit), so cell-level attribution via district covariates carries a
  standard ecological-fallacy caveat. See `feature_selection/README.md` and each per-set Limitations note. *(The
  paper-facing rename of "F_causal" to an explicitly associational label is a pending PI decision.)*

## Reproducibility

- Figures are regenerated by `famail_temporal/analysis/paper_figures.py` (`--feat {hcm,3feat,4feat}`); cleanup
  tables by `famail_temporal/analysis/{dataset_summary,experiment_delta}.py`; the sink decomposition + heatmap by
  `famail_temporal/analysis/{sink_decomposition,sink_heatmap}.py`. All committed.
- The committed config for each set lives in git history of `famail_temporal/config.py`; the cache filenames are
  feature-suffixed so all sets coexist. See `famail_temporal/results/RESULTS_INDEX.md` for the on-disk result dirs.

_This directory is committed to git as the curated paper deliverable (a `.gitignore` negation force-tracks its
`*.json`/`*.csv` data, which the global ignores would otherwise drop). The underlying `results/` tree it copies from
is git-ignored (on-disk only)._
