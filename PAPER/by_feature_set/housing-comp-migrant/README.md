# housing-comp-migrant — PRIMARY equity F_causal  ⏳ RESULTS PENDING

**Role: ★ PRIMARY (the paper's headline metric).** F_causal uses three equity-salient axes
{AvgHousingPricePerSqM, CompPerCapita, **MigrantRatio**} — neighborhood wealth, income, and migrant/hukou population
structure (a real underserved-group axis in Shenzhen). Chosen for **construct validity as a demographic fairness lens**
and because its before-edit F_causal is **higher** than the density-augmented set (so the choice is not the
unfairness-maximizing lens). See `../../feature_selection/README.md` for the full selection rationale.

> **Status (2026-06-29): the re-run is IN PROGRESS.** The editor has confirmed the before-edit metrics
> (**F_causal 0.7988, F_spatial 0.1034**, n_trajectories 95,297, n_active 34,524 — identical corpus to the other
> sets), then runs L1-v2 → weighted-BC (edit / most-fair / placebo) → L2 → variance (~16h total). This directory's
> `figures/`, `tables/`, `data/` and the verified headline numbers will be filled in when the run completes and the
> story is checked. Until then, cite the sensitivity sets in `../housing-gdp-comp/` and
> `../housing-comp-migrant-logpopdensity/`, which bracket this set's expected scale.

## Why this is the most defensible headline set

- **Purely demographic.** Unlike the density-augmented set, it contains **no demand-geography covariate** — every axis
  is an equity/SES/population-structure variable, so "demographic fairness" is not carried by a density control.
- **Not unfairness-maximizing.** Before-edit F_causal 0.799 > the density set's 0.725, so we are not selecting the
  lens that inflates apparent baseline unfairness — this pre-empts the circularity objection.
- **Well-conditioned & targeting-stable.** Max VIF 4.45 (< 10); top-cell Jaccard 0.96 vs the original 3-feature set,
  so the editor flags essentially the same trajectories.

## Expected story (to be confirmed against the run, not asserted)

The two-pillar argument is expected to reproduce at this set's scale (L1 edited fairest faithful; L2 vanilla null;
weighted-BC dose-responsive recovery with the random placebo null on F_causal; model-level null). All statistical
conventions from the top-level README apply (n = 6 Wilcoxon floor = sign-unanimity; lead with t-CIs + dose-response;
n = 5 nulls reported vs the cross-seed noise band; F_causal is associational, 10-district ecological resolution).

**Config:** git commit `16ad5f8` (`famail_temporal/config.py`). **Cache:**
`cache/hat_matrices_T24_thr0.5_feat-housing-comp-migrantratio.pkl`. **Re-run driver:** `results/_rerun_hcm.sh`.
