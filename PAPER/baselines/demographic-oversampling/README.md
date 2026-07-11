# Demographic Oversampling — naive resampling baseline (4th arm)

A **resampling** baseline (not perturbation): duplicate real seeking trajectories originating in
demographically disadvantaged regions (all three `EQUITY_AXES`, the evaluation's own
`region_extremes(frac=1/3)` convention) under **fresh phantom driver IDs**, rebuild the demand + supply
grids **additively on both channels**, and rescore with the identical harness as every other arm. The
naive cousin of the **supply-lift (trim+lift)** editor and a direct empirical probe of the
demand-endogeneity / leveling-down limitation: a duplicate's pickup is *unobserved* demand, so the arm
quantifies **how much apparent fairness pure fabrication buys, at what corpus-inflation cost**. Scored
alongside a random-oversampling **placebo** (identical machinery, sources drawn uniformly over the whole
corpus) that isolates demographic *targeting* from mere corpus *inflation*.

## Read this first
- **[`FINDINGS.md`](FINDINGS.md)** — headline result (targeted +0.0153 vs placebo −0.0172 vs FAMAIL
  +0.0222 at matched budget), dose-response, the DP-explosion mechanism, disclosures, diagnostics
  (including the migrant/comp identical-pool finding), and reproduce commands.

## Contents
- `tables/dose_response.md` — the 9-arm dose-response table (committed copy of the runner's
  `summary.md` output, with provenance).
- `figures/dose_response.png` — two-panel dose-response figure (ΔF_causal | ΔDP-migrant; targeted vs
  placebo; seed mean ± min–max at repeated doses).

## One-line result
Demographic targeting is necessary AND insufficient: targeted oversampling buys **+0.0153** ΔF_causal at
k=10,000 (dose-monotone) — below FAMAIL's **+0.0222** at the same budget — while fabricating **10.5%**
of the corpus; the untargeted placebo *degrades* F_causal (−0.0172) and explodes the DP gap (+2.8).

## Provenance
Selected from the citation-verified lit-scan (`famail_temporal/baselines/DATA_AUG_BASELINE_CANDIDATES.md`,
Candidate 4; Pastaltzidis et al., FAccT '22). Spec:
`docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md` (11 locked decisions);
plan: `docs/superpowers/plans/2026-07-09-demographic-oversampling-baseline.md`. Code:
`famail_temporal/baselines/{demographic_oversampling,run_demographic_oversampling}.py` (23 tests).
Raw arm dirs (gitignored): `famail_temporal/results/*_baseline_demo_oversample_*_shenzhen/` + run log
`famail_temporal/results/demo_oversample_runs.log`. Runner summary output (committed):
`famail_temporal/baselines/demographic_oversampling_results/`.
