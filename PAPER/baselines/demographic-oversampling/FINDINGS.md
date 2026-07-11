# Demographic Oversampling baseline — findings (Shenzhen, 2026-07-10)

Curated record of the 4th Mission-3 baseline arm. All numbers below are transcribed from committed
artifacts (`famail_temporal/baselines/demographic_oversampling_results/summary.md` + the per-arm
`metrics.json` / `external_fairness/external_fairness.json` under
`famail_temporal/results/*_baseline_demo_oversample_*_shenzhen/`) and were independently re-verified by
the branch's final review — nothing here is recomputed.

## 1. Method in one paragraph

Duplicate real seeking trajectories whose **origin cell** lies in a demographically disadvantaged region
(per axis: `region_extremes(frac=1/3)` over the cell demographic values with the harness's
`DISADVANTAGED_HIGH` poles — the *same* definition the external-metrics reporting uses), allocated by
per-axis quotas (dose/3 per equity axis, drawn without replacement with cross-stratum dedupe;
with-replacement fallback flagged). Each duplicate is a **phantom driver**: a fresh namespaced plate ID
(required by the tier-2 distinct-count supply convention — a duplicate under its real plate adds zero
supply where the source driver was already present) displaced by a rigid ±1-cell whole-trajectory shift
(preserves internal adjacency exactly). Grids are rebuilt **additively on both channels**: demand adds one
pickup-event mass at each phantom's pickup (`pickup_mass` convention); supply adds each phantom's tier-2
presence (5×5 neighborhood, distinct per (driver, cell, hour, day), terminal pickup state excluded,
mean-hourly normalization identical to the production `active_taxis` counter — pinned by a fixture test
against the production code path). Scoring is the standard harness: F_causal/F_spatial via the
substituted-grid seam `supply_recount` validated, external metrics (DP/DI/SDR/Theil + paired-bootstrap
CIs) via `run_external_fairness.assemble_results`. The **placebo** uses identical machinery with sources
drawn uniformly over the whole corpus.

**Why additive demand+supply (the load-bearing design decision):** demand-only oversampling is perverse —
it adds demand to already under-served cells and thereby *lowers* their service ratio. Both channels move.

## 2. Headline result

| Arm (d = 10,000, matched to FAMAIL's k = 10,000) | mean ΔF_causal | corpus inflation |
|---|---:|---:|
| **Targeted** demographic oversampling (3 seeds) | **+0.0153** (+0.0175 / +0.0141 / +0.0144) | 10.5% |
| **Placebo** random oversampling (3 seeds) | **−0.0172** (−0.0179 / −0.0168 / −0.0169) | 10.5% |
| **FAMAIL** trim+lift headline (comparator, not recomputed) | **+0.0222** | **0%** |

Three pre-registered readings (spec §1), all realized:

1. **Targeting is necessary:** the placebo — same fabrication, no demographic targeting — *degrades*
   F_causal. The targeted gain is targeting-specific, not a corpus-inflation artifact (the additive
   mirror of the weighted-BC random placebo).
2. **Targeting is insufficient:** dose-monotone (+0.0059 @2,500 → +0.0097 @5,000 → +0.0153 @10,000) but
   still below FAMAIL at the same budget — and it gets there only by fabricating 10.5% of the corpus
   (phantom drivers with unobserved pickups), where FAMAIL redistributes real observed behavior at zero
   inflation.
3. **Ratio metrics are fragile under fabrication:** the placebo's ΔDP explodes (+1.49 @5,000, +2.77 to
   +2.81 @10,000) — see §3.

Full 9-arm table: [`tables/dose_response.md`](tables/dose_response.md). Figure:
[`figures/dose_response.png`](figures/dose_response.png).

## 3. The DP-explosion mechanism (measured, not hypothesized)

From the d10,000 s0 arms' `external_fairness.json`
(`metrics.MigrantRatio.district_extremes.supply_demand_ratio`): uniform fabricated supply raises
`mean_advantaged` by **+3.22** (21.27 → 24.49) while `mean_disadvantaged` rises only **+0.45**
(7.07 → 7.52) — most of the placebo's fabricated supply lands in already-advantaged cells (consistent
with, but not solely explained by, `service_ratio_Y` dividing by `max(demand, DEMAND_FLOOR)`:
floored-demand cells amplify any added supply). Even **targeted** oversampling raises `mean_advantaged`
by **+3.15** alongside the disadvantaged group's own +3.09 lift — targeting concentrates the
*demand-side* draw, but the additive *supply trails* leak into advantaged cells regardless of variant.
Hence targeted ΔDI improves monotonically with dose (+0.035 → +0.083) while targeted ΔDP is mixed
(−0.18 at low dose, +0.06..+0.27 at d10,000): DP is the scale-sensitive gap metric — the same DP≡gap
caveat documented in `PAPER/external-metrics/FINDINGS.md`. ΔTheil is small and positive in every arm.

This is the demand-endogeneity probe working as intended: a baseline that "improves" service ratios by
inventing supply and demand shows exactly where ratio metrics can be gamed, sharpening the case for
FAMAIL's redistribution-of-observed-behavior framing.

## 4. Key diagnostic finding — the corpus cannot supply budget parity

Measured from `duplicates.pkl` (targeted d10,000 s0) and re-derived independently in review:

- **MigrantRatio's and CompPerCapita's disadvantaged-origin pools are the SAME 4,907 trajectories**
  (`pools["MigrantRatio"] == pools["CompPerCapita"]`, exactly; disjoint from Housing's 41,964-trajectory
  pool) — the two axes' bottom-third disadvantaged regions select the same origin trajectories.
- Consequently the with-replacement fallback engages deterministically at the headline dose:
  CompPerCapita's 3,333-quota drains the shared pool first (EQUITY_AXES order), leaving 1,574 for
  MigrantRatio → **1,759 of its draws (~17.6% of the arm) are with-replacement re-duplications**
  (seed-invariant; flagged per the spec's error handling, never silent). Only **8,241 distinct source
  trajectories** appear across the 10,000 duplicates.
- Plainly: the corpus cannot supply the budget-parity dose for the migrant axis without re-duplicating
  already-duplicated trajectories — a limitation of naive oversampling in itself. FAMAIL needs no
  re-duplication at the same budget.

Other diagnostics (all 9 arms): `origin_escape_frac` 0.177–0.189 (rigid ±1 shifts near region borders;
consistent across doses/seeds, a boundary-geometry property, not a bug); `adjacency_violation_rate` 0.0
everywhere (rigid shift preserves adjacency by construction); `n_corpus` = 95,297 seeking trajectories.

## 5. Disclosures (carried from the spec, verbatim in substance)

- **Phantom drivers and their pickups are fabricated, unobserved supply and demand.** Nothing about a
  duplicate was actually observed.
- **Fidelity is NOT scored for this arm** — duplicates are (near-)copies of real trajectories, so
  Fidelity-A/B are trivially non-discriminative by construction; the axis of interest is fairness lift
  vs. corpus inflation.
- **Corpus inflation equals the dose** (`n_edited / n_corpus`), reported per arm, never hidden.
- **SUPPLY_FLOOR asymmetry:** phantom supply is added on top of the already floor-clamped production
  grid (`SUPPLY_FLOOR = 0.1`), so in floor-clamped cells the additive S′ can slightly exceed a true
  recount-plus-phantoms — a conservative convention for the FAMAIL contrast (it can only flatter the
  naive baseline), applied identically before/after so it cannot bias the reported delta.

## 6. Status & what's still pending

Run 2026-07-10, Shenzhen PRIMARY only (SF deferred, same scoping as the tier-2 supply tooling). The
cross-arm 6-row comparison table (raw / FAMAIL / ifgsm / fgsm / random / oversampling) is **deferred until
the three perturbation arms' GPU runs complete** (this arm's `metrics.json` ingestion into
`assemble_baseline_table` is already tested); it will land in `PAPER/baselines/comparison/`.

## 7. Reproduce

Run-book (executed commands + environment notes): `famail_temporal/baselines/STATUS.md`, "4th arm"
section. In short: 9 sequential CPU invocations of
`python -m famail_temporal.baselines.run_demographic_oversampling --variant {targeted|placebo} --dose D
--seed S`, then `--summarize <arm dirs> --out famail_temporal/baselines/demographic_oversampling_results`.
Design spec: `docs/superpowers/specs/2026-07-09-demographic-oversampling-baseline-design.md`; plan:
`docs/superpowers/plans/2026-07-09-demographic-oversampling-baseline.md`.
