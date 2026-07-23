# Reproducibility record: FATE (KDD 2027 submission)

**Paper:** "Mitigating Demonstration Bias via Fairness-Aware Trajectory Editing" (method: **FATE**).
**This document date:** 2026-07-21. **Input chain certified:** 2026-07-15 (see
`paper/reviews/2026-07-15-repro-inputs-audit.md`).

## 1. Purpose and status

This is the map from every headline paper claim to (a) its curated, git-tracked artifact under
`PAPER/`, (b) the raw results directory that produced it, (c) the ledger row that launched it, and
(d) the exact command. It is the record-of-mapping the manuscript's writing conventions point to.

A read-only audit on 2026-07-15 certified the current-era (α\*) input set feeding
`paper/sections/04_experiments.tex`: every checked number traces to a correctly-fingerprinted α\*
artifact and matches its JSON to full precision; all 38 curated `a10` twins are byte-identical
(`cmp`) to their `results/` sources; the ledger carried exactly one outstanding `LAUNCHED` row at
audit time (`S10-REPLICATION`, since `DONE`). The audit verdict: zero correctness-critical
discrepancies. This document is careful assembly against that certified chain, not re-derivation.

Two audit housekeeping items have since been closed on disk (documented in §9, and both are
reflected here): the three previously untracked α\* raw dirs now carry force-tracked provenance
files, and the `tab:featsets` channel-decomposition gap has been run.

## 2. Name and symbol mappings

| In repo / code / artifacts | In the paper | Note |
|---|---|---|
| package `famail` / `famail_temporal` | **FATE** | same method; repo name predates the paper name |
| artifact/JSON key `f_causal` | symbol **F_demo** | renamed 2026-07-20 (author-approved); **code and artifact keys are unchanged** (`f_causal` stays in every JSON). This line plus `paper/README.md` item 2 are the mapping of record. F_demo is **associational**, not a causal-effect claim. |
| `F_spatial`, `F_demo` | fairness scores | **1 = fairest.** Sign-convention erratum dated 2026-05-14: artifacts dated **before** 2026-05-14 carry the inverted sign. All α\* artifacts postdate the erratum and use the current (1 = fairest) convention. |
| `3feat` = {housing, GDP, comp} | **HGC** column | one feature set, two names (audit L2). Before-edit F_demo 0.8069. |
| `4feat` | **PRIMARY + logpopdensity** | i.e. {housing, comp, migrant, **logpopdensity**}. **NOT** 3feat + logpop (audit L1). Before-edit F_demo 0.7253 disambiguates. |
| `trim` | the demand-only editor / editing phase | trim selects via **"demand deficit attribution"** |
| `lift` | the supply-adding editing phase | lift selects via **"supply-gradient attribution"** |

The paper says **"editing phases"** (trim phase, lift phase), never "modes". The two attribution
mechanisms are **"demand deficit attribution"** (trim) and **"supply-gradient attribution"** (lift).

## 3. Environment and determinism

Every α\* artifact directory (raw and curated) carries two provenance files, written by
`famail_temporal.analysis.run_ledger`:

- `environment.json`: `python`, `torch`, `cuda`, `gpu_name`, the full `pip_freeze`, and its
  `pip_freeze_sha256`.
- `PROVENANCE.md`: SHA-256 checksums of every `*.json` / `*.npz` in the directory (timestamped).

Concrete example (git-tracked):
`famail_temporal/results/weighted_bc_sweep/supply_lift_v1_shz_hgc_filtered_6seed/environment.json`
records `python 3.12.3`, `torch 2.11.0+cu130`, `cuda 13.0`, GPU `NVIDIA GeForce RTX 3070`,
`pip_freeze_sha256 b8bca0e41c5d4a9716835a5acc2fab4752dc6b20e1906b9a15332ddf85b294ce`; its sibling
`PROVENANCE.md` carries the checksum block (e.g. `environment.json`
`6927a7d042c129e95469ee212a7b235a21ab06fa3e984efc3534bd8a9a20d68e`).

**Seeds.** Paired-seed design throughout. Weighted-BC and fairness-baseline suites use seeds
`0..5` (n=6); the flagship recovery pools `0..5` with `6..11` (n=12). Variance suites use seeds
`0..4` (n=5), extended to `0..9` (n=10). Bootstrap CIs use `--seed 0`. `config.DEFAULT_SEED = 42`.

**FTZ gotcha (process-global).** Taper-mode runs (`TAIL_LEN > 0`) call
`torch.set_flush_denormal(True)` to avoid denormal-float stalls
(`LIFT_ALGORITHM_REFERENCE.md` §9, `runner.py:387-407`). This is a **process-global** side effect:
a taper-mode `run_experiment` leaves flush-to-zero **on** for any later call in the same process.
The runner CLI is one-shot per process, so this does not affect the committed results; library
callers that reuse a process must be aware of it. Legacy (`TAIL_LEN = 0`) runs keep the historical
FP environment for bit-reproduction. Downstream consumers of the mutated grids treat `|v| < 1e-6`
as zero (documented invariant, predates the supply-lift branch).

## 4. Data

- **Shenzhen (SZ) corpus.** 95,297 trajectories total; **80,427** with length ≥ 3 (the tail-editable
  set), per `PAPER/supply-lift/data/oracle.json`. Migrant-axis, district-extremes grouping defines
  the disadvantaged (D, high-migrant) vs advantaged (A, low-migrant) split.
- **Stuck-GPS cleanup (era-neutral, upstream of the editor).** Per `DATA_INVENTORY.md` §6 and
  `PAPER/shared_cleanup/`: **10 sink cells / 9 plates / 106,677 phantom pickups** removed. Being
  upstream of the editor, this is never superseded by an era change.
- **San Francisco (SF) second dataset.** Cabspotting GPS + ACS demographics (`FAMAIL_CITY=sf12`),
  10,887 trajectories. ~15% of raw SF trajectories carry pre-existing king-move violations (a
  source-data property, not an editor artifact). ACS fills the same PRIMARY feature **names**
  (housing = median home value, comp = per-capita income, migrant = foreign-born share).
- **Demographics.** PRIMARY equity set (`config.DEMOGRAPHIC_FEATURES`):
  `AvgHousingPricePerSqM`, `CompPerCapita`, `MigrantRatio`, assigned at district granularity and
  grouped as district-extremes on the migrant axis.

Data-access terms are not asserted here: the inventory states none, and no licensing language is
invented for this document. Datasets are referenced by name only.

## 5. Era discipline and artifact fingerprinting

Numbers are **not comparable across eras**; several era pairs differ by as little as 0.0004
(`DATA_INVENTORY.md` §1). Never trust a directory name or prose label. Verify an artifact is α\*-era
from the artifact itself:

1. **`config_snapshot`** in its `metrics.json`: `ALPHA_SPATIAL = 0.1`, `ALPHA_CAUSAL = 0.8`,
   `ALPHA_FIDELITY = 0.1`, `TAIL_LEN = 4` (`0` = trim-only, `4` = trim+lift).
2. **The edit-count fingerprint** (`n_trim + n_lift`), which identifies the corpus exactly:
   - **SZ headline:** `2,337 + 7,545 = 9,882` (i.e. 2,455 selected trims minus **118**
     infeasible-reverted). `9,885` = old-α (superseded); `2,455` trim-only = pre-supply-lift.
   - **SF headline:** `1,330 + 629`.

`config.py` spot-check (2026-07-21): `TAIL_LEN = 4`, `EPSILON_BALL = 2.0`, `DEMAND_FLOOR = 0.5`,
`SUPPLY_FLOOR = 0.1`, `LIFT_BUDGET = None` (lift fills `k_total − n_trim`). **The committed
`config.py` ALPHA defaults are `(0.33, 0.33, 0.34)`, the framework calibration defaults, not
α\*.** The α\* weights are applied **per run** via `--override ALPHA_SPATIAL=0.1 --override
ALPHA_CAUSAL=0.8 --override ALPHA_FIDELITY=0.1` (present in every α\* ledger command) and captured
in each artifact's `config_snapshot`. Any re-run must pass these overrides.

**Manuscript-side enforcement.** `paper/lint.sh` carries era guards that fail the build on stale
strings: the old-α set `0.0222|0.0328|0.0310|0.8132|0.0205|0.0278|0.0311|87.4|84.9` and the
trim-only set `0.0144|0.0128|0.0139` outside a labeled ablation/prior-era context. The lint exits 0
on the current manuscript.

## 6. Claim → artifact map (the core)

Every value below is copied verbatim from the 2026-07-15 audit §2c and/or `DATA_INVENTORY.md` §2.
The **command** for any row is the ledger row's `command` cell (verbatim in
`famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md`, keyed by the queue id shown); class templates
are reproduced in §7. All curated artifacts live under `PAPER/supply-lift/data/a10/` unless a full
path is given. Raw dirs are under `famail_temporal/results/` (git-ignored; provenance in `PAPER/`
twins).

### 6.1 Shenzhen editor (§4.2–§4.3)

| Claim (§) | Value | Curated artifact | Raw results dir | Ledger row |
|---|---|---|---|---|
| SZ ΔF_demo (§4.2) | +0.0226 (0.022561); F_demo 0.7988→0.8214 | `shz_a10_metrics.json` | `2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` | B-ALPHA-S10C80 |
| SZ ΔF_spatial (§4.2) | +0.0061 (0.006112) | `shz_a10_metrics.json` | s10_filtered | B-ALPHA-S10C80 |
| SZ trim-only ablation (§4.2) | +0.0146 (mean(Y\|D) flat) | `shz_trimonly_a10_metrics.json` | `2026-07-11T12-11-31_trimonly_a10_shz` | R2a-trimonly-shz |
| SZ mean(Y\|disadv) (§4.3) | 7.073448→7.126312, +0.0529 [+0.0086,+0.0989] | `shz_a10_channel_decomposition.json` | s10_filtered | R0X-s10 |
| SZ channels (§4.2) | supply t1 +0.0176; demand +0.0352 n.s. | `shz_a10_channel_decomposition.json` | s10_filtered | R0X-s10 |
| SZ supply tier-2 recount (§4.2) | +0.0411; MAE 0.0 | `shz_a10_supply_recount.json` | s10_filtered | R0X-s10 |
| SZ external (§4.3) | DI +0.0162; DP −0.890; Theil −0.0087 | `shz_a10_external_fairness.json` | `external_fairness/results/shenzhen-primary-supplylift-s10` | R0X-s10 |

### 6.2 Shenzhen downstream (§4.4)

| Claim (§) | Value | Curated artifact | Raw results dir | Ledger row |
|---|---|---|---|---|
| L1 four-source (§4.4) | raw/edited/bc/gan F_demo 0.7988/0.8214/0.7980/0.8089; GAN Fid-B bimodal (3 hi / 2 lo, 0.171 ±0.129) | `shz_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_shz_5seed` | Q3 |
| WBC dose (§4.4) | w10/w20/w30 +0.0217/+0.0267/+0.0302 (6/6) | `shz_a10_weighted_bc_paired_stats.json` + `_dose_response` | `weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed` | R4-wbc-shz |
| WBC controls (§4.4) | vanilla +0.0022 (p.16); placebo w30 −0.0023; most-fair w30 +0.0009 | `shz_a10_weighted_bc_paired_stats.json` | same as above | R4-wbc-shz |
| WBC dose saturation (§4.4) | w40 +0.0323 / w50 +0.0339 (6/6); w30 = knee | `shz_a10_wbc_dose_ext_paired_stats.json` | `weighted_bc_sweep/alpha_sweep_s10_dose_ext_6seed` | C1-WBC-DOSEEXT |
| n=12 flagship (§4.1/§4.4) | edited_w30 +0.0297±0.0029, 12/12, exact Wilcoxon p=.00049, CI [+0.0278,+0.0315] | `shz_a10_wbc_n12_pooled_stats.json` | pooled `..._6seed` + `alpha_sweep_s10_seeds6to11` | R4-wbc-shz + WBC-N12 |
| variance (§4.4) | +0.0030±0.0022, n=10, 9/10 positive, Wilcoxon p=.0039 | `shz_a10_variance10_aggregate.json` | `variance_suite/supply_lift_shz_10seed` | B2-VAR-SZ |
| rollout attenuation (§4.4) | trim+lift −0.0033 vs trim-only −0.0049 (both 0/6, p=.031) → ~33% | `shz_a10_rollout_summary.json` + `shz_trimonly_a10_rollout_summary.json` | `option_a_rollout_a10` + `option_a_rollout_trimonly_a10` | R5-rollout-a10 + R5b-trimonly-rollout |
| pareto (§4.6) | filter@K=2455 0.7935 vs edit 0.8214 (filtering inverts the gain) | `shz_a10_pareto_points.csv` | `analysis/pareto_supplylift` | Q7-pareto |
| headroom ceiling, pre-build gate (paper: "unconstrained greedy search"; App. C) | +0.786 supply-only; +0.882 full (0.882001) | `oracle.json` | `analysis/supply_lift_oracle_out` | (Stage-0, pre-campaign) |

### 6.3 Baselines and fairness-intervention controls (§4.5)

| Claim (§) | Value | Curated artifact | Raw results dir | Ledger row |
|---|---|---|---|---|
| Perturbation baselines (§4.5) | iFGSM/FGSM/random ΔF_demo −0.0057/+0.0017/+0.0135; adjacency-viol 54.4/91.4/98.8% | `PAPER/baselines/comparison/baseline_table.json` | `2026-07-13T05-42-07/…05-43-40/…05-44-21_baseline_{ifgsm,fgsm,random}_shenzhen` | Q1-arm-ifgsm / -fgsm / -random |
| Demographic oversampling (§4.5) | targeted d10k +0.0153 vs placebo −0.0172; distinct pool 8241 / redup 1759 / inflation 10.5% | `PAPER/baselines/demographic-oversampling/` | `2026-07-10T00-47-33…d10000_s0…` + placebo dirs | B-OS-T10000-S0..2 + B-OS-P10000-S0..2 |
| Fairness reweigh (§4.5) | −0.0227 F_demo, 6/6 neg, p=.031 | `PAPER/baselines/fairness-intervention/fb_reweigh_paired_stats.json` | `weighted_bc_sweep/fairness_baseline_6seed` | FB-REWEIGH |
| Fairness penalty (§4.5) | inert at λ∈{1,3.16,10}; catastrophic at λ=1000 (−0.2053) | `PAPER/baselines/fairness-intervention/fb_penalty_paired_stats.json` + `_probe_sweep` | `weighted_bc_sweep/fairness_penalty_{6seed,probe}` | FB-PENALTY + FB-PENALTY-PROBE |
| Penalty (absolute form) (§4.5) | l10 +0.0008 n.s.; l1000 −0.1293 6/6 | `PAPER/baselines/fairness-intervention/fb_penalty_abs_paired_stats.json` + `_pilot_sweep` | `weighted_bc_sweep/fairness_penalty_abs_{6seed,pilot}` | FB-PENALTY-ABS + FB-PENALTY-ABS-PILOT |
| FB rollout gate (§4.5) | fair_reweigh drain −0.0010 (0/6, p=.031); R5 arm-prefix gate PASSED | `PAPER/baselines/fairness-intervention/fb_rollout_summary.json` | `option_a_rollout_fb` | FB-ROLLOUT |

### 6.4 Feature-set robustness (§4.6, `tab:featsets`)

| Claim (§) | Value | Curated artifact | Raw results dir | Ledger row |
|---|---|---|---|---|
| HGC column (§4.6) | before 0.8069; ΔF_demo +0.0206; DI +0.0147; Theil −0.0080; DP −0.787; Δmean(Y\|disadv) +0.0594 | `shz_hgc_a10_metrics.json` + `_external_fairness.json` | `2026-07-13T04-41-12_supply_lift_v1_shz_hgc_filtered` + `shenzhen-hgc-supplylift` | Q6a-edit/filter + Q7-ext-hgc |
| HGC channel CI (§4.6) | total +0.0594 [+0.0181,+0.1013]; supply t1 +0.0054 [+0.0017,+0.0094]; tier-2 +0.0211 [+0.0181,+0.0247]; demand +0.0539 | `shz_hgc_a10_channel_decomposition.json` + `_supply_recount.json` | `…_hgc_filtered` | A1-HGC + A1P-HGC-RECOUNT/CHAN |
| HGC downstream (§4.6) | WBC w30 +0.0248 (6/6); L1 edited fairest 0.8275; variance +0.0029±0.0038 | `shz_hgc_a10_weighted_bc_*`, `_l1v2_multiseed`, `_variance_aggregate` | `weighted_bc_sweep/…hgc…`, `level1_table_v2/…hgc…`, `variance_suite/…hgc…` | Q8a-wbc/l1v2/var |
| 4FEAT column (§4.6) | before 0.7253; ΔF_demo +0.0220; DI +0.0191; Theil −0.0085; DP −0.886; Δmean(Y\|disadv) +0.1461 | `shz_4feat_a10_metrics.json` + `_external_fairness.json` | `2026-07-13T17-04-22_supply_lift_v1_shz_4feat_filtered` + `shenzhen-4feat-supplylift` | Q6b-edit/filter + Q7-ext-4feat |
| 4FEAT channel CI (§4.6) | total +0.1461 [+0.0900,+0.2039]; supply t1 +0.0608; tier-2 +0.0771 [+0.0706,+0.0836]; demand +0.0853 | `shz_4feat_a10_channel_decomposition.json` + `_supply_recount.json` | `…_4feat_filtered` | A1-4FEAT + A1P-4FEAT-RECOUNT/CHAN |
| 4FEAT downstream (§4.6) | WBC w30 +0.0256 (6/6); L1 edited fairest 0.7473; variance +0.0003±0.0028 | `shz_4feat_a10_weighted_bc_*`, `_l1v2_multiseed`, `_variance_aggregate` | `weighted_bc_sweep/…4feat…`, `level1_table_v2/…4feat…`, `variance_suite/…4feat…` | Q8b-wbc/l1v2/var |

### 6.5 San Francisco replication (§4.7)

| Claim (§) | Value | Curated artifact | Raw results dir | Ledger row |
|---|---|---|---|---|
| SF headline (§4.7) | ΔF_demo +0.0316 (0.031559); ΔF_spatial +0.0139 (0.013872); compliance 87.6 / 85.2 | `sf12_a10_metrics.json` | `2026-07-11T11-31-55_supply_lift_a10_sf12_filtered` | R1-sf-a10 |
| SF trim-only ablation (§4.2) | +0.0144 (coincides with old SZ headline; not it) | `sf12_trimonly_a10_metrics.json` | `2026-07-11T13-43-37_trimonly_a10_sf12` | R2b-trimonly-sf12 |
| SF channels (§4.7) | supply t1 +0.0209; demand −0.0533; total t1 −0.0324; **tier-2 supply +0.1027 [+0.0872,+0.1203] sig; total_tier2 +0.0493 [+0.0185,+0.0790] sig-positive** | `sf12_a10_channel_decomposition.json` + `sf12_a10_supply_recount.json` | `…sf12_filtered` | R1-sf-a10 + D1-RECOUNT + D1-CHAN |
| SF external (§4.7) | Theil −0.0079; DI +0.0058 | `sf12_a10_external_fairness.json` | `external_fairness/results/sf12-supplylift-a10` | R1-sf-a10 |
| SF L1 (§4.7) | edited fairest 0.9067; GAN healthy 0.039 | `sf12_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_sf12_5seed` | Q4 |
| SF WBC dose (§4.7) | +0.0242/+0.0309/+0.0332 | `sf12_a10_weighted_bc_*` | `weighted_bc_sweep/supply_lift_a10_sf12_filtered_6seed` | Q2 |
| SF n=12 flagship (§4.7) | edited_w30 +0.0333±0.0050, 12/12, exact Wilcoxon p=.00049 | `sf12_a10_wbc_n12_pooled_stats.json` | pooled `…sf12_filtered_6seed` + `…sf12_seeds6to11` | Q2 + SF-WBC-N12 |
| SF dose saturation (§4.7) | w40 +0.0370 / w50 +0.0371 (6/6) | `sf12_a10_wbc_dose_ext_paired_stats.json` | `weighted_bc_sweep/supply_lift_a10_sf12_dose_ext_6seed` | SF-WBC-DOSEEXT |
| SF variance (§4.7) | −0.0009±0.0035, n=10 (null) | `sf12_a10_variance10_aggregate.json` | `variance_suite/supply_lift_sf12_10seed` | B2-VAR-SF |
| SF rollout (§4.7) | trim+lift −0.0335 vs trim-only −0.0432 (both 0/6, p=.031) → ~22% | `sf12_a10_rollout_summary.json` + `sf12_trimonly_a10_rollout_summary.json` | `option_a_rollout_sf12_tl` + `…_trimonly` | C3-SF-ROLLOUT-TL + -TO |
| SF baselines (§4.5) | iFGSM/FGSM/random +0.0122/+0.0176/+0.0049; adjacency 72.0/87.0/97.5% | `PAPER/baselines/comparison/sf12_{ifgsm,fgsm,random}_metrics.json` | `2026-07-16T16-08-*_baseline_*_sf12` | B1-IFGSM/FGSM/RANDOM |

## 7. Re-run recipes by experiment class

Command templates are verbatim from the ledger. `<edit>` denotes the filtered α\* corpus dir
(SZ: `famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered`;
SF: `…/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered`). SF variants prefix `FAMAIL_CITY=sf12`
and use `-k 2000`. The α\* weight overrides are mandatory (see §5).

- **Editor run (GPU).**
  `python -m famail_temporal.evaluation.runner -k 10000 --name <tag> --device auto --override ALPHA_SPATIAL=0.1 --override ALPHA_CAUSAL=0.8 --override ALPHA_FIDELITY=0.1`,
  then `python -m famail_temporal.analysis.filter_infeasible_trims --edit-dir <run dir>`.
  Trim-only ablation adds `--override TAIL_LEN=0`. (SZ headline ~7h54m on an RTX 3070.)
- **External fairness (CPU-light, bootstrap 1000).**
  `python -m famail_temporal.baselines.run_external_fairness --edit-dir <edit> --dataset <name> --bootstrap 1000 --seed 0 --delta-supply <edit>/delta_supply_3d.npz`
- **Channel decomposition (CPU, ~seconds).**
  `python -m famail_temporal.analysis.channel_decomposition --edit-dir <edit> --bootstrap 2000 --seed 0 [--tier2-grid <edit>/S_tier2_after.npz]`
- **Supply recount / tier-2 grid (CPU).**
  `python -m famail_temporal.analysis.supply_recount --edit-dir <edit> --persist-grids [--city sf12]`
- **Weighted-BC suite (GPU).**
  `python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir <edit> --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir <out>`
  (dose extension: `--weights 40,50 --placebo 40,50 --most-fair 40,50`; n=12 pool adds a seeds
  `6,7,8,9,10,11 --weights 30` run.)
- **L1v2 identity-aware fidelity (GPU).**
  `python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir <edit> --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir <out>`
- **Variance suite (GPU).**
  `python -m famail_temporal.baselines.run_variance_suite --edit-dir <edit> --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir <out>`
- **Perturbation / data-augmentation baselines (GPU).**
  `python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir <edit> --mode {ifgsm,fgsm,random} --seed 0 --device auto --score-fidelity`
  (iFGSM ran ~50x slower on CPU; use GPU, see ledger Q1-CPU-ABORT.)
- **Demographic oversampling (CPU).**
  `python -m famail_temporal.baselines.run_demographic_oversampling --variant {targeted,placebo} --dose 10000 --seed 0`
- **Alpha-sweep summary tool (CPU).**
  `python -m famail_temporal.analysis.alpha_sweep_summary`
- **Policy rollout (GPU).**
  `python PAPER/external-metrics/scripts/option_a_rollout_eval.py --edit-dir <edit> --out-dir <out>`
  (or `--seeds 0,1,2,3,4,5 --arms raw,edited,edited_w10,edited_w30`).

**End-to-end replication.** `S10-REPLICATION` (ledger `DONE`) re-ran the promoted SZ headline
corpus end-to-end under clean `main`: every metric and count is identical to the committed s10
corpus, `ΔF_demo +0.022561`, `2,337 + 7,545` edits, 118 reverts (see `DATA_INVENTORY.md` §2, row
`s10_replication_metrics.json`). The headline corpus re-derives **exactly**.

## 8. Verification protocol

To re-certify the chain:

1. **Byte-identity of curated twins.** `cmp -s` each `PAPER/supply-lift/data/a10/*` against its
   `results/` source; all 38 were byte-identical at the 2026-07-15 audit (`git ls-files` confirms
   tracking). Two source filenames differ harmlessly from their curated names: the variance twin is
   sourced from `aggregate.json`, the WBC manifest twin from `manifest.json`; both are `cmp`-identical.
2. **Fingerprint check (§5).** For each editor artifact, read `config_snapshot.{ALPHA_*, TAIL_LEN}`
   and the `n_trim + n_lift` count; confirm α\* = `(0.1,0.8,0.1)` / `4` / `9,882` (SZ) or `1,959`
   total (SF: `1,330 + 629`). Never rely on directory names or prose.
3. **Provenance recompute.** Recompute SHA-256 of each artifact and compare to `PROVENANCE.md`;
   confirm `environment.json` `pip_freeze_sha256` matches the recorded value.
4. **Worked example.** `paper/reviews/2026-07-15-repro-inputs-audit.md` is the fully worked
   re-certification (fingerprints §2a, twin verification §2b, claim↔JSON table §2c, ledger
   cross-check §2d/§2e).

## 9. Known gaps and honest caveats

- **`tab:featsets` channel decomposition, CLOSED (was audit M2).** The 2026-07-15 audit and the
  original T17 brief both name this as an open gap ("`channel_decomposition.json` does not exist for
  HGC/4FEAT"; "A1 has NOT been executed"). **That premise is now stale.** Ledger rows `A1-HGC` /
  `A1-4FEAT` (2026-07-16) and `A1P-HGC-CHAN` / `A1P-4FEAT-CHAN` / `A1P-*-RECOUNT` (2026-07-18) ran
  the tier-1 and tier-2 channel decompositions; the curated twins
  `shz_{hgc,4feat}_a10_channel_decomposition.json` + `_supply_recount.json` exist, are git-tracked,
  and carry the CIs mapped in §6.4. The manuscript states the significance in §4.6's prose and,
  as of 2026-07-21, as $^{*}$-markers on the `tab:featsets` rows themselves.
- **SF tier-2 recount, now plumbed.** `LIFT_ALGORITHM_REFERENCE.md` §10/§14 describe SF tier-2 as
  deferred; that is superseded by `D1-RECOUNT` / `D1-CHAN` (2026-07-18, SF-native match path), which
  produced SF tier-2 supply +0.1027 (sig) and total_tier2 +0.0493 (sig-positive). Quote the D1
  values, not the "deferred" note.
- **`config.py` defaults are not α\*.** The committed ALPHA defaults are `(0.33, 0.33, 0.34)`. A
  re-run that omits the `--override ALPHA_*` flags will silently produce a non-α\* corpus. Verify by
  fingerprint (§5), never by trusting the run completed.
- **`PAPER/` prose is unlabeled stratigraphy** (`DATA_INVENTORY.md` §8). Several `.md` docs state
  superseded numbers as current without an era caveat. In particular
  `LIFT_ALGORITHM_REFERENCE.md` §8/§11 report **old-α** results (F_causal +0.0222, mean(Y|D)
  +0.0468, supply +0.0091/+0.0242) and its §6 quotes **115** infeasible-trim reverts, which is the
  **old-α** count (α\* = **118**). Trust `config_snapshot`, not a README. This document uses only the
  era-neutral mechanism content (constants, implementation map, accounting tiers, metric firewall)
  from that reference.
- **Two-name friction (audit L1/L2).** `3feat` = `hgc`/HGC; `4feat` = PRIMARY + logpopdensity
  (not 3feat + logpop). Both are one set with two labels; the mapping in §2 resolves it.
- **Oversampling-arm provenance is partial.** The demographic-oversampling `metrics.json` files
  record only `arm`/`fairness`/`runtime_s` (no git SHA; `env: not recorded`), so their ledger rows
  are backfilled with `env: not recorded`. The arms are CPU, deterministic per `--seed`, and
  reproducible from the command, but carry no environment fingerprint.

## 10. Provenance of this document

Assembled 2026-07-21 by reading `paper/reviews/2026-07-15-repro-inputs-audit.md`,
`PAPER/DATA_INVENTORY.md`, `paper/README.md`, `PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md`,
`famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md`, `famail_temporal/config.py`, `paper/lint.sh`,
and the curated `PAPER/` artifact JSONs. All values are copied verbatim from those sources; none is
reconstructed from memory. This file contains no personal, institutional, or host identifiers and is
safe to copy into an anonymized artifact repository.
