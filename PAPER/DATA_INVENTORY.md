# FAMAIL — experimental data inventory

**One document, every artifact.** What experimental data exists, where it lives, which **era** it
belongs to, and which paper claim (if any) consumes it. Covers the *full valid inventory* — including
runs the paper does **not** currently cite — so a pivot in the argument can be costed against real
data rather than guessed at.

**Every era label below was derived from the artifact's own `config_snapshot` / `run.log`, never from
prose.** Prose in `PAPER/` is unreliable for this (see §8).

Maintained by hand. Last full sweep: **2026-07-13**.

---

## 0. What this is, and what it is not

| document | keyed by | answers | does NOT answer |
|---|---|---|---|
| **this file** | **artifact** | what data exists, its era, where it is, what it backs | how a run was launched |
| `famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md` | queue id (`Q3`, `R4-wbc-shz`) | *process* provenance: exact command, git SHA, env, frozen-editor gate, wall time, and **failed/killed launches** | what a result means; which claim uses it |
| `paper/sections/*.tex` `% src:` comments | paper line | per-number provenance for what is **already written** | anything not in the paper (i.e. anything you might pivot to) |
| `famail_temporal/results/RESULTS_INDEX.md` | — | **SUPERSEDED BY THIS FILE.** Untracked, frozen 2026-06-30, predates supply-lift *and* the α-sweep. 37 of 70 run dirs postdate it. | — |

`results/` is **git-ignored** (7.3 GB, on-disk only). Anything the paper depends on must have a
curated, force-tracked copy under `PAPER/` — that is what `PAPER/**/*.json` + `*.csv` negations in
the root `.gitignore` are for.

---

## 1. Eras — the one thing to get right

The objective weights **α = (spatial, causal, fidelity)** and the presence of the **lift** mechanism
define four eras. Numbers are **not comparable across them**, and several era pairs differ by as
little as 0.0004 — which is precisely how stale values slip into prose.

| era | α | editor | SZ ΔF_causal | status |
|---|---|---|---|---|
| **α\*** ⭐ | **(0.1, 0.8, 0.1)** | trim+lift | **+0.0226** | **CURRENT — all paper reporting** |
| old-α supply-lift | (0.2, 0.7, 0.1) | trim+lift | +0.0222 | **SUPERSEDED** — valid data, do not quote as current |
| pre-supply-lift | (0.2, 0.7, 0.1) | trim-only | +0.0144 | **SUPERSEDED** — but still the definitional "before" for the leveling-down argument |
| pre-cleanup | (0.2, 0.7, 0.1) | trim-only | +0.0128 | **INVALID** — stuck-GPS sinks not filtered. Do not cite. |

**Era-neutral** artifacts (raw-corpus statistics, data cleanup, feature selection) are valid under
*all* eras and are marked **[N]**.

### The fingerprint trick

You never need to trust a label. **The edit count identifies the corpus exactly** (Shenzhen, k=10,000):

| `n_trim + n_lift` | corpus |
|---|---|
| **9,882** (2,337 + 7,545) | **α\*** — `alpha_sweep_s10_c80_f10_filtered` |
| 9,885 (2,340 + 7,545) | old-α supply-lift |
| 2,455 (trim only, no lift) | pre-supply-lift trim-only |

So `run.log: corpus=95297 edited=9882` ⇒ α\*. `edited=2455` ⇒ pre-supply-lift. For any editor run,
`metrics.json → config_snapshot.{ALPHA_SPATIAL,ALPHA_CAUSAL,ALPHA_FIDELITY}` and `TAIL_LEN`
(`0` = trim-only, `4` = trim+lift) settle it outright.

---

## 2. ⭐ THE CURRENT CORPUS (α\*) — everything the paper cites

**Shenzhen headline corpus:** `famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered`
**San Francisco headline corpus:** `famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered`

All curated + git-tracked under **`PAPER/supply-lift/data/a10/`** (34 files; last addition 2026-07-15: HGC L1v2 multiseed).

| artifact (`PAPER/supply-lift/data/a10/`) | source in `results/` | backs |
|---|---|---|
| `shz_a10_metrics.json` | s10_filtered | §4.2 editor: F_causal 0.7988→**0.8214** (+0.0226), F_spatial +0.0061 |
| `shz_a10_channel_decomposition.json` | s10_filtered | §4.2 channels: supply tier-1 +0.0176, demand +0.0352 n.s. |
| `shz_a10_supply_recount.json` | s10_filtered | §4.2 tier-2 recount +0.0411; MAE 0.0 |
| `shz_a10_external_fairness.json` | `external_fairness/results/shenzhen-primary-supplylift-s10/` | §4.3 DI +0.0162, DP −0.890, Theil −0.0087 |
| `shz_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_shz_5seed/` | §4.4 `tab:l1` (edited 0.8214 fairest; Fid-A 0.844) |
| `shz_a10_weighted_bc_paired_stats.json` + `_dose_response` + `_manifest` | `weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed/` | §4.4 dose-response **+0.0217/+0.0267/+0.0302** (6/6) |
| `shz_a10_variance_aggregate.json` | `variance_suite/supply_lift_shz_5seed/` | §4.4 variance +0.0031±0.0022 (n=5) |
| `shz_a10_pareto_points.csv` | `analysis/pareto_supplylift/` | §4.6 filtering INVERTS the gain (0.7988→0.7935) |
| `shz_a10_rollout_summary.json` | `external_fairness/results/option_a_rollout_a10/` | §4.4 allocation boundary (−0.0033 @ w30) ⚠️ see §7 |
| `shz_trimonly_a10_metrics.json` + `_external_fairness.json` | `2026-07-11T12-11-31_trimonly_a10_shz` | §4.2 trim-vs-lift **ablation** (+0.0146; mean(Y\|disadv) flat) |
| `sf12_a10_metrics.json` | sf12 a10_filtered | §4.7 SF +0.0316; **compliance 87.65 / 85.20** ⚠️ see §7 |
| `sf12_a10_channel_decomposition.json` | sf12 a10_filtered | §4.7 SF supply +0.0209, demand −0.0533 |
| `sf12_a10_external_fairness.json` | `external_fairness/results/sf12-supplylift-a10/` | §4.7 SF Theil −0.0079, DI +0.0058 |
| `sf12_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_sf12_5seed/` | §4.7 SF L1 (0.9067 fairest; GAN healthy 0.039) |
| `sf12_a10_weighted_bc_*` (3 files) | `weighted_bc_sweep/supply_lift_a10_sf12_filtered_6seed/` | §4.7 SF WBC +0.0242/+0.0309/+0.0332 |
| `sf12_a10_variance_aggregate.json` | `variance_suite/supply_lift_sf12_5seed/` | §4.7 SF variance null (−0.0025) |
| `sf12_trimonly_a10_*` (2 files) | `2026-07-11T13-43-37_trimonly_a10_sf12` | §4.2 SF ablation (+0.0144 — *coincides* with the old SZ headline; not it) |
| `shz_trimonly_a10_rollout_summary.json` | `external_fairness/results/option_a_rollout_trimonly_a10/` | §4.4 demand-only rollout comparator: −0.0049 @ w30 → **~33% attenuation like-for-like** (added 2026-07-13, run R5b) |
| `shz_hgc_a10_metrics.json` + `_external_fairness.json` | `2026-07-13T04-41-12_supply_lift_v1_shz_hgc_filtered` + `external_fairness/results/shenzhen-hgc-supplylift/` | §4.6 `tab:featsets` HGC column: 0.8069, +0.0206, DI +0.0147, Theil −0.0080 (added 2026-07-13, stages Q6a/Q7-hgc) |
| `shz_hgc_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_shz_hgc_5seed/` | §4.6 `tab:featsets` L1 row, HGC cell: edited fairest 0.8275 + faithful; GAN bimodality seed-identical in all 3 sets (added 2026-07-15, stage Q8a-l1v2) |
| `shz_4feat_a10_weighted_bc_*` (3 files) | `weighted_bc_sweep/supply_lift_v1_shz_4feat_filtered_6seed/` | §4.6 4FEAT: vanilla +0.0011 n.s., w30 +0.0256 (6/6); ⚠️ most-fair w30 sig-positive +0.0072 (~28% of gain) — surfaced 2026-07-14 |
| `shz_4feat_a10_variance_aggregate.json` | `variance_suite/supply_lift_shz_4feat_5seed/` | §4.6 4FEAT variance: null (+0.0003±0.0028, mixed) — differs from PRIMARY's weak positive |
| `shz_4feat_a10_l1v2_multiseed.json` | `level1_table_v2/supply_lift_shz_4feat_5seed/` | §4.6 `tab:featsets` L1 row, 4FEAT cell: edited fairest 0.7473 + faithful; GAN Fid-B bimodality reproduces seed-for-seed (added 2026-07-14, stage Q8b-l1v2) |
| `shz_4feat_a10_metrics.json` + `_external_fairness.json` | `2026-07-13T17-04-22_supply_lift_v1_shz_4feat_filtered` + `external_fairness/results/shenzhen-4feat-supplylift/` | §4.6 `tab:featsets` 4FEAT column: 0.7253, +0.0220, DI +0.0191, Theil −0.0085 (added 2026-07-14, stages Q6b/Q7-4feat) |

**Baselines (α\*-selected edit set, n = 9,882)** — `PAPER/baselines/comparison/baseline_table.{json,md}`,
also `famail_temporal/baselines/baseline_table/`:

| arm | `results/` dir | ΔF_causal |
|---|---|---|
| iFGSM | `2026-07-13T05-42-07_baseline_ifgsm_shenzhen` | −0.0057 |
| FGSM | `2026-07-13T05-43-40_baseline_fgsm_shenzhen` | +0.0017 |
| random jitter | `2026-07-13T05-44-21_baseline_random_shenzhen` | +0.0135 |
| iFGSM `--no-random-start` | `2026-07-13T06-26-19_baseline_ifgsm_shenzhen` | ablation |
| FGSM `--no-random-start` | `2026-07-13T06-27-50_baseline_fgsm_shenzhen` | ablation |

**Weight decision (α\*-era):** `PAPER/objective-motivation/weight-sensitivity/{DECISION.md,
EXTENDED_FRONTIER.md, extended_frontier.json}` — 6-point frontier, ring-2/ring-3 criterion.
⚠️ That dir's `README.md` still names the **retracted** (0.55, 0.35, 0.1) — see §7.

---

## 3. Editor runs — complete era table (from `config_snapshot`)

`tail` = `TAIL_LEN` (0 = trim-only, 4 = trim+lift). `_filtered` = infeasible-trim filter applied;
**always cite the `_filtered` dir**, never the raw one.

### α\* — (0.1, 0.8, 0.1) ⭐
| run dir | tail | ΔF_causal | trim/lift |
|---|---|---|---|
| `2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` ⭐ **SZ headline** | 4 | **+0.02256** | 2337 / 7545 |
| `2026-07-11T11-31-55_supply_lift_a10_sf12_filtered` ⭐ **SF headline** | 4 | **+0.03156** | 1330 / 629 |
| `2026-07-11T12-11-31_trimonly_a10_shz` (ablation) | 0 | +0.01456 | 2455 / 0 |
| `2026-07-11T13-43-37_trimonly_a10_sf12` (ablation) | 0 | +0.01436 | 1371 / 0 |

### α-sweep points — the sensitivity grid (all trim+lift, SZ, k=10,000)
The frontier that motivated the re-anchor. All 5 `_filtered` dirs are fully ledger-wrapped.
**Valid data — cite as the sweep, not as a headline.**

| α | run dir (`_filtered`) | ΔF_causal |
|---|---|---|
| (0.0, 0.9, 0.1) | `2026-07-09T17-11-50_alpha_sweep_s00_c90_f10_filtered` | +0.02210 |
| **(0.1, 0.8, 0.1)** ⭐ | `2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered` | **+0.02256** |
| (0.2, 0.7, 0.1) | *(anchor = the old-α headline run below; no separate run)* | +0.02222 |
| (0.35, 0.55, 0.1) | `2026-07-10T10-32-00_alpha_sweep_s35_c55_f10_filtered` | +0.02168 |
| (0.55, 0.35, 0.1) | `2026-07-10T17-45-40_alpha_sweep_s55_c35_f10_filtered` | +0.02272 |
| (0.8, 0.1, 0.1) | `2026-07-10T23-30-57_alpha_sweep_s80_c10_f10_filtered` | +0.01854 |

Note (0.55) has the *highest* raw ΔF_causal — it was **decided and then retracted** when its tier-1
supply channel went significantly negative. `DECISION.md` has the history. The frontier is flat within
noise across [0, 0.55]; α\* was chosen on the three-ring criterion, not on ΔF_causal alone.

### old-α supply-lift — (0.2, 0.7, 0.1), trim+lift. SUPERSEDED, data valid.
| run dir | ΔF_causal |
|---|---|
| `2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered` | +0.02222 |
| `2026-07-08T22-43-06_supply_lift_v1_sf12_filtered` | +0.03277 |

Curated at `PAPER/supply-lift/data/` (top level, **not** `a10/`). ⚠️ The `supply-lift/` prose
(`README`, `FINDINGS`, `data_provenance`) still describes **these** as current — see §8.

### pre-supply-lift — (0.2, 0.7, 0.1), trim-only, cleaned data. SUPERSEDED.
The three demographic feature sets. **This is the era the paper's §4.6 robustness table is being
re-run out of** (campaign stages Q6–Q8).

| feature set | run dir | before → after |
|---|---|---|
| **hcm** {housing, comp, migrant} — PRIMARY | `2026-06-29T12-06-55_..._cleaned_hcm` | 0.7988 → 0.8132 (+0.0144) |
| **3feat** {housing, GDP, comp} | `2026-06-26T12-32-59_..._cleaned` | 0.8069 → 0.8193 (+0.0124) |
| **4feat** + logpopdensity | `2026-06-28T11-46-12_..._cleaned_4feat` | 0.7253 → 0.7409 (+0.0156) |
| SF dual-claim | `2026-07-01T09-59-11_sf12-dual` | 0.8752 → 0.8891 (+0.0139) |

### pre-cleanup / bring-up — DO NOT CITE
`2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup` (+0.0128) and the ~25 dirs from 2026-04/05/06
(smokes, calibration, multiloop/STE ablations, sf50 variants, α=(0.33,0.33,0.34) defaults). Stuck-GPS
sinks unfiltered. Historical record only.

---

## 4. Downstream suites — each inherits its era from the corpus it consumed

Verified by reading `--edit-dir` out of each suite's own `manifest.json` / `sweep.json`.

| suite (`famail_temporal/results/…`) | consumed corpus | era |
|---|---|---|
| `level1_table_v2/supply_lift_shz_5seed` | s10_filtered | **α\*** ⭐ |
| `level1_table_v2/supply_lift_sf12_5seed` | sf12 a10_filtered | **α\*** ⭐ |
| `weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed` | s10_filtered | **α\*** ⭐ |
| `weighted_bc_sweep/supply_lift_a10_sf12_filtered_6seed` | sf12 a10_filtered | **α\*** ⭐ |
| `variance_suite/supply_lift_shz_5seed` | s10_filtered | **α\*** ⭐ |
| `variance_suite/supply_lift_sf12_5seed` | sf12 a10_filtered | **α\*** ⭐ |
| `analysis/pareto_supplylift` | s10_filtered | **α\*** ⭐ |
| `weighted_bc_sweep/supply_lift_v1_shz_primary_filtered_6seed` | old-α SZ | old-α |
| `level1_table_v2/cleaned_{hcm,,4feat}_5seed` | the 3 cleaned editors | pre-lift |
| `level2_table/cleaned_{hcm,,4feat}_5seed` | the 3 cleaned editors | pre-lift |
| `weighted_bc_sweep/cleaned_{hcm,,4feat}_6seed` | the 3 cleaned editors | pre-lift |
| `variance_suite/cleaned_{hcm,,4feat}_5seed` | the 3 cleaned editors | pre-lift |
| `analysis/pareto_{hcm,cleaned,4feat}` | the 3 cleaned editors | pre-lift |
| `level1_table_v2/sf12_5seed`, `level2_table/sf12_5seed`, `weighted_bc_sweep/sf12_6seed`, `variance_suite/sf12_5seed` | `sf12-dual` | pre-lift |
| `weighted_bc_sweep/{dryrun, full_5seed_w10_w30, sig_6seed_w10_w20_w30, placebo_6seed_w10_w30}` | **2026-05-28 dirty corpus** | **pre-cleanup — INVALID** |

> ⚠️ **The original placebo run (`placebo_6seed_w10_w30`) was on the pre-cleanup corpus.** The
> placebo result the paper relies on is the one inside the *cleaned* and *α\** sweeps
> (`random_w10`/`random_w30` arms), not this dir.

**There is NO α\*-era `level2_table` run.** Vanilla-BC transfer at α\* is measured as the **w1 arm**
inside the weighted-BC sweep (+0.0022, n.s.), not by a separate L2 suite. Do not go looking for one.

---

## 5. External fairness + rollouts

`famail_temporal/baselines/external_fairness/results/` (outside `results/`, also git-ignored).

| dir | corpus | era |
|---|---|---|
| `shenzhen-primary-supplylift-s10` | s10_filtered | **α\*** ⭐ |
| `sf12-supplylift-a10` | sf12 a10_filtered | **α\*** ⭐ |
| `shenzhen-trimonly-a10`, `sf12-trimonly-a10` | the α\* trim-only ablations | **α\*** ⭐ |
| `shenzhen-primary-supplylift-{s00,s35,s80,a55}` | the other sweep points | sweep (frontier) |
| `shz-primary-supplylift-filtered`, `sf12-supplylift-filtered` | old-α | old-α |
| `shenzhen-{primary,gdp-comp,logpop}`, `sf12` | the 3 cleaned + sf12-fair-ce | pre-lift |
| 5 × `baseline-2026-07-13T*` | the perturbation arms | **α\*-selected** ⭐ |

**Rollouts** (policy-level allocation) — identify by `run.log: edited=`:

| dir | `edited=` | corpus | era |
|---|---|---|---|
| `option_a_rollout_a10` | **9,882** | α\* trim+lift | **α\*** ⭐ |
| `option_a_rollout_supplylift` | 9,885 | old-α trim+lift | old-α |
| `option_a_rollout` | **2,455** | **trim-only, pre-supply-lift** | **pre-lift** ⚠️ |
| `option_a_rollout_trimonly_a10` | **2,455** | **α\* trim-only** (`trimonly_a10_shz`) | **α\*** ⭐ (run R5b, 2026-07-13) |

> ✅ RESOLVED 2026-07-13: the cross-era "31%" ratio was withdrawn (commit `78a98f6`), the α\*
> trim-only rollout was run (R5b, `option_a_rollout_trimonly_a10`), and §4.4 now quotes the
> like-for-like **~33% attenuation** (−0.0033 trim+lift vs −0.0049 trim-only, both α\*, both 0/6
> at $p=.031$; commit `3881b4b`). `option_a_rollout` (pre-lift) remains historical data only.

---

## 6. Era-neutral **[N]** — valid under every era

| artifact | curated | what |
|---|---|---|
| `results/analysis/{dataset_summary, cleanup_delta, experiment_cleanup_delta, sink_decomposition, sink_heatmap}` | `PAPER/shared_cleanup/` | stuck-GPS cleanup: 10 sink cells / 9 plates / **106,677** phantom pickups. Upstream of the editor ⇒ never superseded. |
| `results/analysis/fcausal_feature_sensitivity` | `PAPER/feature_selection/` | demographic-feature vetting, VIF/Pareto, 3-way comparison |
| `PAPER/external-metrics/LEVELING_DOWN_MECHANISM.md` | (tracked) | the structural proof (32× leverage, 93% at demand floor) that **motivated supply-lift**. Computed on the pre-lift corpus **by design** — it is the "before". ⚠️ its 2,455/2,455 flow analysis has not been re-derived at α\* (claim independently corroborated; see §7). |
| `PAPER/baselines/demographic-oversampling/` | (tracked) | oversampling arm — α-independent (no editor in the loop) |
| `analysis/supply_lift_oracle_out/oracle.json` | `PAPER/supply-lift/data/oracle.json` | pre-build headroom gate (+0.882 full accounting) |

---

## 7. ⚠️ Known gaps, and what is NOT yet re-run at α\*

1. **The §4.6 feature-set robustness table** is the only open hole in the paper — 8 `TODO(run:)`
   slots at the 2026-07-13 sweep; the **HGC column landed 2026-07-13** (`a668752`) and the
   **4FEAT column 2026-07-14** (Q6b/Q7-4feat), leaving only the **Q8 downstream rows**
   (q8b running; q8a queued).
2. ✅ **RESOLVED 2026-07-13** — the α\*-era trim-only rollout now exists (R5b,
   `option_a_rollout_trimonly_a10`, §5); §4.4 quotes the like-for-like ~33% (commit `3881b4b`).
3. ✅ **RESOLVED 2026-07-13** — both stale §4 numbers fixed (SF compliance → 87.6/85.2; disclosure
   comment updated) in commit `78a98f6`.
4. ✅ **RESOLVED 2026-07-13** — `paper/lint.sh` now carries the old-α era-guard
   (`0.0222|0.0328|…|87\.4|84\.9`); fires zero times on the fixed manuscript (commit `78a98f6`).
5. **No α\*-era `level2_table`** — by design (see §4).
6. ✅ **RESOLVED 2026-07-14** — `PAPER/supply-lift/LIFT_ALGORITHM_REFERENCE.md` is now tracked. It was
   the most load-bearing untracked file in the repo: **12 `% src:` anchors** in `paper/sections/`
   (11 in `03_methodology.tex`, 1 in `04_experiments.tex`), plus `paper/README.md` §§10/13 and
   `DECISION.md` §9.

---

## 8. ⚠️ `PAPER/` prose is unlabeled stratigraphy — read this before quoting any `.md`

The *data* under `PAPER/` is well-curated and era-separated. The **prose is not**, and several docs
state superseded numbers as current with no caveat. **Trust `config_snapshot`, not a README.**

| doc | says | actually |
|---|---|---|
| `PAPER/argument/*` | declares itself **"authoritative — use verbatim"** | **two eras stale** (pre-lift, old-α). Highest leak risk — it is the designated slide/prose source. |
| `PAPER/supply-lift/{README,FINDINGS,data_provenance}.md` | +0.0222 / +0.0328 / "~40%" as current | old-α. The string **`a10` appears in ZERO `.md` files anywhere in `PAPER/`** — the 23 α\* artifacts are tracked but undocumented in prose. |
| `PAPER/README.md` | headline 0.7988→0.8132 | pre-supply-lift; Layout omits 5 of 11 subdirs |
| `PAPER/objective-motivation/weight-sensitivity/README.md` | (0.55, 0.35, 0.1) dominates | **retracted decision.** `DECISION.md` supersedes with α\*. |
| `PAPER/objective-motivation/MOTIVATION.md` | line 5 says α=(0.2,0.7,0.1); line 157 says (0.1,0.8,0.1) | contradicts itself |
| `PAPER/baselines/{README,comparison/README}.md` | perturbation arms "⏳ PENDING GPU" | **they landed 2026-07-13**; the table is populated |
| `PAPER/second-dataset/` | SF +0.0139 | pre-lift trim-only; SF at α\* is **+0.0316** |
| `PAPER/reviews/README.md` | "two adversarial reviews" | there are three (`REVIEW_C_primary.md` is unlisted) |

Both stale numbers found in §4 on 2026-07-13 entered exactly this way: the manuscript cited a
`PAPER/` prose doc that never announced its own era.

---

## 9. Maintenance

- **A new run is not "in the inventory" until it has a row here.** The run ledger records the
  *launch*; this file records the *artifact*.
- On each campaign landing: add the artifact to §2 (if α\*) or the relevant era table, record the
  `results/` path **and** the curated `PAPER/` twin, and name the paper claim it feeds.
- **Verify era from the artifact, not the prose** — `config_snapshot.ALPHA_*` + `TAIL_LEN`, or the
  `n_trim + n_lift` fingerprint (§1).
- When the campaign drains, this file is the natural input to the planned `PAPER/REPRODUCIBILITY.md`
  capstone (T17), which adds the ledger row + exact command per artifact.
