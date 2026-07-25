# Reproducibility-inputs pristine audit — DATA_INVENTORY ↔ §4 ↔ ledger ↔ disk

**Date:** 2026-07-15 · **Scope:** TASK C1 (parallel audit session, read-only).
**Verdict:** ✅ **The α\* input set feeding `paper/sections/04_experiments.tex` is pristine.**
Every §4 / abstract / intro number I checked traces to a correctly-fingerprinted α\*
artifact and matches its JSON to full precision; all 38 curated `a10` twins are
byte-identical (`cmp`) to their `results/` sources; the ledger is hygienic (exactly one
`LAUNCHED` outstanding). **Zero correctness-critical discrepancies.** The findings below
are housekeeping / documentation items for the T17 `REPRODUCIBILITY.md` capstone, ranked
by severity.

This certifies the inputs are safe to draft `PAPER/REPRODUCIBILITY.md` against.

> Method note: era was verified from each artifact's own `config_snapshot`
> (`ALPHA_SPATIAL=0.1, ALPHA_CAUSAL=0.8, ALPHA_FIDELITY=0.1, TAIL_LEN=4`) and the
> `n_trim + n_lift` fingerprint — never from prose or dir names. All comparisons run with
> Python/`cmp` reads only; no GPU, no writes.

---

## 1. Discrepancy list (ranked by severity)

Nothing rises to HIGH or CRITICAL. Ranked MED → INFO.

### M1 (MEDIUM, housekeeping — controller `git add -f`) — 3 newest α\* raw dirs untracked
Within the two `.gitignore`-re-included result trees (`level1_table_v2/`,
`weighted_bc_sweep/`), exactly three α\* dirs have **zero** tracked files, while every
sibling carries 6–7 force-tracked provenance files (`PROVENANCE.md`, `manifest.json`,
`environment.json`, the `*.json` metrics):

| untracked dir | ledger row | landed |
|---|---|---|
| `level1_table_v2/supply_lift_shz_hgc_5seed` | Q8a-l1v2 | 2026-07-15T12:21Z |
| `weighted_bc_sweep/supply_lift_v1_shz_hgc_filtered_6seed` | Q8a-wbc | 2026-07-15T22:32Z |
| `weighted_bc_sweep/supply_lift_v1_shz_4feat_filtered_6seed` | Q8b-wbc | 2026-07-15T03:30Z |

This is **exactly** the known-untracked set named in the audit brief; the sweep found **no
others**. (`variance_suite/*` and `analysis/*` show 0 tracked uniformly because those trees
are not re-included at all — tracking lives in the `PAPER/` twins by design; not a finding.)

- **Paper impact: none.** §4 cites the curated `PAPER/supply-lift/data/a10/` twins, which
  are tracked and verified byte-identical (§2 below). The gap is only that the raw dirs'
  committed `PROVENANCE.md` SHA-256 checksums + `environment.json` (python/torch/CUDA/GPU +
  pip-freeze hash) are missing for these three, which the `REPRODUCIBILITY.md` capstone will
  want.
- **Action (controller):** `git add -f` the `PROVENANCE.md` / `manifest.json` /
  `environment.json` / metrics `*.json` in those three dirs before T17, matching how the
  siblings were committed. (Not this session — read-only.)

### M2 (LOW–MEDIUM, real analytic gap) — `tab:featsets` Δmean(Y|disadv) cells have no CI
`tab:featsets` reports `Δmean(Y|disadv)` = **+0.0594 (HGC)** / **+0.1461 (4FEAT)** as bare
point-deltas. The `% src` comment on line 435 already admits "HGC/4FEAT point-deltas, no
bootstrap run." Confirmed from disk: **`channel_decomposition.json` does not exist** for
either the HGC or 4FEAT corpus (only PRIMARY + SF have one), so those two cells carry no
significance statement.

- The HGC/4FEAT `external_fairness.json` **do** carry bootstrap CIs for DP-gap, DI, and
  Theil (all exclude 0) — `run_external_fairness --bootstrap 1000` produced them. What it
  does **not** bootstrap is the disadvantaged *level* itself; that CI comes from
  `channel_decomposition` (the `total` channel), which was never run for these two corpora.
- **Cheap fix exists** (see runs-menu candidate **A1**): both corpora have
  `delta_supply_3d.npz` on disk, so a tier-1 `channel_decomposition` is a ~minute CPU/light
  run that yields the total Δmean(Y|disadv) CI **and** a supply/demand split for the alternate
  feature sets. This is the single highest value-per-cost item in the runs menu.
- A hostile reviewer *will* ask whether the robustness-table lift-up deltas are significant;
  right now the answer is "not stated for 2 of 3 columns."

### L1 (LOW, documentation clarity) — `DATA_INVENTORY §3` "4feat" label is ambiguous
§3's pre-lift table lists `**4feat** + logpopdensity` immediately under
`**3feat** {housing, GDP, comp}`, which reads as "4feat = 3feat + logpop." It is **not**:
the α\* 4FEAT `config_snapshot.DEMOGRAPHIC_FEATURES` =
`{AvgHousingPricePerSqM, CompPerCapita, MigrantRatio, LogPopDensity}` — i.e.
**PRIMARY {housing, comp, migrant} + LogPopDensity**. The before-edit `F_causal` 0.7253
(shared by the pre-lift and α\* 4feat runs) disambiguates, and the paper column header
`+logpopdens` is correct *relative to PRIMARY*. Purely a label nit; proposed diff in §4.

### L2 (INFO) — feature-set has two names across docs
`{housing, GDP, comp}` is "3feat" in `DATA_INVENTORY §3` but "HGC"/`hgc` in §2, the curated
filenames, and the paper. Same set, same before-edit 0.8069; consistent values, two names.
A one-line cross-reference would remove the friction (proposed diff in §4).

### L3 (INFO, forward-looking) — S10-REPLICATION not yet an inventory row (correct, but flag)
`S10-REPLICATION` is `LAUNCHED` (in progress at audit time). It is correctly **absent** from
DATA_INVENTORY (a run enters only once it has an artifact). But it is a *hedge that can move
the headline*: if the clean-`main` replication of the promoted s10 corpus diverges from
`ΔF_causal = +0.02256`, §4.2's `+0.0226` and the abstract/intro headline change. §7 has no
item tracking this; proposed addition in §4.

---

## 2. What was verified (the pristine evidence)

### 2a. Fingerprints — all four editor corpora are α\*
| corpus | ALPHA (sp,ca,fi) | TAIL | features | n_trim+n_lift |
|---|---|---|---|---|
| SZ headline `…s10_c80_f10_filtered` | (0.1, 0.8, 0.1) | 4 | PRIMARY | 2337+7545 = 9882 |
| SF headline `…supply_lift_a10_sf12_filtered` | (0.1, 0.8, 0.1) | 4 | PRIMARY(ACS) | 1330+629 |
| HGC `…shz_hgc_filtered` | (0.1, 0.8, 0.1) | 4 | {hous, gdp, comp} | — |
| 4FEAT `…shz_4feat_filtered` | (0.1, 0.8, 0.1) | 4 | {hous, comp, migr, logpop} | — |

### 2b. Curated twins — 38/38 byte-identical (`cmp -s`) to their `results/` source
All `PAPER/supply-lift/data/a10/*` (SZ, SF, HGC, 4FEAT, trim-only, rollout families) plus
`PAPER/baselines/comparison/baseline_table.json` and `PAPER/supply-lift/data/oracle.json`
are byte-identical to source and **git-tracked** (verified via `git ls-files`; 38/38).
(Source filenames differ from curated names in two harmless cases — variance twin ←
`aggregate.json`, WBC manifest twin ← `manifest.json` — both `cmp`-identical.)

### 2c. Claim-text ↔ JSON value check (spot-checks all matched to full precision)
Representative of the ~90 numbers verified; **every one matched**:

| §4 claim | paper | JSON | ✓ |
|---|---|---|---|
| SZ F_causal Δ | +0.0226 | 0.022561 | ✓ |
| SZ F_spatial Δ | +0.0061 | 0.006112 | ✓ |
| SZ DI / DP-gap / Theil Δ | +0.0162 / −0.890 / −0.0087 | 0.016212 / −0.890193 / −0.008648 | ✓ |
| SZ mean(Y\|disadv) 7.0734→7.1263 | +0.0529 [+.0086,+.0989] | 7.073448→7.126312, total_ci matches | ✓ |
| channels supply t1 / t2 / demand | +0.0176 / +0.0411 / +0.0352 | 0.017630 / 0.041054 / 0.035235 | ✓ |
| SZ L1 raw/edited/bc/gan F_causal | 0.7988/0.8214/0.7980/0.8089 | exact | ✓ |
| SZ GAN Fid-B bimodal (3 hi / 2 lo) | 0.171, ±0.129 | [.291,.041,.031,.295,.197] | ✓ |
| WBC dose w10/w20/w30 | +0.0217/+0.0267/+0.0302 | 0.021675/0.026682/0.030192 | ✓ |
| WBC vanilla / placebo w30 / most-fair w30 | +0.0022(p.16) / −0.0023 / +0.0009 | exact + p's | ✓ |
| variance SZ/SF/HGC/4FEAT | +0.0031±.0022 / −0.0025±.0029 / +0.0029±.0038 / +0.0003±.0028 | exact | ✓ |
| rollout trim+lift / trim-only w30 | −0.0033 / −0.0049 (0/6, p.031) | −0.003290 / −0.004900 | ✓ (→33% atten.) |
| baselines iFGSM/FGSM/random ΔF_c | −0.0057/+0.0017/+0.0135 | exact | ✓ |
| baselines adjacency-viol % | 54.4/91.4/98.8 | 54.40/91.36/98.78 | ✓ |
| oversample targeted d10k (3-seed) / placebo | +0.0153 / −0.0172 | mean[.0175,.0141,.0144]=.0153 / −.0172 | ✓ |
| oversample distinct pool / reduplications / inflation | 8241 / 1759 / 10.5% | 10000−1759 / 1759 / 0.10494 | ✓ |
| featsets Δ (HGC/4FEAT): before / ΔF_c / DI / DP / Theil / mYd | see tab | all exact | ✓ |
| SF F_causal / F_spatial | +0.0316 / +0.0139 | 0.031559 / 0.013872 | ✓ |
| SF compliance (α\* corrected) | 87.6 / 85.2 | 0.87647 / 0.85197 | ✓ (no stale 87.4/85.0) |
| SF channels supply/demand/total | +0.0209 / −0.0533 / −0.0324 | 0.020926 / −0.053330 / −0.032405 | ✓ |
| pareto filter@K=2455 / edit | 0.7935 / 0.8214 | 0.793539 / 0.821356 | ✓ |
| oracle ceiling | +0.786 / +0.882 | FINDINGS ctrl-decomp / oracle.json 0.882001 | ✓ |

Abstract + intro headline numbers (+0.0226, +0.0176/+0.0411, +0.0316, +0.0162 DI, 10.5%)
all carry `% src` comments pointing to the a10 bundle and trace to inventory rows. **No
orphan claims** in the abstract/intro/§4.

### 2d. Inventory ↔ ledger cross-check — every α\* downstream suite consumed the right corpus
Read `--edit-dir` out of each suite's own `manifest.json`: SZ/SF/HGC/4FEAT × {L1v2, WBC,
variance} + SZ pareto all point at the correctly-fingerprinted α\* corpus. Each campaign
inventory row corresponds to a ledger row whose command/artifact-dir agree
(R0X-s10, R1-sf-a10, R2a/b, Q1–Q8, R5/R5b). No mismatches.

### 2e. Ledger hygiene (read-only)
- Exactly **one** `LAUNCHED` outstanding = `S10-REPLICATION` (the running s10 hedge). ✓
- All dead launches carry terminal statuses (`ABORTED`/`DIED`/`FAILED`/`KILLED`), each
  explained in the "Campaign events" section. ✓
- Live table is **last** in the file and contiguous; no prose below it. ✓
- 36 `DONE` + 20 `DONE (backfilled)` + terminal dead rows. ✓

### 2f. Orphan sweep + §7 gaps
- All 38 `a10` files appear in DATA_INVENTORY §2. **No orphan artifacts** in the a10 set.
- §7's six items re-verified: (1) **zero `TODO(run:)` in §4**, only `TODO(PI-framing)`
  remains ✓; (2) α\* trim-only rollout exists ✓; (3) SF compliance 87.6/85.2 in place, no
  stale value ✓; (4) `paper/lint.sh` old-α guard present and **exits 0** (no violations) ✓;
  (5) no α\*-era `level2_table` — by design ✓; (6) `LIFT_ALGORITHM_REFERENCE.md` tracked ✓.
  §7 is accurate; the one thing it omits is the M2 featsets-CI gap and the L3 hedge (added
  in the proposed diff).

---

## 3. What was NOT exhaustively re-derived (honest coverage limits)
- I spot-checked ~90 of the numeric cells in §4 (all load-bearing + all historically-risky
  ones) — **all matched** — but did not machine-diff every last CI digit in every table.
  Given a 0-discrepancy rate across a wide, adversarially-chosen sample, residual risk is low.
- I did not re-run any bootstrap (that would change nothing already on disk); CIs were read
  from the committed JSONs.
- SF/HGC/4FEAT `median_split` grouping cells were not each individually checked against §4
  prose beyond the migrant/district-extremes headline cells the paper actually reports.

---

## 4. Proposed diff for `PAPER/DATA_INVENTORY.md` (do NOT apply — controller applies)

All changes are additive clarifications; none touches a number.

**(a) §3 pre-supply-lift table — disambiguate the "4feat" label (fixes L1 + L2).**
```
OLD:
| **hcm** {housing, comp, migrant} — PRIMARY | `2026-06-29T12-06-55_..._cleaned_hcm` | 0.7988 → 0.8132 (+0.0144) |
| **3feat** {housing, GDP, comp} | `2026-06-26T12-32-59_..._cleaned` | 0.8069 → 0.8193 (+0.0124) |
| **4feat** + logpopdensity | `2026-06-28T11-46-12_..._cleaned_4feat` | 0.7253 → 0.7409 (+0.0156) |

NEW:
| **hcm** {housing, comp, migrant} — PRIMARY | `2026-06-29T12-06-55_..._cleaned_hcm` | 0.7988 → 0.8132 (+0.0144) |
| **3feat** {housing, GDP, comp} — a.k.a. **hgc** (paper "HGC" column) | `2026-06-26T12-32-59_..._cleaned` | 0.8069 → 0.8193 (+0.0124) |
| **4feat** = PRIMARY + logpopdensity, i.e. {housing, comp, migrant, **logpopdensity**} (NOT 3feat+logpop; before-edit 0.7253 confirms) | `2026-06-28T11-46-12_..._cleaned_4feat` | 0.7253 → 0.7409 (+0.0156) |
```

**(b) §7 — add the two open items the audit surfaced.** Append after current item 6:
```
7. ⚠️ **`tab:featsets` Δmean(Y|disadv) cells carry no CI** — the HGC (+0.0594) and 4FEAT
   (+0.1461) lift-up deltas are point-only; `channel_decomposition.json` was never run for
   those two corpora (only PRIMARY + SF have one). Their `delta_supply_3d.npz` exist, so a
   tier-1 `channel_decomposition` (~1 min CPU each, no config flip, no ALPHA overrides) would
   supply the total CI + supply/demand split. Until then the two robustness columns state no
   significance for the lift-up quantity. (DP-gap/DI/Theil CIs for those columns DO exist in
   their `external_fairness.json`.)
8. ⏳ **S10-REPLICATION hedge in flight** — the clean-`main` end-to-end replication of the
   promoted s10 corpus (ledger row `S10-REPLICATION`, `LAUNCHED` 2026-07-15) is not yet an
   inventory artifact. If it diverges from `ΔF_causal = +0.02256`, §4.2's +0.0226 and the
   abstract/intro headline must be revisited (report both per `DECISION.md`). Add its row to
   §2 once it lands.
```

**(c) §9 maintenance — note the force-track chore (fixes M1).** Append a bullet:
```
- **Three α\* raw dirs are on-disk but not force-committed** (`level1_table_v2/supply_lift_shz_hgc_5seed`,
  `weighted_bc_sweep/supply_lift_v1_shz_{hgc,4feat}_filtered_6seed`). Their siblings carry
  force-tracked `PROVENANCE.md`/`manifest.json`/`environment.json`; these three do not. The
  paper is unaffected (it cites the tracked byte-identical `a10` twins), but T17
  `REPRODUCIBILITY.md` should `git add -f` their provenance files so every α\* artifact has a
  committed checksum + environment record.
```

---

## 5. Bottom line for `REPRODUCIBILITY.md` drafting
Drafting can proceed against the current inputs with confidence. The claim → curated
artifact → ledger row → command chain is intact and self-consistent for **every** §4 number
checked. Before the capstone freezes, the controller should (i) `git add -f` the three raw
dirs (M1), and (ii) optionally run candidate **A1** (featsets channel decomposition) so the
`tab:featsets` lift-up cells gain CIs (M2) — a ~2-minute CPU job that closes the one real
reviewer-facing gap. Neither blocks drafting the chain for the artifacts already curated.
