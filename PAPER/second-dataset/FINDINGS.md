# SF Second Dataset — Complete Findings Report

*The full narrative of the San Francisco second-dataset work: dataset selection, implementation, the editing
(dual-claim) results, the BC/GAN downstream baseline comparisons, and a head-to-head with the Shenzhen primary
dataset. This is the comprehensive findings summary; `README.md` is the terse deliverable index, and the
engineering detail lives in `../../famail_temporal/second_dataset/docs/`. Everything here is on `main`
(the second dataset is kept deliberately separable so it can be swapped wholesale).*

---

## 0. Executive summary

**The FAMAIL method transfers cleanly to a second city, with no algorithm change.** On San Francisco taxi
data the editor makes trajectories **fairer** (F_causal 0.8752 → 0.8891, Δ **+0.0139**) while they stay
**realistic** (F_fidelity **0.968**), and the **entire two-pillar downstream argument reproduces** under the
identical BC/GAN baseline evaluation:

- **Pillar 1** — FAMAIL-edited data is the **fairest faithful source** (higher F_causal than raw/BC-gen/GAN-gen
  while identity-faithful).
- **L2 vanilla transfer** — a driver-conditioned BC trained on edited data does **not** transfer the fairness
  (null), just like Shenzhen.
- **Pillar 2** — **upweighting** the edited demonstrations **recovers** it (ΔF_causal +0.0387 at weight 30,
  6/6 seeds), and the recovery is **sharper** than Shenzhen: both the random-oversampling placebo *and* the
  most-fair-selection control **hurt** fairness (on Shenzhen they were ~null).

One honest divergence from Shenzhen: the WGAN-GP **GAN did not collapse** on SF, so the Shenzhen "GAN
disqualified by distributional collapse" sub-claim does not transfer (§7.3). This does not affect either
pillar. The realism signal (F_fidelity) is **profile-dominated** on SF — but *equally so on Shenzhen*, so it is
a property of the mechanism, not an SF weakness (§5.4).

---

## 1. Why a second dataset, and why San Francisco Cabspotting

The paper's central claim is a **dual claim**: FAMAIL-edited trajectories are simultaneously *fairer* and
*realistic*. Shown on Shenzhen alone, a reviewer asks whether it is a one-dataset artifact. A second dataset
provides external validity.

**The binding constraint is F_fidelity.** Realism is enforced by a pre-trained, **driver-conditioned,
3-stream Siamese discriminator** (seeking-trajectory BiLSTM + driving-trajectory LSTM + an 11-dim driver
profile) that scores whether two **dense per-driver trajectory sequences** come from the same driver. It
**cannot score origin–destination (OD) pairs** and **must be retrained per city**. This eliminates the
obvious US options:

- **OD-only trip records (NYC TLC, Chicago, DC) are INCOMPATIBLE with the dual claim** — they publish
  pickup→dropoff rows with no dense traces and weak/no persistent driver IDs, so they can carry the *fairness*
  half but never the *realism* half. (This is why the Meeting-40 "NYC + Census" idea was set aside.)
- **Dense-trace + persistent-driver-ID data is compatible** (with per-city discriminator retraining).

**San Francisco Cabspotting is the only US dense-trace taxi set** with (i) a **native occupancy flag** (splits
seeking vs. driving for free), (ii) **persistent per-taxi IDs**, and (iii) a **native US-Census/ACS
demographic join** — so it drops into the existing pipeline with **zero algorithm change**. Fallbacks were
Porto ECML/PKDD (#2, non-US, INE demographics) and Rome (#3); DiDi was excluded.

**Dataset.** 536 SF Yellow-Cab taxis, ~11.2M GPS pings, 2008-05-17 → 06-10, format `[lat lon occupancy time]`
(occupancy 1 = driving/fare, 0 = seeking/free).

---

## 2. Implementation (faithful pipeline, no algorithm/metric/fidelity change)

A city-switchable SF pipeline (`FAMAIL_CITY` env var; `shenzhen` default is bit-identical) emits `source_data`
in the **existing loader schema**, so `preprocess → DataBundle → FAMAILObjective` all run unchanged. Key
decisions (detail in `../../famail_temporal/second_dataset/docs/SF_PHASE2_DECISIONS.md`):

| decision | choice | rationale |
|---|---|---|
| **Grid** | faithful constant **0.01°** cells → SF footprint **32×30** (NOT forced to Shenzhen's 48×90) | preserves the ε-ball edit scale; forcing 48×90 would fold/distort the trajectories |
| **Demographics** | **majority-overlap** of ACS 2006–2010 tracts onto cells | matches Shenzhen's district-mapping method (not areal interpolation, which was over-engineered) |
| **Features** | reuse Shenzhen names filled with ACS values: `housing` = median home value (B25077), `comp` = per-capita income (B19301), `migrant` = foreign-born share (B05002) | keeps `config.DEMOGRAPHIC_FEATURES` city-independent; matches the PRIMARY Shenzhen equity set {housing, comp, migrant} |
| **Time** | editor grid `T=24` hourly; trajectory `time_bucket` 1–288 (5-min); `days_in_week=7` | SF taxis run 7 days (Shenzhen data was Mon–Fri) |

---

## 3. The regime discovery and the sf12 subsample

A fairness-only smoke on the **full 536-taxi fleet** returned **baseline F_causal ≈ 0.982 with the editor a
near-no-op** (13/200 edits moved anything). This is not a bug — it is a **fleet-density regime mismatch**:

| quantity | SF full fleet | Shenzhen |
|---|---|---|
| drivers per cell | **0.56** | **0.012** (~47× sparser) |
| 5×5 distinct-taxi supply / cell | mean ~52 (blankets every cell) | calibrated for the sparse sample |
| demand/supply ratio (DSR) | ≈0 everywhere → no service-inequity gradient | has a gradient |

SF is a *near-complete* fleet; Shenzhen is a *50-driver sample*. The 5×5 supply measure **saturates** on the
dense fleet, so the fairness residual becomes supply-noise orthogonal to demographics and F_causal → 1 with
nothing to edit. **Fix = fleet subsampling** to restore Shenzhen's density. Two candidates were compared with a
full-unfair-pool causal-emphasis editor run (`tables/subsample_selection.csv`):

| subsample | drivers | n_active | baseline F_causal | Δ (default α) | Δ (causal-emphasis) | verdict |
|---|---:|---:|---:|---:|---:|---|
| full | 536 | 11,596 | 0.982 | — | — | rejected (saturated) |
| sf50 (count-matched) | 50 | 7,854 | 0.956 | +0.0011 | +0.0041 | rejected (still saturated; cannot headline fairness) |
| **sf12 (density-matched)** | 12 | 4,230 | **0.870** | +0.0085 | **+0.0199** | **CHOSEN** |
| *Shenzhen ref* | 50 | ~34,500 | ~0.82 | — | *~+0.0128* | — |

**Decision: sf12 + causal-emphasis (α_spatial=0.2, α_causal=0.7) + DEMAND_FLOOR=0.5.** sf12's density matches
Shenzhen (~0.012 drivers/cell) and it is the only subsample that produces a publishable fairness gain (larger
than Shenzhen's own +0.0128 from a comparable baseline). `DEMAND_FLOOR=1.0` was tried and is **worse** (crashes
most edits); an isotonic proxy sweep suggested raising the floor but the real editor disproved it — a
methodological lesson: **trust the editor, not cheap proxies** (the 200-edit smoke and isotonic sweep both
understated the achievable gain because they measured *workability*, not *achievable magnitude*).

The 12 sf12 drivers are `[2, 6, 55, 75, 104, 117, 148, 346, 412, 469, 476, 488]`. Supply/demand regime
diagnostic: `figures/sf_supply_demand.png`.

---

## 4. The F_fidelity discriminator (Phase 4)

The realism claim needs a discriminator **trained on SF**. Retraining it surfaced three latent SF-pipeline bugs
(Phase 4 was the *first* time the discriminator-training path consumed SF data — Phase 3 only exercised the
fairness path) and one training-dynamics failure. Full engineering detail:
`../../famail_temporal/second_dataset/docs/SF_PHASE4_DISCRIMINATOR.md`.

### 4.1 Three SF-pipeline day-handling bugs (fixed; fairness baseline provably preserved)
1. **Trajectory `day` column was an absolute epoch-day serial (14016–14040), not day-of-week.** The
   `FeatureNormalizer` cyclically encodes `day` with `days_in_week`; the hardcoded Shenzhen default (5) would
   compute `(14016−1) mod 5` — an arbitrary phase, silently turning the day feature into noise. **Fix:** remap
   col-3 → day-of-week `1..7` (`weekday_from_epoch_day`) and set `days_in_week=7`.
2. **The calendar-day sidecars were sorted-distinct (24 values for 931 trajectories).** The discriminator's
   pair-generator requires them **parallel-per-trajectory** (one day per trajectory) and *raises* otherwise —
   so generation would have crashed. **Fix:** emit parallel-per-trajectory absolute calendar days.
3. **`calendar_day_map.pkl` was missing** (the generator loads it). **Fix:** now emitted (2008-05-17 → 06-10).

**Crucially, the fairness baseline is byte-identical.** The day-of-week remap is confined to the discriminator
corpus; the pickup/dropoff counts (which drive F_causal/F_spatial) keep the absolute day. A full deterministic
rebuild produced **byte-identical** count/demographic/mapping/profile artifacts, and the post-rebuild baseline
**F_causal = 0.8752** is unchanged.

### 4.2 Normalizer config threaded through the checkpoint
`FeatureNormalizer`'s `x_max/y_max/time_buckets/days_in_week` are plain attributes, **not** in the state_dict.
They were added as constructor kwargs (both the training and inference `MultiStreamSiameseDiscriminator`) and
stored in `self.config`, so SF's `(x_max=32, y_max=30, time_buckets=288, days_in_week=7)` bakes into the
checkpoint and round-trips through `load_discriminator`. Backward-compatible: Shenzhen defaults `(49, 89, 288,
5)` reproduce exactly and the real Shenzhen checkpoint loads bit-identical.

### 4.3 The training failure and the fix
The **first run FAILED**: val-AUC **0.495** (chance), train *and* val loss flat at ln 2 for all 13 epochs,
early-stopped at epoch 13. The generated pairs were verified healthy (positive pairs have `‖profile₁−profile₂‖
= 0` exactly; 12 distinct profiles; coords/day correct), so it was not a data problem. An **overfit-256 test**
diagnosed it: the same model/data at lr=1e-3 reached train-AUC 0.93–0.95, proving the failure was **lr=6e-5
being ~12× too low** — the concatenation head's slow warmup kept val-loss flat, and early-stopping (patience
12, on val-loss) fired in the dead zone. **Fix:** lr=1e-3, patience 30, and cap the pathological global-max
padding (one 528-step outlier trajectory forced 97% padding) to seeking 64 / driving 32.

### 4.4 Result — val-AUC 0.998
Retrained with the identical V3 architecture and Ren-aligned protocol as Shenzhen (concatenation combo,
[200,100] BiLSTM, N=5, **1,556,089 params** ≈ Shenzhen's 1,556,337; 10,000 day-based pairs = 5000 pos incl. 500
identical + 5000 neg, split 7500/1500/1000, 12 drivers, 282 usable driver-days). **val-AUC = 0.998** (best
epoch 5, val-loss 0.047; Shenzhen hit 0.982). The 12-identity discriminator trains credibly — the
identity-classification risk is cleared.

---

## 5. Editing results — the dual claim

### 5.1 Headline (`tables/dual_claim_sf12.csv`, `data/sf12_dual_metrics.json`)
Editor causal-emphasis config (α_spatial=0.2, α_causal=0.7, α_fidelity=0.1), **fidelity ON**, `-k 2000` (1371
trajectories edited, 1341 converged, mean 25.3 iters):

| metric | before | after | Δ |
|---|---:|---:|---:|
| **F_causal** (fairness, 1=fairest) | 0.8752 | 0.8891 | **+0.0139 ↑** |
| F_spatial (secondary) | 0.1846 | 0.1817 | −0.0030 ↓ |
| gini_dsr | 0.8266 | 0.8325 | +0.0059 |
| **F_fidelity** (realism) | — | **0.968** | edit-induced Δ ≈ **−1.5e-5** |

**+0.0139 beats Shenzhen's own +0.0128** from a density-comparable baseline. F_fidelity = mean discriminator
P[same driver | original, edited] over the 1371 edits (min 0.922, median 0.979); the **edit itself barely moves
it** (mean drop 1.5e-5) — edited SF trajectories are still recognized as the same driver.

### 5.2 Two ΔF_causal figures, both correct
- **+0.0199** = the *subsample-selection* metric: causal-emphasis over the **entire unfair pool** (~762
  highest-attribution trajectories), fidelity off. This is what chose sf12 over sf50 (§3).
- **+0.0139** = the *dual-claim headline*: `-k 2000` (top-k → 1371 edits), fidelity on.

Different edit subsets, not a regression. The dual-claim run edits *more* trajectories (including lower-impact
ones) than the full-unfair-pool run, so its per-trajectory-averaged gain is smaller.

### 5.3 Fidelity is inert as a gradient (matched control, `data/sf12_fairoff_k2000_metrics.json`)
A matched run with `ALPHA_FIDELITY=0` at the same `-k 2000` gives ΔF_causal **+0.01392** vs **+0.01394** with
fidelity on — a **2e-5** difference. Turning fidelity on costs **zero** fairness and only adds the
per-iteration discriminator forward/backward (33 min vs 4.4 min wall-clock). This confirms the fidelity
gradient is ~0.

### 5.4 The realism caveat — F_fidelity is profile-dominated (shared with Shenzhen)
In the editing use case both branches share the **same driver's profile**, so the discriminator must read the
**seeking trajectory** to notice an edit. A direct probe (`tables/fidelity_sensitivity.csv`) shows it does not:
swapping a *different driver's entire seeking trajectory* into one branch changes the score by ~0
(sf12 −0.0001; gradient **2.6e-11**). **The Shenzhen primary discriminator behaves identically** (−0.0012;
gradient 4.7e-6), matching the earlier `fidelity-grad ≈ 0` gradient-heatmap finding. So F_fidelity is a
**driver-identity-preservation metric** (edits stay within the driver's signature → "realistic"), **not** an
active gradient constraint — a property of the whole mechanism, not an SF regression. **PI decision: report
as-is for parity with Shenzhen;** a stronger seeking-sensitive discriminator (drop the profile stream, or add
same-driver-corrupted-seeking hard negatives) is deferred and would require re-running Shenzhen the same way.
sf12 is *more* saturated (scores exactly 1.0 for identical/different-driver in the isolated probe) because 12
profiles are even more trivially separable than 50; the reported F_fidelity 0.968 (not 1.0) and its 0.92–1.0
variation come from the sampled driving/context streams, not the edited pickup.

---

## 6. Baseline comparison results — the two-pillar story (BC/GAN downstream eval)

The **identical** Shenzhen downstream evaluation was run on the SF edited trajectories. **Identical protocol +
two small backward-compatible plumbing fixes**, and it **reused the existing sf12-dual edit run** (no editor
re-run — the runners consume `<edit-dir>/histories.pkl` + `metrics.json`). Whole suite ≈ **90 min on one RTX
3070** (L1v2 ~15, L2 ~15, weighted-BC ~41, variance ~3). A 1-seed smoke calibrated it at 136 s/seed
(~17× faster than Shenzhen's 38.8 min/seed, because SF's corpus is ~9× smaller and its vocab ~4.5× smaller).

**Plumbing fixes** (the only code changes, both Shenzhen-safe):
1. **City-aware discriminator checkpoint** in the 4 baseline runners — they hardcoded
   `discriminator_checkpoints/default/best.pt`; changed to `config.DISCRIMINATOR_CHECKPOINT_DIR /
   config.DISCRIMINATOR_CHECKPOINT_FILENAME`, which resolves to `sf_12/best.pt` under `FAMAIL_CITY=sf12` and
   stays byte-identical (`default/best.pt`) for Shenzhen. Without this the runs would silently score SF
   trajectories against the wrong (48×90, 50-driver) discriminator.
2. **Variance suite DI-metric skip** — the DI (disparate-impact) metric is a Shenzhen **hukou-district**
   disparity ratio and needs a `district_id_grid`. SF has **no administrative-district abstraction** (its
   demographics are per-cell ACS tracts), so the variance runner now records DI as NaN when there is no
   district grid and computes the district-free metrics (F_causal / F_spatial / localized). Shenzhen unaffected.

**Fidelity-A validation gate PASSED** (real-anchored: matched real-driver pairs 0.958 vs mismatched 0.034,
margin 0.20), so despite the 12-identity concern the discriminator cleanly separates same/different **real**
drivers and **Fidelity-A is trusted** on sf12.

### 6.1 Pillar 1 — L1 data quality (`tables/eval_l1_data_quality.csv`, `data/eval_l1v2_sf12_*.json`)
Four data sources scored on the fairness + fidelity axes:

| source | F_causal | F_spatial | Fidelity-A ↑ | Fidelity-B ↓ (JS vs raw) |
|---|---:|---:|---:|---:|
| raw | 0.8752 | 0.1846 | 0.958 | 0.0000 |
| **edited** | **0.8891** | 0.1817 | **0.958** | 0.1058 |
| bc (MLE-gen) | 0.8789 | 0.1894 | 0.958 | 0.0100 |
| gan (WGAN-GP-gen) | 0.8794 | 0.1856 | 0.958 | 0.0269 |

**FAMAIL-edited is the fairest source** (F_causal 0.889 > raw 0.875 ≈ bc 0.879 ≈ gan 0.879) while
**identity-faithful** (Fidelity-A 0.958 = raw). Edited's Fidelity-B (0.106) is the highest of the non-raw
sources — expected, since the edit deliberately relocates pickups; it is a modest divergence (edits are within
the ε=2-cell ball). ✓ Pillar 1 reproduces Shenzhen. *(GAN's Fidelity-B behavior is the key SF-vs-Shenzhen
divergence — see §7.3.)*

### 6.2 L2 — vanilla transfer (`tables/eval_l2_transfer.csv`, `data/eval_l2_sf12_*.json`)
A driver-conditioned BC (TrajectoryLSTM) is trained on each of raw / edited / bc-gen / gan-gen and its
generated demand is re-scored (5 seeds):

| training source | F_causal (mean ± std) | Fidelity-A | Fidelity-B |
|---|---:|---:|---:|
| raw | 0.8742 ± 0.0053 | 0.9575 | 0.0109 |
| edited | 0.8745 ± 0.0043 | 0.9575 | 0.0130 |
| bcgen | 0.8801 ± 0.0028 | 0.9577 | 0.0164 |
| gangen | 0.8779 ± 0.0015 | 0.9576 | 0.0341 |

**Paired edited − raw ΔF_causal = +0.0004 ± 0.0033 (n=5, Wilcoxon p=0.81, NULL).** Vanilla BC averages the edit
away — exactly Shenzhen's L2 null. This is the null that Pillar 2 overcomes.

### 6.3 Pillar 2 — weighted-BC recovery (`tables/eval_weighted_bc_recovery.csv`, `data/eval_weighted_bc_sf12_*.json`)
Upweighting the edited demonstrations during BC (6 seeds, 10 arms), paired ΔF_causal vs raw:

| arm | Δ vs raw | Wilcoxon p | note |
|---|---:|---:|---|
| edited (w1) | +0.0005 | 0.56 | vanilla baseline — reproduces the L2 null |
| **edited w10** | **+0.0296** | 0.031 | **RECOVERY** (6/6 seeds positive) |
| **edited w20** | **+0.0348** | 0.031 | monotone dose-response |
| **edited w30** | **+0.0387** | 0.031 | largest — **exceeds Shenzhen's +0.0311** |
| random placebo w10 | −0.0071 | 0.031 | oversampling a random non-edited subset **HURTS** |
| random placebo w30 | −0.0095 | 0.031 | |
| most-fair select w10 | −0.0117 | 0.031 | upweighting the already-fairest trajectories **HURTS** |
| most-fair select w20 | −0.0068 | 0.031 | |
| most-fair select w30 | −0.0027 | 0.44 | |

Importance-weighting **recovers** the fairness (monotone +0.0296 → +0.0387, all 6 seeds, Fidelity-A unchanged
at ~0.9576). **Both controls are negative**, i.e. neither generic oversampling nor selecting the already-fair
trajectories reproduces the gain — a **sharper** edit-dominance than Shenzhen, where both controls were ~null.
*(n=6 note: the two-sided Wilcoxon floors at p=0.03125 = all-6-same-sign, so p is a sign-unanimity certificate,
not an effect magnitude; the evidence is the mean Δ, the monotone dose-response, 6/6 sign-consistency, and the
negative controls.)*

### 6.4 Model-level variance (`tables/eval_variance_model_level.csv`, `data/eval_variance_sf12_aggregate.json`)
Paired b0 (raw-corpus BC) vs FAMAIL (edited-corpus BC), MLE-only, 5 seeds:

| metric | b0 | FAMAIL | paired Δ (FAMAIL − b0) |
|---|---:|---:|---:|
| f_causal | 0.8749 ± 0.0022 | 0.8744 ± 0.0025 | **−0.0005 ± 0.0043 (null)** |
| f_spatial | 0.1875 ± 0.0028 | 0.1898 ± 0.0017 | +0.0023 ± 0.0023 |
| f_causal_localized | 0.1376 ± 0.0059 | 0.1241 ± 0.0056 | −0.0135 ± 0.0063 |
| di_primary / di_supplementary | N/A | N/A | N/A (no SF districts) |

The vanilla MLE generator does **not** transmit the edit at the model level (ΔF_causal −0.0005 ± 0.0043),
mirroring Shenzhen's −0.0011 ± 0.0032. This is the model-level companion to the L2 null. (DI is N/A for SF —
§6, fix 2. The terminal-cell JS seed noise floor is 0.021 ± 0.0016.)

---

## 7. SF vs Shenzhen — head-to-head

### 7.1 Side-by-side (Shenzhen = PRIMARY `housing-comp-migrant` set)

| result | Shenzhen (primary) | SF (sf12) | agree? |
|---|---|---|---|
| **Editor ΔF_causal** (causal-emphasis) | 0.7988 → 0.8132 (**+0.0144**) | 0.8752 → 0.8891 (**+0.0139**; +0.0199 full pool) | ✓ |
| **Pillar 1**: edited fairest faithful | edited 0.8132 = fairest; Fidelity-A ≈ raw | edited 0.8891 = fairest; Fidelity-A 0.958 = raw | ✓ |
| **L2 vanilla transfer** (edited−raw ΔF_causal) | −0.0012 (n.s.) | +0.0004 (n.s.) | ✓ null |
| **Pillar 2 weighted-BC** (w10/20/30) | +0.0205 / +0.0278 / **+0.0311** | +0.0296 / +0.0348 / **+0.0387** | ✓ (SF stronger) |
| **random placebo** (control) | ~null (−0.0009 @ w30) | **negative** (−0.0071/−0.0095) | ✓ (SF sharper) |
| **most-fair select** (control) | ~null (+0.0004 @ w30, p=1.0) | **negative** (−0.0117/−0.0068/−0.0027) | ✓ (SF sharper) |
| **model-level variance** (Δ) | −0.0011 ± 0.0032 (null) | −0.0005 ± 0.0043 (null) | ✓ |
| **GAN Fidelity-B** (JS vs raw) | **~0.32 (COLLAPSED → disqualified)** | **0.027 (did NOT collapse)** | ✗ **diverges (§7.3)** |
| discriminator val-AUC | 0.982 | 0.998 | — (SF higher; 12 identities easier) |
| Fidelity-A level | ~0.84–0.85 | 0.958 | — (12 identities more separable; gate still passes) |
| edited fraction of corpus | ~2.6% | ~12.6% (1371/10,887) | — |
| F_fidelity seeking-sensitivity | grad 4.7e-6 (inert) | grad 2.6e-11 (inert) | ✓ both profile-dominated |

**What is the same:** every *directional* conclusion — edited = fairest faithful source, vanilla BC/variance
does not transfer it, weighted-BC recovers it edit-specifically, and F_fidelity is a profile-dominated
identity-preservation metric. **What differs in magnitude:** SF operates at a higher absolute F_causal baseline
(0.875 vs 0.799) and higher Fidelity-A (0.958 vs ~0.845), and its Pillar-2 recovery + negative controls are
*sharper*. **What differs qualitatively:** the GAN collapse (§7.3).

### 7.2 Why SF's Pillar-2 recovery is sharper (interpretation)
SF's edited fraction is ~5× larger than Shenzhen's (12.6% vs 2.6%), so upweighting the edited demos moves the
training distribution more decisively — which both strengthens the edited-arm recovery (+0.0387 vs +0.0311) and
makes the *contrast* arms clearly negative: at high weight the random-placebo and most-fair-select arms
concentrate BC capacity on non-fairness-improving subsets, actively degrading F_causal. On Shenzhen (thinner
edited slice) those arms merely failed to help (~null); on SF they hurt. Either way the qualitative claim —
**the gain is edit-specific, not oversampling and not selection** — holds, and it is *cleaner* on SF. (Caveat:
the larger effective-edited mass at w20/30 is worth watching for over-shift; here Fidelity-A/B stayed healthy
and the w1 baseline stayed null, so the recovery is not a distributional artifact.)

### 7.3 The GAN-did-not-collapse divergence (detailed)
**On Shenzhen**, the WGAN-GP GAN-generated source was **disqualified** from the "faithful sources" comparison
because its **Fidelity-B (Jensen–Shannon divergence of trajectory-statistic distributions vs raw) collapsed to
~0.32** — the adversarial generator free-runs / degenerates (length and coverage collapse), producing
trajectories distributionally far from real. This collapse was used as evidence in the Shenzhen story that
*generative* data can silently degrade, motivating "edit the data rather than generate it."

**On SF, the GAN did NOT collapse.** Its Fidelity-B is **0.0269** — comparable to BC's 0.0100 and far below any
collapse threshold. So on SF, gan-gen is a *faithful* source too, and it is **not disqualified**.

Consequences and interpretation:
- **The core Pillar-1 claim still holds** — edited (0.8891) is the fairest source regardless; it does not
  depend on disqualifying the GAN. If anything, Pillar 1 is *cleaner* on SF: all three non-raw sources are
  faithful, and edited still wins on F_causal without needing a disqualification argument.
- **The Shenzhen "GAN collapse" cautionary sub-narrative does NOT transfer to SF** and should not be claimed
  for the second dataset. This is an honest, reportable difference.
- **Likely cause (hypothesis, not verified):** SF's much smaller **vocabulary** (963 grid-cell tokens =
  32×30+3, vs Shenzhen's 4323 = 48×90+3, ~4.5× smaller) and **corpus** (~10.9k trajectories vs ~95–105k,
  ~9× smaller) make the WGAN-GP adversarial dynamics more stable — a smaller output space and shorter
  trajectories (SF real mean length ~15–18 grid steps) are far easier for the critic/generator to balance, so
  the mode/length collapse that plagued the large-vocab Shenzhen setup does not arise. The generation
  hyperparameters (`MAX_GEN_LEN=64`, `MAX_TRAIN_TOKENS=256`) were tuned to Shenzhen's longer trajectories and
  comfortably bound SF's shorter ones.
- **Not load-bearing:** the two-pillar argument rests on edited-vs-raw (data quality) and the weighted-BC
  recovery-vs-controls, neither of which depends on the GAN collapsing. The GAN arm is a *supporting* baseline;
  its different behavior on SF is a dataset-characterization note, not a threat to the claims.

---

## 8. The two-pillar story of the SF data (narrative)

1. **Editing makes the data fairer while keeping it realistic** (the dual claim, §5): F_causal +0.0139,
   F_fidelity 0.968, no algorithm change. Fairness is the active objective; realism is preserved (the edit
   moves pickups within a 2-cell ball, invisible to the identity discriminator).
2. **The edited data is the fairest *faithful* source** (Pillar 1 / L1, §6.1): higher F_causal than raw,
   BC-generated, and GAN-generated data, while identity-faithful — so the value of editing is not bought by
   sacrificing driver realism, and it beats simply *generating* synthetic data.
3. **But vanilla behavior cloning does not inherit the fairness** (L2 + variance nulls, §6.2/§6.4): a
   driver-conditioned BC trained on the edited corpus produces demand no fairer than one trained on raw — the
   edit is a ~12.6% slice that the loss averages away. This is the negative that makes the method non-trivial.
4. **Importance-weighting recovers it, and does so edit-specifically** (Pillar 2, §6.3): upweighting the edited
   demonstrations drives a monotone, 6/6-seed recovery to +0.0387, while oversampling a random subset or
   selecting the already-fairest trajectories both *hurt* fairness. So the recovered signal is a property of
   the **edits**, not of oversampling or of the data already present.

**Framing:** this positions FAMAIL as a **fairness-oriented data-augmentation method** — edit a small,
targeted slice of the demonstrations, then upweight it during policy training — validated now on **two cities**.

---

## 9. Caveats, limitations, and decisions on record

- **F_fidelity is profile-dominated** (§5.4) → it certifies *driver-identity preservation* under edits, not
  trajectory-shape realism; equally true on Shenzhen. Reported as-is for parity (PI decision); a
  seeking-sensitive discriminator is a deferred option.
- **Small n**: 12 drivers, 5–6 seeds. The edited−raw *data-level* F_causal gap is deterministic (std=0), so it
  is a point comparison with no sampling CI (same as Shenzhen). Downstream inference rests on paired seeds +
  the Wilcoxon sign-unanimity floor + dose-response + negative controls, not on a powered test.
- **The GAN sub-claim does not transfer** (§7.3).
- **F_causal is associational**, not causal — the partial R² of a cross-sectional OLS of the demand-adjusted
  service residual on observational demographics; SF fills the Shenzhen feature *names* with ACS values
  (`migrant` = foreign-born share is an ACS proxy, not hukou). Interpret magnitudes relatively, not
  cross-city-absolute (SF baseline 0.875 ≠ Shenzhen 0.799).
- **DI metric is N/A for SF** — it is a Shenzhen hukou-district disparity ratio; SF has no administrative
  districts (§6, fix 2).
- **Decisions locked:** sf12 (density-matched) over sf50; causal-emphasis + DEMAND_FLOOR=0.5; parity framing
  for F_fidelity; report the GAN divergence honestly rather than force the Shenzhen collapse narrative.
- **Open (paper-level):** whether to invest in a seeking-sensitive discriminator for a stronger realism claim
  (would need a Shenzhen re-run for parity); the `F_causal → F_demo` rename (a Shenzhen-wide agenda item).

---

## 10. Provenance / reproduce

All results are curated in this directory with source provenance (`README.md` §7 has the full artifact→source
table). The underlying `famail_temporal/results/` tree and the SF corpus/checkpoint are git-ignored (on-disk
only). Reproduce commands are in `README.md` §8. Engineering detail:
`../../famail_temporal/second_dataset/docs/` (`SF_SECOND_DATASET_STORY.md`, `SF_PHASE4_DISCRIMINATOR.md`,
`SECOND_DATASET_COMPATIBILITY.md`, `SF_PHASE2_DECISIONS.md`, `SF_SOURCE_SCHEMA.md`). Meeting briefing:
`../../famail_temporal/baselines/MEETING_41_PREP.md`. Git: editor dual claim `4c50a3f`/`52e0568`; BC/GAN eval
`658ae63`; merges on `main`.
