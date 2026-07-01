# SF Phase 4 — Retraining the F_fidelity Discriminator on sf12

*Companion to `SF_SECOND_DATASET_STORY.md`. Records the discriminator
configuration, the SF-pipeline day-handling fixes it required, the training-data
composition, the architecture, and the results (val-AUC = the 12-identity risk
gate). Everything here runs in the worktree `second-dataset-compat` on branch
`sf-phase4-discriminator` with `/home/robert/FAMAIL/.venv/bin/python`.*

**Status: Phase 4 + Phase 5 COMPLETE.** Discriminator retrained on sf12
(val-AUC 0.998, §6.2); F_fidelity is profile-dominated but **consistent with
Shenzhen** (§6.3, PI-approved parity framing); dual-claim run done (§6.4):
**ΔF_causal +0.0139 while F_fidelity 0.968.** Fairness baseline **provably
preserved** (byte-identical count artifacts) — see §2.3.

---

## 1. Goal
Phase 3 built the sf12 fairness pipeline and showed the *fairness* pillar is
strong (ΔF_causal +0.0199 under causal-emphasis, beating Shenzhen's +0.0128).
The *realism* pillar (F_fidelity) needs a discriminator trained **on SF**, because
F_fidelity is a pre-trained, frozen, driver-conditioned 3-stream Siamese
discriminator that must be retrained per city. Phase 4 = train that discriminator
on sf12 and check whether **12 identities** can support a credible same/different
-driver classifier (the one accepted risk of the sf12 decision).

## 2. What Phase 4 revealed about the SF pipeline (and how it was fixed)

Phase 4 is the **first** time the SF corpus is consumed by the discriminator
training path (`discriminator/multi_stream/dataset_generation/generation.py`);
Phase 3 only exercised the fairness path. Three day-handling issues surfaced, all
in `sf_segmentation.py` / `sf_multistream.py`. **PI-approved approach:** rebuild
the corpus with a day-of-week col-3 (mirroring Shenzhen) + thread the normalizer
config through the checkpoint.

### 2.1 Trajectory `day` column was an absolute epoch-day serial, not day-of-week
The discriminator's `FeatureNormalizer` cyclically encodes col-3 as a day-of-week
signal: `day_angle = 2π·(day-1)/days_in_week`. Shenzhen's col-3 is `1..5`
(Mon–Fri) with `days_in_week=5`. SF's col-3 was the **absolute local epoch-day
serial `14016..14040`**, so the hardcoded `days_in_week=5` default computed
`(14016-1) mod 5` — an arbitrary phase, silently turning the day feature into
noise. **Fix:** `sf_multistream.assemble_multistream` now remaps col-3 →
day-of-week `1..7` (`weekday_from_epoch_day`, `Mon=1..Sun=7`; verified
2008-05-17 = epoch 14016 = Saturday = 6), and the discriminator is trained/scored
with `days_in_week=7`.

### 2.2 The `calendar_days` sidecars had the wrong shape (would crash generation)
`generation.py` requires the calendar-day sidecar to be **parallel to the
trajectory list** (one day per trajectory; it *raises* if
`len(calendar_days) != len(trajs)`). SF emitted `sorted({distinct days})` (24
values for 931 trajectories) → generation would crash immediately. **Fix:**
`sf_segmentation.segment_driver` now emits parallel-per-trajectory calendar days
(each trajectory's absolute start day), matching Shenzhen (whose sidecars have
repeats). The sidecars stay **absolute** so Ren pairing groups by true calendar
date; only the trajectory *col-3* becomes day-of-week.

### 2.3 The remap is fairness-neutral — proven by byte-identical count artifacts
The pickup/dropoff counts are derived from seeking-trajectory terminal states
(`sf_build.py`), and `preprocess` builds mean-hourly demand/supply tensors that
divide by `n_days = #distinct day keys`. So the *count* path must keep the
absolute day. The day-of-week remap is therefore confined to
`assemble_multistream` (the discriminator/editor corpus), on **copies**, leaving
segmentation + counts untouched. A full deterministic rebuild of sf12 confirmed:

| artifact | rebuilt vs prior |
|---|---|
| `pickup_dropoff_counts.pkl` | **byte-identical** |
| `active_taxis_5x5_hourly.pkl` | **byte-identical** |
| `cell_demographics.pkl`, grid/driver mappings, `ms_profile_features.pkl` | **byte-identical** |
| `ms_seeking/driving_trajs`, `passenger_seeking` | x/y/time identical; col-3 → day-of-week |
| `ms_*_calendar_days` | sorted-distinct → parallel-per-traj (still absolute) |

Post-rebuild sanity: `preprocess` → `n_active=4230` (unchanged) and baseline
**F_causal = 0.8752, F_spatial = 0.1846** — identical to the prior sf12. The
+0.0199 headline is intact.

### 2.4 `calendar_day_map.pkl` was missing
`generation.py` loads (but does not use for pairing) `calendar_day_map.pkl`.
The SF pipeline didn't emit it. **Fix:** `assemble_multistream` now builds it
(`{epoch_day: 'YYYY-MM-DD'}`); for sf12 it spans 2008-05-17 → 2008-06-10 (25 days).

## 3. Normalizer config threaded through the checkpoint (the R7 reconciliation)
`FeatureNormalizer`'s `x_max/y_max/time_buckets/days_in_week` are plain
attributes, **not** in the `state_dict`, and both `MultiStreamSiameseDiscriminator`
constructors hardcoded `FeatureNormalizer()`. So a checkpoint is only
self-consistent if the model is reconstructed with the same values. We added
`x_max/y_max/time_buckets/days_in_week` kwargs to **both** constructors
(`discriminator/model/model.py` train-side, `famail_temporal/fidelity/model.py`
inference-side), forwarded to the normalizer, and stored in `self.config` so they
are baked into the checkpoint's `model_config`. `load_discriminator` reconstructs
them automatically.

- **SF values:** `x_max=32, y_max=30, time_buckets=288, days_in_week=7` (SF's
  1-indexed 32×30 grid, 7-day week).
- **Backward-compat (verified):** defaults reproduce Shenzhen `(49, 89, 288, 5)`;
  the real Shenzhen `default/best.pt` (whose `model_config` lacks these keys)
  loads bit-identical with `days_in_week=5`. Shenzhen numerics unchanged.

## 4. Training-data composition (Ren day-based pairs)
Identical generation protocol to the Shenzhen V3 discriminator; only the data
source differs. `MultiStreamGenerationConfig(extracted_data_dir=sf_source_12,
positive=5000, negative=5000, identical_ratio=0.1, n_trajs_per_stream=5,
min_trajs_per_day=5, val=0.15, test=0.10, seed=42)`.

| quantity | value |
|---|---|
| drivers (identities) | **12** — `[2,6,55,75,104,117,148,346,412,469,476,488]` |
| usable driver-days (≥5 seek & ≥5 drive) | **282** (21–25 per driver; all 12 have ≥2) |
| pairs | 5000 positive (incl. 500 identical) + 5000 negative = **10,000** |
| split | train **7500** / val **1500** / test **1000** |
| agent coverage | 12/12 drivers in both positive and negative pairs |
| seeking length (global-max pad) | 528 · driving length 108 · profile dim 11 |
| positive class ratio | ≈0.50 (train 0.496, val 0.509) |

Pairs: positive = same driver, two different calendar days; negative = two
different drivers; identical = same driver/day (10%). Data at
`source_data/second_dataset/discriminator_datasets/sf12/{train,val,test}.npz`
(gitignored).

## 5. Architecture + training config
V3 `MultiStreamSiameseDiscriminator`, **identical to Shenzhen production**:

| knob | value |
|---|---|
| streams | seeking + driving + profile |
| per-stream LSTM | `[200,100]` bidirectional, dropout 0.2 |
| per-traj projection | Dense(48, ReLU); N=5 trajs/stream |
| profile encoder | 11 → (64,32) → 8 |
| combination | **concatenation** (Ren-style) |
| classifier | (64,32,8) → 1 |
| **trainable params** | **1,556,089** (Shenzhen: 1,556,337) |
| normalizer | **x_max=32, y_max=30, time_buckets=288, days_in_week=7** |

Training: `lr=6e-5`, batch 32, weight_decay 1e-4, ≤100 epochs, early-stopping
patience 12 (on val-loss), ReduceLROnPlateau, AMP on, seed 42, RTX 3070, ≈54 s/epoch.
best.pt = lowest val-loss epoch → `famail_temporal/discriminator_checkpoints/sf12/best.pt`.

## 6. Results

### 6.1 First run FAILED, then diagnosed (lr too low)
The initial run at the Shenzhen example lr (`6e-5`) **did not learn**: val-AUC
**0.495** (chance), train+val loss flat at ln 2 for all 13 epochs, early-stopped
at epoch 13. Root cause (proven by an overfit-256 test: same model/data at
lr 1e-3 → train-AUC 0.93–0.95): `lr=6e-5` is ~12× too low, and the concatenation
head's slow warmup kept val-loss flat, so early-stopping (patience 12 on
val-loss) fired in the dead zone. The generated pairs were verified healthy
(positive pairs have `‖p1−p2‖=0`; 12 distinct profiles; coords/day correct).

### 6.2 Retrain — val-AUC = 0.998 (12-identity gate PASSES)
Fix: `lr=1e-3`, early-stop patience 30, and capped the pathological 528-step
padding (one outlier trajectory) to seeking 64 / driving 32.

| metric | value |
|---|---|
| **best val-AUC** | **0.998** (best epoch 5 by val-loss 0.047) |
| max val-AUC | 1.000 |
| val accuracy | 0.983 (pos 1.00, neg 0.966) · F1 0.984 |
| epochs | 35 (early-stopped) · ~35 s/epoch · RTX 3070 |
| checkpoint | `discriminator_checkpoints/sf_12/best.pt` (note **`sf_12`**, the config-canonical suffix; normalizer `(32,30,288,7)` baked into `model_config`, round-trips through `load_discriminator`) |

The 12-identity discriminator trains credibly — val-AUC 0.998 exceeds Shenzhen's
0.982. So the *identity-classification* risk is cleared.

### 6.3 The real gate — F_fidelity is a PROFILE SHORTCUT (seeking-insensitive)
Raw AUC is not the fidelity gate. In the editing use case both branches always
share the **same driver's profile**; the fidelity term must read the **seeking**
trajectory to detect an edit. A direct probe (hold branch-1 = driver A; vary
branch-2 slot-0 seeking) shows it does **not**:

| probe | **sf12** | Shenzhen (`default/best.pt`) |
|---|---|---|
| score: A-identical seeking | 1.0000 | 0.9315 |
| score: different-driver B seeking in A's context | 1.0000 | 0.9189 |
| mean score **drop** (identical − B-seeking), N combos | **−0.0001** (n=88) | **−0.0012** (n=392) |
| `∂score/∂(slot-0 seeking x,y)` | **2.6e-11** | **4.7e-6** |
| seeking-sensitive? | **No** | **No** |

**The discriminator classifies same/different-driver almost entirely on the
profile stream and ignores the seeking trajectory.** Consequently F_fidelity
returns ~constant (~1.0 for sf12) under any within-ε-ball pickup edit, and its
gradient w.r.t. the edited cell is ≈0.

**This is a property of the whole fidelity mechanism, NOT an sf12 regression:**
the Shenzhen primary discriminator shows the identical pattern (drop −0.0012,
grad 4.7e-6), matching the earlier gradient-heatmap finding (`fidelity-grad ≈ 0`,
α-independence). F_fidelity therefore functions as a **driver-identity-preservation
metric** (edits stay within the driver's signature → "realistic"), not as an
active gradient constraint on trajectory shape. sf12 is more saturated (scores
exactly 1.0) because 12 profiles are even more trivially separable than 50.

**Implication for Phase 5 / the dual claim:** F_fidelity for sf12 will report
high/constant, supporting "edited trajectories remain realistic (preserve driver
identity)" in the *same limited sense* as the primary dataset. Whether to (a)
report as-is for parity, or (b) strengthen the discriminator (drop the profile
stream / add same-driver-corrupted-seeking hard negatives) to make F_fidelity a
genuine trajectory-realism signal — is a PI/framing decision (see the session
notes; deviating from the Shenzhen discriminator design would warrant re-running
Shenzhen for parity).

### 6.4 Phase 5 — the dual-claim end-to-end run (sf12, causal-emphasis, fidelity ON)
`FAMAIL_CITY=sf12 runner --name sf12-dual -k 2000 --device cuda --override
ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1`
(results: `famail_temporal/results/2026-07-01T09-59-11_sf12-dual/`).

| metric | before | after | Δ |
|---|---:|---:|---:|
| **F_causal** (fairness) | 0.8752 | 0.8891 | **+0.0139 ↑** |
| F_spatial | 0.1846 | 0.1817 | −0.0030 ↓ |
| gini_dsr | 0.8266 | 0.8325 | +0.0059 |
| **F_fidelity** (realism, mean disc P[same driver \| orig, edited] over 1371 edits) | — | **0.968** | edit-induced Δ ≈ **−1.5e-5** |

1371 trajectories edited (1341 converged, mean 25.3 iters). **The dual claim
holds on SF: edited trajectories improve fairness (ΔF_causal +0.0139, beating
Shenzhen's +0.0128) while remaining realistic (F_fidelity 0.968; the edit itself
barely moves it, drop ~1e-5).** F_fidelity variation across trajectories
(0.92–1.0) comes from the sampled driving/context streams, not the edited pickup
— consistent with §6.3's profile-dominance; the ε-ball pickup shift is invisible
to the discriminator, so fairness edits cost ~0 realism.

**Matched fidelity-OFF comparison (isolates fidelity's effect).** Same run with
`ALPHA_FIDELITY=0` (`sf12-fairoff-k2000`):

| run | ΔF_causal @ `-k 2000` (1371 edits) | wall-clock |
|---|---:|---:|
| fidelity **OFF** (α_fid=0) | **+0.01392** | 4.4 min |
| fidelity **ON** (α_fid=0.1) | **+0.01394** | 33 min |

**Fidelity is inert** — turning it on changes ΔF_causal by 2e-5 (no fairness
cost) and only adds the per-iteration discriminator forward/backward (7× slower),
confirming the ~0 fidelity gradient. (Note: this +0.0139 at `-k 2000` differs
from the prior fidelity-OFF headline +0.0199, which edited the *entire unfair
pool* ~762 highest-attribution trajectories — a different, higher-impact subset,
not a regression.)

## 7. Reproduce
```bash
cd /home/robert/FAMAIL/.claude/worktrees/second-dataset-compat
PY=/home/robert/FAMAIL/.venv/bin/python
# (1) corpus already rebuilt in place (sf_source_12); to redo from raw:
PYTHONPATH=$(pwd) $PY -m famail_temporal.second_dataset.data.source_generation.sf_build   # full; sf12 via driver_ids
FAMAIL_CITY=sf12 $PY -m famail_temporal.preprocess --force
# (2) generate Ren pairs (scratch script) -> discriminator_datasets/sf12/
# (3) train (scratch script, PYTHONPATH=worktree) -> discriminator_checkpoints/sf12/best.pt
# tests
$PY -m pytest famail_temporal/second_dataset/ -q     # 21 pass
```

## 8. Files touched (all on branch `sf-phase4-discriminator`)
- `second_dataset/.../sf_segmentation.py` — parallel-per-traj absolute calendar sidecars.
- `second_dataset/.../sf_multistream.py` — `weekday_from_epoch_day`, col-3→dow remap, `calendar_day_map`.
- `second_dataset/.../sf_build.py` — write `calendar_day_map.pkl`.
- `discriminator/model/model.py` + `famail_temporal/fidelity/model.py` — normalizer config kwargs + `self.config`.
- `second_dataset/.../tests/{test_sf_segmentation,test_sf_multistream}.py` — +4 tests (21 total).
