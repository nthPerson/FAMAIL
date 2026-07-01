# second-dataset (San Francisco Cabspotting) — external validity of the dual claim

**Role: SECOND DATASET (self-contained, deliberately separable).** This directory holds the full
realism+fairness **dual-claim** result on a *second* city, so the paper's central claim is not shown on
Shenzhen alone. It is kept **isolated from the Shenzhen (primary) deliverable** on purpose: if the second
dataset is later swapped for a different one, this directory can be replaced wholesale without touching
`../by_feature_set/`, `../shared_cleanup/`, `../feature_selection/`, or `../reviews/` (all Shenzhen-only).

> **The dual claim holds on SF:** the FAMAIL editor makes SF taxi trajectories **fairer** (F_causal
> **0.8752 → 0.8891, Δ +0.0139**) while they remain **realistic** (F_fidelity **0.968**; the edit itself
> barely moves it, ~1e-5) — a larger fairness gain than Shenzhen's own +0.0128 from a density-comparable
> baseline. **No change to the FAMAIL algorithm, fairness metric, or fidelity architecture.**

## 1. Why San Francisco Cabspotting (dataset selection)
The realism half of the claim is enforced by **F_fidelity = a pre-trained, driver-conditioned, 3-stream
Siamese discriminator** over **dense per-driver trajectory sequences** (seeking + driving + 11-dim profile).
It cannot score origin–destination (OD) pairs and must be retrained per city. That eliminates every
OD-only US source (NYC TLC, Chicago, DC): they have no dense traces and weak driver IDs, so they can carry
the *fairness* half but not the *dual* claim. **SF Cabspotting is the only US dense-trace taxi set with a
native occupancy flag (free seeking/driving split), persistent per-taxi IDs, and a native US-Census/ACS
demographic join** — so it drops into the existing pipeline with zero algorithm change. (Porto ECML/PKDD,
non-US, was the #2 fallback.) 536 taxis, ~11.2M GPS pings, 2008-05-17→06-10.

## 2. Headline numbers (`data/sf12_dual_metrics.json`)
Editor **causal-emphasis** config (α_spatial=0.2, α_causal=0.7, α_fidelity=0.1), **fidelity ON**, `-k 2000`
(1371 trajectories edited, 1341 converged). See `tables/dual_claim_sf12.csv`.

| metric | before | after | Δ |
|---|---:|---:|---:|
| **F_causal** (fairness, 1=fairest) | 0.8752 | 0.8891 | **+0.0139 ↑** |
| F_spatial (secondary) | 0.1846 | 0.1817 | −0.0030 ↓ |
| gini_dsr | 0.8266 | 0.8325 | +0.0059 |
| **F_fidelity** (realism) | — | **0.968** | edit-induced Δ ≈ **−1.5e-5** |

**Fidelity is inert as a gradient — proven by a matched run.** With `ALPHA_FIDELITY=0` at the same `-k 2000`
(`data/sf12_fairoff_k2000_metrics.json`), ΔF_causal = **+0.01392** vs **+0.01394** with fidelity on — a 2e-5
difference. Turning fidelity on costs zero fairness and only adds the per-iteration discriminator pass (33 min
vs 4.4 min). This mirrors Shenzhen's `fidelity-grad ≈ 0` (α-independence) property; see §5.

## 3. Why the sf12 subsample (regime discovery)
On the **full 536-taxi fleet**, F_causal ≈ 0.982 and the editor is a near-no-op: SF is a *near-complete*
fleet (0.56 drivers/cell) vs Shenzhen's *50-driver sample* (0.012/cell, ~47× sparser). The 5×5 distinct-taxi
supply measure — calibrated for the sparse sample — **saturates** on the dense fleet, so the service residual
becomes supply-noise orthogonal to demographics and F_causal → 1 with nothing to edit. **Fix = fleet
subsampling** to restore Shenzhen's density. `tables/subsample_selection.csv`: **sf12** (12 drivers,
~0.012/cell, Shenzhen-density-matched) gives the largest editable gain (causal-emphasis full-pool ΔF_causal
+0.0199) and was chosen over sf50 (count-matched, still saturated at +0.0041). Supply/demand heatmap:
`figures/sf_supply_demand.png`.

> Note two different ΔF_causal figures, both correct: **+0.0199** was the subsample-*selection* metric
> (causal-emphasis over the *entire unfair pool*, ~762 highest-attribution trajectories, fidelity off);
> **+0.0139** is the *dual-claim headline* (`-k 2000` → 1371 edits, fidelity on). Different edit subsets,
> not a regression.

## 4. The F_fidelity discriminator (`data/sf12_discriminator_*.json`)
Retrained on sf12 with the **identical V3 architecture and Ren-aligned training protocol** as the Shenzhen
discriminator (3-stream, concatenation combo, [200,100] BiLSTM, N=5, 1.556M params; 10k day-based pairs,
7500/1500/1000 split, 12 drivers). **val-AUC 0.998** (Shenzhen 0.982) — the 12-identity classifier trains
credibly. Two SF-specific configuration points, baked into the checkpoint's `model_config` so inference
matches training: **FeatureNormalizer (x_max=32, y_max=30, days_in_week=7)** for SF's 32×30 grid and 7-day
week; day-of-week trajectory encoding. (Full engineering detail — 3 latent SF-pipeline day bugs fixed, the
normalizer plumbing, the lr fix that took val-AUC 0.495→0.998 — is in
`famail_temporal/second_dataset/docs/SF_PHASE4_DISCRIMINATOR.md`, kept out of this deliverable to preserve
separability.)

## 5. The honest caveat a reviewer will probe — F_fidelity is profile-dominated
`tables/fidelity_sensitivity.csv`. In the editing use case both branches share the **same driver's profile**;
the discriminator must read the **seeking trajectory** to notice an edit. A direct probe shows it does not:
swapping a *different driver's entire seeking trajectory* into one branch changes the score by ~0
(sf12 −0.0001; gradient 2.6e-11). **The Shenzhen primary discriminator behaves identically** (−0.0012;
grad 4.7e-6) — matching the prior `fidelity-grad ≈ 0` gradient-heatmap finding. So F_fidelity is a
**driver-identity-preservation metric** (edits stay within the driver's signature → "realistic"), not an
active gradient constraint — **a property of the whole mechanism, shared with the primary dataset, not an SF
regression.** *(PI decision: report as-is for parity with Shenzhen; a stronger seeking-sensitive
discriminator, e.g. dropping the profile stream or adding same-driver-corrupted-seeking hard negatives, is
deferred and would require re-running Shenzhen the same way.)* The reported F_fidelity 0.968 (not saturated
at 1.0; variation 0.92–1.0 comes from sampled driving/context, not the edited pickup) is a believable,
Shenzhen-like value; the meaningful realism quantity is the **~0 edit-induced change**.

## 6. Contents → source provenance
| artifact | source (git-ignored `famail_temporal/…`) |
|---|---|
| `data/sf12_dual_metrics.json` | `results/2026-07-01T09-59-11_sf12-dual/metrics.json` (fidelity ON) |
| `data/sf12_fairoff_k2000_metrics.json` | `results/2026-07-01T13-00-55_sf12-fairoff-k2000/metrics.json` (matched α_fid=0) |
| `data/sf12_discriminator_training.json` | `discriminator_checkpoints/sf_12/sf12_training_summary.json` (val-AUC 0.998) |
| `data/sf12_discriminator_history.json` | `discriminator_checkpoints/sf_12/history.json` (per-epoch curves) |
| `data/sf12_pair_generation.json` | `source_data/second_dataset/discriminator_datasets/sf12/generation_summary.json` |
| `figures/sf_supply_demand.png` | `results/sf_diagnostics/sf_supply_demand.png` (regime diagnostic) |
| `tables/{dual_claim_sf12,subsample_selection,fidelity_sensitivity}.csv` | derived from the above |

## 7. Config / reproduce (SF data is git-ignored, lives in `famail_temporal/source_data/second_dataset/`)
- **City switch:** `FAMAIL_CITY=sf12` (config resolves the `sf_source_12` corpus, `cache/sf_12`, and the
  `discriminator_checkpoints/sf_12/best.pt` checkpoint). Default `shenzhen` is bit-identical to the primary.
- **Dual-claim run:** `FAMAIL_CITY=sf12 python -m famail_temporal.evaluation.runner --name sf12-dual -k 2000
  --device cuda --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1`
- **Pipeline + discriminator build:** see `famail_temporal/second_dataset/docs/` (SF_SECOND_DATASET_STORY.md,
  SF_PHASE4_DISCRIMINATOR.md, SF_PHASE2_DECISIONS.md, SF_SOURCE_SCHEMA.md).
- Merged to `main` in commits `4c50a3f` (feat) / `52e0568` (merge).

## 8. What stays OUTSIDE this deliverable (separability)
The reusable SF machinery is NOT duplicated here: the pipeline code + docs live in
`famail_temporal/second_dataset/`; the `FAMAIL_CITY` switch is in `famail_temporal/config.py`; the raw data /
corpus / checkpoint are git-ignored under `famail_temporal/{source_data/second_dataset, cache, discriminator_checkpoints/sf_12}`.
This directory holds only the **curated results** for the paper — so replacing the second dataset means
regenerating just this directory.
