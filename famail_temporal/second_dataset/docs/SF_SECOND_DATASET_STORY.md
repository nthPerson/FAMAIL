# FAMAIL Second Dataset (San Francisco) — The Full Story

*A unified narrative of the SF second-dataset work: why, what we built, what we
discovered, the key results, and where we are. Companion detail docs live beside
this one in `famail_temporal/second_dataset/docs/`.*

**Status (2026-07-01):** Phases 3–5 **COMPLETE**. Pipeline built & verified;
F_causal supply/demand regime resolved via fleet subsampling; **sf12 +
causal-emphasis** decided; discriminator retrained on sf12 (**val-AUC 0.998**);
**dual claim demonstrated on SF — ΔF_causal +0.0139 (beats Shenzhen's +0.0128)
while F_fidelity 0.968.** Discriminator details + the profile-dominance finding
(shared with Shenzhen, PI-approved parity framing) in `SF_PHASE4_DISCRIMINATOR.md`.

---

## 1. Goal
Add a **second dataset** to the FAMAIL paper so the central claim — *edited taxi
trajectories stay realistic (F_fidelity) while improving fairness (F_spatial,
F_causal)* — is shown on more than just Shenzhen, **without contorting the
algorithm**.

## 2. Why San Francisco Cabspotting
A deep read of `famail_temporal` established that the *realism* claim is enforced
by **F_fidelity = a pre-trained, driver-conditioned, 3-stream Siamese
discriminator** over **dense per-driver trajectory sequences** (seeking + driving
+ 11-dim profile). It **cannot score OD pairs** and **must be retrained per city**.

Consequence (see `SECOND_DATASET_COMPATIBILITY.md`): **all OD-only US data (NYC
TLC, Chicago, DC) is INCOMPATIBLE** for the dual claim — no dense traces, weak
driver IDs. The dataset ranking became:
1. **SF Cabspotting** — the *only* US dense-trace set with a **native occupancy
   flag** (splits seeking vs. driving for free), **persistent per-taxi IDs**, and
   a **native US Census/ACS join**. Zero algorithm change.
2. Porto ECML/PKDD (non-US, INE demographics); 3. Rome; DiDi excluded.

**Dataset:** 536 SF Yellow-Cab taxis, ~11.2M GPS pings, 2008-05-17 → 06-10.
Format `[lat lon occupancy time]` (occupancy 1=driving/fare, 0=seeking/free).

## 3. What we built (Phase 3 — all TDD, zero change to algorithm/fairness/fidelity)
A city-switchable SF pipeline emitting `source_data` in the **existing loader
schema**, so `preprocess.py` → `DataBundle.load()` → the unchanged
`FAMAILObjective` all work as-is.

| Decision | Choice |
|---|---|
| **Grid** (D1) | Faithful constant **0.01°** cells (matches Shenzhen `GRID_SIZE_DEG`), so the ε-ball edit scale is preserved. SF footprint → **32×30** (NOT 48×90; forcing 48×90 would fold data + distort scale). |
| **Demographics** (D2, revised) | **Majority-overlap** of ACS 2006–2010 tracts onto cells (matches Shenzhen's district mapping). `housing`=median home value (B25077), `comp`=per-capita income (B19301), `migrant`=foreign-born share (B05002). Reuses Shenzhen feature *names* → `config.DEMOGRAPHIC_FEATURES` unchanged. |
| **Vintage** (D3) | ACS **2006–2010** (centered on the 2008 traces); 2010 tract geometry. |
| **Time** (D4) | Editor grid `T=24` hourly; trajectory `time_bucket` 1–288 (5-min); `days_in_week=7`. |

**Modules** (in `famail_temporal/second_dataset/data/source_generation/`):
`sf_raw_loader` → `sf_segmentation` (occupancy/gap split; reproduces de-risk
counts exactly: seeking 441,361 / driving 461,318) → `sf_demographics`
(majority-overlap, geopandas) → `sf_grid_counts` (5×5 distinct-taxi supply) →
`sf_multistream` (11-dim profiles) → `sf_build` (assembler). Plus a **`FAMAIL_CITY`
config switch** (`shenzhen` default = bit-identical; `sf`/`sf50`/`sf12` variants
= 32×30, isolated `sf_source*` + `cache/sf*` dirs, SF features).

**Bug found & fixed:** the editor reads a trajectory's pickup as `states[-1]`;
pickups must be counted at each seeking trajectory's **terminal cell** (not the
occ=1 transition), else the editor subtracts mass from empty cells → crash.

**Verified end-to-end:** `sf_build` → `preprocess` (n_active≈13k) →
`DataBundle.load()` → unchanged `FAMAILObjective` returns finite SF fairness.

## 4. The discovery — F_causal saturates on the full fleet
A fairness-only editor smoke on the full 536-taxi fleet: **baseline F_causal ≈
0.982 (near-max), editor a no-op (13/200 edits moved anything).** Not a bug — a
**regime mismatch**:

| quantity | SF (full fleet) | meaning |
|---|---|---|
| demand (pickups)/cell | median **0**, mean 1.3 | sparse |
| cells demand-clamped | **85%** | |
| 5×5 supply (distinct taxis) | mean **52** | blankets every cell |
| DSR = demand/supply | **≈0 everywhere** | no service-inequity gradient |

**Root cause = fleet density.** SF is a *near-complete* fleet (**0.56
drivers/cell**); Shenzhen is a *50-driver sample* (**0.012/cell, ~47× sparser**).
The 5×5 supply measure, calibrated for the sparse sample, **saturates** on the
dense fleet → the fairness residual becomes supply-noise, orthogonal to
demographics → F_causal ≈ 1, nothing to edit. Heatmaps: `sf_supply_demand_heatmap.py`
→ `results/sf_diagnostics/`.

## 5. The fix — fleet subsampling, and the sf50-vs-sf12 comparison
Subsampling de-saturates supply and restores an editable gradient. Two candidates
(nested, seed 42, fixed grid, majority-overlap demographics):
- **sf50** = Shenzhen **count**-matched (50 drivers).
- **sf12** = Shenzhen **density**-matched (~0.012 drivers/cell → ~12 drivers).

### Baseline + editor workability (200-edit smoke, DEMAND_FLOOR=0.5)
| variant | drivers | n_active | baseline F_causal | edits moved (of 200) |
|---|---|---|---|---|
| full | 536 | 11,596 | 0.982 | 13 |
| sf50 | 50 | 7,854 | 0.957 | 161 |
| **sf12** | 12 | 4,230 | **0.875** | **183** |

`DEMAND_FLOOR=1.0` is **worse** (crashes 180/200 edits at sf50; smaller gains) —
**keep 0.5.** (An isotonic-proxy sweep suggested raising the floor; the real
editor disproved it — trust the editor, not the proxy.)

### The decisive measurement — full-k ΔF_causal (edit the ENTIRE unfair pool, GPU, `evaluation.runner`)
| subsample | baseline F_causal | **ΔF_causal (default α=.33)** | **ΔF_causal (causal-emphasis α_ca=.7)** |
|---|---|---|---|
| sf50 (count) | 0.956 | +0.0011 | +0.0041 |
| **sf12 (density)** | 0.870 | +0.0085 | **+0.0199** |
| *Shenzhen headline (ref)* | ~0.82 | — | *~+0.0128* |

Editing the full pool widened the gap to **~7.6×** (default) and, under Shenzhen's
**causal-emphasis** headline config, **sf12 reaches ΔF_causal +0.0199 — larger
than Shenzhen's own +0.0128** from a comparable baseline (0.870 ≈ 0.82). **sf50
cannot headline fairness** even with causal-emphasis (saturated baseline caps it).

> Methodological lesson: cheap proxies (200-edit smoke, isotonic sweep) all
> understated the gap because they measured *workability*, not *achievable
> magnitude*. Only the full-k causal-emphasis run — the actual headline config —
> revealed that the density match is the *only* subsample producing a publishable
> fairness result.

## 6. Decision
**sf12 (density-matched) + causal-emphasis + DEMAND_FLOOR=0.5.**
- **Fairness pillar: STRONG** (ΔF_causal +0.0199, beats Shenzhen; the pillar that
  was in doubt on SF).
- **Realism pillar: the one accepted risk** — retraining the discriminator on **12
  identities** (Shenzhen used 50). Tested in Phase 4 via val-AUC; fallback = a
  ~20–25-driver subsample if 12 won't train a credible discriminator.

## 7. Where we are + next steps
- **Phase 3: DONE** (pipeline + regime resolution + decision).
- **Phase 4 (next, GPU): retrain the Multi-Stream Siamese discriminator on sf12.**
  Training pipeline located at **`/home/robert/FAMAIL/discriminator/model/`**
  (`train.py`, `trainer.py`, `dataset.py`, `dataset_generation_tool/`). Steps:
  (1) generate SF same/different-driver pairs from sf12's `ms_*` corpus;
  (2) configure the architecture for the SF grid (x_max/y_max → 32×30 extent = the
  R7 reconciliation; `days_in_week=7`; 12 drivers); (3) train → drop
  `discriminator_checkpoints/sf12/best.pt`; **check val-AUC** (12-identity gate,
  Shenzhen hit 0.982).
- **Phase 5 (GPU): the dual-claim run** — the same `evaluation.runner` with
  **fidelity ON** + the sf12 checkpoint + causal-emphasis → *edited SF trajectories
  are realistic (F_fidelity) while improving fairness (F_causal)*.

## 8. Reproduce / key commands (use the `.venv`: `/home/robert/FAMAIL/.venv/bin/python`)
```bash
# Build the SF subsamples (fixed grid, majority-overlap demographics)
python -m famail_temporal.second_dataset.data.source_generation.sf_build   # full sf_source
# (sf50/sf12 built via the nested-subsample driver script; see git history)

# Preprocess a variant (CPU)
FAMAIL_CITY=sf12 python -m famail_temporal.preprocess --force

# Fairness-only full-k editor run (GPU, causal-emphasis) — the sf12 headline
FAMAIL_CITY=sf12 python -m famail_temporal.evaluation.runner \
  --name sf12-fair-ce -k 2000 --device cuda \
  --override ALPHA_FIDELITY=0 --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7

# Tests
python -m pytest famail_temporal/second_dataset/ -q
```

## 9. What stays outside this directory (not separable)
- **`famail_temporal/config.py`** — the `FAMAIL_CITY` switch is core shared config.
- **`famail_temporal/source_data/second_dataset/`** — the SF data + `.census_api_key`
  (gitignored); the pipeline reads/writes it via `config.SOURCE_DATA_DIR`.
- **`docs/superpowers/plans/2026-06-29-sf-second-dataset.md`** — the repo-level plan.
- Discriminator training code lives in the parent monorepo `discriminator/`.

## 10. Companion docs (this directory)
- `SECOND_DATASET_COMPATIBILITY.md` — dataset survey + why SF; F_fidelity mechanism.
- `SF_PHASE2_DECISIONS.md` — D1–D4 (grid, demographics, vintage, time).
- `SF_SOURCE_SCHEMA.md` — the exact `source_data` artifact contract.
- `SF_PHASE3_RESULTS.md` — findings 1–3 (pickup bug, regime, fleet comparison) in detail.
- Scripts: `sf_cabspotting_derisk.py`, `sf_cabspotting_r4_probe.py`,
  `build_sf_demographics.py`, `sf_supply_demand_heatmap.py`, `sf_regime_sweep.py`.
