# SF Second Dataset — Phase 3 Results & Findings (2026-06-30)

**Status:** Phase 3 (data pipeline) **complete and verified**. Two findings surfaced
that need decisions before the GPU phases (4 retrain, 5 edit-run): one fixed
(pickup consistency), one **open and material** (SF supply/demand regime).

## What was built (all TDD, default-city suite green: 386 passed)

A city-switchable SF pipeline emitting `source_data` in the existing loader schema,
with **zero change to `algorithm/`, `fairness/`, or `fidelity/`**:

| Component | File | Tests |
|---|---|---|
| Raw loader | `data/source_generation/sf_raw_loader.py` | 2 |
| Grid + occupancy/gap segmentation | `sf_config.py`, `sf_segmentation.py` | 5 |
| Demographics (pop-weighted areal interpolation, geopandas) | `sf_demographics.py` | 3 |
| Grid counts (pickup/dropoff + 5×5 supply) | `sf_grid_counts.py` | 3 |
| Multi-stream + 11-dim profiles | `sf_multistream.py` | 3 |
| Assembler | `sf_build.py` | (integration) |
| `FAMAIL_CITY` config switch | `config.py` | regression |

**Verified end-to-end:** `sf_build` → `preprocess` (n_active=13,032) → `DataBundle.load()`
→ the **unchanged `FAMAILObjective`** computes finite SF fairness on the 32×30 grid.

## Finding 1 (FIXED) — pickup-cell consistency

The editor reads a trajectory's pickup as `states[-1]` (`Trajectory.pickup`). The first
build counted pickups at the occ=1 transition cell (one ping after a seeking trajectory
ends), so the editor subtracted mass from near-empty cells → `compute_fspatial`
negative-value crash on **12/20** edits. Fixed by counting pickups/dropoffs at each
trajectory's **terminal cell**. Post-fix: **0/50** edits crash.

## Finding 2 (OPEN — needs PI decision) — SF supply/demand regime saturates F_causal

A fairness-only editor smoke (`alpha_fidelity=0`, CPU, 50 top-attributed trajectories,
30 iters) runs cleanly but yields **0/50 improvements** and **no change** in fairness:

```
BASELINE  F_spatial = 0.148   F_causal = 0.975 (near-max)
EDITED    F_spatial = 0.148   F_causal = 0.975   (delta 0.0000)
```

Diagnosis (measured):

| quantity | SF | implication |
|---|---|---|
| demand D (mean-hourly pickups) | mean 1.30, **median 0.00**, p90 1.24 | demand is sparse per 1 km cell |
| cells below `DEMAND_FLOOR=0.5` | **85.2%** | most cells demand-clamped |
| supply S (5×5 distinct taxis) | mean **52.6**, median 12.5, max 388 | supply pool dwarfs demand |
| Y = S / clamp(D) | mean **~60** | Shenzhen regime is far lower (~O(1)) |

**Mechanism:** with 85% of cells demand-clamped, `Y = S/D ≈ 2·S` is **supply-driven**, so the
fairness residual `R = Y − g₀(D)` is dominated by supply noise and is **orthogonal to the
demographics** → `F_causal ≈ 1` (the g₀(D) fit is correspondingly weak: power R² 0.018
all-cells / 0.46 signal-regime). There is essentially **no demographic-fairness signal for
the editor to improve**. This is a *regime* mismatch, not a pipeline bug: the SF
supply (5×5 distinct taxis over a full hour) and demand (terminal-cell pickups) are on
incomparable scales versus Shenzhen.

**This directly bears on the paper's "improves fairness" half on SF.** Options to discuss
with the PI (each an F_causal *intermediate-calculation* change → algorithm-change protocol):
1. **Re-scale supply/demand** so `Y = S/D` lands in a Shenzhen-like O(1) regime (e.g., define
   supply per-cell-hour rather than 5×5 distinct-over-hour, or normalize S and D consistently).
2. **Lower/retune `DEMAND_FLOOR`** and/or aggregate demand over coarser time so fewer cells clamp.
3. **Coarsen the grid** (fewer, larger cells) so per-cell demand is denser — trades against the
   faithful-1 km-cell decision (D1).
4. Accept it as a **finding**: "the method's fairness effect is dataset-dependent; SF taxi
   service shows little demographic-fairness signal at 1 km resolution" — and lean on Shenzhen
   for the fairness-improvement claim, using SF only for realism/transfer.

## Finding 2b — heatmaps + signal-existence test (2026-06-30)

Concentration heatmaps (`docs/sf_supply_demand_heatmap.py` →
`results/sf_diagnostics/sf_supply_demand.png`) show the mechanism directly: **demand is
highly concentrated** (top-10 cells = 50% of pickups; 51% of cells empty; a downtown hotspot)
while the **5×5 supply blankets all 960 cells** (~100× larger) → **DSR ≈ 0 everywhere**.

Root cause is a **fleet-density mismatch**: SF is a near-complete fleet (**536 taxis, 0.56
drivers/cell**) vs Shenzhen's **50-driver sample (0.012 drivers/cell, ~47× sparser)**. The
`active_taxis` = "distinct taxis in the 5×5 neighborhood over the hour" measure, calibrated
for the sparse sample, **saturates** on the dense fleet.

**Signal-existence test** (Pearson r of service quantities vs demographics over 13,032 active
units): a demographic-service association **exists but is weak** — supply vs housing/income
**r ≈ 0.19**, demand vs demographics r ≈ 0.08, and the **DSR signal F_causal actually uses is
only r ≈ 0.05**. So even a supply-measure fix would surface only a **modest** fairness signal;
SF taxis mildly favor wealthier/downtown areas, but the effect is small.

**Implication for the two-pillar claim:** SF is a **strong fit for the realism/method-transfer
pillar** (dense per-driver traces → discriminator retrain; the whole editor runs un-contorted)
but a **weak fit for the fairness-improvement pillar** (little demographic-service inequity to
edit, and the DSR formulation barely captures what exists). Leaning SF → *realism/transfer*
and Shenzhen → *fairness magnitude* is the honest split.

**Recommendation:** do **not** start the GPU retrain (Phase 4) until the regime question is
resolved — a retrained discriminator is wasted effort if the fairness signal is absent/weak.
Bring the heatmaps + the signal-existence numbers to Dr. Zhang. If the fairness claim on SF is
required, test a supply re-definition (per-cell instead of 5×5, or a fleet subsample to
Shenzhen density) — but expect a modest signal given r ≈ 0.19 is the ceiling.

*Reproduce: `FAMAIL_CITY=sf python -m famail_temporal.preprocess --force` then the smoke in
the session log; demand/supply diagnostic inline.*

## Finding 3 (2026-06-30) — fleet subsample RESOLVES the regime; density-match wins

Following the PI-approved plan, demographics were switched to **majority-overlap** (matching
Shenzhen; `sf_demographics`), and two fleet subsamples were built (fixed grid, nested,
seed 42): **sf50** (Shenzhen count-matched) and **sf12** (Shenzhen *density*-matched,
~0.012 drivers/cell). Editor-workability test = fairness-only edits on the top-200
attributed trajectories (`alpha_fidelity=0`, CPU).

| variant | drivers | n_active | baseline F_causal | edits that moved | ΔF_causal | ΔF_spatial |
|---|---|---|---|---|---|---|
| full | 536 | 11,596 | 0.982 | **13/200** | +0.00007 | +0.0002 |
| **sf50** (count) | 50 | 7,854 | 0.957 | **161/200** | +0.00132 | +0.0006 |
| **sf12** (density) | 12 | 4,230 | **0.875** | **183/200** | **+0.00326** | +0.0002 |

**Conclusions:**
- **Subsampling fixes the regime.** Fewer drivers de-saturate supply and restore an editable
  gradient: baseline F_causal 0.982 (full) → 0.957 (50) → **0.875 (12), essentially Shenzhen's
  ~0.82 regime**. The editor goes from **13/200** edits (full, no-op) to **183/200** (sf12).
- **Density-match (12) is the most workable for edits** — closest-to-Shenzhen F_causal and the
  largest improvement. This validates your density argument.
- **DEMAND_FLOOR=1.0 is WORSE, not better** (it *lowers* the F_causal proxy but hurts the actual
  editor): at floor 1.0, sf50 **crashes on 180/200 edits** and sf12's gain shrinks (+0.00025 vs
  +0.00326 at 0.5). **The proxy sweep misled on the floor — keep DEMAND_FLOOR=0.5.**

**Two-pillar tradeoff (the remaining choice):**
- **sf12** maximizes the *fairness* signal but gives the discriminator only **12 identity classes
  / ~11k trajectories** — thin for the realism-retrain pillar (Shenzhen used 50).
- **sf50** is **Shenzhen-matched for the discriminator** (50 identities / ~43k trajectories) AND
  still workable for editing (161/200, real +0.00132 gain).

**Recommendation:** **sf50 + DEMAND_FLOOR=0.5** as the balanced choice for the *dual* claim
(Shenzhen-matched discriminator corpus + workable fairness editing); **sf12** if the
fairness-improvement magnitude is the priority and a 12-identity discriminator proves trainable.
Either way, **the editor now makes real edits → the GPU discriminator retrain (Phase 4) is
finally warranted** (it needs the parent-monorepo training pipeline).
