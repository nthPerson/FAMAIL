# Filtered supply-lift results — PROVENANCE

Derived from: `famail_temporal/results/2026-07-08T22-43-06_supply_lift_v1_sf12`
Source experiment_id: `2026-07-08T22-43-06_supply_lift_v1_sf12`  ·  git_sha: `8605915`
Tool: `famail_temporal.analysis.filter_infeasible_trims`  ·  user decision: 2026-07-08

## Rule

An edit is applied only when a king-compliant repair exists (max(|dx|,|dy|) <= 1 on every consecutive step). Trim edits that fell back to the legacy pickup-only perturbation (tapered-tail repair infeasible) are reverted to their originals, making trim symmetric with lift (which already skips on infeasible).

## Why

The G4 adjacency sweep found exactly 47 modified trajectories that violate king-move adjacency — all trim edits that fell back to the legacy pickup-only perturbation because their tapered-tail repair was infeasible (the G3 trade-off). Lift mode already *skips* such edits; this post-process makes trim symmetric by reverting those 47 trajectories to their originals. After filtering, G4 must be 100% king-compliant.

The published legacy trim numbers remain reproducible via `TAIL_LEN=0`; this tool modifies nothing in the source directory.

**Non-reoptimized survivors:** the surviving edits were NOT re-optimized after removing the reverted trims — their optimization saw the reverted edits' intermediate demand perturbations in the sequential base grid. The filtered grids are exact for "these surviving edits applied to base," which is not byte-identical to a from-scratch skip-on-infeasible run. Approved trade-off (2026-07-08) to avoid a multi-hour GPU re-run; the coupling is negligible (the reverted edits were pickup-only single-cell-mass moves).

## Edit counts

| | source | filtered |
|---|---|---|
| n_trim | 1371 | 1324 |
| n_lift | 629 | 629 |
| n_skipped_infeasible_trim | 0 (fell back) | 47 |
| total edits | 2000 | 1953 |

## King-move compliance (absolute + edit-relative)

Violators are identified by **replaying the modifier's fallback decision** (`apply_tail_perturbation` on the original with the recovered integer pickup offset and this run's TAIL_LEN/GRID_DIMS; `None` = fallback) — exact by construction and city-robust. Raw source data is not necessarily 100% king-compliant (SF Cabspotting-derived trajectories have ~15% baseline violations from GPS gaps of up to ~18 cells — a source-data property, unrelated to editing), so *absolute* compliance of the edited corpus can only be judged against the original-corpus baseline; the cross-city G4 statement is **edit-relative compliance** (fraction of edits introducing zero new violations), which must be 100% post-filter. Note a fallback can introduce no NEW violation (<=1-cell legacy move, or altering an already-violating raw step) yet still break the skip-on-infeasible rule — such fallbacks are reverted too.

| | source (pre-filter) | filtered |
|---|---|---|
| absolute — modified corpus | 1707/2000 (85.35%) | 1707/1953 (87.40%) |
| absolute — ORIGINAL corpus baseline | 1702/2000 (85.10%) | 1659/1953 (84.95%) |
| edit-relative (zero new violations) | 1956/2000 (97.80%) | 1953/1953 (**100.00%**) |

## Fairness metrics (recomputed from filtered histories)

| metric | source before | source after | filtered before | filtered after | filtered Δ |
|---|---|---|---|---|---|
| f_spatial | 0.184629 | 0.185207 | 0.184629 | 0.202652 | +0.018023 |
| f_causal | 0.875151 | 0.897494 | 0.875151 | 0.907916 | +0.032765 |
| gini_dsr | 0.826567 | 0.824360 | 0.826567 | 0.789490 | -0.037076 |
| gini_asr | 0.804175 | 0.805225 | 0.804175 | 0.805205 | +0.001029 |

## Supply totals (filtered ΔS)

- added: 109.7433
- removed: 109.8267

## ΔS reconstruction equivalence (load-bearing check)

Rebuilding ΔS from ALL source histories (float32 accumulator, histories order, mirroring `modifier._hard_tail_delta_supply`) reproduces the persisted `delta_supply_3d.npz`:

- max abs diff: 0.000e+00
- sum(recon)=-0.083331, sum(persisted)=-0.083331
- allclose(atol=1e-5, rtol=1e-4): True

The filtered `delta_supply_3d.npz` is rebuilt from scratch from the surviving histories (never subtracted in place).

## Reverted trajectory ids (47)

```
1886, 4529, 4817, 3858, 4810, 5276, 5655, 263, 2404, 2771, 2141, 2538, 4073, 8822, 566, 3806, 449, 2881, 9513, 5788, 7986, 6097, 4108, 7727, 5726, 6564, 8474, 988, 5882, 3155, 4191, 4050, 4923, 7887, 951, 5052, 2456, 8618, 9223, 8286, 7026, 6534, 4160, 1595, 8319, 2524, 4561
```

