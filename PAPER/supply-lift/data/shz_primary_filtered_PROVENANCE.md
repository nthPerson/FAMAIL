# Filtered supply-lift results — PROVENANCE

Derived from: `famail_temporal/results/2026-07-08T14-03-03_supply_lift_v1_shz_primary`
Source experiment_id: `2026-07-08T14-03-03_supply_lift_v1_shz_primary`  ·  git_sha: `85c6dbc`
Tool: `famail_temporal.analysis.filter_infeasible_trims`  ·  user decision: 2026-07-08

## Rule

An edit is applied only when a king-compliant repair exists (max(|dx|,|dy|) <= 1 on every consecutive step). Trim edits that fell back to the legacy pickup-only perturbation (tapered-tail repair infeasible) are reverted to their originals, making trim symmetric with lift (which already skips on infeasible).

## Why

The G4 adjacency sweep found exactly 115 modified trajectories that violate king-move adjacency — all trim edits that fell back to the legacy pickup-only perturbation because their tapered-tail repair was infeasible (the G3 trade-off). Lift mode already *skips* such edits; this post-process makes trim symmetric by reverting those 115 trajectories to their originals. After filtering, G4 must be 100% king-compliant.

The published legacy trim numbers remain reproducible via `TAIL_LEN=0`; this tool modifies nothing in the source directory.

**Non-reoptimized survivors:** the surviving 9,885 edits were NOT re-optimized after removing the 115 — their optimization saw the 115's intermediate demand perturbations in the sequential base grid. The filtered grids are exact for "these 9,885 edits applied to base," which is not byte-identical to a from-scratch skip-on-infeasible run. Approved trade-off (2026-07-08) to avoid a multi-hour GPU re-run; the coupling is negligible (the 115 were pickup-only single-cell-mass moves).

## Edit counts

| | source | filtered |
|---|---|---|
| n_trim | 2455 | 2340 |
| n_lift | 7545 | 7545 |
| n_skipped_infeasible_trim | 0 (fell back) | 115 |
| total edits | 10000 | 9885 |

## Fairness metrics (recomputed from filtered histories)

| metric | source before | source after | filtered before | filtered after | filtered Δ |
|---|---|---|---|---|---|
| f_spatial | 0.103427 | 0.103273 | 0.103427 | 0.109785 | +0.006357 |
| f_causal | 0.798795 | 0.819723 | 0.798795 | 0.821013 | +0.022218 |
| gini_dsr | 0.898093 | 0.898583 | 0.898093 | 0.885558 | -0.012535 |
| gini_asr | 0.895053 | 0.894872 | 0.895053 | 0.894873 | -0.000180 |

## Supply totals (filtered ΔS)

- added: 1724.5834
- removed: 1727.3334

## ΔS reconstruction equivalence (load-bearing check)

Rebuilding ΔS from ALL source histories (float32 accumulator, histories order, mirroring `modifier._hard_tail_delta_supply`) reproduces the persisted `delta_supply_3d.npz`:

- max abs diff: 0.000e+00
- sum(recon)=-2.750000, sum(persisted)=-2.750000
- allclose(atol=1e-5, rtol=1e-4): True

The filtered `delta_supply_3d.npz` is rebuilt from scratch from the surviving histories (never subtracted in place).

## Reverted trajectory ids (115)

```
58948, 70351, 52891, 60790, 87881, 71706, 85960, 66557, 90674, 18461, 5350, 62594, 6687, 56504, 58139, 34977, 71434, 10841, 70077, 16602, 35495, 62492, 21703, 48107, 41158, 53544, 6474, 24556, 52029, 36644, 73768, 82281, 1545, 19996, 70103, 7022, 65903, 51378, 81118, 63165, 12478, 3788, 68392, 5537, 49536, 36847, 37000, 28233, 162, 49279, 3347, 23175, 92607, 92070, 79702, 14720, 66875, 17796, 27181, 66931, 63164, 50027, 32666, 22431, 4090, 13902, 42214, 32442, 13096, 77814, 27568, 32143, 68104, 54259, 48878, 77206, 27604, 42118, 62227, 51823, 28777, 88895, 62356, 61546, 90445, 77306, 34120, 196, 52370, 9797, 24302, 59807, 3277, 50398, 51825, 56035, 66779, 2958, 92758, 66945, 20497, 52874, 72585, 78411, 1324, 20983, 49933, 53474, 19770, 88151, 49472, 80048, 48501, 77946, 78851
```

