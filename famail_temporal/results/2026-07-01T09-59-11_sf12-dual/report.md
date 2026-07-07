# Experiment Report - `2026-07-01T09-59-11_sf12-dual`

- **Timestamp (UTC):** 2026-07-01T17:32:08+00:00
- **Git SHA:** `de41d1c`  **(dirty)**
- **Command line:** `/home/robert/FAMAIL/.claude/worktrees/second-dataset-compat/famail_temporal/evaluation/runner.py --name sf12-dual -k 2000 --device cuda --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1`

## Config

| Param | Value |
|---|---|
| ACCEPT_RULE | objective |
| ACTIVE_SUPPLY_THRESHOLD | 0.5 |
| **ALPHA_CAUSAL** | **0.7** |
| **ALPHA_FIDELITY** | **0.1** |
| **ALPHA_SPATIAL** | **0.2** |
| ANNEAL_TEMPERATURE | True |
| CACHE_DIR | /home/robert/FAMAIL/.claude/worktrees/second-dataset-compat/famail_temporal/cache/sf_12 |
| CITY | sf12 |
| CONVERGENCE_TOL | 1e-06 |
| DEFAULT_SEED | 42 |
| DEMAND_FLOOR | 0.5 |
| DEMOGRAPHIC_FEATURES | ['AvgHousingPricePerSqM', 'CompPerCapita', 'MigrantRatio'] |
| DIAGNOSTICS_ENABLED | True |
| DISCRIMINATOR_CHECKPOINT_DIR | /home/robert/FAMAIL/.claude/worktrees/second-dataset-compat/famail_temporal/discriminator_checkpoints |
| DISCRIMINATOR_CHECKPOINT_FILENAME | sf_12/best.pt |
| EPS | 1e-08 |
| EPSILON_BALL | 2.0 |
| EPSILON_CAP | 2.0 |
| GRID_DIMS | [32, 30] |
| ITERATIVE_TOPK_MAX_EDITS | 1 |
| MAX_ITERATIONS | 50 |
| MAX_ROUNDS | 1 |
| MIN_ACTIVE_UNITS_PER_BLOCK | 10 |
| MIN_TOTAL_ACTIVE_UNITS | 100 |
| N_TIME_BUCKETS | 288 |
| PACKAGE_ROOT | /home/robert/FAMAIL/.claude/worktrees/second-dataset-compat/famail_temporal |
| PATIENCE | 10 |
| ROUND_CONVERGENCE_TOL | None |
| ROUND_PATIENCE | 2 |
| SOFT_NEIGHBORHOOD_SIZE | 5 |
| SOURCE_DATA_DIR | /home/robert/FAMAIL/.claude/worktrees/second-dataset-compat/famail_temporal/source_data/second_dataset/sf_source_12 |
| STEP_SIZE_ALPHA | 0.1 |
| STE_ENABLED | False |
| SUPPLY_FLOOR | 0.1 |
| T | 24 |
| TAU_MAX | 1.0 |
| TAU_MIN | 0.1 |
| TIME_BLOCKS | [['hour_00', 0, 1], ['hour_01', 1, 2], ['hour_02', 2, 3], ['hour_03', 3, 4], ['hour_04', 4, 5], ['hour_05', 5, 6], ['hour_06', 6, 7], ['hour_07', 7, 8], ['hour_08', 8, 9], ['hour_09', 9, 10], ['hour_10', 10, 11], ['hour_11', 11, 12], ['hour_12', 12, 13], ['hour_13', 13, 14], ['hour_14', 14, 15], ['hour_15', 15, 16], ['hour_16', 16, 17], ['hour_17', 17, 18], ['hour_18', 18, 19], ['hour_19', 19, 20], ['hour_20', 20, 21], ['hour_21', 21, 22], ['hour_22', 22, 23], ['hour_23', 23, 24]] |

## Dataset

| n_trajectories | n_drivers | n_active_units | k_modified |
|---|---|---|---|
| 10887 | 12 | 4230 | 1371 |

## Fairness

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `f_spatial` | 0.1846 | 0.1817 | -0.0030 down |
| `f_causal` | 0.8752 | 0.8891 | +0.0139 up |
| `gini_dsr` | 0.8266 | 0.8325 | +0.0059 up |
| `gini_asr` | 0.8042 | 0.8042 | +0.0000 - |

## Convergence

- Converged: 1341 / 1371
- Mean total iterations: 25.31
- Mean final gradient norm: 0.0000

## Top 10 modified trajectories

| rank | trajectory_id | driver_id | original_pickup_cell_x | original_pickup_cell_y | modified_pickup_cell_x | modified_pickup_cell_y | delta_x | delta_y | converged | total_iterations | initial_objective | final_objective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7170 | 346 | 23 | 11 | 25 | 10 | 2 | -1 | True | 42 | 0.7446003556251526 | 0.7446298599243164 |
| 2 | 10614 | 488 | 23 | 11 | 24 | 10 | 1 | -1 | True | 50 | 0.7495693564414978 | 0.7496099472045898 |
| 3 | 5424 | 117 | 25 | 5 | 26 | 6 | 1 | 1 | True | 24 | 0.7475480437278748 | 0.7475640773773193 |
| 4 | 9187 | 469 | 25 | 5 | 26 | 6 | 1 | 1 | True | 25 | 0.7434369325637817 | 0.7434508800506592 |
| 5 | 10045 | 476 | 25 | 5 | 26 | 6 | 1 | 1 | True | 24 | 0.7496474385261536 | 0.7496587634086609 |
| 6 | 10480 | 488 | 25 | 5 | 26 | 6 | 1 | 1 | True | 22 | 0.7496594190597534 | 0.7496688365936279 |
| 7 | 3539 | 75 | 27 | 6 | 26 | 6 | -1 | 0 | True | 14 | 0.7494066953659058 | 0.7494029998779297 |
| 8 | 6143 | 148 | 27 | 6 | 26 | 6 | -1 | 0 | True | 11 | 0.7437183260917664 | 0.743717610836029 |
| 9 | 6670 | 148 | 27 | 6 | 26 | 6 | -1 | 0 | True | 11 | 0.7434931993484497 | 0.7434923052787781 |
| 10 | 1753 | 6 | 26 | 10 | 24 | 10 | -2 | 0 | True | 46 | 0.7487958073616028 | 0.7488343119621277 |

## Key findings

- F_spatial regressed by -0.0030.
- F_causal improved by +0.0139.
- ASR Gini unchanged - only pickups are modified by the framework.

## Artifacts

| Artifact | Path | Size (bytes) |
|---|---|---:|
| attribution_distribution | `attribution_distribution.npz` | 50334 |
| augmented_trajs_after | `augmented_trajs_after.pkl` | 6873435 |
| augmented_trajs_before | `augmented_trajs_before.pkl` | 6873435 |
| convergence_curve | `convergence_curve.npz` | 3972 |
| grid_after | `grid_after.pkl` | 392432 |
| grid_before | `grid_before.pkl` | 392432 |
| histories | `histories.pkl` | 5591789 |
| modified_trajectory_ids | `modified_trajectory_ids.json` | 121644 |
| per_unit_attribution_csv | `per_unit_attribution.csv` | 838665 |
| trajectories_csv | `trajectories.csv` | 557588 |

