# Experiment Report - `2026-05-28T08-51-32_k-10000_causal_emphasis_no-dedup`

- **Timestamp (UTC):** 2026-05-28T17:54:28+00:00
- **Git SHA:** `8cb252e`  **(dirty)**
- **Command line:** `/home/robert/FAMAIL/famail_temporal/evaluation/runner.py --name k-10000_causal_emphasis_no-dedup -k 10000 --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1`

## Config

| Param | Value |
|---|---|
| ACTIVE_SUPPLY_THRESHOLD | 0.5 |
| **ALPHA_CAUSAL** | **0.7** |
| **ALPHA_FIDELITY** | **0.1** |
| **ALPHA_SPATIAL** | **0.2** |
| ANNEAL_TEMPERATURE | True |
| CACHE_DIR | /home/robert/FAMAIL/famail_temporal/cache |
| CONVERGENCE_TOL | 1e-06 |
| DEFAULT_SEED | 42 |
| DEMAND_FLOOR | 0.5 |
| DEMOGRAPHIC_FEATURES | ['AvgHousingPricePerSqM', 'GDPperCapita', 'CompPerCapita'] |
| DIAGNOSTICS_ENABLED | True |
| DISCRIMINATOR_CHECKPOINT_DIR | /home/robert/FAMAIL/famail_temporal/discriminator_checkpoints |
| DISCRIMINATOR_CHECKPOINT_FILENAME | default/best.pt |
| EPS | 1e-08 |
| EPSILON_BALL | 2.0 |
| GRID_DIMS | [48, 90] |
| MAX_ITERATIONS | 50 |
| MIN_ACTIVE_UNITS_PER_BLOCK | 10 |
| MIN_TOTAL_ACTIVE_UNITS | 100 |
| N_TIME_BUCKETS | 288 |
| PACKAGE_ROOT | /home/robert/FAMAIL/famail_temporal |
| PATIENCE | 10 |
| SOFT_NEIGHBORHOOD_SIZE | 5 |
| SOURCE_DATA_DIR | /home/robert/FAMAIL/famail_temporal/source_data |
| STEP_SIZE_ALPHA | 0.1 |
| SUPPLY_FLOOR | 0.1 |
| T | 24 |
| TAU_MAX | 1.0 |
| TAU_MIN | 0.1 |
| TIME_BLOCKS | [['hour_00', 0, 1], ['hour_01', 1, 2], ['hour_02', 2, 3], ['hour_03', 3, 4], ['hour_04', 4, 5], ['hour_05', 5, 6], ['hour_06', 6, 7], ['hour_07', 7, 8], ['hour_08', 8, 9], ['hour_09', 9, 10], ['hour_10', 10, 11], ['hour_11', 11, 12], ['hour_12', 12, 13], ['hour_13', 13, 14], ['hour_14', 14, 15], ['hour_15', 15, 16], ['hour_16', 16, 17], ['hour_17', 17, 18], ['hour_18', 18, 19], ['hour_19', 19, 20], ['hour_20', 20, 21], ['hour_21', 21, 22], ['hour_22', 22, 23], ['hour_23', 23, 24]] |

## Dataset

| n_trajectories | n_drivers | n_active_units | k_modified |
|---|---|---|---|
| 105401 | 50 | 34524 | 3773 |

## Fairness

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `f_spatial` | 0.0822 | 0.0824 | +0.0003 up |
| `f_causal` | 0.8052 | 0.8180 | +0.0128 up |
| `gini_dsr` | 0.9384 | 0.9378 | -0.0006 down |
| `gini_asr` | 0.8973 | 0.8973 | +0.0000 - |

## Convergence

- Converged: 3765 / 3773
- Mean total iterations: 21.84
- Mean final gradient norm: 0.0000

## Top 10 modified trajectories

| rank | trajectory_id | driver_id | original_pickup_cell_x | original_pickup_cell_y | modified_pickup_cell_x | modified_pickup_cell_y | delta_x | delta_y | converged | total_iterations | initial_objective | final_objective |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25471 | 14 | 13 | 35 | 13 | 33 | 0 | -2 | True | 38 | 0.6705049872398376 | 0.670516848564148 |
| 2 | 66447 | 31 | 13 | 35 | 13 | 33 | 0 | -2 | True | 27 | 0.672407865524292 | 0.6724188327789307 |
| 3 | 67558 | 31 | 15 | 38 | 14 | 37 | -1 | -1 | True | 46 | 0.6706266403198242 | 0.6706382632255554 |
| 4 | 99987 | 46 | 15 | 38 | 15 | 37 | 0 | -1 | True | 46 | 0.6735671758651733 | 0.6735865473747253 |
| 5 | 65729 | 31 | 13 | 35 | 12 | 35 | -1 | 0 | True | 44 | 0.6721726059913635 | 0.6721892356872559 |
| 6 | 66568 | 31 | 14 | 39 | 15 | 37 | 1 | -2 | True | 46 | 0.6717105507850647 | 0.6717321276664734 |
| 7 | 78198 | 38 | 14 | 39 | 15 | 37 | 1 | -2 | True | 38 | 0.6709097623825073 | 0.6709244251251221 |
| 8 | 2142 | 1 | 14 | 38 | 15 | 37 | 1 | -1 | True | 44 | 0.6703882813453674 | 0.6704109907150269 |
| 9 | 56330 | 27 | 15 | 38 | 15 | 37 | 0 | -1 | True | 38 | 0.67188560962677 | 0.6718931198120117 |
| 10 | 56863 | 27 | 14 | 39 | 15 | 37 | 1 | -2 | True | 39 | 0.6732690334320068 | 0.6732887029647827 |

## Key findings

- F_spatial improved by +0.0003.
- F_causal improved by +0.0128.
- ASR Gini unchanged - only pickups are modified by the framework.

## Artifacts

| Artifact | Path | Size (bytes) |
|---|---|---:|
| augmented_trajs_after | `augmented_trajs_after.pkl` | 97571123 |
| augmented_trajs_before | `augmented_trajs_before.pkl` | 97571123 |
| grid_after | `grid_after.pkl` | 1763321 |
| grid_before | `grid_before.pkl` | 1763321 |
| histories | `histories.pkl` | 14496105 |
| modified_trajectory_ids | `modified_trajectory_ids.json` | 349234 |
| per_unit_attribution_csv | `per_unit_attribution.csv` | 6990352 |
| trajectories_csv | `trajectories.csv` | 856127 |

