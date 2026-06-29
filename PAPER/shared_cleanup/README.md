# shared_cleanup — data cleanup & F_spatial (demographic-independent)

**Purpose.** The stuck-GPS data-cleanup artifacts and the spatial-fairness (F_spatial) decomposition. These are
**demographic-INDEPENDENT**: F_spatial uses grid channel 0 (spatial attribution), and the cleanup is a filter on raw
GPS pickup sinks — neither depends on the F_causal demographic feature set. **These artifacts are therefore valid for
all three feature sets** and are factored out here rather than duplicated three times.

## What the cleanup did

Raw Shenzhen taxi data contained per-driver "stuck-GPS" pickup sinks: single driver plates parked at one cell
emitting thousands of phantom "pickups" with almost no matching drop-offs (a GPS/meter artifact, not real demand).
A signature rule (n_pickups ≥ 1000 ∧ dropoff_ratio < 0.02 per (plate, rounded-coord)) flagged **10 calibrated sink
cells across 9 driver plates**; filtering them removed **106,677 phantom pickups**. The PI decided (Meeting 40) to
filter all of them and re-run the full pipeline.

## Contents

### `tables/`
| file | content | source |
|---|---|---|
| `dataset_summary.md` | dirty-vs-clean removal stats (removal rate, phantom pickups, 10 sink cells) | `source_data{,_dirty}/processing_metadata.json` via `analysis/dataset_summary.py` |
| `sink_f_spatial_decomposition.md` | per-sink share of the F_spatial recovery (headline sink dominates locally) | the two editor runs' `grid_before.pkl` (channel 0) via `analysis/sink_decomposition.py` |
| `cleanup_delta_editor.csv` | editor-level dirty-vs-clean F_spatial / F_causal delta | the two editor `metrics.json` |
| `experiment_cleanup_delta.md` | dirty-vs-clean L1/L2/wbc/variance headline numbers (cleanup changed no conclusion) | the experiment dirs vs the dirty baselines via `analysis/experiment_delta.py` |

### `figures/`
| file | shows | source |
|---|---|---|
| `sink_spatial_attr_before_after.png` | per-cell spatial αᵢ dirty vs cleaned, sinks circled (South-at-bottom) | the two editor runs' `grid_before.pkl` |

### `data/`
| file | source |
|---|---|
| `dataset_summary.json` | `results/analysis/dataset_summary/` |
| `sink_f_spatial_decomposition.json` | `results/analysis/sink_decomposition/` |

## Read-this-before-citing notes

- **Removal rate denominator.** `removal_rate = n_removed / total_extracted`, where `total_extracted = seeking +
  driving` trajectories (dirty 0.4975, clean 0.3895). It is over **all** extracted trajectories — **not** seeking
  alone. Do not read "38.95%" as "39% of *seeking* trips removed" (the seeking-only fraction is ~90%, and even that
  over-attributes because some removed trajectories are driving). `dataset_summary.md` now lists `total_extracted`
  explicitly so the denominator is visible.
- **Per-cell vs net F_spatial (no 416% paradox).** The headline sink at grid **(29,53)** recovers **+0.0885
  locally** (its per-cell F_spatial contribution), but the **net global** F_spatial recovery is only **+0.0213**. The
  difference is a **−0.0783 redistribution residual** spread across non-sink cells: the 10 sinks sum to +0.0996, and
  reallocating their removed mass pulls back −0.0783, netting +0.0213. So the sink's local gain is ~4× the net
  recovery *by construction*; it is not an inconsistency. `sink_f_spatial_decomposition.md` shows the full
  reconciliation.
- **F_causal in `experiment_cleanup_delta.md` is 3-feature.** The dirty-vs-clean experiment comparison was validated
  under the original 3-feature set {housing, GDP, comp} (the set in force when the cleanup was decided). It is an
  apples-to-apples comparison at a constant feature set, and the dirty-vs-clean conclusions are feature-set-invariant
  — but its absolute F_causal values (e.g. clean edited 0.8193) are **not** comparable to the other feature sets'
  headline tables. The file header states this. The unweighted `edited` (w=1) arm's n=6-floor significance flips
  dirty→clean (direction preserved); the report discloses this rather than hiding it.
- **F_spatial is robust to the cleanup direction.** Net F_spatial recovery is small and positive; the editor's
  spatial-fairness conclusions are unchanged by the cleanup.
