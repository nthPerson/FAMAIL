# `data/` — Data pipeline: producer (raw GPS → source datasets) and consumer (source → cache tensors)

## Purpose

Everything the rest of `famail_temporal/` needs in order to load data sits under this directory.
Two distinct responsibilities live side by side:

| Side | What it does | Input | Output |
|---|---|---|---|
| **Producer** (`source_generation/`) | Takes raw GPS pickle files and generates the 8 source datasets that land in `famail_temporal/source_data/` | `raw_data/taxi_record_*.pkl` (monthly GPS records) | `famail_temporal/source_data/*.pkl` (pickup_dropoff, active_taxis, trajectories, multi-stream, profile, calendars, driver mapping) + `processing_metadata.json` |
| **Consumer** (the rest of this directory) | Aggregates the 8 source datasets into the canonical `(48, 90, T)` tensors and the active-unit index, then exposes them via `DataBundle` | `famail_temporal/source_data/*.pkl` (the producer's outputs) | `DataBundle` instance; writes intermediate tensors to `cache/` |

The producer and consumer are intentionally decoupled:

- The producer is one-shot and offline — regenerate only when raw GPS or convention changes.
- The consumer is load-time and online — called at the start of every experiment via `DataBundle.load()`.
- **Nothing downstream reads raw GPS directly.** All of `algorithm/`, `fairness/`, `evaluation/` see their inputs through `DataBundle`.

> **Serialization note.** Both sides use Python `.pkl` files, since this is the format already consumed and produced by the project's trusted tooling. All `.pkl` I/O in this directory is against project-internal paths only; no code here deserializes files from external or untrusted sources.

---

## Directory layout

```
famail_temporal/data/
├── README.md                        # this file
├── __init__.py
│
│   ── CONSUMER SIDE ────────────────────────────────────────────────
├── loader.py                        # DataBundle dataclass + .load() entry point
├── aggregation.py                   # raw .pkl → (48, 90, T) tensors; hour↔block mapping
├── active_mask.py                   # UnitIndexMap; compute_active_mask() filter
├── cache_io.py                      # typed save/load helpers with config-encoded names
├── demographics.py                  # demographic feature loading + validation
│
│   ── PRODUCER SIDE ────────────────────────────────────────────────
└── source_generation/               # unified raw-GPS → source-dataset tool
    ├── README.md                    # producer-side architecture
    ├── SOURCE_DATASET_GENERATION_QUICKSTART.md   # researcher-facing guide
    ├── __main__.py                  # `python -m famail_temporal.data.source_generation`
    ├── cli.py                       # run_generation + argparse entry
    ├── config.py                    # constants (grid, time, neighborhood, day filter)
    ├── raw_loader.py                # load + concat taxi_record_*.pkl → DataFrame
    ├── quantization.py              # gps_to_grid, seconds_to_time_bucket, timestamp_to_day
    ├── transitions.py               # per-driver passenger-indicator transition detection
    ├── event_stream.py              # build the single enriched event-stream DataFrame
    ├── views/                       # per-output-file view modules (see views/README.md)
    ├── removal.py                   # RemovalRecord + RemovalSummary dataclasses
    ├── invariants.py                # per-trajectory + systemic invariant enforcement
    ├── writer.py                    # serialize the 10 output artifacts
    └── tests/                       # TDD unit + golden end-to-end tests
```

---

## Consumer-side files

| File | Role |
|---|---|
| `loader.py` | `DataBundle` dataclass and `.load()` class method — the single entry point for the rest of the system |
| `aggregation.py` | `source_data/*.pkl` → `(48, 90, T)` tensors; `hour_to_block_index()` helper |
| `active_mask.py` | `UnitIndexMap` dataclass; `compute_active_mask()` which applies the two-rule filter |
| `cache_io.py` | Typed save/load helpers that encode config parameters into filenames |
| `demographics.py` | Loads `cell_demographics.pkl`; validates that every active cell has finite demographic values |

---

## Consumer-side key design choices

### 1. Canonical active-unit ordering (cell-major, block-within-cell)

The set of active `(cell, t)` units is enumerated once at preprocess time in a deterministic order:
cells traversed in row-major order (x=0..47, y=0..89, flat index 0..4319), and within each cell
the T time blocks are listed in block order (0..T-1). This ordering is serialized in
`cache/unit_index_map_*.pkl` and asserted at every load boundary:

```
unit_map.n_units == hat_matrices['I_minus_H_demo'].shape[0]
```

The `UnitIndexMap.flat_lookup` array (shape `48*90*T`) maps every possible `(cell, t)` to its
index in `[0, N)`, or `-1` if inactive. All arrays in R^N — pickup counts, supply ratios,
attribution scores, hat-matrix rows — share this ordering.

### 2. Unified mean-hourly aggregation

All three base tensors use the same aggregation rule: sum 5-minute buckets to hourly within each
time block, mean across hours in the block, then mean across weekdays. The result is a
**mean-hourly rate** for each `(cell, t)`:

| Tensor | Semantic |
|---|---|
| `pickup_3d` | Mean hourly pickups per `(cell, block)` |
| `dropoff_3d` | Mean hourly dropoffs per `(cell, block)` |
| `active_taxis_3d` | Mean hourly active taxis per `(cell, block)` |

Gini is scale-invariant, so `F_spatial` is numerically identical regardless of whether
sum- or mean-aggregated tensors are used. `F_causal` requires `g_0(D)` to be fit at the same
scale D is evaluated at — the re-fit at block-mean scale (rather than reusing the fit from the
V2 codebase) eliminates the dual-tensor design where metrics ran at different scales.

### 3. Active filter at `(cell, t)` granularity

A unit `(c, t)` is active iff all three conditions hold:

1. `active_taxis_3d[c, t] > ACTIVE_SUPPLY_THRESHOLD` (default 0.5)
2. `valid_mask[c]` is True — cell is inside the Shenzhen boundary (from
   `source_data/grid_to_district_mapping.pkl`)
3. No `NaN` in any selected demographic feature for cell `c`

The `DEMAND_FLOOR = 0.01` is applied inside `Y = S/D` computation (not as an activity
criterion), so near-zero demand in an active unit does not cause Y to explode.

This is a temporal generalization of the current 2D filter: a cell that has taxis only during
morning peak will have morning-peak units active and other-block units inactive — the fairness
metrics see only the time-blocks where service is actually present.

### 4. `g_0(D)` is re-fit at block-mean scale

`g_0(D)` approximates the baseline `Y = S/D` relationship attributable to pure demand response
(ignoring demographics). Fitting at the same scale as evaluation ensures the hat-matrix
projection in `F_causal` operates on residuals with mean zero. Using a power basis
`[1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)]` captures the hyperbolic `Y ~ a/D` shape with four
linear parameters compatible with the hat-matrix algebra in `fairness/`.

---

## Producer-side overview

`source_generation/` is the one tool that turns raw taxi GPS pickles (`raw_data/taxi_record_07_50drivers.pkl`, `_08_`, `_09_`) into all 8 source datasets + driver-index mapping + processing metadata. It replaces three legacy tools (`pickup_dropoff_counts/`, `active_taxis/`, `new_all_trajs/`) with a single pipeline whose cross-file consistency holds by construction: every output derives from one enriched event-stream DataFrame produced in one pass.

The full design rationale lives in [`docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md`](../../docs/superpowers/specs/2026-04-20-unified-source-data-generation-design.md). The producer-side architecture (single event stream → deterministic views) is documented in [`source_generation/README.md`](source_generation/README.md). For running the tool and interpreting its outputs, see [`source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md`](source_generation/SOURCE_DATASET_GENERATION_QUICKSTART.md).

**When do you regenerate?** Only when one of:
- Raw GPS data is replaced or updated.
- A convention decision changes (e.g., day filter, active-taxi semantic, trajectory endpoint).
- An upstream bug is discovered and fixed.

Regeneration triggers a required follow-up: re-run `python -m famail_temporal.preprocess --force` so the cache is rebuilt from the new source files, and retrain the v3 discriminator on the new multi-stream files before running any F_fidelity experiments.

---

## Consumer API surface

```python
from famail_temporal.data.loader import DataBundle, UnitIndexMap
from famail_temporal.data.aggregation import hour_to_block_index

# Load (or rebuild) all preprocessed data
bundle = DataBundle.load()                          # uses cache if current
bundle = DataBundle.load(force_rebuild_cache=True)  # always reprocess

# Navigate the active-unit index
unit_idx = bundle.unit_map.from_cell_time(cell_flat, t_block)  # -1 if inactive
cell, t  = bundle.unit_map.to_cell_time(unit_idx)

# Map a raw time bucket to a block index
t_block = hour_to_block_index(time_bucket_index)  # int in [0, T)
```

**`DataBundle` fields (frozen dataclass):**

| Field | Shape / Type | Description |
|---|---|---|
| `pickup_3d` | (48, 90, T) float32 | Mean hourly pickups |
| `dropoff_3d` | (48, 90, T) float32 | Mean hourly dropoffs |
| `active_taxis_3d` | (48, 90, T) float32 | Mean hourly active taxis |
| `mask_3d` | (48, 90, T) bool | Active-unit mask |
| `unit_map` | `UnitIndexMap` | Canonical ordering |
| `n_hours_per_block` | (T,) int | Hours in each time block |
| `n_days` | int | Weekdays in dataset (e.g., 65) |
| `g0_func` | Callable | Fitted power-basis g_0(D) |
| `hat_matrices` | dict | I_minus_H_demo, M, etc. |
| `trajectories` | List[Trajectory] | 50 drivers' trajectories |
| `multi_stream` | MultiStreamData | Discriminator context inputs |
| `discriminator` | torch.nn.Module | Loaded fidelity model (eval mode) |

---

## Producer API surface

```bash
# CLI — regenerate all 8 source datasets from raw GPS
python -m famail_temporal.data.source_generation \
    --input-dir raw_data/ \
    --output-dir famail_temporal/source_data/
```

```python
# Programmatic
from pathlib import Path
from famail_temporal.data.source_generation.cli import run_generation

result = run_generation(
    input_dir=Path("raw_data/"),
    output_dir=Path("famail_temporal/source_data/"),
)
print(f"Kept: {result.n_seeking_kept} seeking + {result.n_driving_kept} driving")
print(f"Removed: {result.n_removals} trajectories (see processing_metadata.json)")
```

---

## Dependencies

**Consumer side:**
- `config.py` — grid dims, time blocks, thresholds
- `fairness/g0_power_basis.py` — `fit_g0` (called during preprocess)
- `fairness/hat_matrices.py` — `precompute_hat_matrices` (called during preprocess)
- `utils/trajectory.py` — `Trajectory`, `TrajectoryState`
- `fidelity/checkpoint.py` — `load_discriminator`
- Standard library: `pickle`, `pathlib`
- Third-party: `numpy`, `torch`

**Producer side** (see [`source_generation/README.md`](source_generation/README.md) for full list):
- Standard library: `pickle`, `pathlib`, `json`, `subprocess`
- Third-party: `pandas`, `numpy`

No imports from outside `famail_temporal/`.

---

## Paper-section hook

This module corresponds to the **"Data Preparation"** subsection of the Methods section. The
unified mean-hourly aggregation rule and the active-unit filter definition are the two
consumer-side subsections most likely to need prose explanation for reviewers. The producer side
(source-dataset generation) is an appendix-worthy methodological contribution in its own right:
unifying three legacy tools into one consistent pipeline, enforcing a dataset-level invariant
(every trajectory's pickup cell has a non-zero pickup count) by construction, and removing known
indexing inconsistencies between files. The `UnitIndexMap` canonical ordering is an
implementation detail, but may warrant a sentence in the supplementary about reproducibility.
