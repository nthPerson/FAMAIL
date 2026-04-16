# `data/` — Ingestion and canonical active-unit representation

## Purpose

Load raw pickle files from `raw_data/`, aggregate them to the `(48, 90, T)` spatiotemporal grid,
determine which `(cell, t)` units are active, fix a canonical ordering over those units, and expose
everything through the `DataBundle` dataclass. All downstream modules receive their inputs from
`DataBundle.load()` — nothing reads raw files directly.

---

## Files

| File | Role |
|---|---|
| `loader.py` | `DataBundle` dataclass and `.load()` class method — the single entry point for the rest of the system |
| `aggregation.py` | `raw_data/*.pkl` → `(48, 90, T)` tensors; `hour_to_block_index()` helper |
| `active_mask.py` | `UnitIndexMap` dataclass; `compute_active_mask()` which applies the two-rule filter |
| `cache_io.py` | Typed save/load helpers that encode config parameters into filenames |

---

## Key design choices

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

## API surface

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

## Dependencies

- `config.py` — grid dims, time blocks, thresholds
- `fairness/g0_power_basis.py` — `fit_g0` (called during preprocess)
- `fairness/hat_matrices.py` — `precompute_hat_matrices` (called during preprocess)
- `utils/trajectory.py` — `Trajectory`, `TrajectoryState`
- `fidelity/checkpoint.py` — `load_discriminator`
- Standard library: `pickle`, `pathlib`
- Third-party: `numpy`, `torch`

No imports from outside `famail_temporal/`.

---

## Paper-section hook

This module corresponds to the **"Data Preparation"** subsection of the Methods section. The
unified mean-hourly aggregation rule and the active-unit filter definition are the two subsections
most likely to need prose explanation for reviewers. The `UnitIndexMap` canonical ordering is an
implementation detail, but may warrant a sentence in the supplementary about reproducibility.
