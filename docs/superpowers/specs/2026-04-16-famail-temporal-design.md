# FAMAIL-Temporal Design Specification

**Date:** 2026-04-16
**Status:** Draft — awaiting user review
**Target directory:** `famail_temporal/` (new, standalone)
**Supersedes (does not modify):** the existing `trajectory_modification/` and `objective_function/` modules in this repository

---

## 1. Executive summary

`famail_temporal/` is a ground-up, dependency-free rewrite of the FAMAIL trajectory modification algorithm. Its purpose is to extend the fairness metrics along the time axis, so that the algorithm can distinguish fairness at the `(cell, time_block)` level rather than only at the `(cell)` level. It also serves as the paper-replication codebase: concise, heavily documented, and self-contained.

### Scope at a glance

| Area | Decision |
|---|---|
| Fidelity term | Included — opaque pre-trained checkpoint ported into the new directory |
| Time-block granularity | `T = 4` default: morning peak, midday, evening peak, night; configurable |
| Fairness aggregation | **Pooled** — one Gini / one F_causal over all active `(cell, t)` units |
| Active-unit filter | `(cell, t)` granularity with `active_taxis > 0.5` threshold |
| Causal formulation | Option B (demographic hat-matrix projection) — only formulation |
| Attribution | Per-unit Option B decomposition → per-trajectory via pickup inheritance |
| Perturbation target | Pickup-only (terminal state of trajectory) |
| Cache strategy | Config-encoded filenames in `cache/` + cache `README.md` |
| Testing | Real tests (no "skip because it's research code" excuses) |
| Dashboard | Deferred until algorithm is verified |
| Directory name | `famail_temporal/` |

### Why this rewrite exists

The current `trajectory_modification/` and `objective_function/` modules aggregate all fairness metrics to a 2D `(48, 90)` grid — the temporal dimension (day, time-of-day) is collapsed before the objective sees it. This prevents the algorithm from detecting fairness disparities that manifest only at certain times of day (e.g., night-time underservice of demographically-distinct neighborhoods). The rewrite lifts this limitation by treating each `(cell, time_block)` as the fundamental unit of fairness analysis, while preserving the soft-cell-assignment gradient-flow design and the ST-iFGSM modification algorithm.

Secondary motivation: the current codebase has accumulated ~4,000 lines in a single `utils.py`, multiple causal formulations behind string dispatches, and layered stability fixes that are hard to audit. The rewrite collapses these into a single, interpretable implementation suitable for paper publication.

---

## 2. Design decisions (recorded during brainstorming)

Decisions that shaped the design, in order of discussion:

| # | Decision | Rationale |
|---|---|---|
| 1 | Include F_fidelity; port the discriminator as an opaque pre-trained checkpoint | Preserves the full three-term objective without pulling in thousands of lines of training code |
| 2 | `T=4` time blocks (morning peak, midday, evening peak, night), configurable via `config.py` | Maps to transport-research commute regimes; reviewer-legible; memory-feasible for pooled hat matrices |
| 3 | Active filtering at `(cell, t)` granularity, `supply > 0.5`, `DEMAND_FLOOR = 0.01` | Natural generalization of current 2D filter; catches time-specific underservice |
| 4 | Attribution: per-unit via Option B decomposition, per-trajectory via pickup `(cell, t)` inheritance | Per-unit scores sum to `1 - F_causal` — a publishable decomposition property |
| 5 | Perturbation target: pickup-only | Moderates complexity; path-aware attribution deferred to later work |
| 6 | Re-fit `g₀(D)` on block-mean scale; unify to one aggregation rule | Eliminates dual-tensor design and runtime rescale; cleaner math |
| 7 | Cache: config-encoded filenames + `cache/README.md` | Prevents silent staleness without automatic invalidation machinery |
| 8 | Tests: math invariants + bug-class guards + integration; fast tests in <10s | Catches the classes of bugs the current codebase hit; promotes refactor confidence |
| 9 | Dashboard: deferred | Keeps scope tight; algorithm verification first |
| 10 | Directory: `famail_temporal/`; sub-modules grouped by concern | Matches the paper's narrative structure |
| 11 | `fairness/causal_option_b.py` → `fairness/causal.py`; `checkpoints/` → `discriminator_checkpoints/` | No formulation qualifier needed when there's only one formulation |
| 12 | Per-subdirectory `README.md` files linked from top-level `README.md` | Self-documenting codebase; seeds future paper sections |

---

## 3. Architecture

### 3.1 The pooled unit abstraction

A **unit** is a pair `(c, t)` where `c ∈ [0, 48×90)` is a flattened cell index and `t ∈ [0, T)` is a time block. A unit is **active** iff its mean active-taxis within that time block exceeds `ACTIVE_SUPPLY_THRESHOLD = 0.5` and the cell is within the Shenzhen valid mask.

Let `N = |active units|`. For `T=4` and Shenzhen, `N` is empirically expected to be `6,000–8,000`. All downstream math — Gini over `ℝ^N`, hat matrices in `ℝ^{N×N}`, attribution in `ℝ^N` — operates on this single vector of active units.

### 3.2 End-to-end data flow

```
raw_data/*.pkl
      │
      ▼
preprocess.py  (one-time)
      │
      ├─► cache/pickup_counts_T4.pkl             (48, 90, T) float32
      ├─► cache/dropoff_counts_T4.pkl            (48, 90, T) float32
      ├─► cache/active_taxis_T4.pkl              (48, 90, T) float32
      ├─► cache/active_mask_T4_thr0.5.pkl        (48, 90, T) bool, n_active = N
      ├─► cache/unit_index_map_T4_thr0.5.pkl     canonical ordering
      ├─► cache/hat_matrices_T4_thr0.5_feat-*.pkl  dict of (N, N) arrays
      └─► cache/g0_power_basis_T4_thr0.5.pkl     fitted coefficients
      │
      ▼
data.loader.DataBundle.load()
      │
      ▼
algorithm.objective.FAMAILObjective(bundle)
   ├─ fairness.spatial.compute_fspatial   (pooled Gini over active units)
   ├─ fairness.causal.compute_fcausal     (pooled Option B over active units)
   └─ fidelity.compute.compute_ffidelity  (discriminator similarity)
      │
      ▼
algorithm.modifier.TrajectoryModifier    (ST-iFGSM; perturbs pickup only)
      │
      ▼
algorithm.attribution                    (per-unit → per-trajectory ranking)
```

### 3.3 Architectural invariants

1. **One active-unit ordering.** The flattened active unit vector has a canonical order (cell-major, then time-block) fixed at preprocess time. All arrays that live in `ℝ^N` — `R`, `Y`, `D`, `S`, demographics, hat matrices — use this order.
2. **Single grid↔unit conversion point.** The `(48, 90, T) → (N,)` gather happens in exactly one place: `algorithm/objective.py::forward()`. `fairness/` modules never see grid dimensions.
3. **Gradient flow only through `pickup_counts`.** The only tensor that varies during ST-iFGSM is `soft_pickup_3d`, and only in the `[:, :, t*]` slice where `t*` is the pickup's time block. All other inputs are frozen.
4. **No external dependencies.** No imports from outside `famail_temporal/`. Only `torch`, `numpy`, `scikit-learn`, `pytest`.

---

## 4. Configuration — `config.py`

Single source of truth for every reviewer-visible knob. Cache filenames encode the values of this config so multiple configurations coexist without invalidation.

Key values:

```python
# Grid (fixed by dataset)
GRID_DIMS = (48, 90)
N_TIME_BUCKETS = 288

# Time blocks — configurable, night wraparound encoded as end_hour > 24
TIME_BLOCKS = [
    ("morning_peak", 7, 10),
    ("midday",      10, 16),
    ("evening_peak",16, 20),
    ("night",       20, 31),   # 31 = 24 + 7
]
T = len(TIME_BLOCKS)

# Active filter
ACTIVE_SUPPLY_THRESHOLD = 0.5
DEMAND_FLOOR = 0.01
SUPPLY_FLOOR = 0.1

# Demographics used in the Option B hat matrix
DEMOGRAPHIC_FEATURES = ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita"]

# Objective weights
ALPHA_SPATIAL, ALPHA_CAUSAL, ALPHA_FIDELITY = 0.33, 0.33, 0.34

# ST-iFGSM
STEP_SIZE_ALPHA = 0.1
EPSILON_BALL = 2.0
MAX_ITERATIONS = 50
CONVERGENCE_TOL = 1e-6

# Soft cell assignment
SOFT_NEIGHBORHOOD_SIZE = 5
TAU_MAX, TAU_MIN = 1.0, 0.1
ANNEAL_TEMPERATURE = True

# Numerical stability
EPS = 1e-8
MIN_ACTIVE_UNITS_PER_BLOCK = 10
MIN_TOTAL_ACTIVE_UNITS = 100

def cache_suffix(include_features: bool = False) -> str:
    ...   # builds "T4_thr0.5" or "T4_thr0.5_feat-housing-gdp-comp"
```

---

## 5. Data pipeline

### 5.1 Aggregation rules (unified)

All three base tensors use the same aggregation rule. `g₀(D)` is re-fit at block-mean scale during preprocessing.

| Tensor             | Within-block rule                     | Across-days rule | Semantic                                   |
|--------------------|---------------------------------------|------------------|--------------------------------------------|
| `pickup_3d`        | SUM 5-min → hourly, MEAN across hours | MEAN             | Mean hourly pickups per `(cell, block)`    |
| `dropoff_3d`       | Same                                  | MEAN             | Mean hourly dropoffs per `(cell, block)`   |
| `active_taxis_3d`  | MEAN across hours                     | MEAN             | Mean hourly active taxis per `(cell, block)` |

Gini is scale-invariant, so F_spatial is numerically identical regardless of whether sum- or mean-aggregated tensors are used. F_causal requires g₀(D) to be fit at the same scale D is evaluated at — hence the re-fit.

### 5.2 `UnitIndexMap` (canonical active-unit ordering)

```python
@dataclass(frozen=True)
class UnitIndexMap:
    cell_indices:       np.ndarray   # (N,) int32  flat cell id
    time_block_indices: np.ndarray   # (N,) int8   block id
    flat_lookup:        np.ndarray   # (48·90·T,) int32, inactive = -1
    n_units:            int
    n_active_cells:     int
    units_per_block:    np.ndarray   # (T,) int

    def to_flat_cell(self, unit_idx) -> int: ...
    def to_time_block(self, unit_idx) -> int: ...
    def to_cell_time(self, unit_idx) -> tuple[int, int]: ...
    def from_cell_time(self, cell, t) -> int: ...     # -1 if inactive
```

Ordering rule: **cell-major, then time-block** within each cell. This ordering is built once at preprocess time, serialized, and asserted at every load boundary.

### 5.3 Active-unit filter

A unit `(c, t)` is active iff:
1. `active_taxis_3d[c, t] > ACTIVE_SUPPLY_THRESHOLD`, AND
2. `valid_mask[c]` is True (cell is inside Shenzhen, from `grid_to_district_mapping.pkl`), AND
3. No `NaN` in any selected demographic feature for `c`.

The demand floor `DEMAND_FLOOR = 0.01` is applied inside the Y = S/D computation (not as an activity criterion) so near-zero demand in an active unit doesn't explode Y.

### 5.4 `DataBundle`

```python
@dataclass(frozen=True)
class DataBundle:
    # Base tensors (mean-hourly-aggregated)
    pickup_3d:        np.ndarray    # (48, 90, T) float32
    dropoff_3d:       np.ndarray    # (48, 90, T) float32
    active_taxis_3d:  np.ndarray    # (48, 90, T) float32
    mask_3d:          np.ndarray    # (48, 90, T) bool
    unit_map:         UnitIndexMap

    # Aggregation metadata (used by modifier to scale pickup deltas correctly)
    n_hours_per_block: np.ndarray   # (T,) int — hours covered by each block
    n_days:            int          # days in the dataset (e.g., 65 weekdays)

    # Causal machinery
    g0_func:          Callable
    hat_matrices:     Dict[str, np.ndarray]   # {'I_minus_H_demo', 'M', ...}

    # Trajectories and multi-stream context
    trajectories:     List[Trajectory]
    multi_stream:     MultiStreamData

    # Discriminator (loaded from discriminator_checkpoints/)
    discriminator:    torch.nn.Module

    @classmethod
    def load(cls, force_rebuild_cache: bool = False) -> "DataBundle": ...
```

#### `MultiStreamData`

A small frozen dataclass defined in `fidelity/context.py`, bundling the five multi-stream inputs:

```python
@dataclass(frozen=True)
class MultiStreamData:
    driving_trajs:    Dict[int, List]          # {driver_idx: [trajs]}, 1-indexed coords
    seeking_trajs:    Dict[int, List]          # {driver_idx: [trajs]}, 1-indexed coords
    profile_features: Dict[int, np.ndarray]    # {driver_idx: ndarray(11,)}, z-score normalized
    seeking_days:     Dict[int, List[int]]     # {driver_idx: [cal_day_per_traj]}
    driving_days:     Dict[int, List[int]]     # {driver_idx: [cal_day_per_traj]}
```

Lifted from the current `MultiStreamDataLoader.load()` return dict; wrapping it in a dataclass makes the `DataBundle` field typed and immutable.

---

## 6. Fairness metrics

### 6.1 F_spatial (pooled Gini)

Inputs are length-N vectors already restricted to active units.

```
DSR = pickup_N / active_taxis_N
ASR = dropoff_N / active_taxis_N
F_spatial = 1 - 0.5·(Gini(DSR) + Gini(ASR))
```

Gini computed via the differentiable pairwise formula `G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ)`.

### 6.2 F_causal (pooled Option B)

```
D = max(demand_N, DEMAND_FLOOR)
Y = supply_N / D
R = Y - g₀(D)
F_causal = R'(I - H_demo)R / R'MR
```

Where `H_demo = X_demo (X_demo' X_demo)⁻¹ X_demo'` projects onto the standardized demographic feature space (with intercept), and `M = I - 11'/N` is the centering matrix.

### 6.3 Per-unit attribution

Under Option B, `1 - F_causal = r²_demo` admits the exact decomposition

```
r²_demo = Σᵢ [(MR)ᵢ² - ((I-H)R)ᵢ²] / R'MR
```

each term being unit `i`'s contribution to demographic-explained variance. The sum property holds because both `M` and `(I - H)` are idempotent. A signed version `signed_attribution_i = attribution_i · sign((HR)ᵢ)` distinguishes over- vs. under-service.

### 6.4 g₀(D) — power basis fit

Power basis `[1, 1/(D+1), 1/√(D+1), √(D+1)]` captures the hyperbolic `Y ≈ a/D` relationship with four linear parameters. Fit at the active-unit block-mean scale during preprocess. Isotonic regression is also fit for diagnostic comparison; the two must agree within a configurable tolerance.

---

## 7. Fidelity integration

### 7.1 Port scope

From `discriminator/model/model.py` (1,297 lines, 8 classes), only port:
- `FeatureNormalizer`
- `SiameseLSTMEncoder`
- `ProfileEncoder`
- `MultiStreamSiameseDiscriminator`

All training-mode branches, training loops, dataset classes, and the 5 deprecated architectures are excluded.

### 7.2 The cuDNN backward-in-inference workaround

cuDNN's RNN backward requires training mode, but we need inference-mode behavior (no dropout) while still allowing gradient flow through the LSTM for ST-iFGSM. The port preserves the `torch.backends.cudnn.flags(enabled=False)` context manager around the discriminator forward pass so gradients flow correctly through the LSTM in inference mode.

### 7.3 Multi-stream context (Decisions 1–4 preserved from current code)

- Decision 1: Both Siamese branches represent the same driver
- Decision 2: Seeking fill strategy = 'sample' (N=5 seeking trajectories, slot 0 is the target)
- Decision 3: Coordinate conversion — V3 trained on 1-indexed coords, modifier is 0-indexed; add +1 when injecting
- Decision 4: Gradient flow through slot 0 of `x2` only

### 7.4 Checkpoint handling

Checkpoint is loaded once at `DataBundle.load()`, set to inference mode via PyTorch's standard `.eval()` convention, and all parameters are frozen with `requires_grad=False`. The canonical checkpoint is `discriminator_checkpoints/default/best.pt`; provenance is recorded in `discriminator_checkpoints/README.md`.

---

## 8. Algorithm

### 8.1 Soft cell assignment

Unchanged in essence from current implementation: Gaussian softmax over a `(2k+1) × (2k+1)` neighborhood with temperature annealing from `TAU_MAX` to `TAU_MIN`. The output is still a 2D probability distribution; the caller places it into `soft_pickup_3d[:, :, t_block]` via the **delta-tensor pattern**:

```
delta = torch.zeros_like(base_3d)
# fill delta[:, :, t_block] with probs from SoftCellAssignment
out = base_3d + delta    # autograd-safe
```

#### Mass-balance under mean-hourly aggregation

Because `pickup_3d` is mean-hourly-aggregated (not raw integer counts), a single trajectory's pickup contributes an amount `pickup_mass = 1.0 / (n_hours_per_block[t*] × n_days)` to `pickup_3d[c, t*]`, not `1.0`. The modifier therefore scales both the subtraction and the soft distribution by `pickup_mass`:

```
pickup_mass = 1.0 / (bundle.n_hours_per_block[t*] * bundle.n_days)

# Subtract trajectory's contribution at original cell
delta[orig_cx, orig_cy, t*] -= pickup_mass

# Distribute via soft assignment (probs_2d sums to 1)
for (di, dj) in neighborhood:
    delta[orig_cx + di, orig_cy + dj, t*] += probs_2d[di + k, dj + k] * pickup_mass
```

`pickup_mass` is uniform across all cells within a block, so the mass balance is exact: total subtracted equals total added, preserving the mean-aggregated interpretation of the modified `pickup_3d`.

Using uniform `n_hours_per_block[t*] × n_days` (not a per-cell observation count) is a deliberate modeling choice — it treats every cell as having the same nominal observation budget per block, avoiding per-cell scale-factor tracking. Cells with missing data appear as inactive in `mask_3d` and are excluded from the metrics entirely, so the uniform assumption introduces no bias in the fairness computation.

### 8.2 `FAMAILObjective.forward()`

```python
pickup_N       = soft_pickup_3d[mask_3d]
dropoff_N      = dropoff_3d[mask_3d]
active_taxis_N = active_taxis_3d[mask_3d]

f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_taxis_N)

with torch.no_grad():
    g0_D_N = g0_func(clamp(pickup_N, min=DEMAND_FLOOR))
f_causal, _ = compute_fcausal(pickup_N, active_taxis_N, g0_D_N, I_minus_H, M)

if ALPHA_FIDELITY > 0:
    f_fidelity, _ = compute_ffidelity(discriminator, tau, tau_prime, ms_kwargs)
else:
    f_fidelity = 0.0

total = ALPHA_SPATIAL·f_spatial + ALPHA_CAUSAL·f_causal + ALPHA_FIDELITY·f_fidelity
```

### 8.3 ST-iFGSM loop (`TrajectoryModifier`)

Per trajectory (selected by attribution rank):
1. Determine time block `t* = hour_to_block_index(hour_of(pickup.time_bucket))`
2. Compute `pickup_mass = 1.0 / (n_hours_per_block[t*] * n_days)` — the uniform mass factor for this block (Section 8.1)
3. Subtract this trajectory's contribution from `_base_pickup_3d[orig_c, t*] -= pickup_mass`
4. For each iteration `iter ∈ [0, MAX_ITERATIONS)`:
   a. Anneal temperature
   b. Build `pickup_tensor = (x, y)` with `requires_grad=True`
   c. Compute soft probs → inject `pickup_mass · probs_2d` into `_base_pickup_3d[:, :, t*]` via delta pattern
   d. Forward + backward through `FAMAILObjective`
   e. Apply `δ = clip(α·sign(∇), -ε, ε)`; update cumulative `δ`
   f. Clip pickup to grid bounds; check convergence
5. Persist change to shared `_base_pickup_3d` for next trajectory (subtract `pickup_mass` at original cell, add `pickup_mass` at final cell)

### 8.4 Attribution → trajectory ranking

```python
per_unit = per_unit_attribution(R, I_minus_H_demo, M)   # (N,)

for each trajectory:
    cell, t_block = pickup_cell_and_time_block(trajectory)
    unit_idx = unit_map.from_cell_time(cell, t_block)
    score = per_unit[unit_idx] if unit_idx >= 0 else 0.0

rank by score (descending); select top-k with score > 0
```

### 8.5 Pickup-in-inactive-unit safeguard

At `modify_single()` entry: if the soft-assignment neighborhood around the pickup contains no active units in `t*`, log a warning and skip the trajectory. Top-k selected trajectories pass this check trivially.

---

## 9. Testing plan

Tests fall into three categories, organized by motivation.

### 9.1 Mathematical invariants

- Equal DSR → `Gini = 0`, `F_spatial = 1`
- One-hot DSR → `Gini → (N-1)/N`
- `R ∈ span(X_demo)` → `F_causal = 0`
- `R ⊥ X_demo` → `F_causal = 1`
- `Σᵢ per_unit_attribution_i == 1 - F_causal` within `EPS`
- `(I - H)² == (I - H)`; `M² == M`
- `rank(H_demo) == n_features + 1`

### 9.2 Bug-class regression guards

- Gradient flow through full pooled objective (non-zero, non-NaN)
- Gradient only flows through the `t*` slice of `soft_pickup_3d`
- LSTM backward succeeds in inference mode (cuDNN workaround)
- Canonical unit ordering stable across runs
- Inactive pickups score 0 in attribution
- Hat matrix shape matches unit count

### 9.3 Integration

- Fixed-seed 5-iteration convergence (metrics improve or plateau monotonically)
- Cross-trajectory baseline update (order-dependence test)
- ε-ball respected after 50 iterations
- Inactive-pickup trajectory skipped without errors

Fixtures live in `tests/conftest.py` and `tests/synthetic/fixtures.py`. Synthetic tests run in <10 seconds; slow tests (full `DataBundle.load()`) behind `@pytest.mark.slow`.

---

## 10. Stability safeguards (consolidated)

**At preprocess time:**
- `mask_3d.sum() >= MIN_TOTAL_ACTIVE_UNITS`
- `units_per_block[t] >= MIN_ACTIVE_UNITS_PER_BLOCK` ∀t
- `rank(H_demo) == n_features + 1`
- `np.isfinite(demographics[active_mask]).all()`
- `g₀(D)` max/min ratio bounded

**At load time:**
- `unit_map.n_units == hat_matrices['I_minus_H_demo'].shape[0]`
- `pickup_3d.shape == dropoff_3d.shape == active_taxis_3d.shape == (48, 90, T)`
- `DataBundle` is `frozen=True` (immutable)

**At every forward pass:**
- `D := max(D, DEMAND_FLOOR)` before `Y = S/D`
- `EPS` added to every divisor
- All scalar fairness metrics clamped to `[0, 1]`
- `torch.where(ss_tot < EPS, 1.0, ss_res / ss_tot)` degenerate guard

**At every ST-iFGSM step:**
- `cumulative_delta` clipped to `[-ε, ε]`
- Pickup clipped to grid bounds
- Zero-gradient iterations are no-ops (no error)

---

## 11. Identified snags and resolutions

| # | Snag | Resolution |
|---|---|---|
| S1 | Order-dependence via shared `_base_pickup_3d` | Intentional; documented. Attribution computed once before modifications. |
| S2 | Missing `architecture_config` in checkpoint | Added via one-time preprocessing; `load_discriminator` raises specifically if missing. |
| S3 | In-place ops on `.clone()` could break autograd | Use delta-tensor pattern: `out = base + delta` with `delta` scatter-filled. |
| S4 | `torch.where` gradient semantics for degenerate branch | `torch.where` is gradient-safe; gradient flows through selected branch. |
| S5 | All-inactive neighborhood → no gradient signal | Startup check in `modify_single`; top-k selection avoids trivially. |
| S6 | `g₀` clip boundaries are constants | Intentional — gradient flows through `Y = S/D`, not `g₀(D)`. |
| S7 | `torch.from_numpy` memory-sharing | Explicit `.copy()` during `DataBundle.load()`; frozen dataclass. |
| S8 | 0-indexed modifier vs. 1-indexed discriminator | Preserved `+1` in `fidelity/context.py`; guard test. |
| S9 | NaN demographics outside Shenzhen | Filtered in `active_mask.compute()`. |
| S10 | Reproducibility | `utils/seeding.py::set_all_seeds(seed)` called at every top-level script. |
| S11 | Per-iteration `g₀(D_N)` performance | Negligible for N=8000; flagged for profiling if needed. |
| S12 | `float64` pinv vs. `float32` GPU | Hat matrices built `float64`, cast `float32` at load; diagnostic test confirms agreement. |
| S13 | Mass-balance of one trajectory under mean-hourly aggregation | Use uniform `pickup_mass = 1/(n_hours_per_block[t*] · n_days)` as the per-pickup scale factor. Documented in §8.1. Test: after modifying one trajectory, `pickup_3d.sum()` is unchanged within `EPS`. |

---

## 12. Final directory layout

```
famail_temporal/
├── README.md                         # top-level; links every sub-README + quickstart
├── requirements.txt                  # torch, numpy, scikit-learn, pytest
├── config.py                         # single source of truth
├── preprocess.py                     # raw_data/ → cache/
│
├── data/
│   ├── README.md                     # active-unit invariant, aggregation rules
│   ├── __init__.py
│   ├── loader.py                     # DataBundle + .load()
│   ├── aggregation.py                # raw pickles → (48, 90, T) + hour_to_block_index
│   └── active_mask.py                # UnitIndexMap + active computation
│
├── fairness/
│   ├── README.md                     # pooled metrics derivation, per-unit attribution math
│   ├── __init__.py
│   ├── spatial.py                    # pairwise_gini + compute_fspatial
│   ├── causal.py                     # compute_fcausal + per_unit_attribution(_signed)
│   ├── hat_matrices.py               # precompute + compute_fcausal_torch
│   └── g0_power_basis.py             # G0Function + fit()
│
├── fidelity/
│   ├── README.md                     # discriminator port, cuDNN workaround rationale
│   ├── __init__.py
│   ├── model.py                      # MultiStreamSiameseDiscriminator + 3 encoders
│   ├── checkpoint.py                 # load_discriminator()
│   ├── context.py                    # MultiStreamContextBuilder
│   └── compute.py                    # compute_ffidelity() with cuDNN workaround
│
├── algorithm/
│   ├── README.md                     # ST-iFGSM loop, gradient flow diagram
│   ├── __init__.py
│   ├── soft_cell_assignment.py       # SoftCellAssignment + inject_soft_counts_into_3d
│   ├── attribution.py                # per_unit → per_trajectory ranking
│   ├── objective.py                  # FAMAILObjective (orchestrator)
│   └── modifier.py                   # TrajectoryModifier (ST-iFGSM loop)
│
├── utils/
│   ├── README.md                     # seeding, trajectory utilities
│   ├── __init__.py
│   ├── seeding.py                    # set_all_seeds()
│   └── trajectory.py                 # Trajectory + TrajectoryState
│
├── raw_data/                         # gitignored contents
│   └── README.md                     # file list, sources, sizes, schemas
│
├── cache/                            # gitignored contents
│   └── README.md                     # filename scheme + artifact table
│
├── discriminator_checkpoints/        # gitignored contents
│   ├── README.md                     # canonical checkpoint + provenance
│   └── default/best.pt
│
└── tests/
    ├── README.md                     # test plan, fast vs. slow markers, fixtures
    ├── __init__.py
    ├── conftest.py
    ├── test_aggregation.py
    ├── test_active_mask.py
    ├── test_data_loader.py
    ├── test_g0_power_basis.py
    ├── test_hat_matrices.py
    ├── test_spatial_fairness.py
    ├── test_causal_fairness.py
    ├── test_soft_cell_assignment.py
    ├── test_attribution.py
    ├── test_fidelity.py
    ├── test_gradient_flow.py
    ├── test_modifier_integration.py
    └── synthetic/
        ├── __init__.py
        └── fixtures.py
```

Target: ~50 files, none exceeding ~500 lines.

---

## 13. Per-subdirectory README contents

Each sub-README follows a consistent template. Headings in order: **Purpose**, **Files**, **Key design choices**, **API surface**, **Dependencies**, **Paper-section hook**.

### 13.1 `famail_temporal/README.md` (top level)

**Purpose:** One-paragraph overview of the algorithm. Links to sub-READMEs.

**Quickstart:** Installation, `python -m famail_temporal.preprocess`, `pytest`.

**Running the algorithm:** Minimal code example importing `DataBundle`, `FAMAILObjective`, `TrajectoryModifier`.

**Design spec:** Link to this document.

**Table of contents:** Links to each sub-README with a one-sentence description.

### 13.2 `data/README.md`

**Purpose:** Ingest raw pickles and produce the canonical active-unit representation.

**Files:** `loader.py`, `aggregation.py`, `active_mask.py`.

**Key design choices:**
- The canonical active-unit ordering (cell-major, block-within-cell)
- The unified aggregation rule (mean hourly rate within block, mean across days)
- Why `g₀(D)` is re-fit at block-mean scale (eliminates dual-tensor design)
- Active filter at `(cell, t)` granularity with the two sub-rules

**API surface:** `DataBundle`, `DataBundle.load()`, `UnitIndexMap`, `hour_to_block_index()`.

**Dependencies:** `utils/`, `fairness.g0_power_basis`, `fairness.hat_matrices` (preprocessing only).

**Paper-section hook:** "Data Preparation" in Methods.

### 13.3 `fairness/README.md`

**Purpose:** Pooled fairness metrics over active `(cell, t)` units.

**Files:** `spatial.py`, `causal.py`, `hat_matrices.py`, `g0_power_basis.py`.

**Key design choices:**
- N-vector inputs only — no grid geometry awareness
- Pooled Gini (single Gini over all active units, not per-block)
- Option B as the sole causal formulation
- The per-unit attribution decomposition with the `Σᵢ attribution_i = 1 - F_causal` property
- Why `g₀(D)` uses the power basis (for hat-matrix compatibility) with isotonic as diagnostic

**API surface:** `compute_fspatial`, `compute_fcausal`, `compute_fcausal_torch`, `per_unit_attribution`, `per_unit_attribution_signed`, `precompute_hat_matrices`, `G0Function`, `fit_g0`.

**Dependencies:** `config`. Nothing else.

**Paper-section hook:** "Fairness Metrics" in Methods; per-unit attribution section surfaces in Results.

### 13.4 `fidelity/README.md`

**Purpose:** Discriminator-based realism check for modified trajectories.

**Files:** `model.py`, `checkpoint.py`, `context.py`, `compute.py`.

**Key design choices:**
- Checkpoint is opaque (no training code ported)
- Which 4 of 8 discriminator classes are ported, and why the other 4 are excluded
- The cuDNN backward-in-inference workaround (why it's necessary, what breaks without it)
- The four multi-stream context builder decisions (preserved verbatim)
- How `ALPHA_FIDELITY = 0` cleanly bypasses the entire fidelity pathway

**API surface:** `load_discriminator`, `MultiStreamContextBuilder`, `compute_ffidelity`.

**Dependencies:** `config`, `utils/trajectory`.

**Paper-section hook:** "Fidelity Term" subsection in Methods; checkpoint provenance in Supplementary.

### 13.5 `algorithm/README.md`

**Purpose:** Orchestration of the objective and the ST-iFGSM trajectory modification loop.

**Files:** `soft_cell_assignment.py`, `attribution.py`, `objective.py`, `modifier.py`.

**Key design choices:**
- The single grid↔unit conversion point inside `FAMAILObjective.forward()`
- `pickup_N` carries gradient for both spatial and causal terms
- Delta-tensor pattern for autograd-safe 3D injection
- Per-trajectory inheritance of per-unit attribution
- Cross-trajectory ordering semantics via `_base_pickup_3d`
- The pickup-in-inactive-unit safeguard

**Gradient flow diagram** (ASCII): `pickup_tensor (x,y) → SoftCellAssignment → probs_2d → delta tensor [:, :, t*] → base + delta → gather via mask_3d → pickup_N → [F_spatial, F_causal] → weighted sum → total → backward → pickup_tensor.grad`.

**API surface:** `FAMAILObjective`, `TrajectoryModifier`, `compute_per_unit_attribution`, `rank_trajectories`, `select_top_k`, `SoftCellAssignment`, `inject_soft_counts_into_3d`.

**Dependencies:** `data/`, `fairness/`, `fidelity/`, `config`, `utils/`.

**Paper-section hook:** "Algorithm" in Methods; attribution pipeline in Results.

### 13.6 `utils/README.md`

**Purpose:** Shared utilities with no domain-specific knowledge.

**Files:** `seeding.py`, `trajectory.py`.

**Key design choices:**
- Seed management: one `set_all_seeds(seed)` call covers `random`, `numpy`, `torch`, `torch.cuda`, and the multi-stream context sampler
- `Trajectory` / `TrajectoryState` are the same dataclasses as in current code, lifted verbatim (already clean)

**API surface:** `set_all_seeds`, `Trajectory`, `TrajectoryState`.

**Dependencies:** `config` only.

**Paper-section hook:** Briefly mentioned in Reproducibility appendix.

### 13.7 `tests/README.md`

**Purpose:** Test organization, running conventions, and fixture inventory.

**Files:** Per-module test files + `conftest.py` + `synthetic/fixtures.py`.

**Key design choices:**
- Three test categories: math invariants, bug-class regression guards, integration
- Fast tests (<10s) use `synthetic_bundle` fixture
- Slow tests (`@pytest.mark.slow`) use real `DataBundle.load()`; skipped by default
- `seeded` autouse fixture sets seeds before every test

**Running the tests:**
- `pytest` — fast tests only
- `pytest --run-slow` — everything

**Paper-section hook:** Reproducibility appendix describes the test suite; specific invariant tests may be referenced in Methods.

### 13.8 `raw_data/README.md`

**Purpose:** Inventory of raw input files copied from the parent project.

**Content:** Table listing each file, size (approx.), source location in the original `source_data/` or `discriminator/multi_stream/extracted_data/`, SHA256 fingerprint for reproducibility, brief schema description.

### 13.9 `cache/README.md`

**Purpose:** Describe the filename scheme and what each cached artifact contains.

**Content:** Naming convention (`{artifact}_T{T}_thr{threshold}[_feat-{features}].pkl`), table of all artifact types, regeneration instructions, note that caches with different suffixes coexist (no automatic invalidation).

### 13.10 `discriminator_checkpoints/README.md`

**Purpose:** Document which checkpoint is canonical and how to swap.

**Content:** Canonical checkpoint provenance (which training run produced it, with date); the expected `architecture_config` dict embedded in the checkpoint; instructions for substituting a different checkpoint by editing `config.DISCRIMINATOR_CHECKPOINT_FILENAME`.

---

## 14. Out of scope (explicitly)

The following are **not** part of this rewrite and remain in their current locations or are deferred:

- Discriminator training (`discriminator/model/train.py`, etc.)
- Multi-stream extraction pipeline (`discriminator/multi_stream/extraction/`, `dataset_generation/`)
- Streamlit dashboards for any component
- Path-aware trajectory attribution (deferred; current attribution uses pickup inheritance only)
- Non-pickup trajectory perturbation (deferred)
- `T = 24` hourly granularity (deferred; pooled hat matrices at T=24 would require QR-based implementation to avoid 18GB memory)
- Dayofweek-stratified analysis (deferred)

---

## 15. Success criteria

The rewrite is "done" when:

1. `python -m famail_temporal.preprocess` completes without errors on the Shenzhen dataset and produces all cache artifacts with correctly-encoded filenames.
2. `pytest` (fast tests only) passes in under 10 seconds.
3. `pytest --run-slow` passes in under 2 minutes.
4. `DataBundle.load()` produces a bundle where `unit_map.n_units` matches `hat_matrices['I_minus_H_demo'].shape[0]`.
5. A single-trajectory `modify_single()` call produces a 50-iteration history where the total objective strictly improves or plateaus, and `f_spatial`, `f_causal`, `f_fidelity` are all in `[0, 1]` throughout.
6. The per-unit attribution vector's sum equals `1 - F_causal` within `EPS` (property test).
7. All 12 sub-READMEs exist and render correctly on GitHub.

---

## 16. Next steps

Upon user approval of this spec, the immediate next step is to invoke the `superpowers:writing-plans` skill to produce a detailed, ordered implementation plan — including a recommended build order that respects the dependency graph (configs and data schemas first, then fairness math, then fidelity, then the orchestrating algorithm, then integration tests, then sub-READMEs).
