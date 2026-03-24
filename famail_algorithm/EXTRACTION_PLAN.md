# FAMAIL Algorithm — Extraction & Porting Plan

> **Date**: 2026-03-15
> **Status**: Draft — documentation only, no code written yet
> **Purpose**: Extract all core trajectory modification logic into a self-contained `famail_algorithm/` module

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [Proposed Module Structure](#2-proposed-module-structure)
3. [Component Inventory & Extraction Map](#3-component-inventory--extraction-map)
4. [Phase 1: Attribution (Trajectory Selection)](#4-phase-1-attribution-trajectory-selection)
5. [Phase 2: Modification (ST-iFGSM Editing)](#5-phase-2-modification-st-ifgsm-editing)
6. [Unified Algorithm Runner](#6-unified-algorithm-runner)
7. [Objective Function Components](#7-objective-function-components)
8. [Soft Cell Assignment](#8-soft-cell-assignment)
9. [Discriminator & Fidelity](#9-discriminator--fidelity)
10. [Data Loading & DataBundle](#10-data-loading--databundle)
11. [Global Metrics & Evaluation](#11-global-metrics--evaluation)
12. [Configuration & Hyperparameters](#12-configuration--hyperparameters)
13. [Data Preparation Pipeline](#13-data-preparation-pipeline)
14. [Discriminator Training Pipeline](#14-discriminator-training-pipeline)
15. [Naming Conventions & Terminology](#15-naming-conventions--terminology)
16. [Cross-Module Dependency Summary](#16-cross-module-dependency-summary)
17. [Migration Checklist](#17-migration-checklist)
18. [Open Questions](#18-open-questions)

---

## 1. Overview & Goals

### 1.1 What This Module Is

`famail_algorithm/` is the **self-contained implementation** of the FAMAIL Fairness-Aware Trajectory Editing Algorithm. It contains everything needed to:

1. **Load** preprocessed taxi trajectory and fairness data
2. **Select** trajectories for modification based on fairness attribution scores (Phase 1)
3. **Modify** selected trajectories using gradient-based perturbation (Phase 2)
4. **Evaluate** before/after fairness metrics

### 1.2 Why Extract

The algorithm logic is currently scattered across multiple directories:

| Current Location | What Lives There |
|-----------------|-----------------|
| `trajectory_modification/` | Core modifier, objective, data loader, metrics, trajectory types |
| `trajectory_modification/dashboard.py` | Phase 1 attribution logic (LIS, DCD, scoring, selection) |
| `objective_function/soft_cell_assignment/` | Differentiable soft cell assignment module |
| `objective_function/causal_fairness/utils.py` | Hat matrix math, g₀(D) fitting, R² computation, demographic feature engineering |
| `objective_function/causal_fairness/config.py` | Causal fairness configuration |
| `discriminator/model/model.py` | SiameseLSTM discriminator architectures |

This makes the algorithm difficult to understand as a whole, hard to debug, and unsuitable for research paper submission where reviewers need to inspect a coherent codebase.

### 1.3 Design Principles

1. **Self-contained**: All algorithm code in one directory. No imports from `objective_function/` or `discriminator/` — those modules' relevant code is ported into `famail_algorithm/`.
2. **Unified pipeline**: Phase 1 (attribution) and Phase 2 (modification) run sequentially in a single call, not as separate user-triggered actions.
3. **Auditable**: File organization follows the algorithm structure so a reviewer can trace from the paper's math to the code.
4. **Configurable**: All hyperparameters exposed through a single configuration dataclass.
5. **Reproducible**: Random seeds, configuration serialization, and deterministic execution.

### 1.4 What This Document Covers

This document maps every function, class, constant, and data dependency that must be ported into `famail_algorithm/`. It specifies:
- **Source location**: Exact file and line range in the current codebase
- **Target location**: Proposed file in `famail_algorithm/`
- **Changes needed**: Any modifications from the current implementation (e.g., Phase 1+2 unification, naming changes)
- **Dependencies**: What each component requires

---

## 2. Proposed Module Structure

```
famail_algorithm/
├── __init__.py                  # Public API exports
├── EXTRACTION_PLAN.md           # This document
│
├── ── Core Algorithm ──
├── algorithm.py                 # Unified Phase 1 + Phase 2 runner
├── attribution.py               # Phase 1: LIS, DCD, scoring, top-k selection
├── modifier.py                  # Phase 2: ST-iFGSM loop (single + batch)
│
├── ── Objective Function ──
├── objective.py                 # Combined objective: L = α₁F_sp + α₂F_ca + α₃F_fi
├── spatial_fairness.py          # F_spatial: Gini-based equitable distribution
├── causal_fairness.py           # F_causal: Demographic Residual Independence (hat matrix)
├── fidelity.py                  # F_fidelity: Discriminator-based behavioral authenticity
│
├── ── Differentiable Infrastructure ──
├── soft_cell.py                 # SoftCellAssignment: differentiable grid assignment
│
├── ── Data & Types ──
├── trajectory.py                # Trajectory, TrajectoryState dataclasses
├── data_loader.py               # DataBundle, loader classes, g₀(D) fitting
├── config.py                    # All configuration dataclasses
│
├── ── Evaluation ──
├── metrics.py                   # GlobalMetrics, FairnessSnapshot, before/after tracking
│
└── ── Discriminator Model ──
    ├── discriminator_adapter.py # DiscriminatorAdapter wrapper
    └── model.py                 # SiameseLSTM architecture definitions (V1 + V2)
```

### 2.1 File Responsibilities

| File | Responsibility | Primary Source |
|------|---------------|---------------|
| `algorithm.py` | **NEW**: Unified entry point that runs Phase 1 → Phase 2 → Evaluation | New code combining `dashboard.py` workflow + `modifier.py` batch logic |
| `attribution.py` | Phase 1 scoring and selection | `trajectory_modification/dashboard.py` functions (lines 227–531) |
| `modifier.py` | ST-iFGSM perturbation loop | `trajectory_modification/modifier.py` |
| `objective.py` | Combined differentiable objective | `trajectory_modification/objective.py` |
| `spatial_fairness.py` | Gini coefficient computation | Extracted from `trajectory_modification/objective.py` (`_pairwise_gini`, `compute_spatial_fairness`) |
| `causal_fairness.py` | Hat matrix math, g₀(D), R² | `objective_function/causal_fairness/utils.py` (subset) + `objective.py` causal methods |
| `fidelity.py` | Discriminator-based scoring | Extracted from `trajectory_modification/objective.py` (`compute_fidelity`) |
| `soft_cell.py` | Gaussian soft assignment | `objective_function/soft_cell_assignment/module.py` |
| `trajectory.py` | Core data structures | `trajectory_modification/trajectory.py` |
| `data_loader.py` | Data loading and g₀(D) fitting | `trajectory_modification/data_loader.py` |
| `config.py` | Configuration classes | New, consolidating params from `modifier.py`, `objective.py`, dashboard sidebar |
| `metrics.py` | Global fairness tracking | `trajectory_modification/metrics.py` |
| `discriminator_adapter.py` | Discriminator wrapper | `trajectory_modification/discriminator_adapter.py` |
| `model.py` | SiameseLSTM architecture | `discriminator/model/model.py` (SiameseLSTMDiscriminator, V2, FeatureNormalizer, Encoder) |

---

## 3. Component Inventory & Extraction Map

### 3.1 Functions Currently in `trajectory_modification/dashboard.py` (Phase 1 Logic)

These are pure algorithm functions that should NOT be in a dashboard file. They must be extracted to `attribution.py`.

| Function | Lines | Purpose | Target |
|----------|-------|---------|--------|
| `compute_cell_lis_scores(cell_counts)` | 227–248 | LIS = \|cᵢ - μ\| / μ per cell | `attribution.py` |
| `compute_cell_dcd_scores(demand, supply, g_function, ...)` | 251–347 | DCD per cell (baseline + DRI) | `attribution.py` |
| `compute_trajectory_attribution_scores(trajectories, ...)` | 350–475 | Combined LIS+DCD per trajectory | `attribution.py` |
| `select_top_k_by_attribution(scores, k, method)` | 478–530 | Top-k or diverse selection | `attribution.py` |

### 3.2 Classes in `trajectory_modification/modifier.py`

| Class/Function | Purpose | Target | Changes |
|---------------|---------|--------|---------|
| `ModificationResult` | Per-iteration result dataclass | `modifier.py` | None |
| `ModificationHistory` | Full trajectory modification trace | `modifier.py` | None |
| `TrajectoryModifier` | ST-iFGSM loop engine | `modifier.py` | Remove dashboard coupling |
| `TrajectoryModifier.set_global_state()` | Initialize global counts | `modifier.py` | None |
| `TrajectoryModifier.modify_single()` | Core ST-iFGSM for one trajectory | `modifier.py` | None |
| `TrajectoryModifier.modify_batch()` | Sequential batch modification | `modifier.py` | None |
| `TrajectoryModifier._compute_soft_pickup_counts()` | Differentiable count computation | `modifier.py` | None |
| `TrajectoryModifier._compute_heuristic_gradient()` | Fallback gradient | `modifier.py` | None |
| `TrajectoryModifier._get_annealed_temperature()` | Temperature schedule | `modifier.py` | None |
| `TrajectoryModifier._clip_to_grid()` | Grid bounds enforcement | `modifier.py` | None |
| `TrajectoryModifier._update_counts()` | Pickup count bookkeeping | `modifier.py` | None |

### 3.3 Classes in `trajectory_modification/objective.py`

| Class/Function | Purpose | Target | Changes |
|---------------|---------|--------|---------|
| `FAMAILObjective(nn.Module)` | Combined objective L | `objective.py` | Refactor to use sub-modules |
| `FAMAILObjective.compute_spatial_fairness()` | F_spatial via Gini | Move core math to `spatial_fairness.py` |
| `FAMAILObjective._pairwise_gini()` | Differentiable Gini | `spatial_fairness.py` |
| `FAMAILObjective.compute_causal_fairness()` | F_causal dispatch | `objective.py` (calls `causal_fairness.py`) |
| `FAMAILObjective._compute_baseline_causal()` | Baseline R² | `causal_fairness.py` |
| `FAMAILObjective._compute_option_b_causal()` | DRI hat matrix | `causal_fairness.py` |
| `FAMAILObjective._compute_option_c_causal()` | Partial ΔR² | `causal_fairness.py` |
| `FAMAILObjective.compute_fidelity()` | Discriminator scoring | `fidelity.py` |
| `FAMAILObjective.forward()` | Main forward pass | `objective.py` |
| `MissingDataError` | Exception | `objective.py` |
| `MissingComponentError` | Exception | `objective.py` |
| `InsufficientDataError` | Exception | `objective.py` |

### 3.4 Classes in `trajectory_modification/trajectory.py`

| Class | Purpose | Target | Changes |
|-------|---------|--------|---------|
| `TrajectoryState` | Single state: (x, y, time, day) | `trajectory.py` | None |
| `Trajectory` | Full trajectory with perturbation methods | `trajectory.py` | None |

### 3.5 Classes in `trajectory_modification/data_loader.py`

| Class/Function | Purpose | Target | Changes |
|---------------|---------|--------|---------|
| `DataBundle` | All data needed for algorithm | `data_loader.py` | None |
| `TrajectoryLoader` | Load trajectory pickle | `data_loader.py` | None |
| `PickupDropoffLoader` | Load pickup/dropoff counts | `data_loader.py` | None |
| `ActiveTaxisLoader` | Load active taxi counts | `data_loader.py` | None |
| `GFunctionLoader` | Fit g₀(D) from data | `data_loader.py` | Port causal_fairness.utils dependencies inline |
| `find_workspace_root()` | Locate project root | `data_loader.py` | None |
| `load_pickle()` | Generic pickle loader | `data_loader.py` | None |
| `DEFAULT_PATHS` | File path constants | `data_loader.py` | None |
| `GRID_DIMS` | (48, 90) constant | `config.py` |  |

### 3.6 Classes in `trajectory_modification/metrics.py`

| Class | Purpose | Target | Changes |
|-------|---------|--------|---------|
| `FairnessSnapshot` | Snapshot of all fairness scores | `metrics.py` | None |
| `GlobalMetrics` | Track metrics across modifications | `metrics.py` | Port `compute_fcausal_option_b_numpy` inline |

### 3.7 Classes in `trajectory_modification/discriminator_adapter.py`

| Class | Purpose | Target | Changes |
|-------|---------|--------|---------|
| `DiscriminatorAdapter` | Load and wrap discriminator | `discriminator_adapter.py` | Update import to local `model.py` |

### 3.8 Functions from `objective_function/causal_fairness/utils.py`

These functions are imported by `data_loader.py`, `objective.py`, and `metrics.py`. They must be ported into `causal_fairness.py` within `famail_algorithm/`.

**Required for g₀(D) fitting (used by `data_loader.py`):**

| Function | Purpose |
|----------|---------|
| `extract_demand_from_counts(pickup_dropoff_data)` | Extract pickup counts from combined dict |
| `aggregate_to_period(data, period_type, aggregator)` | Temporal aggregation (hourly, daily, etc.) |
| `compute_service_ratios(demand, supply, min_demand, ...)` | Y = S/D per cell |
| `extract_demand_ratio_arrays(demand, ratios)` | Convert dicts to aligned numpy arrays |
| `estimate_g_power_basis(demands, ratios)` | Fit g₀(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1) |
| `estimate_g_isotonic(demands, ratios)` | Fit isotonic regression g₀(D) (for baseline) |

**Required for hat matrix computation (used by `data_loader.py`):**

| Function | Purpose |
|----------|---------|
| `enrich_demographic_features(demo_grid, feature_names)` | Add derived features (GDPperCapita, etc.) |
| `build_power_basis_features(demands, include_intercept)` | [1, 1/(D+1), 1/√(D+1), √(D+1)] matrix |
| `build_hat_matrix(X)` | H = X @ pinv(X) |
| `build_centering_matrix(n)` | M = I - 11'/n |
| `precompute_hat_matrices(demands, demographic_features, feature_names)` | Full hat matrix pipeline → Dict |

**Required for differentiable F_causal (used by `objective.py`):**

| Function | Purpose |
|----------|---------|
| `compute_fcausal_option_b_torch(R, I_minus_H, M, eps)` | Differentiable DRI: R'(I-H)R / R'MR |
| `compute_fcausal_option_c_torch(Y, I_minus_H_red, I_minus_H_full, M, eps)` | Differentiable partial ΔR² |

**Required for numpy metrics (used by `metrics.py`):**

| Function | Purpose |
|----------|---------|
| `compute_fcausal_option_b_numpy(R, I_minus_H, M, eps)` | Non-differentiable DRI for evaluation |

### 3.9 Classes from `objective_function/soft_cell_assignment/module.py`

| Class/Function | Purpose | Target |
|---------------|---------|--------|
| `SoftCellAssignment(nn.Module)` | Differentiable Gaussian soft cell assignment | `soft_cell.py` |
| `update_counts_with_soft_assignment()` | Update grid counts with soft probs | `soft_cell.py` |

### 3.10 Classes from `discriminator/model/model.py`

| Class | Purpose | Target |
|-------|---------|--------|
| `FeatureNormalizer` | Convert 4D raw → 6D normalized features | `model.py` |
| `SiameseLSTMEncoder` | Shared LSTM encoder for Siamese architecture | `model.py` |
| `SiameseLSTMDiscriminator` | V1 discriminator (concatenation combination) | `model.py` |
| `SiameseLSTMDiscriminatorV2` | V2 discriminator (difference combination, preferred) | `model.py` |

---

## 4. Phase 1: Attribution (Trajectory Selection)

### 4.1 Current State

Phase 1 logic lives entirely in `trajectory_modification/dashboard.py` (lines 227–531) as four standalone functions. These are pure algorithm code with no Streamlit dependencies — they operate on numpy arrays and return plain Python dicts/lists. This makes extraction straightforward.

### 4.2 Target: `attribution.py`

The four functions port directly with no changes to their logic:

#### `compute_cell_lis_scores(cell_counts: np.ndarray) -> np.ndarray`
- **Source**: `dashboard.py:227–248`
- **Math**: LIS_c = |N_c - μ| / μ where N_c is cell count and μ is global mean
- **Input**: [48, 90] count array (pickup or dropoff)
- **Output**: [48, 90] LIS scores (≥ 0)

#### `compute_cell_dcd_scores(...) -> np.ndarray`
- **Source**: `dashboard.py:251–347`
- **Math (DRI formulation)**: DCD_c = |R̂_c| where R̂ = H_demo @ R, R = Y - g₀(D)
- **Math (baseline)**: DCD_c = |Y_c - g(D_c)|
- **Input**: demand/supply grids, g_function, formulation flag, hat matrices
- **Output**: [48, 90] DCD scores (≥ 0)
- **Note**: The `causal_formulation` parameter dispatches to baseline or DRI logic

#### `compute_trajectory_attribution_scores(...) -> List[Dict]`
- **Source**: `dashboard.py:350–475`
- **Logic**:
  1. Compute cell-level LIS (pickup + dropoff) and DCD grids
  2. For each trajectory: LIS_τ = max(LIS_pickup, LIS_dropoff), DCD_τ = DCD[pickup_cell]
  3. Normalize both to [0, 1]
  4. Combined_τ = w_LIS × LIS_norm + w_DCD × DCD_norm
- **Parameters**: lis_weight, dcd_weight, normalize flag, causal_formulation

#### `select_top_k_by_attribution(scores, k, method) -> List[int]`
- **Source**: `dashboard.py:478–530`
- **Methods**:
  - `top_k`: Sort by combined score descending, take first k
  - `diverse`: Greedy selection with cell-level penalty (β=0.5) to spread across areas

### 4.3 Changes from Current Implementation

1. **Rename parameter**: `causal_formulation="option_b"` → `causal_formulation="dri"` (see Section 15)
2. **Add type hints** for `Callable` parameter (g_function)
3. **No other logic changes** — the functions are already pure and well-tested

---

## 5. Phase 2: Modification (ST-iFGSM Editing)

### 5.1 Current State

Phase 2 is implemented in `trajectory_modification/modifier.py` as the `TrajectoryModifier` class. This is well-structured and ports almost directly.

### 5.2 Target: `modifier.py`

#### `TrajectoryModifier` class
- **Source**: `trajectory_modification/modifier.py`
- **Constructor parameters**:
  - `objective_fn: nn.Module` — the FAMAILObjective
  - `grid_dims: Tuple[int, int] = (48, 90)`
  - `alpha: float = 0.1` — ST-iFGSM step size
  - `epsilon: float = 2.0` — max perturbation per dimension
  - `max_iterations: int = 50`
  - `convergence_threshold: float = 1e-6`
  - `gradient_mode: str = 'soft_cell'` — `'soft_cell'` or `'heuristic'`
  - `temperature: float = 1.0`
  - `temperature_annealing: bool = True`
  - `tau_max: float = 1.0`
  - `tau_min: float = 0.1`
  - `neighborhood_size: int = 5`

#### Core algorithm (ST-iFGSM loop in `modify_single`):
1. Clone trajectory, initialize cumulative_δ = [0, 0]
2. For each iteration:
   a. Anneal temperature: τ_t = τ_max × (τ_min/τ_max)^(t/(T-1))
   b. Create pickup_tensor with requires_grad=True
   c. Compute soft pickup counts (differentiable)
   d. Forward pass through objective → L
   e. Backward pass → gradient ∂L/∂pickup
   f. ST-iFGSM step: δ = α × sign(grad), clip cumulative_δ to ε-ball
   g. Apply perturbation from ORIGINAL position (not running copy)
   h. Update global counts if cell changed
   i. Check convergence: |L_t - L_{t-1}| < threshold
3. Return ModificationHistory

#### `modify_batch`: Sequential modification of multiple trajectories
- Syncs base_pickup_counts after each trajectory so next trajectory sees updated world state

### 5.3 Changes from Current Implementation

1. **Import path**: `SoftCellAssignment` imported from local `soft_cell.py` instead of `objective_function.soft_cell_assignment`
2. **Constructor**: Accept `FAMAILConfig` object instead of individual parameters (optional — can support both)
3. **No logic changes** — the ST-iFGSM loop is correct and well-documented

### 5.4 Key Data Structures

#### `ModificationResult` (per-iteration snapshot)
```
Fields:
  trajectory: Trajectory         # Current modified trajectory state
  objective_value: float         # L = α₁F_sp + α₂F_ca + α₃F_fi
  f_spatial: float
  f_causal: float
  f_fidelity: float
  gradient_norm: float
  perturbation: np.ndarray       # [δx, δy] cumulative from original
```

#### `ModificationHistory` (full trajectory trace)
```
Fields:
  original: Trajectory           # Unmodified input trajectory
  modified: Trajectory           # Final modified trajectory
  iterations: List[ModificationResult]
  converged: bool
  total_iterations: int
  final_objective: float
```

---

## 6. Unified Algorithm Runner

### 6.1 Current State

Currently, the user workflow in the dashboard runs Phase 1 and Phase 2 as **separate button clicks**:
1. User clicks "Compute Attribution Scores" → Phase 1 runs, stores selected indices in session state
2. User clicks "Run Modification" → Phase 2 runs on the stored indices

### 6.2 Target: `algorithm.py`

The new module unifies this into a single entry point. The proposed API:

```python
class FAMAILAlgorithm:
    """Unified FAMAIL trajectory editing pipeline."""

    def __init__(self, config: FAMAILConfig):
        """Initialize with full configuration."""

    def run(self, bundle: DataBundle) -> AlgorithmResult:
        """
        Execute the full FAMAIL trajectory editing pipeline.

        Steps:
        1. Initialize global state from DataBundle
        2. Compute initial fairness snapshot
        3. Phase 1: Attribution scoring + top-k selection
        4. Phase 2: ST-iFGSM modification of selected trajectories
        5. Compute final fairness snapshot
        6. Return results

        Args:
            bundle: DataBundle with all required data

        Returns:
            AlgorithmResult with modification histories, metrics, etc.
        """

    def run_attribution_only(self, bundle: DataBundle) -> AttributionResult:
        """Run Phase 1 only (for analysis/debugging)."""

    def run_modification_only(
        self,
        bundle: DataBundle,
        selected_indices: List[int],
    ) -> ModificationResult:
        """Run Phase 2 only on pre-selected trajectories."""
```

### 6.3 AlgorithmResult dataclass

```python
@dataclass
class AlgorithmResult:
    # Phase 1 outputs
    attribution_scores: List[Dict]        # Per-trajectory attribution scores
    selected_indices: List[int]           # Indices selected for modification

    # Phase 2 outputs
    modification_histories: List[ModificationHistory]  # Per-trajectory results

    # Evaluation
    initial_snapshot: FairnessSnapshot    # Before modification
    final_snapshot: FairnessSnapshot      # After modification

    # Configuration
    config: FAMAILConfig                  # Full config used for this run
```

### 6.4 Unified Flow (Pseudocode)

```
FUNCTION FAMAILAlgorithm.run(bundle):
    # ── Initialize ──
    global_state ← initialize from bundle (pickup_grid, dropoff_grid, active_taxis, etc.)
    metrics_tracker ← GlobalMetrics(g_function, hat_matrices, ...)
    metrics_tracker.initialize_from_data(pickup_grid, dropoff_grid, active_taxis)
    initial_snapshot ← metrics_tracker.compute_snapshot()

    # ── Phase 1: Attribution ──
    attribution_scores ← compute_trajectory_attribution_scores(
        trajectories=bundle.trajectories,
        pickup_counts=bundle.pickup_grid,
        dropoff_counts=bundle.dropoff_grid,
        supply_counts=bundle.active_taxis_grid,
        g_function=bundle.g_function,
        lis_weight=config.lis_weight,
        dcd_weight=config.dcd_weight,
        causal_formulation=config.causal_formulation,
        hat_matrices=bundle.hat_matrices,
        g0_power_basis_func=bundle.g0_power_basis_func,
        active_cell_indices=bundle.active_cell_indices,
    )

    selected_indices ← select_top_k_by_attribution(
        trajectory_scores=attribution_scores,
        k=config.top_k,
        selection_method=config.selection_method,
    )

    # ── Phase 2: Modification ──
    objective ← FAMAILObjective(
        alpha_spatial=config.alpha_spatial,
        alpha_causal=config.alpha_causal,
        alpha_fidelity=config.alpha_fidelity,
        g_function=bundle.g_function,
        discriminator=loaded_discriminator,   # If checkpoint_path provided
        causal_formulation=config.causal_formulation,
        hat_matrices=bundle.hat_matrices,
        g0_power_basis_func=bundle.g0_power_basis_func,
        active_cell_indices=bundle.active_cell_indices,
    )

    modifier ← TrajectoryModifier(
        objective_fn=objective,
        alpha=config.alpha,
        epsilon=config.epsilon,
        max_iterations=config.max_iterations,
        convergence_threshold=config.convergence_threshold,
        gradient_mode=config.gradient_mode,
        temperature=config.temperature,
        temperature_annealing=config.temperature_annealing,
        tau_max=config.tau_max,
        tau_min=config.tau_min,
    )

    modifier.set_global_state(
        pickup_counts=tensor(bundle.pickup_grid),
        dropoff_counts=tensor(bundle.dropoff_grid),
        active_taxis=tensor(bundle.active_taxis_grid),
        causal_demand=tensor(bundle.causal_demand_grid),
        causal_supply=tensor(bundle.causal_supply_grid),
    )

    histories ← []
    FOR EACH idx IN selected_indices:
        traj ← bundle.trajectories[idx]
        history ← modifier.modify_single(traj)
        histories.append(history)

        # Update metrics tracker
        IF history.iterations:
            orig_cell ← (traj.states[-1].x_grid, traj.states[-1].y_grid)
            mod_cell ← (history.modified.states[-1].x_grid, history.modified.states[-1].y_grid)
            metrics_tracker.update_pickup(orig_cell, mod_cell, fidelity_score=...)

    # ── Evaluation ──
    final_snapshot ← metrics_tracker.compute_snapshot()

    RETURN AlgorithmResult(
        attribution_scores, selected_indices,
        histories, initial_snapshot, final_snapshot, config
    )
```

---

## 7. Objective Function Components

### 7.1 Combined Objective (`objective.py`)

The `FAMAILObjective(nn.Module)` computes:

```
L = α₁ × F_spatial + α₂ × F_causal + α₃ × F_fidelity
```

All terms are in [0, 1] where higher = better (fairer/more realistic).

**Source**: `trajectory_modification/objective.py`

The objective delegates to three sub-components. In the new module, the math for each term is separated into its own file for auditability, but `FAMAILObjective` remains the single `nn.Module` that composes them.

**Constructor parameters** (from current `objective.py`):
- `alpha_spatial`, `alpha_causal`, `alpha_fidelity`: Weighting coefficients (default ~0.33 each)
- `grid_dims`: (48, 90) for Shenzhen
- `neighborhood_size`: Soft cell window (default 5 → 5×5)
- `temperature`: Initial soft assignment temperature
- `g_function`: Frozen g₀(D) for F_causal
- `discriminator`: Frozen SiameseLSTM for F_fidelity
- `causal_formulation`: `"baseline"`, `"dri"` (formerly `"option_b"`), or `"partial_r2"` (formerly `"option_c"`)
- `hat_matrices`: Pre-computed constant matrices for DRI formulation
- `g0_power_basis_func`: Power basis g₀(D) for DRI formulation
- `active_cell_indices`: Boolean mask of cells with valid demographics

**Key method — `forward()`**:
```python
def forward(
    self,
    pickup_counts,      # [48, 90] — differentiable soft counts
    dropoff_counts,     # [48, 90]
    supply,             # [48, 90] — for F_causal (mean-aggregated)
    active_taxis,       # [48, 90] — for F_spatial
    tau_features=None,  # [1, seq_len, 4] — original trajectory
    tau_prime_features=None,  # [1, seq_len, 4] — modified trajectory
    causal_demand=None, # [48, 90] — demand for F_causal (mean-aggregated)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Returns (total_loss, {f_spatial, f_causal, f_fidelity})"""
```

### 7.2 F_spatial — Spatial Fairness (`spatial_fairness.py`)

**Source**: `trajectory_modification/objective.py` methods `compute_spatial_fairness()` and `_pairwise_gini()`

**Formula**: F_spatial = 1 - 0.5 × (Gini(DSR) + Gini(ASR))

Where:
- DSR_c = pickups_c / active_taxis_c (Departure Service Rate)
- ASR_c = dropoffs_c / active_taxis_c (Arrival Service Rate)
- Gini = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ) (differentiable pairwise formula)

**Key implementation details**:
- Computed over **service-active cells only**: `(pickups > ε) | (dropoffs > ε) | (active_taxis > 0.5)` (~2000 of 4320 cells)
- The 0.5 threshold distinguishes real taxi activity from the artificial 0.1 floor applied in data_loader
- Pairwise Gini is O(n²) — the dominant computational cost per ST-iFGSM iteration

**Functions to extract**:
- `pairwise_gini(values: torch.Tensor) -> torch.Tensor` — differentiable Gini
- `compute_spatial_fairness(pickup_counts, dropoff_counts, active_taxis) -> torch.Tensor` — full F_spatial

### 7.3 F_causal — Causal Fairness (`causal_fairness.py`)

This is the most complex component. It must consolidate code from two current locations:
1. **Differentiable implementations**: `objective_function/causal_fairness/utils.py`
2. **Objective integration**: `trajectory_modification/objective.py`
3. **NumPy evaluation**: `trajectory_modification/metrics.py` (uses numpy version)

#### 7.3.1 DRI Formulation (formerly "Option B") — PREFERRED

**Formula**: F_causal = R'(I-H)R / R'MR

Where:
- R = Y - g₀(D) — demand-controlled residuals (Y = supply/demand)
- H = X(X'X)⁻¹X' — hat matrix projecting onto demographic feature space
- M = I - 11'/N — centering matrix
- (I-H) and M are pre-computed constants (demographics are fixed)

**Interpretation**: Fraction of demand-adjusted service variation NOT explained by demographics. F=1 is perfectly fair.

**Functions to port from `objective_function/causal_fairness/utils.py`**:

```
# Hat matrix construction (called once during data loading)
build_power_basis_features(demands, include_intercept=True) -> np.ndarray
build_hat_matrix(X) -> np.ndarray
build_centering_matrix(n) -> np.ndarray
precompute_hat_matrices(demands, demographic_features, feature_names) -> Dict

# g₀(D) fitting (called once during data loading)
estimate_g_power_basis(demands, ratios) -> Tuple[Callable, Dict]
estimate_g_isotonic(demands, ratios) -> Tuple[Callable, Dict]

# Differentiable computation (called every ST-iFGSM iteration)
compute_fcausal_option_b_torch(R, I_minus_H, M, eps) -> torch.Tensor

# Non-differentiable evaluation (called for before/after metrics)
compute_fcausal_option_b_numpy(R, I_minus_H, M, eps) -> Dict[str, float]
```

**Functions to port from `objective.py`**:
```
# Integration methods from FAMAILObjective
_compute_option_b_causal(demand, supply) -> torch.Tensor
_get_hat_matrix_tensors(device, dtype) -> Dict[str, torch.Tensor]
```

#### 7.3.2 Baseline Formulation

**Formula**: F_causal = R² = 1 - Var(Y - g₀(D)) / Var(Y)

Where g₀(D) is an isotonic regression fitted on (D, Y) pairs.

**Functions to port from `objective_function/causal_fairness/utils.py`**:
```
compute_r_squared(y_true, y_pred, eps) -> float             # NumPy
compute_r_squared_torch(y_true, y_pred, eps) -> torch.Tensor # Differentiable
estimate_g_isotonic(demands, ratios) -> Tuple[Callable, Dict]
```

#### 7.3.3 Partial ΔR² Formulation (formerly "Option C")

**Formula**: F_causal = 1 - ΔR² = 1 - (R²_full - R²_red)

**Functions to port**:
```
compute_fcausal_option_c_torch(Y, I_minus_H_red, I_minus_H_full, M, eps) -> torch.Tensor
compute_fcausal_option_c_numpy(Y, I_minus_H_red, I_minus_H_full, M, eps) -> Dict
```

#### 7.3.4 Supporting Data Functions

These are used by `data_loader.py` to prepare data for F_causal:

```
# Service ratio computation
extract_demand_from_counts(pickup_dropoff_data) -> Dict[Tuple, int]
aggregate_to_period(data, period_type, aggregator) -> Dict
compute_service_ratios(demand, supply, min_demand, ...) -> Dict[Tuple, float]
extract_demand_ratio_arrays(demand, ratios) -> Tuple[np.ndarray, np.ndarray, List]

# Demographic feature engineering
enrich_demographic_features(demo_grid, feature_names) -> Tuple[np.ndarray, List[str]]
```

### 7.4 F_fidelity — Behavioral Fidelity (`fidelity.py`)

**Source**: `trajectory_modification/objective.py` method `compute_fidelity()`

**Formula**: F_fidelity = Discriminator(τ, τ') ∈ [0, 1]

The discriminator evaluates whether the modified trajectory τ' is indistinguishable from the original τ (same-agent classification).

**Function to extract**:
```
compute_fidelity(tau_features, tau_prime_features, discriminator) -> torch.Tensor
```

**Key details**:
- Input features: [batch, seq_len, 4] where 4 = (x_grid, y_grid, time_bucket, day_index)
- Discriminator is in eval mode with requires_grad=False for its parameters
- Gradients still flow through the input features (τ' carries grad)
- If α₃ = 0, fidelity computation is skipped (no discriminator needed)
- If no discriminator loaded, returns tensor(0.0) as fallback

---

## 8. Soft Cell Assignment

### 8.1 Purpose

Solves the non-differentiability of hard grid cell assignment. When the optimizer moves a pickup location continuously, the count in the destination cell changes discretely — breaking gradient flow. SoftCellAssignment replaces this hard assignment with a Gaussian softmax over a neighborhood of cells.

### 8.2 Source

`objective_function/soft_cell_assignment/module.py`

### 8.3 Target: `soft_cell.py`

#### `SoftCellAssignment(nn.Module)`

**Constructor**:
```python
SoftCellAssignment(
    grid_dims: Tuple[int, int] = (48, 90),
    neighborhood_size: int = 5,           # Window = 5×5
    initial_temperature: float = 1.0,
    eps: float = 1e-8,
)
```

**Key methods**:
- `forward(location, original_cell) -> probs [batch, ns, ns]`
  - Computes Gaussian softmax: σ_c(x,y) = exp(-d²/(2τ²)) / Z
  - `location`: [batch, 2] continuous (x, y) with requires_grad=True
  - `original_cell`: [batch, 2] integer center of the neighborhood
  - Returns probability distribution over neighborhood cells

- `set_temperature(temperature: float)` — update τ
- `get_annealed_temperature(iteration, total, tau_max, tau_min) -> float`
  - Exponential: τ_t = τ_max × (τ_min/τ_max)^(t/(T-1))

**Also port**:
- `update_counts_with_soft_assignment(base_counts, soft_probs, original_cell, subtract_original)` — functional helper for updating count grids

### 8.4 Gradient Flow

```
pickup_location (requires_grad=True)
    → SoftCellAssignment.forward() → probs [5×5]
        → scatter into count grid → soft_pickup_counts [48×90]
            → FAMAILObjective.forward() → L
                → L.backward() → ∂L/∂pickup_location
```

---

## 9. Discriminator & Fidelity

### 9.1 Model Architecture (`model.py`)

**Source**: `discriminator/model/model.py`

Port the following classes:

#### `FeatureNormalizer`
- Converts 4D raw features to 6D normalized: [x/49, y/89, sin(2πt/288), cos(2πt/288), sin(2πd/5), cos(2πd/5)]
- Input: [batch, seq_len, 4] → Output: [batch, seq_len, 6]

#### `SiameseLSTMEncoder`
- Stacked LSTM layers with variable hidden dims per layer
- Bidirectional support
- Dropout between layers
- Output: trajectory embedding [batch, emb_dim]

#### `SiameseLSTMDiscriminator` (V1)
- Constructor: `(lstm_hidden_dims=(200,100), dropout=0.2, bidirectional=True, classifier_hidden_dims=(64,32,8))`
- Combination: concatenation of embeddings
- Forward: `(x1, x2, mask1=None, mask2=None) -> [batch, 1]` probability

#### `SiameseLSTMDiscriminatorV2` (V2 — Preferred)
- Constructor: adds `combination_mode="difference"` parameter
- Combination modes: `"difference"` (|emb1-emb2|), `"distance"` (+cosine_sim, euclidean), `"hybrid"`
- Key improvement: Identical trajectories naturally produce zero difference → high similarity score

### 9.2 Adapter (`discriminator_adapter.py`)

**Source**: `trajectory_modification/discriminator_adapter.py`

#### `DiscriminatorAdapter`
- **Constructor**: `(checkpoint_path, device='cpu', threshold=0.5)`
- **Methods**:
  - `load_checkpoint(path)` — auto-detects V1 vs V2 from saved config
  - `evaluate(tau_original, tau_modified) -> float` — similarity score
  - `is_same_agent(tau_original, tau_modified) -> bool`
  - `get_similarity_with_grad(tau_features, tau_prime_features) -> torch.Tensor` — for optimization

### 9.3 Import Change

Current: `from discriminator.model import SiameseLSTMDiscriminator, SiameseLSTMDiscriminatorV2`

New: `from .model import SiameseLSTMDiscriminator, SiameseLSTMDiscriminatorV2`

---

## 10. Data Loading & DataBundle

### 10.1 Source

`trajectory_modification/data_loader.py`

### 10.2 Target: `data_loader.py`

The `DataBundle` dataclass and its loaders port with one key change: the imports from `objective_function.causal_fairness.utils` are replaced with imports from local `causal_fairness.py`.

#### `DataBundle` fields

```python
@dataclass
class DataBundle:
    # Core trajectory data
    trajectories: List[Trajectory]

    # Spatial fairness grids (SUM-aggregated — total counts)
    pickup_grid: np.ndarray              # [48, 90]
    dropoff_grid: np.ndarray             # [48, 90]

    # Supply data
    active_taxis_data: Dict[Tuple, int]  # Raw hourly dict
    active_taxis_grid: np.ndarray        # [48, 90] MEAN-aggregated, floored at 0.1

    # Causal fairness grids (MEAN-aggregated — per-period averages for Y=S/D scale)
    causal_demand_grid: Optional[np.ndarray]   # [48, 90]
    causal_supply_grid: Optional[np.ndarray]   # [48, 90]

    # g₀(D) function
    g_function: Callable                 # Isotonic regression (for baseline)
    g0_power_basis_func: Optional[Callable]  # Power basis (for DRI formulation)
    g_function_diagnostics: Optional[Dict]

    # Hat matrices (for DRI formulation)
    hat_matrices: Optional[Dict[str, Any]]  # {I_minus_H_demo, M, I_minus_H_red, I_minus_H_full, ...}
    active_cell_indices: Optional[np.ndarray]  # Boolean mask

    # Demographics (for hat matrix construction)
    demographics_grid: Optional[np.ndarray]  # [48, 90, n_features]
    demographic_feature_names: Optional[List[str]]

    # Raw data (retained for reference)
    pickup_dropoff_data: Dict[Tuple, Tuple[int, int]]  # 1-indexed
    grid_dims: Tuple[int, int] = (48, 90)
```

#### `DataBundle.load_default()` — Critical Data Pipeline

This class method orchestrates all data loading and preprocessing:

1. **Load trajectories**: `TrajectoryLoader.load_passenger_seeking()` → List[Trajectory]
   - Source: `source_data/passenger_seeking_trajs_45-800.pkl`
   - Converts 0-indexed state arrays to TrajectoryState objects

2. **Load pickup/dropoff counts**: `PickupDropoffLoader.load()` → Dict
   - Source: `source_data/pickup_dropoff_counts.pkl`
   - Note: Uses 1-indexed coordinates; aggregate_to_grid converts to 0-indexed

3. **Aggregate spatial grids**: `PickupDropoffLoader.aggregate_to_grid(data, 'sum')` → [48, 90]
   - SUM aggregation for spatial fairness (total counts across all time/days)

4. **Load active taxis**: `ActiveTaxisLoader.load('hourly')` → Dict
   - Source: `source_data/active_taxis_5x5_hourly.pkl`
   - Aggregate to grid with MEAN, floor at 0.1

5. **Fit g(D) via isotonic regression**: `GFunctionLoader.estimate_from_data()`
   - Calls causal_fairness utils: extract_demand → aggregate_to_period('hourly') → compute_service_ratios → estimate_g_isotonic
   - Returns (g_function, diagnostics)

6. **Create causal grids** (MEAN-aggregated for proper Y=S/D scale):
   - `PickupDropoffLoader.aggregate_to_grid(data, 'mean')` → causal_demand_grid
   - `ActiveTaxisLoader.aggregate_to_grid(data, 'mean')` → causal_supply_grid

7. **Load demographics**: `source_data/cell_demographics.pkl`
   - Contains demographics_grid [48, 90, 13] and feature_names

8. **Enrich demographics**: `enrich_demographic_features()` → [48, 90, 20]
   - Adds derived features: GDPperCapita, CompPerCapita, MigrantRatio, LogGDP, LogHousingPrice, LogCompensation, LogPopDensity

9. **Fit g₀(D) power basis**: `estimate_g_power_basis(demands, ratios)` → g0_func
   - g₀(D) = β₀ + β₁/(D+1) + β₂/√(D+1) + β₃√(D+1)

10. **Pre-compute hat matrices**: `precompute_hat_matrices(demands, demo_features, names)` → Dict
    - Computes I_minus_H_demo, M, and optionally I_minus_H_red, I_minus_H_full

### 10.3 Coordinate System (CRITICAL)

**In pickle files** (pickup_dropoff_counts, active_taxis):
- x_grid: 1–48, y_grid: 1–90 (1-indexed)

**In memory** (arrays, Trajectory objects):
- x_grid: 0–47, y_grid: 0–89 (0-indexed)

**Geographic convention**:
- Origin (0,0) = south-west corner
- x_grid increases northward (latitude)
- y_grid increases eastward (longitude)

The loaders handle the 1→0 index conversion via `aggregate_to_grid()`.

---

## 11. Global Metrics & Evaluation

### 11.1 Source

`trajectory_modification/metrics.py`

### 11.2 Target: `metrics.py`

#### `FairnessSnapshot` dataclass
```python
@dataclass
class FairnessSnapshot:
    gini_coefficient: float
    r_squared: float           # For F_causal
    mean_fidelity: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    combined_objective: float  # L = α₁F_sp + α₂F_ca + α₃F_fi
    num_trajectories_modified: int
```

#### `GlobalMetrics` class
- Tracks system-wide fairness as trajectories are modified
- Uses **NumPy** (not PyTorch) for efficient metric computation
- Maintains copies of pickup_counts, supply_counts, active_taxis
- Methods:
  - `initialize_from_data(pickup, supply, active_taxis)`
  - `update_pickup(old_cell, new_cell, fidelity_score=None)`
  - `compute_gini() -> float` — DSR-based Gini
  - `compute_f_causal() -> float` — dispatches to DRI or baseline
  - `compute_mean_fidelity() -> float`
  - `compute_snapshot() -> FairnessSnapshot`
  - `get_improvement() -> Dict[str, float]` — delta from first to last snapshot

### 11.3 Changes

- Import `compute_fcausal_option_b_numpy` from local `causal_fairness.py` instead of `objective_function.causal_fairness.utils`

---

## 12. Configuration & Hyperparameters

### 12.1 Target: `config.py`

Currently, configuration is spread across constructor parameters in `TrajectoryModifier`, `FAMAILObjective`, dashboard sidebar widgets, and data loading parameters. The new module consolidates everything into a single configuration class.

```python
@dataclass
class FAMAILConfig:
    """Complete configuration for the FAMAIL trajectory editing algorithm."""

    # ── Phase 1: Attribution ──
    top_k: int = 10                          # Number of trajectories to select
    selection_method: str = 'top_k'          # 'top_k' or 'diverse'
    lis_weight: float = 0.5                  # Weight for LIS in attribution
    dcd_weight: float = 0.5                  # Weight for DCD in attribution
    diversity_penalty: float = 0.5           # β for diverse selection mode

    # ── Phase 2: ST-iFGSM ──
    alpha: float = 0.1                       # Step size
    epsilon: float = 3.0                     # Max perturbation per dimension (grid cells)
    max_iterations: int = 50                 # Max ST-iFGSM iterations per trajectory
    convergence_threshold: float = 1e-6      # |ΔL| convergence test

    # ── Objective Weights ──
    alpha_spatial: float = 0.33              # Weight for F_spatial
    alpha_causal: float = 0.33              # Weight for F_causal
    alpha_fidelity: float = 0.34            # Weight for F_fidelity

    # ── F_causal Formulation ──
    causal_formulation: str = 'dri'          # 'baseline', 'dri', or 'partial_r2'

    # ── Gradient Mode ──
    gradient_mode: str = 'soft_cell'         # 'soft_cell' or 'heuristic'
    neighborhood_size: int = 5               # Soft cell window (5×5)

    # ── Temperature Annealing ──
    temperature: float = 1.0                 # Initial temperature
    temperature_annealing: bool = True       # Enable annealing
    tau_max: float = 1.0                     # Annealing start
    tau_min: float = 0.1                     # Annealing end

    # ── Discriminator ──
    discriminator_checkpoint: Optional[str] = None  # Path to trained .pt file
    device: str = 'auto'                     # 'auto', 'cuda', or 'cpu'

    # ── Data Loading ──
    max_trajectories: int = 100              # Limit trajectories loaded
    max_drivers: Optional[int] = None        # Limit drivers loaded
    active_taxis_period: str = 'hourly'      # 'hourly' or 'time_bucket'
    causal_aggregation: str = 'mean'         # Aggregation for causal grids

    # ── Grid ──
    grid_dims: Tuple[int, int] = (48, 90)    # Shenzhen grid dimensions

    def validate(self):
        """Validate configuration consistency."""
        assert self.alpha > 0
        assert self.epsilon > 0
        assert self.max_iterations > 0
        assert self.top_k > 0
        total = self.alpha_spatial + self.alpha_causal + self.alpha_fidelity
        assert abs(total - 1.0) < 0.01, f"Objective weights should sum to ~1.0, got {total}"
        assert self.causal_formulation in ('baseline', 'dri', 'partial_r2')
        assert self.gradient_mode in ('soft_cell', 'heuristic')
        assert self.selection_method in ('top_k', 'diverse')

    def normalize_weights(self):
        """Normalize objective weights to sum to 1.0."""
        total = self.alpha_spatial + self.alpha_causal + self.alpha_fidelity
        if total > 0:
            self.alpha_spatial /= total
            self.alpha_causal /= total
            self.alpha_fidelity /= total
```

### 12.2 Default Parameters Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `top_k` | 10 | 1–100 | Trajectories to modify |
| `selection_method` | `'top_k'` | `top_k`, `diverse` | Selection strategy |
| `lis_weight` | 0.5 | 0.0–1.0 | Attribution LIS weight |
| `dcd_weight` | 0.5 | 0.0–1.0 | Attribution DCD weight |
| `alpha` | 0.1 | 0.01–1.0 | ST-iFGSM step size |
| `epsilon` | 3.0 | 1.0–10.0 | Max perturbation (grid cells) |
| `max_iterations` | 50 | 10–200 | Max iterations per trajectory |
| `convergence_threshold` | 1e-6 | 1e-8–1e-2 | Convergence tolerance |
| `alpha_spatial` | 0.33 | 0.0–1.0 | F_spatial weight |
| `alpha_causal` | 0.33 | 0.0–1.0 | F_causal weight |
| `alpha_fidelity` | 0.34 | 0.0–1.0 | F_fidelity weight |
| `causal_formulation` | `'dri'` | `baseline`, `dri`, `partial_r2` | F_causal variant |
| `gradient_mode` | `'soft_cell'` | `soft_cell`, `heuristic` | Gradient computation |
| `temperature` | 1.0 | 0.01–5.0 | Soft assignment temperature |
| `tau_max` | 1.0 | 0.1–5.0 | Annealing start temperature |
| `tau_min` | 0.1 | 0.01–1.0 | Annealing end temperature |

---

## 13. Data Preparation Pipeline

The algorithm requires several preprocessed datasets. These are prepared by separate tools/scripts that are NOT part of `famail_algorithm/` but must be run beforehand.

### 13.1 Required Data Files

| File | Source Tool | Description |
|------|-----------|-------------|
| `source_data/passenger_seeking_trajs_45-800.pkl` | External | Passenger-seeking trajectories (50 drivers) |
| `source_data/pickup_dropoff_counts.pkl` | `pickup_dropoff_counts/processor.py` | Aggregated pickup/dropoff events |
| `source_data/active_taxis_5x5_hourly.pkl` | `active_taxis/generation.py` | Active taxi counts per cell/hour |
| `source_data/cell_demographics.pkl` | `data/demographic_data/` notebook | Per-cell demographic features |
| `source_data/grid_to_district_mapping.pkl` | `data/geo_data/` notebook | Grid cell → Shenzhen district mapping |

### 13.2 Pickup/Dropoff Count Preparation

**Tool**: `pickup_dropoff_counts/processor.py`
**Dashboard**: `streamlit run pickup_dropoff_counts/app.py`

**Input**: Raw GPS data files in `raw_data/`:
- `taxi_record_07_50drivers.pkl` (July 2016)
- `taxi_record_08_50drivers.pkl` (August 2016)
- `taxi_record_09_50drivers.pkl` (September 2016)

**Processing**:
1. Parse GPS records: each is [plate_id, lat, lon, seconds_since_midnight, passenger_indicator, timestamp]
2. Spatial quantization: GPS → 48×90 grid (0.01° ≈ 1.1 km cells)
3. Temporal quantization: time → 288 five-minute buckets per day
4. Event detection: pickup = passenger 0→1, dropoff = passenger 1→0
5. Aggregate by (x_grid, y_grid, time_bucket, day)

**Output format**: `{(x, y, time_bucket, day): (pickup_count, dropoff_count), ...}`
- **1-indexed** coordinates: x=1–48, y=1–90, time=1–288, day=1–6

**Run**:
```bash
cd pickup_dropoff_counts
python processor.py --output ../source_data/pickup_dropoff_counts.pkl
```

### 13.3 Active Taxis Preparation

**Tool**: `active_taxis/generation.py`
**Dashboard**: `streamlit run active_taxis/app.py`

**Input**: Same raw GPS files as pickup/dropoff

**Processing**:
1. Parse GPS records (same as above)
2. For each (cell, time_period), count unique taxis present in a 5×5 neighborhood
3. Neighborhood size k=2 means the 5×5 grid centered on each cell

**Output format**: `{(x, y, hour, day): count, ...}`
- **1-indexed** coordinates
- Hourly aggregation (24 hours) is the default and recommended

**Configuration**:
```python
ActiveTaxisConfig(
    neighborhood_size=2,     # 5×5 neighborhood
    period_type='hourly',    # Recommended
    exclude_sunday=True,
)
```

**Run**:
```bash
# Via Streamlit dashboard (recommended — interactive config)
streamlit run active_taxis/app.py

# Programmatic
python -c "
from active_taxis import ActiveTaxisConfig, generate_active_taxi_counts, save_output
from pathlib import Path
config = ActiveTaxisConfig(neighborhood_size=2, period_type='hourly')
counts, stats = generate_active_taxi_counts(config)
save_output(counts, stats, config, Path('source_data/active_taxis_5x5_hourly.pkl'))
"
```

### 13.4 Demographic Data Preparation

**Source data**: `data/demographic_data/all_demographics_by_district.csv`
- 14 demographic features across 10 Shenzhen districts
- Must be sourced from Shenzhen Statistical Bureau or equivalent

**Grid mapping**: `data/geo_data/create_48x90_grid_map_districts.ipynb`
- ArcGIS grid-to-district intersection analysis
- Maps each 48×90 grid cell to a Shenzhen district
- Output: `source_data/grid_to_district_mapping.pkl`

**Demographics merge**: `data/demographic_data/join_grid-to-district_and_demographics.ipynb`
- Joins grid mapping with district demographics
- Output: `source_data/cell_demographics.pkl`
  - `demographics_grid`: [48, 90, 13] array
  - `feature_names`: list of 13 feature names

**Feature enrichment** (done automatically by DataBundle.load_default):
- Adds 7 derived features → [48, 90, 20] grid
- Derived: GDPperCapita, CompPerCapita, MigrantRatio, LogGDP, LogHousingPrice, LogCompensation, LogPopDensity

### 13.5 Trajectory Data

**File**: `source_data/passenger_seeking_trajs_45-800.pkl`

**Format**: `{driver_id: [trajectory, trajectory, ...], ...}`
- Each trajectory is a list of states: `[x_grid, y_grid, time_bucket, day_index]`
- 0-indexed coordinates
- Trajectory states go from passenger-seeking start (states[0]) to pickup (states[-1])
- Only the pickup location (last state) is modified by the algorithm

---

## 14. Discriminator Training Pipeline

The discriminator model must be trained before the algorithm can use F_fidelity. This is a separate pipeline that produces a checkpoint file.

### 14.1 Training Data Generation

**Tool**: `discriminator/dataset_generation_tool/`
**Dashboard**: `streamlit run discriminator/dataset_generation_tool/app.py`

**Input**: `source_data/all_trajs.pkl` — complete trajectory data (50 drivers, 126-element state vectors)

**Output**: `.npz` files with trajectory pairs:
- `x1`: [N, L, 4] first trajectories (x_grid, y_grid, time_bucket, day_index)
- `x2`: [N, L, 4] second trajectories
- `label`: [N] binary labels (1=same agent, 0=different agent)
- `mask1`, `mask2`: [N, L] boolean masks for variable-length sequences

**Key configuration** (`GenerationConfig`):
- `positive_pairs`: Number of same-agent pairs (e.g., 5000)
- `negative_pairs`: Number of different-agent pairs (e.g., 20000)
- `identical_pair_ratio`: Fraction of positive pairs where both elements are the SAME trajectory (e.g., 0.15) — critical for V2 model training
- `negative_strategy`: `"random"`, `"temporal_hard"`, `"spatial_hard"`, `"mixed_hard"`

### 14.2 Model Training

**Tool**: `discriminator/model/train.py`
**Dashboard**: `streamlit run discriminator/model/training_dashboard.py`

**Recommended configuration** (based on best-performing checkpoint):
```bash
python discriminator/model/train.py \
    --data-dir discriminator/datasets/<dataset_name>/ \
    --model-version v2 \
    --combination-mode difference \
    --lstm-hidden-dims 200,100 \
    --classifier-dims 64,32,8 \
    --dropout 0.2 \
    --batch-size 64 \
    --lr 0.001 \
    --epochs 100 \
    --early-stopping 10 \
    --output discriminator/model/checkpoints \
    --experiment-name <experiment_name>
```

**Output**: Checkpoint directory containing:
- `best.pt` — best model weights + config
- `config.json` — training configuration
- `history.json` — per-epoch metrics
- `results.json` — final evaluation

### 14.3 Using a Trained Discriminator

The algorithm loads the discriminator via `DiscriminatorAdapter`:
```python
config = FAMAILConfig(
    discriminator_checkpoint='discriminator/model/checkpoints/<experiment>/best.pt',
    alpha_fidelity=0.34,  # Non-zero to enable fidelity
)
```

If `discriminator_checkpoint` is None or `alpha_fidelity` is 0, the fidelity term is skipped.

### 14.4 Current Best Checkpoint

```
discriminator/model/checkpoints/pass-seek_5000-20000_(84ident_72same_44diff)/best.pt
```
- Architecture: SiameseLSTMDiscriminatorV2, combination_mode='difference'
- Performance: 84% identical, 72% same-agent, 44% different-agent accuracy
- Training data: 5000 positive + 20000 negative pairs

---

## 15. Naming Conventions & Terminology

### 15.1 F_causal Formulation Naming

The internal codename "Option B" originated from the formulation comparison process (Options A1, A2, B, C, D) documented in `FCAUSAL_FORMULATIONS.md`. Now that Option B is the chosen formulation, the name should reflect what it IS, not how it was selected.

**Proposed rename**:

| Old Name | New Name (Code) | Full Descriptive Name |
|----------|----------------|----------------------|
| `option_b` | `dri` | **Demographic Residual Independence** |
| `option_c` | `partial_r2` | Partial ΔR² |
| `baseline` | `baseline` | Baseline (R² with isotonic g₀) |

**Rationale**: "Demographic Residual Independence" (DRI) accurately describes what the formulation measures — the independence of demand-controlled residuals from demographic features. The abbreviation `dri` is concise for code while the full name is used in documentation and UI.

**DCD Attribution rename** (related):
- Under the DRI formulation, DCD computes |R̂_c| where R̂ = H_demo @ R — the magnitude of the demographically-explained residual
- This could be called "Demographic Residual Attribution" but the DCD name is fine since it's already documented

### 15.2 Code Convention Changes

| Current | Proposed | Where |
|---------|----------|-------|
| `causal_formulation="option_b"` | `causal_formulation="dri"` | FAMAILConfig, FAMAILObjective, attribution functions |
| `causal_formulation="option_c"` | `causal_formulation="partial_r2"` | FAMAILConfig, FAMAILObjective |
| `compute_fcausal_option_b_torch()` | `compute_fcausal_dri_torch()` | causal_fairness.py |
| `compute_fcausal_option_b_numpy()` | `compute_fcausal_dri_numpy()` | causal_fairness.py |
| `compute_fcausal_option_c_torch()` | `compute_fcausal_partial_r2_torch()` | causal_fairness.py |
| `_compute_option_b_causal()` | `_compute_dri_causal()` | objective.py |
| `I_minus_H_demo` | `I_minus_H_demo` | No change (already descriptive) |
| `H_demo` | `H_demo` | No change |

---

## 16. Cross-Module Dependency Summary

### 16.1 External Python Dependencies

| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| `torch` | ≥2.0 | All | Differentiable computation, model inference |
| `numpy` | ≥1.24 | All | Array operations |
| `scikit-learn` | ≥1.3 | causal_fairness.py, data_loader.py | IsotonicRegression, StandardScaler, LinearRegression |
| `scipy` | ≥1.10 | causal_fairness.py | `scipy.stats.binned_statistic` (for binning g method) |

**NOT required in the algorithm module** (dashboard-only):
- streamlit, plotly, matplotlib, pandas, altair

### 16.2 Internal Dependencies (within famail_algorithm/)

```
algorithm.py
├── attribution.py     (Phase 1)
├── modifier.py        (Phase 2)
├── objective.py       (Combined objective)
│   ├── spatial_fairness.py
│   ├── causal_fairness.py
│   └── fidelity.py
├── metrics.py         (Evaluation)
├── data_loader.py     (DataBundle)
├── config.py          (FAMAILConfig)
└── trajectory.py      (Data types)

modifier.py
├── soft_cell.py       (Differentiable grid assignment)
├── objective.py       (Forward + backward)
└── trajectory.py

objective.py
├── spatial_fairness.py
├── causal_fairness.py
├── fidelity.py
└── soft_cell.py

data_loader.py
├── trajectory.py
├── causal_fairness.py  (g₀ fitting, hat matrices, service ratios)
└── config.py

discriminator_adapter.py
└── model.py           (SiameseLSTM architectures)

metrics.py
└── causal_fairness.py  (NumPy F_causal computation)
```

### 16.3 Dependencies on External Data Files

```
famail_algorithm/ reads from:
├── source_data/passenger_seeking_trajs_45-800.pkl    (trajectories)
├── source_data/pickup_dropoff_counts.pkl              (pickup/dropoff events)
├── source_data/active_taxis_5x5_hourly.pkl            (supply proxy)
├── source_data/cell_demographics.pkl                  (demographics grid)
├── source_data/grid_to_district_mapping.pkl           (district mapping)
└── discriminator/model/checkpoints/<exp>/best.pt      (trained discriminator)
```

All paths are relative to the workspace root, located dynamically by `find_workspace_root()`.

---

## 17. Migration Checklist

### Phase 0: Setup
- [ ] Create `famail_algorithm/` directory
- [ ] Create `famail_algorithm/__init__.py` with public API exports
- [ ] Create `famail_algorithm/config.py` with `FAMAILConfig` dataclass

### Phase 1: Core Data Types (No Dependencies)
- [ ] Port `trajectory.py` (TrajectoryState, Trajectory)
- [ ] Verify: `Trajectory.apply_perturbation()`, `clone()`, `to_tensor()`, `to_discriminator_format()`

### Phase 2: Differentiable Infrastructure
- [ ] Port `soft_cell.py` (SoftCellAssignment, update_counts_with_soft_assignment)
- [ ] Verify: gradient flow through forward(), temperature annealing

### Phase 3: Causal Fairness Math
- [ ] Port `causal_fairness.py` — consolidate from `objective_function/causal_fairness/utils.py`:
  - [ ] Hat matrix functions: `build_power_basis_features`, `build_hat_matrix`, `build_centering_matrix`, `precompute_hat_matrices`
  - [ ] g₀(D) fitting: `estimate_g_power_basis`, `estimate_g_isotonic`
  - [ ] Service ratio functions: `extract_demand_from_counts`, `aggregate_to_period`, `compute_service_ratios`, `extract_demand_ratio_arrays`
  - [ ] Demographic features: `enrich_demographic_features`
  - [ ] Differentiable F_causal: `compute_fcausal_dri_torch` (renamed from option_b), `compute_fcausal_partial_r2_torch` (renamed from option_c)
  - [ ] NumPy F_causal: `compute_fcausal_dri_numpy`, `compute_r_squared`
- [ ] Rename all `option_b` → `dri`, `option_c` → `partial_r2` in function names and parameters

### Phase 4: Objective Function Components
- [ ] Port `spatial_fairness.py` — extract from `objective.py`: `pairwise_gini`, `compute_spatial_fairness`
- [ ] Port `fidelity.py` — extract from `objective.py`: `compute_fidelity`
- [ ] Port `objective.py` — `FAMAILObjective(nn.Module)` with imports from local sub-modules

### Phase 5: Discriminator
- [ ] Port `model.py` — from `discriminator/model/model.py`: FeatureNormalizer, SiameseLSTMEncoder, SiameseLSTMDiscriminator, SiameseLSTMDiscriminatorV2
- [ ] Port `discriminator_adapter.py` — update import path to local `model.py`

### Phase 6: Data Loading
- [ ] Port `data_loader.py` — DataBundle, TrajectoryLoader, PickupDropoffLoader, ActiveTaxisLoader, GFunctionLoader
- [ ] Update imports: replace `objective_function.causal_fairness.utils` → local `causal_fairness`
- [ ] Verify: `DataBundle.load_default()` produces correct grids and hat matrices

### Phase 7: Attribution (Phase 1 Logic)
- [ ] Port `attribution.py` — extract from `dashboard.py`:
  - [ ] `compute_cell_lis_scores`
  - [ ] `compute_cell_dcd_scores` (with DRI support)
  - [ ] `compute_trajectory_attribution_scores`
  - [ ] `select_top_k_by_attribution`
- [ ] Rename `causal_formulation="option_b"` → `"dri"` in parameters

### Phase 8: Modifier (Phase 2 Logic)
- [ ] Port `modifier.py` — TrajectoryModifier, ModificationResult, ModificationHistory
- [ ] Update imports: SoftCellAssignment from local `soft_cell.py`

### Phase 9: Metrics
- [ ] Port `metrics.py` — FairnessSnapshot, GlobalMetrics
- [ ] Update imports: causal fairness numpy function from local `causal_fairness.py`

### Phase 10: Unified Algorithm Runner
- [ ] Create `algorithm.py` — FAMAILAlgorithm with `run()` method
- [ ] Implement unified Phase 1 → Phase 2 → Evaluation pipeline
- [ ] Create `AlgorithmResult` dataclass

### Phase 11: Public API
- [ ] Define `__init__.py` exports:
  ```python
  from .algorithm import FAMAILAlgorithm, AlgorithmResult
  from .config import FAMAILConfig
  from .data_loader import DataBundle
  from .trajectory import Trajectory, TrajectoryState
  from .modifier import TrajectoryModifier, ModificationHistory, ModificationResult
  from .objective import FAMAILObjective
  from .metrics import GlobalMetrics, FairnessSnapshot
  from .attribution import (
      compute_cell_lis_scores,
      compute_cell_dcd_scores,
      compute_trajectory_attribution_scores,
      select_top_k_by_attribution,
  )
  from .discriminator_adapter import DiscriminatorAdapter
  ```

### Phase 12: Integration Testing
- [ ] Verify: `DataBundle.load_default()` works from `famail_algorithm/`
- [ ] Verify: Full `FAMAILAlgorithm.run()` pipeline completes without errors
- [ ] Verify: Results match the existing `trajectory_modification/dashboard.py` output
- [ ] Verify: All gradient flows work (soft cell → objective → backward)
- [ ] Verify: DRI formulation produces same F_causal values as current `option_b`

### Phase 13: Dashboard Update (Optional)
- [ ] Update `trajectory_modification/dashboard.py` to import from `famail_algorithm/` instead of scattered sources
- [ ] Or: Create a new lightweight dashboard for `famail_algorithm/`

---

## 18. Open Questions

### 18.1 For Discussion

1. **Module name confirmation**: Is `famail_algorithm` the right name, or would something else be better? (e.g., `famail_core`, `trajectory_editor`)

2. **F_causal naming**: Is "Demographic Residual Independence" (DRI) the right descriptive name for the Option B formulation? Alternative: "Demographic Independence Test" (DIT)?

3. **Dashboard relationship**: Should `trajectory_modification/dashboard.py` be updated to import from `famail_algorithm/`, or should a new minimal dashboard be created?

4. **Backward compatibility**: Should the old `trajectory_modification/` module be kept as-is (with imports pointing to `famail_algorithm/`), or deprecated?

5. **Option C inclusion**: Should the partial ΔR² formulation be ported? It has a known limitation (hat matrices contain demand features that change during optimization). It could be included for completeness or excluded to simplify.

6. **Test suite**: Should unit tests be created as part of this extraction? Currently, testing is minimal — this would be an opportunity to add systematic tests for each component.

7. **Re-attribution between trajectories**: The computational analysis (Section 7.4 in the pseudocode doc) shows re-attribution is computationally free (0.0025% overhead). Should the unified runner re-compute attribution scores after each trajectory modification to ensure subsequent selections reflect the updated world state?

---

## Appendix A: Files to Read During Implementation

For each target file in `famail_algorithm/`, the implementer should read these source files:

| Target | Source Files to Read |
|--------|---------------------|
| `algorithm.py` | `trajectory_modification/dashboard.py` (workflow), `trajectory_modification/modifier.py` (batch logic) |
| `attribution.py` | `trajectory_modification/dashboard.py:227–531` |
| `modifier.py` | `trajectory_modification/modifier.py` (entire file) |
| `objective.py` | `trajectory_modification/objective.py` (entire file) |
| `spatial_fairness.py` | `trajectory_modification/objective.py:241–343` |
| `causal_fairness.py` | `objective_function/causal_fairness/utils.py` (subset — see Section 3.8), `trajectory_modification/objective.py:384–587` |
| `fidelity.py` | `trajectory_modification/objective.py:589–633` |
| `soft_cell.py` | `objective_function/soft_cell_assignment/module.py` |
| `trajectory.py` | `trajectory_modification/trajectory.py` (entire file) |
| `data_loader.py` | `trajectory_modification/data_loader.py` (entire file) |
| `config.py` | `trajectory_modification/modifier.py` constructor, `trajectory_modification/objective.py` constructor, `trajectory_modification/dashboard.py` sidebar widgets |
| `metrics.py` | `trajectory_modification/metrics.py` (entire file) |
| `discriminator_adapter.py` | `trajectory_modification/discriminator_adapter.py` (entire file) |
| `model.py` | `discriminator/model/model.py:30–780` (FeatureNormalizer, Encoder, V1, V2) |

## Appendix B: Reference Documents

| Document | Path | Relevance |
|----------|------|-----------|
| Algorithm pseudocode | `trajectory_modification/pseudocode_docs/algorithm_pseudocode.md` | Complete pseudocode for all phases |
| Option B formulation | `objective_function/causal_fairness/f_causal_term_reformulation/chosen_formulation/OPTION_B_FORMULATION.md` | Mathematical foundation for DRI F_causal |
| Formulation comparison | `objective_function/causal_fairness/f_causal_term_reformulation/FCAUSAL_FORMULATIONS.md` | All F_causal options compared |
| Changelog | `CHANGELOG.md` | History of algorithm fixes and decisions |
| CLAUDE.md | `CLAUDE.md` | Project-wide conventions and setup |
