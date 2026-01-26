# FAMAIL Trajectory Modification Tool

This module implements the **Fairness-Aware Trajectory Editing Algorithm** (Modified ST-iFGSM) for improving spatial and causal fairness in taxi service distribution.

## Algorithm

The ST-iFGSM algorithm modifies trajectory pickup locations to maximize the combined fairness objective:

```
L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
```

Where:
- **F_spatial** = 1 - Gini(DSR) measures spatial equity of service distribution
- **F_causal** = max(0, R²) measures causal relationship between demand and supply
- **F_fidelity** = f(τ, τ') measures behavioral similarity to original trajectory

### Perturbation Formula

```
δ = clip(α · sign[∇L], -ε, ε)
```

- α: Step size (default: 0.1)
- ε: Maximum perturbation per dimension (default: 3.0 grid cells)
- ∇L: Gradient of objective w.r.t. pickup location

## Module Structure

```
trajectory_modification/
├── __init__.py           # Module exports
├── trajectory.py         # Trajectory and TrajectoryState dataclasses
├── modifier.py           # TrajectoryModifier with ST-iFGSM loop
├── objective.py          # FAMAILObjective combined function
├── discriminator_adapter.py  # Wrapper for trained discriminator
├── metrics.py            # GlobalMetrics tracker
├── data_loader.py        # Data loading utilities
└── dashboard.py          # Streamlit dashboard
```

## Quick Start

### 1. Basic Usage

```python
from trajectory_modification import (
    DataBundle,
    TrajectoryModifier,
    FAMAILObjective,
    GlobalMetrics,
)
import torch

# Load data
bundle = DataBundle.load_default(max_trajectories=100)

# Create objective function
objective = FAMAILObjective(
    alpha_spatial=0.33,
    alpha_causal=0.33,
    alpha_fidelity=0.34,
    g_function=bundle.g_function,
)

# Create modifier
modifier = TrajectoryModifier(
    objective_fn=objective,
    alpha=0.1,
    epsilon=3.0,
    max_iterations=50,
)

# Set global state
tensors = bundle.to_tensors()
modifier.set_global_state(
    pickup_counts=tensors['pickup_counts'],
    supply_counts=tensors['supply_counts'],
    active_taxis=tensors['active_taxis'],
)

# Modify a trajectory
traj = bundle.trajectories[0]
history = modifier.modify_single(traj)

print(f"Converged: {history.converged}")
print(f"Final objective: {history.final_objective:.4f}")
```

### 2. With Discriminator (Fidelity)

```python
from trajectory_modification import DiscriminatorAdapter

# Load discriminator
adapter = DiscriminatorAdapter()
adapter.load_checkpoint('discriminator/model/checkpoints/')

# Create objective with discriminator
objective = FAMAILObjective(
    alpha_spatial=0.33,
    alpha_causal=0.33,
    alpha_fidelity=0.34,
    g_function=bundle.g_function,
    discriminator=adapter.model,
)
```

### 3. Track Global Metrics

```python
from trajectory_modification import GlobalMetrics

# Initialize metrics
metrics = GlobalMetrics(g_function=bundle.g_function)
metrics.initialize_from_data(
    bundle.pickup_counts,
    bundle.supply_counts,
    bundle.active_taxis,
)

# Get initial snapshot
initial = metrics.compute_snapshot()
print(f"Initial Gini: {initial.gini_coefficient:.4f}")

# After modifications, update and check improvement
final = metrics.compute_snapshot()
improvement = metrics.get_improvement()
print(f"Gini change: {improvement['delta_gini']:+.4f}")
```

## Dashboard

Run the Streamlit dashboard for interactive testing:

```bash
cd /home/robert/FAMAIL
streamlit run trajectory_modification/dashboard.py
```

The dashboard provides:
- Trajectory selection and visualization
- Algorithm parameter tuning
- Real-time modification execution
- Fairness metrics comparison
- Heatmap visualizations

## Components

### Trajectory
Core data structure representing a taxi trajectory with grid-based states.

```python
@dataclass
class TrajectoryState:
    x_grid: float      # Grid x coordinate (0-47)
    y_grid: float      # Grid y coordinate (0-89)
    time_bucket: int   # Time period (0-143 for 10-min buckets)
    day_index: int     # Day of week (0-6)

@dataclass
class Trajectory:
    states: List[TrajectoryState]
    driver_id: int
```

### TrajectoryModifier
Implements the ST-iFGSM modification loop.

```python
modifier = TrajectoryModifier(
    objective_fn=objective,
    alpha=0.1,           # Step size
    epsilon=3.0,         # Max perturbation
    max_iterations=50,   # Maximum iterations
    convergence_threshold=1e-4,
)
```

### FAMAILObjective
Combined objective function with all three fairness terms.

```python
objective = FAMAILObjective(
    alpha_spatial=0.33,    # Weight for spatial fairness
    alpha_causal=0.33,     # Weight for causal fairness
    alpha_fidelity=0.34,   # Weight for fidelity
    g_function=g_func,     # Fitted g(d) function
    discriminator=disc,    # Trained discriminator model
)
```

### DiscriminatorAdapter
Wrapper for loading and using trained discriminator models.

```python
adapter = DiscriminatorAdapter()
adapter.load_checkpoint(checkpoint_dir)
similarity = adapter.evaluate(tau_features, tau_prime_features)
```

### GlobalMetrics
Tracks system-wide fairness metrics as trajectories are modified.

```python
metrics = GlobalMetrics(
    grid_dims=(48, 90),
    g_function=g_func,
    alpha_weights=(0.33, 0.33, 0.34),
)
```

## Data Requirements

The module expects the following files in the workspace:

- `source_data/all_trajs.pkl` or `source_data/passenger_seeking_trajs_45-800.pkl` - Trajectory data
- `source_data/latest_traffic.pkl` - Supply/demand grid data
- `discriminator/model/checkpoints/` - Trained discriminator checkpoints (optional)
- `objective_function/causal_fairness/g_function_params.json` - g(d) function parameters (optional)

## Dependencies

- PyTorch (for differentiable objective)
- NumPy
- Streamlit (for dashboard)
- Plotly (for visualizations)
