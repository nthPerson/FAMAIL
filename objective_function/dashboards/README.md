# FAMAIL Integrated Objective Function Dashboard

A comprehensive Streamlit dashboard for testing, visualizing, and understanding the FAMAIL objective function for trajectory optimization.

## Overview

This dashboard provides interactive tools to:

1. **Verify Gradient Flow**: Test that gradients flow correctly through all components
2. **Visualize Soft Cell Assignment**: See how soft assignment enables differentiability
3. **Explore Attribution Methods**: Understand how LIS and DCD combine to select trajectories
4. **Run Integration Tests**: Validate the complete pipeline end-to-end

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run integrated_dashboard.py
```

The dashboard will open at `http://localhost:8501`.

## Tabs

### 1. 🔄 Gradient Flow

- **Mathematical Formulation**: View the combined objective function formula
- **Gradient Flow Diagram**: Visualize how gradients backpropagate
- **Gradient Verification Tests**: Run automatic tests to verify gradient validity
- **Temperature Annealing Analysis**: See how gradients change with temperature

### 2. 🔲 Soft Cell Assignment

- **Formula Explanation**: Understand the Gaussian softmax assignment
- **Interactive Visualization**: Move a point and see real-time soft assignments
- **Temperature Comparison**: Compare soft assignments at different τ values

### 3. 🎯 Attribution Methods

- **LIS (Local Inequality Score)**: Spatial fairness attribution
- **DCD (Demand-Conditional Deviation)**: Causal fairness attribution
- **Combined Attribution**: Weight LIS and DCD to rank trajectories
- **Trajectory Selection**: Select top trajectories for modification

### 4. 🧪 Integration Testing

- **Basic Gradient Flow**: Verify basic backpropagation
- **Temperature Annealing**: Test across temperature schedule
- **Attribution Consistency**: Validate attribution logic
- **Full Pipeline**: End-to-end integration test

## Configuration (Sidebar)

- **Grid Configuration**: Set grid dimensions (default 48×90)
- **Objective Weights**: Adjust α_spatial, α_causal, α_fidelity
- **Temperature**: Set soft assignment temperature τ
- **Attribution Weights**: Balance LIS vs DCD for trajectory selection

## Architecture

```
dashboards/
├── INTEGRATED_DASHBOARD_DESIGN.md  # Design document
├── integrated_dashboard.py          # Main Streamlit app
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── components/
    ├── __init__.py
    ├── combined_objective.py        # DifferentiableFAMAILObjective
    ├── gradient_flow.py             # Gradient verification utilities
    └── attribution_integration.py   # LIS + DCD attribution
```

## Key Classes

### DifferentiableFAMAILObjective

Combined objective function with full gradient support:

```python
from components import DifferentiableFAMAILObjective

objective = DifferentiableFAMAILObjective(
    alpha_spatial=0.33,
    alpha_causal=0.33,
    alpha_fidelity=0.34,
    grid_dims=(48, 90),
    temperature=1.0,
)

total, terms = objective(pickup_coords, dropoff_coords)
total.backward()  # Gradients flow to coordinates
```

### Attribution Functions

```python
from components import (
    compute_combined_attribution,
    select_trajectories_for_modification,
)

# Compute attribution scores
result = compute_combined_attribution(
    trajectories=trajectories,
    pickup_counts=pickup_counts,
    dropoff_counts=dropoff_counts,
    supply_counts=supply_counts,
    g_function=g_function,
    lis_weight=0.5,
    dcd_weight=0.5,
)

# Select top trajectories
selected = select_trajectories_for_modification(
    result, 
    n_trajectories=10,
)
```

## Mathematical Formulations

### Combined Objective

$$\mathcal{L} = \alpha_1 \cdot F_{\text{causal}} + \alpha_2 \cdot F_{\text{spatial}} + \alpha_3 \cdot F_{\text{fidelity}}$$

### Soft Cell Assignment

$$\sigma(p|\tau) = \frac{\exp(-\|loc - cell_p\|^2 / \tau)}{\sum_{q \in \mathcal{N}} \exp(-\|loc - cell_q\|^2 / \tau)}$$

### Local Inequality Score (LIS)

$$LIS_i = \frac{|c_i - \mu|}{\mu}$$

### Demand-Conditional Deviation (DCD)

$$DCD_i = |Y_i - g(D_i)|$$

where $Y_i = S_i / D_i$ is the service ratio.

## Data Loading

The dashboard supports loading trajectory data from:

1. **Synthetic Data**: Generate random trajectories with configurable clustering
2. **Real Data**: Load from `all_trajs.pkl` files in the workspace

## Development

To extend the dashboard:

1. Add new components in `components/`
2. Export from `components/__init__.py`
3. Add new tabs in `integrated_dashboard.py`

## Dependencies

- **streamlit**: Web dashboard framework
- **torch**: Differentiable computation
- **plotly**: Interactive visualizations
- **scikit-learn**: Isotonic regression for g(d)
- **numpy/pandas**: Data manipulation
