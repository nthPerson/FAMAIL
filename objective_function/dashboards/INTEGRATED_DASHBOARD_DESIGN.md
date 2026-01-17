# FAMAIL Integrated Objective Function Dashboard

## Design Document

**Version**: 1.0.0  
**Date**: 2026-01-16  
**Status**: Implementation Plan  
**Author**: FAMAIL Research Team

---

## 1. Overview

### 1.1 Purpose

This dashboard provides a comprehensive tool for testing, visualizing, and understanding the complete FAMAIL objective function. It integrates all three terms (Spatial Fairness, Causal Fairness, Fidelity) and demonstrates how gradients flow through the system to guide trajectory modification.

### 1.2 Key Objectives

1. **Validate gradient flow** through the entire objective function
2. **Demonstrate soft cell assignment** as the key enabler of differentiability
3. **Test attribution methods** for trajectory selection
4. **Prepare for integration** into the trajectory modification algorithm

---

## 2. Architecture

### 2.1 Module Structure

```
objective_function/dashboards/
├── INTEGRATED_DASHBOARD_DESIGN.md    # This design document
├── integrated_dashboard.py           # Main Streamlit dashboard
├── components/
│   ├── __init__.py
│   ├── gradient_flow.py             # Gradient visualization utilities
│   ├── combined_objective.py        # Combined objective function module
│   └── attribution_integration.py    # LIS + DCD combined attribution
└── tests/
    └── test_integration.py          # Integration tests
```

### 2.2 Data Flow

```
                    Raw Trajectory Data (all_trajs.pkl)
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │   Trajectory Coordinate Tensor    │
                    │   requires_grad=True              │
                    └─────────────┬─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │      Soft Cell Assignment       │
                    │   σ_c(x,y) = exp(-d²/2τ²)/Z    │
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
    ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
    │   Spatial     │     │   Causal      │     │   Fidelity    │
    │   Fairness    │     │   Fairness    │     │   (Discrim)   │
    │   (Gini)      │     │   (R²)        │     │               │
    └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
            │                     │                     │
            ▼                     ▼                     ▼
        F_spatial              F_causal              F_fidelity
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │   L = α₁F_causal + α₂F_spatial  │
                    │       + α₃F_fidelity            │
                    └─────────────┬───────────────────┘
                                  │
                                  ▼
                           L.backward()
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │   ∂L/∂x, ∂L/∂y per trajectory   │
                    │   (Attribution Scores)          │
                    └─────────────────────────────────┘
```

---

## 3. Dashboard Tabs

### Tab 1: Objective Function Gradient Flow

**Purpose**: Validate and visualize gradient flow through each term and the combined objective.

**Sections**:

1. **Mathematical Formulations**
   - Display formulas for each term with gradient derivations
   - Explain chain rule through soft cell assignment
   - Show computational graph

2. **Per-Term Gradient Tests**
   - Spatial Fairness: Test pairwise Gini gradients
   - Causal Fairness: Test R² gradients with frozen g(d)
   - Fidelity: Test discriminator gradients

3. **Combined Objective Test**
   - Create synthetic trajectory tensor
   - Forward pass through all terms
   - Backward pass and gradient analysis
   - Verify gradients are non-zero and finite

4. **Visualizations**
   - Gradient magnitude heatmaps
   - Per-term contribution breakdown
   - Gradient flow diagram (interactive)

### Tab 2: Soft Cell Assignment

**Purpose**: Demonstrate how soft cell assignment enables differentiability.

**Sections**:

1. **The Differentiability Problem**
   - Show why hard assignment breaks gradients
   - Visualize the "step function" problem

2. **Soft Assignment Solution**
   - Interactive kernel visualization
   - Temperature parameter effects
   - Compare soft vs hard assignment outputs

3. **Spatial Fairness Application**
   - How soft counts feed into Gini
   - End-to-end gradient demo for spatial term

4. **Causal Fairness Application**
   - How soft demand counts work
   - Frozen g(d) interaction
   - End-to-end gradient demo for causal term

5. **Temperature Annealing**
   - Annealing schedules comparison
   - Effect on convergence
   - Recommended settings

### Tab 3: Attribution Methods

**Purpose**: Show how LIS and DCD work together to select trajectories for modification.

**Sections**:

1. **Attribution Method Formulas**
   - LIS (Local Inequality Score): $\text{LIS}_\tau = \sum_c \sigma_c(\tau) \cdot |x_c - \bar{x}|$
   - DCD (Demand-Conditional Deviation): $\text{DCD}_\tau = \sum_c \sigma_c(\tau) \cdot R_c^2$

2. **Combined Attribution Score**
   - Formula: $\text{Attribution}_\tau = w_1 \cdot \text{LIS}_\tau + w_2 \cdot \text{DCD}_\tau$
   - Or gradient-based: $\text{Attribution}_\tau = \|\nabla_\tau \mathcal{L}\|$

3. **Trajectory Selection Tool**
   - Load real trajectories from all_trajs.pkl
   - Compute combined attribution scores
   - Rank and display top-N trajectories
   - Visualize selected trajectories on map

4. **Analysis Tools**
   - Distribution of attribution scores
   - Correlation between LIS and DCD
   - Geographic pattern analysis
   - Per-driver aggregation

### Tab 4: Integration Testing

**Purpose**: Full end-to-end tests with real data.

**Sections**:

1. **Data Loading**
   - Load all_trajs.pkl
   - Load auxiliary data (pickup_dropoff_counts, active_taxis)
   - Validate data alignment

2. **Objective Function Computation**
   - Compute combined objective value
   - Display per-term breakdown
   - Compare with baseline (unmodified trajectories)

3. **Gradient-Based Trajectory Selection**
   - Select top-N trajectories by gradient magnitude
   - Show which specific points have highest gradients
   - Predict effect of modifications

4. **Simulated Modification**
   - Apply small perturbation to selected trajectories
   - Recompute objective
   - Verify improvement

---

## 4. Implementation Details

### 4.1 Combined Objective Function Module

```python
class DifferentiableFAMAILObjective(nn.Module):
    """
    Combined differentiable objective function for FAMAIL.
    
    L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity
    """
    
    def __init__(
        self,
        alpha_spatial: float = 0.33,
        alpha_causal: float = 0.33,
        alpha_fidelity: float = 0.34,
        grid_dims: Tuple[int, int] = (48, 90),
        neighborhood_size: int = 5,
        temperature: float = 1.0,
        g_function: Optional[Callable] = None,
        discriminator: Optional[nn.Module] = None,
    ):
        ...
    
    def forward(
        self,
        pickup_coords: torch.Tensor,
        dropoff_coords: torch.Tensor,
        trajectory_features: Optional[torch.Tensor] = None,
        supply_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined objective.
        
        Returns:
            total_objective: Scalar loss (higher = better)
            term_values: Dict with individual term values
        """
        ...
```

### 4.2 Combined Attribution Score

```python
def compute_combined_attribution(
    trajectory_coords: torch.Tensor,
    service_counts: torch.Tensor,
    demand_tensor: torch.Tensor,
    supply_tensor: torch.Tensor,
    g_function: Callable,
    weight_lis: float = 0.5,
    weight_dcd: float = 0.5,
    grid_dims: Tuple[int, int] = (48, 90),
    neighborhood_size: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined LIS + DCD attribution score for trajectories.
    
    Args:
        trajectory_coords: Coordinates for each trajectory [n_traj, n_points, 2]
        service_counts: Service counts per cell [grid_x, grid_y]
        demand_tensor: Demand per cell [grid_x, grid_y]
        supply_tensor: Supply per cell [grid_x, grid_y]
        g_function: Pre-fitted g(d) function
        weight_lis: Weight for LIS component
        weight_dcd: Weight for DCD component
        
    Returns:
        attribution_scores: Score per trajectory [n_traj]
    """
    ...
```

### 4.3 Trajectory Selection

```python
def select_trajectories_for_modification(
    trajectories: Dict[str, List[np.ndarray]],
    auxiliary_data: Dict[str, Any],
    n_select: int = 10,
    method: str = 'gradient',  # 'gradient', 'lis', 'dcd', 'combined'
    **kwargs,
) -> List[Tuple[str, int, float]]:
    """
    Select trajectories that most affect global fairness.
    
    Args:
        trajectories: Dict mapping driver_id to list of trajectories
        auxiliary_data: pickup_dropoff_counts, active_taxis, etc.
        n_select: Number of trajectories to select
        method: Selection method
        
    Returns:
        List of (driver_id, trajectory_idx, attribution_score)
    """
    ...
```

---

## 5. Test Scenarios

### 5.1 Gradient Flow Tests

| Test | Description | Expected Result |
|------|-------------|-----------------|
| T1.1 | Spatial term gradients | Non-zero gradients for pickup/dropoff coords |
| T1.2 | Causal term gradients | Non-zero gradients for pickup coords (supply frozen) |
| T1.3 | Fidelity term gradients | Non-zero gradients for trajectory features |
| T1.4 | Combined objective | All term gradients flow correctly |
| T1.5 | Gradient magnitude | Finite, non-NaN values |

### 5.2 Attribution Tests

| Test | Description | Expected Result |
|------|-------------|-----------------|
| T2.1 | LIS consistency | Higher LIS for trajectories in unequal areas |
| T2.2 | DCD consistency | Higher DCD for trajectories with large residuals |
| T2.3 | Combined ranking | Top trajectories affect fairness most |
| T2.4 | Modification effect | Editing top trajectories improves fairness |

### 5.3 Integration Tests

| Test | Description | Expected Result |
|------|-------------|-----------------|
| T3.1 | Data loading | All data loads without errors |
| T3.2 | Full forward pass | Objective computed successfully |
| T3.3 | Full backward pass | Gradients computed for all trajectories |
| T3.4 | End-to-end optimization | Objective improves after gradient step |

---

## 6. Usage

### Starting the Dashboard

```bash
cd objective_function/dashboards
streamlit run integrated_dashboard.py
```

### Configuration Options

- **Alpha weights**: Adjust relative importance of terms
- **Temperature**: Control soft assignment sharpness
- **Selection method**: Choose attribution method for trajectory selection
- **Number to select**: How many trajectories to identify

---

## 7. Dependencies

```
torch>=2.0.0
streamlit>=1.28.0
plotly>=5.0.0
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
```

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-16 | Initial design document |

---

*This document guides the implementation of the integrated objective function dashboard.*
