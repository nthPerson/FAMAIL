# FAMAIL Objective Function

## Overview

The FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) objective function evaluates and guides trajectory modification to improve fairness in taxi service distribution while maintaining trajectory realism.

$$
\mathcal{L}(\mathcal{T}') = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

Where all terms output values in **[0, 1]** (higher = better), and the optimization direction is **maximization**.

---

## Key Design Principle: End-to-End Differentiability

> **All objective function terms are fully differentiable** to support gradient-based attribution in the Trajectory Modification Algorithm.

This design choice enables:
- **Gradient-based trajectory modification**: Identify exactly which spatial changes improve fairness
- **Efficient optimization**: Analytical gradients instead of expensive finite differences
- **Attribution analysis**: Understand how each location contributes to overall fairness

### Differentiability Summary

| Term | Differentiability Approach |
|------|---------------------------|
| **Spatial Fairness** ($F_{\text{spatial}}$) | Pairwise Gini coefficient (avoids sorting) |
| **Causal Fairness** ($F_{\text{causal}}$) | Pre-computed frozen $g(d)$ lookup + differentiable $R^2$ |
| **Fidelity** ($F_{\text{fidelity}}$) | Native PyTorch (ST-SiameseNet discriminator) |

---

## Directory Structure

```
objective_function/
├── README.md                           # This file
├── TERM_INTERFACE_SPECIFICATION.md     # Standard interface for all terms
├── INTEGRATION_DEVELOPMENT_PLAN.md     # Integration layer documentation
│
├── spatial_fairness/
│   ├── DEVELOPMENT_PLAN.md             # Spatial fairness design & implementation
│   └── (term implementation TBD)
│
├── causal_fairness/
│   ├── DEVELOPMENT_PLAN.md             # Causal fairness design & implementation
│   └── (term implementation TBD)
│
└── fidelity/
    └── (term implementation TBD - wraps discriminator/model)
```

---

## Terms

### 1. Spatial Fairness ($F_{\text{spatial}}$)

**Purpose**: Measures equitable geographic coverage of taxi service.

**Core Metric**: Complementary Gini coefficient (1 - Gini) of service rates across grid cells.

**Differentiability**: Uses pairwise absolute differences instead of sorted-index formulation:

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

**Documentation**: [spatial_fairness/DEVELOPMENT_PLAN.md](spatial_fairness/DEVELOPMENT_PLAN.md)

### 2. Causal Fairness ($F_{\text{causal}}$)

**Purpose**: Measures whether service allocation is explained by demand (legitimate) vs. contextual factors (potentially unfair).

**Core Metric**: $R^2$ coefficient measuring variance explained by demand.

**Differentiability**: Pre-computed frozen $g(d)$ lookup table + differentiable variance computation.

**Documentation**: [causal_fairness/DEVELOPMENT_PLAN.md](causal_fairness/DEVELOPMENT_PLAN.md)

### 3. Fidelity ($F_{\text{fidelity}}$)

**Purpose**: Ensures modified trajectories remain realistic and indistinguishable from original expert trajectories.

**Core Metric**: ST-SiameseNet discriminator similarity score.

**Differentiability**: Native PyTorch (nn.LSTM, nn.Linear, nn.Sigmoid).

**Documentation**: [fidelity/DEVELOPMENT_PLAN.md](fidelity/DEVELOPMENT_PLAN.md) (TBD)

---

## Usage

### Basic Computation (NumPy)

```python
from objective_function.integration import FAMAILObjectiveFunction, IntegrationConfig

# Create objective function
config = IntegrationConfig(
    alpha_causal=0.33,
    alpha_spatial=0.33,
    alpha_fidelity=0.34,
)
objective_fn = FAMAILObjectiveFunction(config)

# Compute objective value
value = objective_fn.compute(trajectories, auxiliary_data)
```

### Differentiable Computation (PyTorch)

```python
import torch
from objective_function.integration import DifferentiableFAMAILObjective

# Setup differentiable module
objective_module = DifferentiableFAMAILObjective(
    spatial_term=spatial_term,
    causal_term=causal_term,
    fidelity_term=fidelity_term,
)

# Forward pass (returns scalar tensor)
trajectory_coords = torch.randn(100, 50, 2, requires_grad=True)
objective, term_values = objective_module(trajectory_coords, auxiliary_data)

# Backward pass (compute gradients)
objective.backward()

# Gradient attribution
gradient = trajectory_coords.grad  # Shape: (100, 50, 2)
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [TERM_INTERFACE_SPECIFICATION.md](TERM_INTERFACE_SPECIFICATION.md) | Standard interface all terms must implement |
| [INTEGRATION_DEVELOPMENT_PLAN.md](INTEGRATION_DEVELOPMENT_PLAN.md) | Integration layer design and implementation |
| [spatial_fairness/DEVELOPMENT_PLAN.md](spatial_fairness/DEVELOPMENT_PLAN.md) | Spatial fairness term design |
| [causal_fairness/DEVELOPMENT_PLAN.md](causal_fairness/DEVELOPMENT_PLAN.md) | Causal fairness term design |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2.0 | 2026-01-12 | Mandatory end-to-end differentiability for all terms |
| 1.1.0 | 2026-01-12 | Removed Quality Term (overlap with Fidelity) |
| 1.0.0 | 2026-01-09 | Initial framework with 4 terms |

---

*For questions or contributions, see the main FAMAIL project documentation.*
