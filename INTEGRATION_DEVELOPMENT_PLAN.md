# FAMAIL Objective Function Integration Development Plan

## Document Metadata

| Property | Value |
|----------|-------|
| **Document Title** | Integration Development Plan |
| **Version** | 1.1.0 |
| **Last Updated** | 2026-01-12 |
| **Status** | Development Planning |
| **Author** | FAMAIL Research Team |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Complete Objective Function](#2-complete-objective-function)
3. [Term Dependencies and Data Flow](#3-term-dependencies-and-data-flow)
4. [Integration Architecture](#4-integration-architecture)
5. [Weighted Combination Strategy](#5-weighted-combination-strategy)
6. [Integration Implementation](#6-integration-implementation)
7. [ST-iFGSM Integration Guidance](#7-st-ifgsm-integration-guidance)
8. [Validation and Testing](#8-validation-and-testing)
9. [Monitoring and Diagnostics](#9-monitoring-and-diagnostics)
10. [Development Roadmap](#10-development-roadmap)

---

## 1. Overview

### 1.1 Purpose

This document describes the **integration layer** that combines the four objective function terms into a unified optimization objective for the FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) system.

The integration layer is responsible for:
1. **Orchestrating** term computations
2. **Combining** term values with configurable weights
3. **Providing** gradients/signals to the trajectory editing framework
4. **Monitoring** optimization progress across all terms

### 1.2 Scope

| In Scope | Out of Scope |
|----------|--------------|
| Term integration and weighting | Individual term implementations |
| Unified objective function API | ST-iFGSM perturbation mechanics |
| Multi-objective monitoring | Model training procedures |
| Configuration management | Data preprocessing |

### 1.3 Prerequisites

Before implementing integration, the following must be complete:

- [ ] Spatial Fairness term ([spatial_fairness/DEVELOPMENT_PLAN.md](spatial_fairness/DEVELOPMENT_PLAN.md))
- [ ] Causal Fairness term ([causal_fairness/DEVELOPMENT_PLAN.md](causal_fairness/DEVELOPMENT_PLAN.md))
- [ ] Fidelity term ([fidelity/DEVELOPMENT_PLAN.md](fidelity/DEVELOPMENT_PLAN.md))
- [ ] Common interface ([TERM_INTERFACE_SPECIFICATION.md](TERM_INTERFACE_SPECIFICATION.md))

### 1.4 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Optimization | Maximization | Aligns with ST-iFGSM framework |
| Term combination | Weighted linear | Simple, interpretable, tunable |
| Normalization | All terms [0,1] | Fair weighting comparison |
| Configuration | Dataclass-based | Type-safe, documented defaults |

---

## 2. Complete Objective Function

### 2.1 Mathematical Definition

The complete FAMAIL objective function is:

$$
\mathcal{L}(\mathcal{T}') = \alpha_1 F_{\text{causal}}(\mathcal{T}') + \alpha_2 F_{\text{spatial}}(\mathcal{T}') + \alpha_3 F_{\text{fidelity}}(\mathcal{T}')
$$

Where:
- $\mathcal{T}'$ = edited trajectory set
- $\alpha_i$ = weight for term $i$ (hyperparameters)
- $F_i(\cdot)$ = objective function term (all normalized to [0, 1])

### 2.2 Optimization Direction

**MAXIMIZE** the objective function:
$$
\mathcal{T}'^* = \arg\max_{\mathcal{T}'} \mathcal{L}(\mathcal{T}')
$$

**Interpretation**:
- $\mathcal{L} = 1.0$: Perfect trajectories (all terms maximized)
- $\mathcal{L} = 0.0$: Worst possible trajectories

### 2.3 Term Summary

| Term | Symbol | Purpose | Range |
|------|--------|---------|-------|
| Causal Fairness | $F_{\text{causal}}$ | Service allocation driven by demand | [0, 1] |
| Spatial Fairness | $F_{\text{spatial}}$ | Equitable geographic coverage | [0, 1] |
| Fidelity | $F_{\text{fidelity}}$ | Trajectories appear realistic | [0, 1] |

### 2.4 Weight Constraints

The weights are non-negative but do **not** need to sum to 1 (allows for scale adjustment):

$$
\alpha_i \geq 0 \quad \forall i \in \{1, 2, 3\}
$$

**Typical configurations**:

| Configuration | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | Purpose |
|--------------|------------|------------|------------|---------|
| Balanced | 0.33 | 0.33 | 0.34 | Equal importance |
| Fairness-focused | 0.40 | 0.40 | 0.20 | Prioritize fairness |
| Realism-focused | 0.20 | 0.20 | 0.60 | Prioritize realism |
| Research | 1.0 | 1.0 | 1.0 | Unweighted sum |

---

## 3. Term Dependencies and Data Flow

### 3.1 Data Dependencies

```
                    ┌──────────────────────────────────────┐
                    │           Data Sources               │
                    └──────────────────┬───────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  all_trajs.pkl  │         │ pickup_dropoff  │         │ latest_volume_  │
│                 │         │  _counts.pkl    │         │  pickups.pkl    │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         ├───────────────────────────┼───────────────────────────┤
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│    Fidelity     │         │ Spatial Fairness│         │ Causal Fairness │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                                     ▼
                          ┌─────────────────┐
                          │   Integration   │
                          │     Layer       │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │    𝓛(𝒯')       │
                          │  (Objective)    │
                          └─────────────────┘
```

### 3.2 Data Requirements by Term

| Term | Primary Data | Secondary Data | Computed From |
|------|--------------|----------------|---------------|
| Spatial Fairness | pickup_dropoff_counts.pkl | - | Trajectories |
| Causal Fairness | pickup_dropoff_counts.pkl | latest_volume_pickups.pkl | Both |
| Fidelity | Discriminator model | - | Trajectories |

### 3.3 Computation Order

Terms can be computed **in parallel** (no inter-term dependencies):

```
                    ┌───────────────────┐
                    │   Input: 𝒯'       │
                    └─────────┬─────────┘
                              │
       ┌──────────┬──────────┼──────────┐
       │          │          │          │
       ▼          ▼          ▼          │
   F_causal  F_spatial  F_fidelity     │
       │          │          │          │
       └──────────┴──────────┘          │
                              │         │
                              ▼         │
                    ┌───────────────────┐          │
                    │   Weighted Sum    │◄─────────┘
                    │   𝓛 = Σ αᵢFᵢ     │    (weights)
                    └───────────────────┘
```

---

## 4. Integration Architecture

### 4.1 Class Structure

```
objective_function/
├── base.py                    # Abstract base classes
├── integration.py             # Integration layer
├── TERM_INTERFACE_SPECIFICATION.md
├── INTEGRATION_DEVELOPMENT_PLAN.md
│
├── spatial_fairness/
│   ├── __init__.py
│   ├── term.py               # SpatialFairnessTerm
│   └── DEVELOPMENT_PLAN.md
│
├── causal_fairness/
│   ├── __init__.py
│   ├── term.py               # CausalFairnessTerm
│   └── DEVELOPMENT_PLAN.md
│
└── fidelity/
    ├── __init__.py
    ├── term.py               # FidelityTerm
    └── DEVELOPMENT_PLAN.md
```

### 4.2 Integration Class Design

```python
# File: objective_function/integration.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import pickle
import numpy as np

from objective_function.base import ObjectiveFunctionTerm
from objective_function.spatial_fairness.term import SpatialFairnessTerm
from objective_function.causal_fairness.term import CausalFairnessTerm
from objective_function.fidelity.term import FidelityTerm


@dataclass
class IntegrationConfig:
    """Configuration for the complete objective function."""
    
    # Term weights
    alpha_causal: float = 0.33
    alpha_spatial: float = 0.33
    alpha_fidelity: float = 0.34
    
    # Term configs (optional overrides)
    spatial_config: Optional[Dict[str, Any]] = None
    causal_config: Optional[Dict[str, Any]] = None
    fidelity_config: Optional[Dict[str, Any]] = None
    
    # Data paths
    data_dir: str = "source_data"
    
    # Computation settings
    parallel_computation: bool = True
    cache_data: bool = True


class FAMAILObjectiveFunction:
    """
    Complete FAMAIL objective function integrating all terms.
    
    𝓛(𝒯') = α₁F_causal + α₂F_spatial + α₃F_fidelity
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._terms: Dict[str, ObjectiveFunctionTerm] = {}
        self._data_cache: Dict[str, Any] = {}
        
        self._initialize_terms()
    
    def _initialize_terms(self) -> None:
        """Initialize all objective function terms."""
        # Create term instances with configs
        self._terms['spatial'] = SpatialFairnessTerm(
            self.config.spatial_config or {}
        )
        self._terms['causal'] = CausalFairnessTerm(
            self.config.causal_config or {}
        )
        self._terms['fidelity'] = FidelityTerm(
            self.config.fidelity_config or {}
        )
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute the complete objective function value.
        
        Args:
            trajectories: Edited trajectory set
            auxiliary_data: Optional pre-loaded data
        
        Returns:
            Objective function value (higher = better)
        """
        # Load auxiliary data if not provided
        if auxiliary_data is None:
            auxiliary_data = self._load_auxiliary_data()
        
        # Compute each term
        term_values = {}
        
        term_values['causal'] = self._terms['causal'].compute(
            trajectories, auxiliary_data
        )
        term_values['spatial'] = self._terms['spatial'].compute(
            trajectories, auxiliary_data
        )
        term_values['fidelity'] = self._terms['fidelity'].compute(
            trajectories, auxiliary_data
        )
        
        # Weighted combination
        objective = (
            self.config.alpha_causal * term_values['causal'] +
            self.config.alpha_spatial * term_values['spatial'] +
            self.config.alpha_fidelity * term_values['fidelity']
        )
        
        return objective
    
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compute with detailed breakdown of all terms.
        
        Returns:
            Dictionary with overall value and per-term breakdowns
        """
        if auxiliary_data is None:
            auxiliary_data = self._load_auxiliary_data()
        
        # Get detailed breakdown from each term
        breakdowns = {}
        for name, term in self._terms.items():
            breakdowns[name] = term.compute_with_breakdown(
                trajectories, auxiliary_data
            )
        
        # Compute weighted sum
        term_values = {
            name: bd['value'] for name, bd in breakdowns.items()
        }
        
        weights = {
            'causal': self.config.alpha_causal,
            'spatial': self.config.alpha_spatial,
            'fidelity': self.config.alpha_fidelity,
        }
        
        objective = sum(
            weights[name] * value 
            for name, value in term_values.items()
        )
        
        return {
            'value': objective,
            'term_values': term_values,
            'weights': weights,
            'weighted_contributions': {
                name: weights[name] * term_values[name]
                for name in term_values
            },
            'term_breakdowns': breakdowns,
        }
    
    def _load_auxiliary_data(self) -> Dict[str, Any]:
        """Load required auxiliary data."""
        if self.config.cache_data and self._data_cache:
            return self._data_cache
        
        data_dir = Path(self.config.data_dir)
        
        auxiliary_data = {}
        
        # Load pickup_dropoff_counts
        pdc_path = data_dir / "pickup_dropoff_counts.pkl"
        if pdc_path.exists():
            with open(pdc_path, 'rb') as f:
                auxiliary_data['pickup_dropoff_counts'] = pickle.load(f)
        
        # Load latest_volume_pickups
        lvp_path = data_dir / "latest_volume_pickups.pkl"
        if lvp_path.exists():
            with open(lvp_path, 'rb') as f:
                auxiliary_data['latest_volume_pickups'] = pickle.load(f)
        
        if self.config.cache_data:
            self._data_cache = auxiliary_data
        
        return auxiliary_data
    
    def update_weights(
        self,
        alpha_causal: Optional[float] = None,
        alpha_spatial: Optional[float] = None,
        alpha_fidelity: Optional[float] = None,
    ) -> None:
        """Update term weights dynamically."""
        if alpha_causal is not None:
            self.config.alpha_causal = alpha_causal
        if alpha_spatial is not None:
            self.config.alpha_spatial = alpha_spatial
        if alpha_fidelity is not None:
            self.config.alpha_fidelity = alpha_fidelity
```

---

## 5. Weighted Combination Strategy

### 5.1 Weight Selection Principles

**Balanced Approach**:
- Start with equal weights: $\alpha_i = 0.25$
- Provides baseline for comparison
- Fair treatment of all objectives

**Domain-Driven Approach**:
- Prioritize based on research goals
- Higher fairness weights for equity studies
- Higher fidelity for practical deployment

**Empirical Approach**:
- Grid search over weight combinations
- Pareto front analysis
- Cross-validation on held-out trajectories

### 5.2 Weight Normalization Options

#### Option A: Normalized Weights (Sum = 1)

$$
\alpha_i' = \frac{\alpha_i}{\sum_j \alpha_j}
$$

**Pros**: Objective bounded by [0, 1]  
**Cons**: Changing one weight affects all others

#### Option B: Unnormalized Weights

Use weights directly without normalization.

**Pros**: Independent weight adjustment  
**Cons**: Objective range varies with total weight

**Recommendation**: Use normalized weights for interpretability.

### 5.3 Multi-Objective Considerations

The weighted sum approach converts multi-objective optimization to single-objective, but it's important to monitor individual terms:

```python
def analyze_pareto_efficiency(
    solutions: List[Dict[str, float]]
) -> List[int]:
    """
    Identify Pareto-optimal solutions.
    
    A solution is Pareto-optimal if no other solution
    dominates it (better in all objectives).
    """
    pareto_indices = []
    
    for i, sol_i in enumerate(solutions):
        dominated = False
        for j, sol_j in enumerate(solutions):
            if i == j:
                continue
            
            # Check if sol_j dominates sol_i
            all_better = all(
                sol_j[k] >= sol_i[k] for k in sol_i
            )
            some_strictly_better = any(
                sol_j[k] > sol_i[k] for k in sol_i
            )
            
            if all_better and some_strictly_better:
                dominated = True
                break
        
        if not dominated:
            pareto_indices.append(i)
    
    return pareto_indices
```

---

## 6. Integration Implementation

### 6.1 Implementation Steps

```
ALGORITHM: Compute FAMAIL Objective Function
═══════════════════════════════════════════════════════════════════════

INPUT:
  - 𝒯': Edited trajectory set
  - config: IntegrationConfig with weights and term configs

OUTPUT:
  - 𝓛: Objective function value (float)

STEPS:

1. LOAD AUXILIARY DATA
   ─────────────────────────────────
   Load pickup_dropoff_counts.pkl
   Load latest_volume_pickups.pkl
   (Cache if enabled)

2. COMPUTE TERMS (in parallel if enabled)
   ─────────────────────────────────
   F_causal ← CausalFairnessTerm.compute(𝒯', data)
   F_spatial ← SpatialFairnessTerm.compute(𝒯', data)
   F_fidelity ← FidelityTerm.compute(𝒯', data)

3. COMBINE WITH WEIGHTS
   ─────────────────────────────────
   𝓛 = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity

4. RETURN 𝓛
```

### 6.2 Parallel Computation

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_parallel(
    self,
    trajectories: Dict[str, List[List[List[float]]]],
    auxiliary_data: Dict[str, Any]
) -> float:
    """Compute all terms in parallel."""
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                self._terms[name].compute,
                trajectories,
                auxiliary_data
            ): name
            for name in self._terms
        }
        
        term_values = {}
        for future in as_completed(futures):
            name = futures[future]
            term_values[name] = future.result()
    
    # Weighted combination
    weights = {
        'causal': self.config.alpha_causal,
        'spatial': self.config.alpha_spatial,
        'fidelity': self.config.alpha_fidelity,
    }
    
    return sum(weights[k] * term_values[k] for k in term_values)
```

### 6.3 Usage Examples

#### 6.3.1 Basic Usage

```python
from objective_function.integration import (
    FAMAILObjectiveFunction, 
    IntegrationConfig
)

# Create with default configuration
config = IntegrationConfig()
objective_fn = FAMAILObjectiveFunction(config)

# Load trajectories
with open('source_data/all_trajs.pkl', 'rb') as f:
    trajectories = pickle.load(f)

# Compute objective
value = objective_fn.compute(trajectories)
print(f"Objective value: {value:.4f}")
```

#### 6.3.2 Custom Weights

```python
# Fairness-focused configuration
config = IntegrationConfig(
    alpha_causal=0.40,
    alpha_spatial=0.40,
    alpha_fidelity=0.20,
)

objective_fn = FAMAILObjectiveFunction(config)
```

#### 6.3.3 Detailed Analysis

```python
# Get full breakdown
breakdown = objective_fn.compute_with_breakdown(trajectories)

print(f"Overall objective: {breakdown['value']:.4f}")
print("\nTerm values:")
for name, value in breakdown['term_values'].items():
    weight = breakdown['weights'][name]
    contribution = breakdown['weighted_contributions'][name]
    print(f"  {name}: {value:.4f} × {weight:.2f} = {contribution:.4f}")

print("\nDetailed breakdowns available for each term...")
```

---

## 7. ST-iFGSM Integration Guidance

### 7.1 Overview

The ST-iFGSM (Spatio-Temporal iterative Fast Gradient Sign Method) framework is used for trajectory editing. This section provides high-level guidance on how the objective function integrates with trajectory perturbation.

### 7.2 Interface with Trajectory Editor

```python
# Conceptual interface (details TBD based on ST-iFGSM implementation)

class TrajectoryEditor:
    """
    ST-iFGSM-based trajectory editing framework.
    """
    
    def __init__(
        self,
        objective_fn: FAMAILObjectiveFunction,
        epsilon: float,
        num_iterations: int,
    ):
        self.objective_fn = objective_fn
        self.epsilon = epsilon
        self.num_iterations = num_iterations
    
    def edit_trajectories(
        self,
        original_trajectories: Dict[str, List[List[List[float]]]]
    ) -> Dict[str, List[List[List[float]]]]:
        """
        Edit trajectories to maximize objective function.
        
        This is a placeholder for the actual ST-iFGSM implementation.
        """
        edited = original_trajectories.copy()
        
        for iteration in range(self.num_iterations):
            # Compute current objective
            current_value = self.objective_fn.compute(edited)
            
            # Compute gradient/perturbation direction
            # (ST-iFGSM specific implementation)
            perturbations = self._compute_perturbations(edited)
            
            # Apply perturbations
            edited = self._apply_perturbations(edited, perturbations)
            
            # Log progress
            new_value = self.objective_fn.compute(edited)
            print(f"Iteration {iteration}: {current_value:.4f} → {new_value:.4f}")
        
        return edited
    
    def _compute_perturbations(self, trajectories):
        """Compute perturbation direction (ST-iFGSM specific)."""
        # Implementation depends on ST-iFGSM methodology
        raise NotImplementedError
    
    def _apply_perturbations(self, trajectories, perturbations):
        """Apply perturbations within epsilon ball."""
        # Implementation depends on trajectory representation
        raise NotImplementedError
```

### 7.3 Gradient Considerations

**Note**: Most terms are not differentiable (discrete operations). The integration layer supports two approaches:

#### Approach A: Numerical Gradients

```python
def estimate_gradient(
    objective_fn: FAMAILObjectiveFunction,
    trajectories: Dict,
    delta: float = 1e-6
) -> Dict:
    """
    Estimate gradient via finite differences.
    
    Expensive but works for non-differentiable objectives.
    """
    base_value = objective_fn.compute(trajectories)
    gradients = {}
    
    for driver_id, driver_trajs in trajectories.items():
        driver_grads = []
        for traj_idx, traj in enumerate(driver_trajs):
            traj_grad = []
            for state_idx, state in enumerate(traj):
                state_grad = []
                for feat_idx in range(len(state)):
                    # Perturb feature
                    original = state[feat_idx]
                    state[feat_idx] = original + delta
                    
                    new_value = objective_fn.compute(trajectories)
                    gradient = (new_value - base_value) / delta
                    state_grad.append(gradient)
                    
                    # Restore
                    state[feat_idx] = original
                
                traj_grad.append(state_grad)
            driver_grads.append(traj_grad)
        gradients[driver_id] = driver_grads
    
    return gradients
```

#### Approach B: Proxy Gradient

Use differentiable proxy for gradient direction, verify with actual objective:

```python
def proxy_gradient_step(
    objective_fn: FAMAILObjectiveFunction,
    proxy_fn: Callable,  # Differentiable approximation
    trajectories: Dict,
    step_size: float
) -> Dict:
    """
    Use proxy gradient but verify with actual objective.
    """
    # Get gradient from proxy
    proxy_grad = proxy_fn.gradient(trajectories)
    
    # Line search to find good step
    best_trajectories = trajectories
    best_value = objective_fn.compute(trajectories)
    
    for scale in [0.1, 0.5, 1.0, 2.0]:
        stepped = apply_gradient(trajectories, proxy_grad, step_size * scale)
        value = objective_fn.compute(stepped)
        
        if value > best_value:
            best_value = value
            best_trajectories = stepped
    
    return best_trajectories
```

### 7.4 Optimization Loop

High-level optimization structure:

```
ALGORITHM: FAMAIL Trajectory Optimization
═══════════════════════════════════════════════════════════════════════

INPUT:
  - 𝒯: Original expert trajectories
  - objective_fn: FAMAILObjectiveFunction
  - max_iterations: int
  - convergence_threshold: float

OUTPUT:
  - 𝒯'*: Optimized trajectories

INITIALIZE:
  𝒯' ← copy(𝒯)
  best_value ← objective_fn.compute(𝒯')
  best_trajectories ← 𝒯'

FOR iteration = 1 to max_iterations:
    
    # Get detailed breakdown for monitoring
    breakdown ← objective_fn.compute_with_breakdown(𝒯')
    log(iteration, breakdown)
    
    # Compute perturbation direction (method-specific)
    Δ ← compute_perturbation(𝒯', objective_fn)
    
    # Apply perturbation
    𝒯' ← 𝒯' + Δ
    
    # Ensure validity constraints
    𝒯' ← project_to_valid(𝒯')
    
    # Track best
    current_value ← objective_fn.compute(𝒯')
    IF current_value > best_value:
        best_value ← current_value
        best_trajectories ← copy(𝒯')
    
    # Convergence check
    IF |current_value - prev_value| < convergence_threshold:
        BREAK

RETURN best_trajectories
```

---

## 8. Validation and Testing

### 8.1 Unit Tests

```python
class TestFAMAILObjectiveFunction:
    """Tests for the integrated objective function."""
    
    def test_initialization(self):
        """Test that all terms are initialized."""
        config = IntegrationConfig()
        obj_fn = FAMAILObjectiveFunction(config)
        
        assert 'spatial' in obj_fn._terms
        assert 'causal' in obj_fn._terms
        assert 'fidelity' in obj_fn._terms
    
    def test_compute_range(self):
        """Test that objective is in expected range."""
        config = IntegrationConfig()
        obj_fn = FAMAILObjectiveFunction(config)
        
        # Create dummy trajectories
        trajectories = create_dummy_trajectories()
        
        value = obj_fn.compute(trajectories)
        
        # Should be in valid range
        assert value >= 0
        # Upper bound depends on weight normalization
    
    def test_weight_impact(self):
        """Test that weights affect the objective."""
        trajectories = create_dummy_trajectories()
        
        # High spatial weight
        config1 = IntegrationConfig(
            alpha_spatial=1.0, alpha_causal=0, 
            alpha_fidelity=0
        )
        obj1 = FAMAILObjectiveFunction(config1)
        
        # High causal weight  
        config2 = IntegrationConfig(
            alpha_causal=1.0, alpha_spatial=0,
            alpha_fidelity=0
        )
        obj2 = FAMAILObjectiveFunction(config2)
        
        value1 = obj1.compute(trajectories)
        value2 = obj2.compute(trajectories)
        
        # Different weights should give different results
        # (unless spatial and causal happen to be equal)
        # This tests that weights are being applied
```

### 8.2 Integration Tests

```python
def test_with_real_data():
    """Full integration test with actual data."""
    import pickle
    
    # Load real data
    with open('source_data/all_trajs.pkl', 'rb') as f:
        trajectories = pickle.load(f)
    
    # Convert to expected format
    trajectories = {str(k): v for k, v in trajectories.items()}
    
    # Create objective function
    config = IntegrationConfig(data_dir='source_data')
    obj_fn = FAMAILObjectiveFunction(config)
    
    # Compute with breakdown
    result = obj_fn.compute_with_breakdown(trajectories)
    
    print("Integration Test Results:")
    print(f"  Overall: {result['value']:.4f}")
    for name, value in result['term_values'].items():
        print(f"  {name}: {value:.4f}")
    
    # Assertions
    assert result['value'] >= 0
    assert all(0 <= v <= 1 for v in result['term_values'].values())
```

### 8.3 Regression Tests

```python
def test_regression_baseline():
    """Ensure results are consistent across runs."""
    import json
    
    trajectories = load_test_trajectories()
    config = IntegrationConfig()
    obj_fn = FAMAILObjectiveFunction(config)
    
    result = obj_fn.compute_with_breakdown(trajectories)
    
    # Load expected baseline
    with open('tests/baselines/integration_baseline.json') as f:
        baseline = json.load(f)
    
    # Check within tolerance
    assert abs(result['value'] - baseline['value']) < 0.001
    
    for name in result['term_values']:
        expected = baseline['term_values'][name]
        actual = result['term_values'][name]
        assert abs(actual - expected) < 0.001, f"{name} regression"
```

---

## 9. Monitoring and Diagnostics

### 9.1 Logging Framework

```python
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class OptimizationStep:
    """Record of a single optimization step."""
    iteration: int
    timestamp: datetime
    objective_value: float
    term_values: Dict[str, float]
    weighted_contributions: Dict[str, float]
    trajectory_stats: Dict[str, Any]


class OptimizationLogger:
    """Logger for optimization progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.history: List[OptimizationStep] = []
        self.log_file = log_file
        
        # Configure logging
        self.logger = logging.getLogger('FAMAIL.optimization')
        self.logger.setLevel(logging.INFO)
    
    def log_step(
        self,
        iteration: int,
        breakdown: Dict[str, Any],
        trajectory_stats: Dict[str, Any]
    ) -> None:
        """Log an optimization step."""
        step = OptimizationStep(
            iteration=iteration,
            timestamp=datetime.now(),
            objective_value=breakdown['value'],
            term_values=breakdown['term_values'],
            weighted_contributions=breakdown['weighted_contributions'],
            trajectory_stats=trajectory_stats,
        )
        
        self.history.append(step)
        
        # Log summary
        self.logger.info(
            f"Step {iteration}: L={step.objective_value:.4f} | "
            f"causal={step.term_values['causal']:.3f} | "
            f"spatial={step.term_values['spatial']:.3f} | "
            f"fidelity={step.term_values['fidelity']:.3f}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.history:
            return {}
        
        initial = self.history[0]
        final = self.history[-1]
        
        return {
            'total_iterations': len(self.history),
            'initial_objective': initial.objective_value,
            'final_objective': final.objective_value,
            'improvement': final.objective_value - initial.objective_value,
            'improvement_pct': (
                (final.objective_value - initial.objective_value) 
                / max(initial.objective_value, 1e-10) * 100
            ),
            'term_improvements': {
                name: final.term_values[name] - initial.term_values[name]
                for name in initial.term_values
            }
        }
```

### 9.2 Visualization

```python
import matplotlib.pyplot as plt

def plot_optimization_progress(logger: OptimizationLogger):
    """Plot optimization progress over iterations."""
    iterations = [s.iteration for s in logger.history]
    objectives = [s.objective_value for s in logger.history]
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Overall objective
    axes[0].plot(iterations, objectives, 'b-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Objective Value')
    axes[0].set_title('FAMAIL Optimization Progress')
    axes[0].grid(True, alpha=0.3)
    
    # Individual terms
    term_names = ['causal', 'spatial', 'fidelity']
    colors = ['red', 'green', 'blue']
    
    for name, color in zip(term_names, colors):
        values = [s.term_values[name] for s in logger.history]
        axes[1].plot(iterations, values, color=color, label=name)
    
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Term Value')
    axes[1].set_title('Individual Term Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_progress.png', dpi=150)
    plt.show()
```

---

## 10. Development Roadmap

### 10.1 Phase 1: Foundation (Prerequisites)

Complete individual term implementations:

- [ ] **P1.1**: Implement Spatial Fairness Term
- [ ] **P1.2**: Implement Causal Fairness Term
- [ ] **P1.3**: Implement Fidelity Term
- [ ] **P1.4**: Validate each term independently

**Duration**: 3-4 weeks  
**Deliverables**: Working term implementations with tests

### 10.2 Phase 2: Integration Layer (Weeks 5-6)

- [ ] **P2.1**: Implement `IntegrationConfig` dataclass
- [ ] **P2.2**: Implement `FAMAILObjectiveFunction` class
- [ ] **P2.3**: Implement data loading and caching
- [ ] **P2.4**: Implement `compute()` method
- [ ] **P2.5**: Implement `compute_with_breakdown()` method
- [ ] **P2.6**: Add parallel computation support

**Duration**: 2 weeks  
**Deliverables**: Working integration layer

### 10.3 Phase 3: Testing and Validation (Weeks 7-8)

- [ ] **P3.1**: Unit tests for integration
- [ ] **P3.2**: Integration tests with real data
- [ ] **P3.3**: Performance benchmarking
- [ ] **P3.4**: Weight sensitivity analysis
- [ ] **P3.5**: Create baseline regression tests

**Duration**: 2 weeks  
**Deliverables**: Tested and validated integration

### 10.4 Phase 4: Monitoring and Tools (Week 9)

- [ ] **P4.1**: Implement OptimizationLogger
- [ ] **P4.2**: Create visualization tools
- [ ] **P4.3**: Add export/serialization
- [ ] **P4.4**: Documentation

**Duration**: 1 week  
**Deliverables**: Monitoring tools and documentation

### 10.5 Phase 5: ST-iFGSM Integration (Weeks 10-12)

- [ ] **P5.1**: Define interface with ST-iFGSM
- [ ] **P5.2**: Implement gradient estimation
- [ ] **P5.3**: Integration testing
- [ ] **P5.4**: End-to-end optimization runs
- [ ] **P5.5**: Hyperparameter tuning

**Duration**: 3 weeks  
**Deliverables**: Complete FAMAIL system

---

## Appendix

### A.1 Configuration Examples

```python
# Example configurations for different use cases

# Research baseline
BALANCED_CONFIG = IntegrationConfig(
    alpha_causal=0.33,
    alpha_spatial=0.33,
    alpha_fidelity=0.34,
)

# Fairness study
FAIRNESS_CONFIG = IntegrationConfig(
    alpha_causal=0.40,
    alpha_spatial=0.40,
    alpha_fidelity=0.20,
)

# Deployment readiness
DEPLOYMENT_CONFIG = IntegrationConfig(
    alpha_causal=0.25,
    alpha_spatial=0.25,
    alpha_fidelity=0.50,
)

# Ablation study (one term only)
SPATIAL_ONLY_CONFIG = IntegrationConfig(
    alpha_causal=0.0,
    alpha_spatial=1.0,
    alpha_fidelity=0.0,
)
```

### A.2 Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| Objective always 0 | Missing data | Check data loading |
| Objective > 1 | Unnormalized weights | Normalize or adjust |
| Term value NaN | Division by zero | Add epsilon handling |
| Slow computation | No caching | Enable cache_data |
| Memory error | Large trajectories | Process in batches |

### A.3 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial integration development plan |
| 1.1.0 | 2026-01-12 | Removed Quality Term ($F_{\text{quality}}$) from objective function; Updated to 3-term formulation |

---

*This document provides the integration framework for the FAMAIL objective function. It should be used in conjunction with the individual term development plans.*
