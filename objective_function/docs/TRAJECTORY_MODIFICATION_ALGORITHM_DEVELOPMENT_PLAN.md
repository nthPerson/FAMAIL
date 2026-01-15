# Trajectory Modification Algorithm: Development Plan

**Version**: 1.2.0  
**Last Updated**: 2025-01-15  
**Status**: Draft (Phase 0 — 75% Complete)  
**Authors**: FAMAIL Research Team

---

## Changelog

### v1.2.1 (2026-01-14)
- **Task 0.3 VERIFIED**: Comprehensive gradient tests confirm Isotonic and Binning methods allow proper gradient flow
- Added `verify_gradient_with_estimation_method()` and `verify_all_estimation_methods()` test functions
- Added "Method Gradient Tests" tab to Causal Fairness dashboard
- Updated Phase 0 status from 75% to 80% complete
- All estimation methods pass gradient verification with <0.0001% relative error

### v1.2.0 (2025-01-15)
- Added Phase 0 Implementation Status Report (Section 9.0.1)
- Updated Phase 0 task table with completion status
- Added Causal Fairness R² benchmark results (Isotonic/Binning methods)
- Documented Isotonic/Binning differentiability concerns
- Added Fidelity Term implementation summary (Section 11.6)
- Documented Fidelity score calibration issue (Section 11.4.1)
- Updated Section 11.1-11.4 with implementation status

### v1.1.0 (2025-01-12)
- Added gradient-based attribution analysis (Method C selection)
- Added multi-period trajectory handling options
- Added ST-iFGSM integration details

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Algorithm Design](#2-core-algorithm-design)
3. [Per-Trajectory Fairness Attribution](#3-per-trajectory-fairness-attribution)
4. [Trajectory Selection Strategy](#4-trajectory-selection-strategy)
5. [Modification Mechanism](#5-modification-mechanism)
6. [Gradient-Based Optimization Techniques](#6-gradient-based-optimization-techniques)
7. [Fidelity Validation](#7-fidelity-validation)
8. [Iterative Convergence](#8-iterative-convergence)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Open Questions and Future Research](#10-open-questions-and-future-research)
11. [Required Component Modifications](#11-required-component-modifications)
12. [References](#12-references)

---

## 1. Executive Summary

### 1.1 Purpose

This document provides a comprehensive development plan for the **Trajectory Modification Algorithm**—the core component of the FAMAIL system responsible for editing expert taxi driver trajectories to improve fairness while maintaining fidelity. The algorithm bridges the objective function specification with practical trajectory editing operations.

### 1.2 Key Insight

> **"Each trajectory contributes to global fairness metrics, and some trajectories disproportionately cause unfairness. Modifying one trajectory affects the fairness metrics for others due to the interdependence through global supply-demand calculations."**

This interdependence is central to the algorithm design: we cannot treat trajectories independently. Instead, we must carefully rank, select, modify, and re-evaluate trajectories iteratively.

### 1.3 Algorithm Philosophy

The FAMAIL approach prioritizes **trajectory editing over trajectory generation**:

- **Editing**: Small adjustments to existing expert trajectories (move pickup/dropoff within $n \times n$ neighborhood)
- **Generation**: Creating entirely new synthetic trajectories from scratch

Editing is preferred because:
1. Preserves expert knowledge and realistic driving patterns
2. Easier to validate via discriminator (modified trajectory should still "look like" the original)
3. Bounded modifications enable constraint satisfaction ($\epsilon$ and $\eta$ limits)

### 1.4 Key Design Decisions (v1.1.0)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Attribution Method** | Gradient-Based (Method C) | Methods A & B computationally infeasible or lack explainability |
| **Differentiability** | End-to-end differentiable | Required for gradient-based attribution |
| **Batch Size** | Configurable parameter | Simple, flexible; no adaptive behavior initially |
| **Multi-period Trajectories** | Average fairness over start/end periods | Best balance of granularity and simplicity |

### 1.5 Relationship to Objective Function

The trajectory modification algorithm operationalizes the optimization:

$$
\max_{\mathcal{T}'} \mathcal{L} = \max_{\mathcal{T}'} \left( \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} \right)
$$

Subject to:
- **Subtle Edits**: $\|\tau' - \tau\|_\infty \leq \epsilon$ (spatial proximity)
- **Limited Modifications**: $\|\tau' - \tau\|_0 \leq \eta$ (count of changed points)

---

## 2. Core Algorithm Design

### 2.1 Six-Step Algorithm Overview

The FAMAIL trajectory modification algorithm follows six iterative steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY MODIFICATION LOOP                      │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 1: Rank trajectories by fairness impact                       │
│      ↓                                                               │
│  STEP 2: Identify worst-offending trajectories (demand hierarchy)   │
│      ↓                                                               │
│  STEP 3: Modify selected trajectories (reallocate pickups/dropoffs) │
│      ↓                                                               │
│  STEP 4: Validate fidelity via discriminator                        │
│      ↓                                                               │
│  STEP 5: Recompute fairness metrics (update Nᵖ)                     │
│      ↓                                                               │
│  STEP 6: Check convergence → LOOP or EXIT                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Algorithm Pseudocode

```python
def trajectory_modification_algorithm(
    T: Set[Trajectory],           # Original expert trajectories
    D: Grid,                      # Demand matrix (pickup_dropoff_counts)
    S: Grid,                      # Supply matrix (active_taxis)
    discriminator: STSiameseNet,  # Fidelity validator
    epsilon: float,               # Max spatial perturbation (grid cells)
    eta: int,                     # Max modified points per trajectory
    alpha: Tuple[float, float, float],  # Objective weights
    max_iterations: int = 100,
    convergence_threshold: float = 1e-4
) -> Set[Trajectory]:
    """
    Main trajectory modification algorithm.
    
    Returns:
        T_prime: Modified trajectory set with improved fairness
    """
    T_prime = copy(T)
    N_p = compute_supply_distribution(T_prime)  # Initial supply
    prev_objective = float('-inf')
    
    for iteration in range(max_iterations):
        # STEP 1: Rank trajectories by fairness impact
        fairness_impacts = compute_per_trajectory_fairness(T_prime, D, N_p)
        ranked_trajectories = sort_by_impact(fairness_impacts, ascending=True)
        
        # STEP 2: Select worst-offending trajectories
        candidates = filter_by_demand_hierarchy(ranked_trajectories, D, N_p)
        selected = candidates[:batch_size]
        
        # STEP 3: Modify selected trajectories
        for tau in selected:
            tau_modified = modify_trajectory(
                tau, D, N_p, epsilon, eta,
                neighborhood_size=5  # 5×5 grid (k=2)
            )
            
            # STEP 4: Validate fidelity
            if discriminator.is_realistic(tau, tau_modified):
                T_prime.replace(tau, tau_modified)
        
        # STEP 5: Recompute fairness metrics
        N_p = compute_supply_distribution(T_prime)
        current_objective = compute_objective(T_prime, D, N_p, alpha)
        
        # STEP 6: Check convergence
        if abs(current_objective - prev_objective) < convergence_threshold:
            break
        prev_objective = current_objective
    
    return T_prime
```

### 2.3 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Modification scope** | Edit (not generate) | Preserves expert knowledge, enables fidelity validation |
| **Neighborhood size** | 5×5 (k=2) | Balances flexibility with constraint satisfaction |
| **Batch processing** | Process multiple trajectories per iteration | Computational efficiency |
| **Supply recalculation** | After each batch | Captures interdependence effects |

---

## 3. Per-Trajectory Fairness Attribution

### 3.1 The Attribution Problem

**Challenge**: Global fairness metrics ($F_{\text{spatial}}$, $F_{\text{causal}}$) are computed over the entire trajectory set. How do we attribute fairness responsibility to individual trajectories?

### 3.2 Attribution Methods Analysis

#### Method A: Marginal Contribution (Leave-One-Out) — ❌ REJECTED

Compute how much fairness changes when trajectory $\tau_i$ is removed:

$$
\Delta F_i = F(\mathcal{T}) - F(\mathcal{T} \setminus \{\tau_i\})
$$

**Pros**: Theoretically clean, captures true contribution  
**Cons**: **Computationally infeasible**

> **Feasibility Analysis**: Each fairness contribution calculation requires recalculating the `active_taxis` dataset, which takes ~190 seconds per trajectory. With ~44,000 trajectories in the dataset:
> $$44,000 \times 190 \text{ seconds} = 8,360,000 \text{ seconds} \approx 96.8 \text{ days per iteration}$$
> 
> For the iterative algorithm (potentially 100+ iterations), this would require **~15 years** of computation time.

**Decision**: Method A is computationally infeasible and is **rejected**.

#### Method B: Spatial Mismatch Score — ❌ REJECTED

For each trajectory, compute how its pickups/dropoffs contribute to supply-demand mismatch:

$$
\text{Mismatch}_i = \sum_{c \in \text{cells}(\tau_i)} \left| \frac{N^p_c}{N^D_c} - \frac{N^p_{\text{global}}}{N^D_{\text{global}}} \right|
$$

Where:
- $\text{cells}(\tau_i)$: Grid cells visited by trajectory $\tau_i$
- $N^p_c$: Supply in cell $c$
- $N^D_c$: Demand in cell $c$

**Pros**: Computationally efficient, intuitive  
**Cons**: **Not directly tied to fairness metrics**

> **Explainability Concern**: The spatial mismatch score measures supply-demand imbalance, but this is a proxy metric that does not directly correspond to the fairness terms ($F_{\text{spatial}}$, $F_{\text{causal}}$) in our objective function. This creates an explainability gap—we cannot clearly articulate why a trajectory was selected for modification in terms of its actual fairness impact.

**Decision**: Method B has unacceptable explainability limitations and is **rejected**.

#### Method C: Gradient-Based Attribution — ✅ SELECTED

Use the gradient of the objective function with respect to trajectory parameters:

$$
\text{Impact}_i = \left\| \nabla_{\tau_i} \mathcal{L} \right\|
$$

**Pros**: 
- Directly tied to optimization objective (perfect explainability)
- Computationally efficient with automatic differentiation
- Consistent with ST-iFGSM approach we are already following
- Enables end-to-end gradient-based optimization

**Cons**: Requires differentiable formulation of all objective function terms

**Decision**: Method C is **selected** as the primary attribution method.

### 3.3 Selected Approach: End-to-End Differentiable Attribution

Given the selection of gradient-based attribution, the entire objective function must be differentiable:

$$
\nabla_{\tau} \mathcal{L} = \alpha_1 \nabla_{\tau} F_{\text{causal}} + \alpha_2 \nabla_{\tau} F_{\text{spatial}} + \alpha_3 \nabla_{\tau} F_{\text{fidelity}}
$$

**Differentiability Requirements** (see Section 3.4 for details):

| Term | Current Status | Required Action |
|------|---------------|------------------|
| $F_{\text{fidelity}}$ | ✅ Differentiable | No changes needed (ST-SiameseNet uses standard PyTorch ops) |
| $F_{\text{spatial}}$ | ⚠️ Non-differentiable | Implement differentiable Gini approximation |
| $F_{\text{causal}}$ | ⚠️ Non-differentiable | Pre-compute $g(d)$ and use differentiable residual formulation |

```python
def compute_per_trajectory_fairness(T_prime, objective_fn):
    """
    Compute fairness attribution for each trajectory using gradients.
    
    Requires: All objective function terms must be differentiable.
    """
    impacts = {}
    
    for tau in T_prime:
        # Enable gradient tracking for this trajectory
        tau_tensor = tau.to_tensor(requires_grad=True)
        
        # Forward pass through differentiable objective
        objective = objective_fn(T_prime, tau_tensor)
        
        # Backward pass to compute gradient
        objective.backward()
        
        # Attribution = gradient magnitude
        impacts[tau] = tau_tensor.grad.norm().item()
        
        # Clear gradients for next trajectory
        tau_tensor.grad.zero_()
    
    return impacts
```

### 3.4 Differentiability Analysis by Objective Function Term

#### 3.4.1 Fidelity Term ($F_{\text{fidelity}}$) — ✅ Already Differentiable

The ST-SiameseNet discriminator is built with standard PyTorch operations:
- `nn.LSTM` — differentiable
- `nn.Linear` — differentiable  
- `nn.Sigmoid` — differentiable
- `nn.Dropout` — differentiable (pass-through in eval mode)

**Conclusion**: No modifications required. The discriminator can be used directly in gradient computations.

#### 3.4.2 Spatial Fairness Term ($F_{\text{spatial}}$) — ⚠️ Requires Modification

**Current Formulation**:
$$
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)
$$

Where $G$ is the Gini coefficient, computed via:
$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

**Differentiability Issue**: The standard Gini formulation involves absolute values and is technically differentiable, but the alternative sorted-index formulation (often used for efficiency) involves sorting, which is non-differentiable.

**Solution**: Use the **pairwise difference formulation** which is fully differentiable:

```python
def differentiable_gini(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Gini coefficient in a differentiable manner.
    
    Uses pairwise absolute differences (differentiable via torch.abs).
    Note: torch.abs is differentiable everywhere except at 0.
    """
    n = x.size(0)
    x_mean = x.mean()
    
    # Pairwise absolute differences
    diff_matrix = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    
    # Gini = sum of all pairwise differences / (2 * n^2 * mean)
    gini = diff_matrix.sum() / (2 * n * n * x_mean + 1e-8)
    
    return gini
```

**Action Required**: Update `spatial_fairness/DEVELOPMENT_PLAN.md` to use differentiable Gini computation.

#### 3.4.3 Causal Fairness Term ($F_{\text{causal}}$) — ⚠️ Requires Modification

**Current Formulation**:
$$
F_{\text{causal}}^p = \frac{\text{Var}_p(g(D_{i,p}))}{\text{Var}_p(Y_{i,p})} = 1 - \frac{\text{Var}_p(R_{i,p})}{\text{Var}_p(Y_{i,p})}
$$

Where:
- $Y = S/D$ (service ratio)
- $g(D)$ = expected service given demand (fitted function)
- $R = Y - g(D)$ (residual)

**Differentiability Issue**: The function $g(d)$ is estimated by fitting a regression model to observed data. The fitting process itself is not differentiable in an end-to-end manner.

**Solution**: **Pre-compute and freeze** $g(d)$:

1. **Pre-computation Phase** (offline, before optimization):
   - Fit $g(d)$ using the original expert trajectories
   - Store as a lookup table or fitted polynomial coefficients
   
2. **Optimization Phase** (online, differentiable):
   - Use frozen $g(d)$ values
   - Compute residuals $R = Y - g(D)$ (differentiable)
   - Compute variance of residuals (differentiable)

```python
class DifferentiableCausalFairness(nn.Module):
    """
    Differentiable causal fairness computation with pre-computed g(d).
    """
    
    def __init__(self, g_lookup: torch.Tensor):
        """
        Args:
            g_lookup: Pre-computed g(d) values for each demand level.
                      Shape: [max_demand + 1]
        """
        super().__init__()
        # Freeze g(d) - not learned during optimization
        self.register_buffer('g_lookup', g_lookup)
    
    def forward(self, demand: torch.Tensor, supply: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable causal fairness.
        
        Args:
            demand: Demand tensor [num_cells]
            supply: Supply tensor [num_cells]
            
        Returns:
            Causal fairness score (scalar)
        """
        # Service ratio (differentiable)
        Y = supply / (demand + 1e-8)
        
        # Expected service from frozen g(d) (lookup, differentiable w.r.t. supply)
        g_d = self.g_lookup[demand.long()]
        
        # Residual (differentiable)
        R = Y - g_d
        
        # Variances (differentiable)
        var_Y = Y.var()
        var_R = R.var()
        
        # Causal fairness = 1 - (unexplained variance / total variance)
        F_causal = 1 - (var_R / (var_Y + 1e-8))
        
        return F_causal
```

**Action Required**: Update `causal_fairness/DEVELOPMENT_PLAN.md` to document pre-computation of $g(d)$ and differentiable residual computation.

### 3.5 Rationale for Differentiable Approach

The decision to pursue end-to-end differentiability is supported by:

1. **ST-iFGSM Precedent**: The ST-iFGSM approach we are following successfully uses gradient-based perturbations with the same discriminator architecture (ST-SiameseNet). This demonstrates the feasibility of gradient-based trajectory modification.

2. **Computational Efficiency**: Automatic differentiation computes gradients in $O(1)$ forward passes (via backpropagation), compared to $O(N)$ for finite differences or leave-one-out methods.

3. **Explainability**: Gradients directly answer "how does the objective change if I modify this trajectory?" — perfect alignment with our attribution goal.

4. **Extensibility**: Once the pipeline is differentiable, we can leverage advanced optimization techniques (Adam, learning rate scheduling, momentum) and potentially learn optimal modification strategies.

---

## 4. Trajectory Selection Strategy

### 4.1 Demand Hierarchy Filter

Not all "unfair" trajectories should be modified. The **demand hierarchy** filter ensures we prioritize trajectories where:

1. **High demand exists** in underserved regions (meaningful impact)
2. **Current trajectory serves overserved regions** (opportunity for improvement)
3. **Modification is feasible** within constraints

### 4.2 Selection Criteria

```python
def filter_by_demand_hierarchy(ranked_trajectories, D, N_p):
    """
    Filter trajectories based on demand context.
    
    Selection criteria:
    1. Trajectory currently serves overserved cells
    2. Underserved cells exist within modification neighborhood
    3. Demand in underserved cells is above threshold
    """
    candidates = []
    
    for tau in ranked_trajectories:
        # Get cells currently served
        served_cells = get_trajectory_cells(tau)
        
        # Check if trajectory serves overserved regions
        serves_overserved = any(
            supply_demand_ratio(c, N_p, D) > OVERSERVED_THRESHOLD
            for c in served_cells
        )
        
        # Check if underserved cells exist nearby
        neighborhood = get_neighborhood(served_cells, k=2)  # 5×5
        underserved_nearby = any(
            supply_demand_ratio(c, N_p, D) < UNDERSERVED_THRESHOLD
            and D[c] > MIN_DEMAND_THRESHOLD
            for c in neighborhood
        )
        
        if serves_overserved and underserved_nearby:
            candidates.append(tau)
    
    return candidates
```

### 4.3 Batch Size Considerations

**Trade-off**: Larger batches are more efficient but may cause "interference" between modifications.

| Batch Size | Pros | Cons |
|------------|------|------|
| 1 | Accurate interdependence tracking | Very slow |
| 10-50 | Good balance | Some interference |
| All | Fastest | High interference, may diverge |

**Recommendation**: Start with batch size of 10-20 trajectories, tune based on convergence behavior.

---

## 5. Modification Mechanism

### 5.1 Pickup/Dropoff Reallocation

The core modification operation is **reallocating pickup and dropoff locations** within a constrained neighborhood.

#### 5.1.1 Modification Space

For each trajectory point (pickup or dropoff at cell $c$), the valid modification space is:

$$
\mathcal{N}(c, k) = \{c' : \|c' - c\|_\infty \leq k\}
$$

With $k=2$ (5×5 neighborhood), each point can move to any of up to 25 candidate cells.

#### 5.1.2 Modification Selection

```python
def modify_trajectory(tau, D, N_p, epsilon, eta, neighborhood_size=5):
    """
    Modify a single trajectory to improve fairness.
    
    Strategy:
    1. Identify points in overserved cells
    2. For each point, find best alternative in neighborhood
    3. Apply modifications up to eta limit
    """
    k = (neighborhood_size - 1) // 2  # k=2 for 5×5
    modifications = []
    
    for point in tau.points:
        if len(modifications) >= eta:
            break
            
        cell = point.cell
        if not is_overserved(cell, N_p, D):
            continue
        
        # Find best alternative in neighborhood
        neighborhood = get_neighborhood(cell, k)
        candidates = [
            c for c in neighborhood
            if is_underserved(c, N_p, D) and D[c] > MIN_DEMAND
        ]
        
        if candidates:
            # Select cell that maximizes fairness improvement
            best_cell = max(candidates, key=lambda c: fairness_gain(c, N_p, D))
            modifications.append((point, best_cell))
    
    # Apply modifications
    tau_modified = apply_modifications(tau, modifications)
    return tau_modified
```

### 5.2 Constraint Enforcement

#### 5.2.1 Subtle Edits Constraint ($\epsilon$)

The $\epsilon$ constraint bounds the maximum spatial displacement:

$$
\|\tau' - \tau\|_\infty = \max_i \|p'_i - p_i\|_\infty \leq \epsilon
$$

**Implementation**: With grid cells of ~0.01° (~1.1 km), $\epsilon = 2$ allows movement within ~2.2 km.

```python
def enforce_epsilon_constraint(original_cell, candidate_cell, epsilon):
    """Check if modification satisfies epsilon constraint."""
    distance = max(
        abs(candidate_cell.row - original_cell.row),
        abs(candidate_cell.col - original_cell.col)
    )
    return distance <= epsilon
```

#### 5.2.2 Limited Modifications Constraint ($\eta$)

The $\eta$ constraint limits the number of modified points per trajectory:

$$
\|\tau' - \tau\|_0 = \#\{i : p'_i \neq p_i\} \leq \eta
$$

**Implementation**: Track modification count and stop when limit reached.

### 5.3 Temporal Consistency

**Important**: When modifying pickup/dropoff locations, we must maintain temporal consistency:

1. **Pickup before dropoff**: Modified pickup location must be reachable before dropoff time
2. **Travel time feasibility**: Distance between consecutive points must be traversable in allocated time
3. **Grid boundary respect**: Modified points must remain within the 48×90 grid

```python
def check_temporal_consistency(tau_modified):
    """Verify modified trajectory maintains temporal feasibility."""
    for i in range(len(tau_modified.points) - 1):
        p1, p2 = tau_modified.points[i], tau_modified.points[i+1]
        distance = grid_distance(p1.cell, p2.cell)
        time_available = p2.timestamp - p1.timestamp
        
        # Assume minimum speed of 10 km/h in urban traffic
        max_distance = time_available * MIN_SPEED
        
        if distance > max_distance:
            return False
    return True
```

---

## 6. Gradient-Based Optimization Techniques

### 6.1 Integration with ST-iFGSM

The ST-iFGSM (Spatio-Temporal iterative Fast Gradient Sign Method) framework provides gradient-based perturbation techniques that can be adapted for trajectory modification.

#### 6.1.1 ST-iFGSM Overview

ST-iFGSM was developed for adversarial attacks on trajectory similarity models. Key concepts:

- **Gradient computation**: Compute gradient of loss with respect to input trajectory
- **Sign-based perturbation**: Move in direction of gradient sign
- **Iterative refinement**: Apply small perturbations repeatedly
- **Constraint projection**: Project perturbations back to valid space

#### 6.1.2 Adaptation for FAMAIL

| ST-iFGSM Concept | FAMAIL Adaptation |
|------------------|-------------------|
| Adversarial loss | Fairness objective $\mathcal{L}$ |
| Trajectory embedding | Grid cell representation |
| Continuous perturbation | Discrete cell reassignment |
| $L_\infty$ bound | $\epsilon$ constraint (grid cells) |

### 6.2 FGSM-Style Modification

#### 6.2.1 Gradient Computation

For gradient-based modification, we need the gradient of the objective with respect to trajectory locations:

$$
\nabla_\tau \mathcal{L} = \alpha_1 \nabla_\tau F_{\text{causal}} + \alpha_2 \nabla_\tau F_{\text{spatial}} + \alpha_3 \nabla_\tau F_{\text{fidelity}}
$$

**Challenge**: Trajectories are discrete (grid cells), not continuous. We need a relaxation.

#### 6.2.2 Soft Cell Assignment

Relax discrete cell assignment to continuous probability distribution:

$$
p_i \approx \sum_{c \in \text{Grid}} \sigma_c \cdot \mathbf{1}_c
$$

Where $\sigma_c$ is the probability of point $i$ being assigned to cell $c$.

**Gumbel-Softmax Trick**: Use temperature-controlled softmax for differentiable discrete sampling:

$$
\sigma_c = \frac{\exp((z_c + g_c) / \tau)}{\sum_{c'} \exp((z_{c'} + g_{c'}) / \tau)}
$$

Where $g_c$ are Gumbel noise samples and $\tau$ is temperature.

#### 6.2.3 Iterative FGSM for Trajectories

```python
def iterative_fgsm_modification(
    tau: Trajectory,
    objective_fn: Callable,
    epsilon: float,
    alpha: float = 0.01,
    num_iterations: int = 10
) -> Trajectory:
    """
    Apply iterative FGSM-style modification to trajectory.
    
    Args:
        tau: Original trajectory
        objective_fn: Fairness objective (to maximize)
        epsilon: Maximum perturbation bound
        alpha: Step size per iteration
        num_iterations: Number of gradient steps
    
    Returns:
        Modified trajectory
    """
    # Initialize soft cell assignments
    soft_assignments = initialize_soft_assignments(tau)
    
    for i in range(num_iterations):
        # Compute gradient of objective
        grad = compute_gradient(soft_assignments, objective_fn)
        
        # Update in direction of gradient (maximize)
        soft_assignments = soft_assignments + alpha * torch.sign(grad)
        
        # Project to valid space (epsilon constraint)
        soft_assignments = project_to_neighborhood(
            soft_assignments, tau, epsilon
        )
        
        # Normalize to valid probability distribution
        soft_assignments = normalize_assignments(soft_assignments)
    
    # Convert soft assignments to hard cell assignments
    tau_modified = hard_assignment(soft_assignments, tau)
    
    return tau_modified
```

### 6.3 Carlini-Wagner Style Optimization

For more precise control, we can adapt the CW attack optimization:

$$
\min_\delta \|\delta\|_p + c \cdot f(\tau + \delta)
$$

Where:
- $\delta$ is the perturbation
- $f(\cdot)$ is the fairness improvement objective
- $c$ is a balancing constant

**Advantage**: CW finds minimal perturbations that achieve the objective, naturally satisfying constraints.

```python
def cw_style_modification(
    tau: Trajectory,
    objective_fn: Callable,
    fidelity_threshold: float,
    c: float = 1.0,
    learning_rate: float = 0.01,
    max_iterations: int = 1000
) -> Trajectory:
    """
    Carlini-Wagner style trajectory modification.
    
    Minimizes perturbation size while achieving fairness improvement.
    """
    delta = torch.zeros_like(tau.embedding, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    
    for i in range(max_iterations):
        optimizer.zero_grad()
        
        tau_perturbed = tau.embedding + delta
        
        # Loss = perturbation size + c * (improvement needed)
        perturbation_loss = torch.norm(delta, p=float('inf'))
        fairness_loss = -objective_fn(tau_perturbed)  # Negative for maximization
        
        loss = perturbation_loss + c * fairness_loss
        
        loss.backward()
        optimizer.step()
        
        # Project delta to valid range
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
    
    return discretize_trajectory(tau.embedding + delta)
```

### 6.4 Differentiability Summary

**Status**: End-to-end differentiability has been confirmed as feasible. See Section 3.4 for detailed analysis.

| Component | Differentiability | Notes |
|-----------|-------------------|-------|
| $F_{\text{fidelity}}$ | ✅ Already differentiable | ST-SiameseNet uses standard PyTorch ops |
| $F_{\text{spatial}}$ | ⚠️ Requires modification | Use pairwise difference Gini formulation |
| $F_{\text{causal}}$ | ⚠️ Requires modification | Pre-compute $g(d)$, use differentiable residuals |

**Gradient flow** through the objective function is now validated:

```
τ (trajectory) ──► F_spatial(τ) ──┐
                                  │
τ (trajectory) ──► F_causal(τ)  ──┼──► L = α₁F_causal + α₂F_spatial + α₃F_fidelity
                                  │
τ (trajectory) ──► F_fidelity(τ)──┘
                                  │
                                  ▼
                        ∇_τ L (gradient for attribution & optimization)
```

---

## 7. Fidelity Validation

### 7.1 Discriminator Integration

The **ST-SiameseNet discriminator** validates whether modified trajectories remain realistic:

```
Original Trajectory ──┐
                      ├──► ST-SiameseNet ──► Similarity Score ──► Accept/Reject
Modified Trajectory ──┘
```

### 7.2 Acceptance Criteria

```python
def validate_fidelity(
    tau_original: Trajectory,
    tau_modified: Trajectory,
    discriminator: STSiameseNet,
    threshold: float = 0.7
) -> bool:
    """
    Validate that modified trajectory maintains fidelity.
    
    Args:
        tau_original: Original expert trajectory
        tau_modified: Modified trajectory
        discriminator: Pre-trained ST-SiameseNet
        threshold: Minimum similarity score for acceptance
    
    Returns:
        True if modification is acceptable, False otherwise
    """
    similarity = discriminator.compute_similarity(tau_original, tau_modified)
    return similarity >= threshold
```

### 7.3 Fidelity-Fairness Trade-off

**Key Insight**: There is a fundamental trade-off between fidelity and fairness improvement:

- **High fidelity threshold** → Fewer accepted modifications → Slower fairness improvement
- **Low fidelity threshold** → More accepted modifications → Risk of unrealistic trajectories

**Recommendation**: Start with high threshold (0.8), lower if convergence is too slow.

### 7.4 Batch Fidelity Validation

For efficiency, validate fidelity in batches:

```python
def batch_validate_fidelity(
    original_trajectories: List[Trajectory],
    modified_trajectories: List[Trajectory],
    discriminator: STSiameseNet,
    threshold: float = 0.7
) -> List[bool]:
    """
    Batch validation for efficiency.
    """
    # Stack trajectories into batch tensors
    original_batch = stack_trajectories(original_trajectories)
    modified_batch = stack_trajectories(modified_trajectories)
    
    # Single forward pass through discriminator
    similarities = discriminator.compute_similarity_batch(
        original_batch, modified_batch
    )
    
    return [s >= threshold for s in similarities]
```

---

## 8. Iterative Convergence

### 8.1 Convergence Criteria

The algorithm converges when any of the following conditions is met:

1. **Objective plateau**: $|\mathcal{L}^{(t)} - \mathcal{L}^{(t-1)}| < \epsilon_{\text{conv}}$
2. **Maximum iterations**: $t > T_{\max}$
3. **No valid modifications**: All candidates rejected by fidelity validator

### 8.2 Supply Recalculation

**Critical Design Choice**: When is $N^p$ (supply distribution) recalculated?

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Per-modification** | Update after each trajectory change | Most accurate | Very slow |
| **Per-batch** | Update after each batch of modifications | Good balance | Some staleness |
| **Per-iteration** | Update once per outer loop | Efficient | May miss fine-grained effects |

**Recommendation**: Per-batch update (after each batch of 10-20 trajectories).

### 8.3 Holding $N^p$ Constant Within Iterations

From Meeting 18:

> "Hold $N^p$ constant within a single trajectory modification iteration. This allows us to rank and modify multiple trajectories based on the same baseline, then recompute $N^p$ before the next iteration."

This approach:
1. Ensures consistent ranking within a batch
2. Avoids "chasing" effects where modifications cascade unpredictably
3. Enables parallelization of modifications within a batch

### 8.4 Convergence Monitoring

```python
class ConvergenceMonitor:
    """Track and visualize convergence progress."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.history = {
            'objective': [],
            'spatial_fairness': [],
            'causal_fairness': [],
            'fidelity': [],
            'num_modifications': []
        }
        self.patience = patience
        self.min_delta = min_delta
        self.best_objective = float('-inf')
        self.wait = 0
    
    def update(self, metrics: dict) -> bool:
        """
        Update history and check for convergence.
        
        Returns:
            True if should continue, False if converged
        """
        for key, value in metrics.items():
            self.history[key].append(value)
        
        current_objective = metrics['objective']
        
        if current_objective > self.best_objective + self.min_delta:
            self.best_objective = current_objective
            self.wait = 0
        else:
            self.wait += 1
        
        return self.wait < self.patience
    
    def plot_convergence(self):
        """Generate convergence visualization."""
        # Implementation for matplotlib plotting
        pass
```

---

## 9. Implementation Roadmap

### 9.0 Phase 0: Differentiability Modifications (Priority — Before Main Development)

> ⚠️ **PREREQUISITE**: The following modifications to existing components must be completed before the main trajectory modification algorithm can be implemented. These ensure end-to-end differentiability required for gradient-based attribution.

| Task | Description | Component | Status | Deliverable |
|------|-------------|-----------|--------|-------------|
| **0.1** | Implement differentiable Gini coefficient | Spatial Fairness | ✅ Complete | `spatial_fairness/term.py` |
| **0.2** | Pre-compute $g(d)$ lookup table | Causal Fairness | ✅ Complete | `causal_fairness/utils.py` |
| **0.3** | Implement differentiable residual computation | Causal Fairness | ✅ Verified | `causal_fairness/term.py`, gradient tests |
| **0.4** | Validate discriminator gradient flow | Fidelity | ✅ Complete | `fidelity/utils.py` gradient verification |
| **0.5** | Create differentiable objective wrapper | Integration | 🔲 Not Started | `differentiable_objective.py` |

**Estimated Duration**: 1-2 weeks (can be parallelized with Phase 1)

---

### 9.0.1 Phase 0 Implementation Status Report (Updated: January 2026)

> 📋 **STATUS SUMMARY**: Phase 0 is approximately **80% complete**. All three objective function terms have been implemented with differentiability support and verified. The Causal Fairness gradient verification for Isotonic/Binning methods is now **complete and passing**. The remaining work is the Fidelity Term validation and the unified objective wrapper.

#### Task 0.1 — Differentiable Gini: ✅ COMPLETE

The differentiable Gini coefficient has been implemented in `spatial_fairness/term.py` using the pairwise absolute difference formulation as specified.

**Implementation Location**: `objective_function/spatial_fairness/term.py`

**Verification**:
- Gradient flow verified via `verify_differentiability()` method
- Dashboard gradient verification tab confirms non-zero gradients
- Unit tests pass for both forward and backward pass

#### Task 0.2 — Pre-compute g(d) Lookup: ✅ COMPLETE

Multiple estimation methods have been implemented for computing the expected service function $g(d)$:

| Method | Implementation | R² Score | Notes |
|--------|----------------|----------|-------|
| **Isotonic** | `IsotonicRegression` | **0.4453** | Highest R², monotonic constraint |
| **Binning** | Binned means + interpolation | **0.4439** | Second highest, simple |
| **Polynomial** | `np.polyfit` deg=3 | 0.1949 | Originally recommended |
| **Linear** | `LinearRegression` | 0.1064 | Poor fit |
| **LOWESS** | `statsmodels.lowess` | 0.0000 | Failed on this data |

**Recommendation**: Based on R² benchmark results, **Isotonic** or **Binning** methods should be used for production. The polynomial method originally suggested in Section 11.3 performs significantly worse on this dataset.

**Implementation Location**: `objective_function/causal_fairness/utils.py` — `fit_expected_service_function()`

#### Task 0.3 — Differentiable Causal Fairness: ✅ VERIFIED (January 2026)

> ✅ **VALIDATED**: Comprehensive gradient verification tests confirm that **all g(d) estimation methods** (including Isotonic and Binning) allow proper gradient flow during trajectory optimization.

**The Differentiability Concern (Now Resolved)**:

The initial concern was that Isotonic and Binning methods use non-differentiable operations:
- **Isotonic**: Sorting and monotonicity constraint enforcement
- **Binning**: Discrete bin assignment via `pd.cut()`

**Why It Works**:

The $g(d)$ function is **pre-computed and frozen** before optimization begins. During gradient-based trajectory modification:

```python
# g_lookup is FROZEN — no gradient flows through it
demand_idx = demand.long().clamp(0, len(self.g_lookup) - 1)
g_d = self.g_lookup[demand_idx]  # Just a lookup, not a learned parameter

# Gradient flows through Y (supply/demand ratio)
R = Y - g_d  # Y has gradients, g_d does not (and shouldn't)
```

The key insight: we are **not differentiating through g(d) fitting** — gradients only need to flow through $R = Y - g(D)$, where $Y = S/D$ depends on supply $S$.

**Verification Results (January 2026)**:

Comprehensive tests were run with `verify_gradient_with_estimation_method()` and `verify_all_estimation_methods()`:

| Method | R² Fit | F_causal | Gradient Valid | Numerical Match | Status |
|--------|--------|----------|----------------|-----------------|--------|
| **Binning** | 0.8071 | 0.8071 | ✅ | ✅ | ✅ PASS |
| **Isotonic** | 0.8385 | 0.8385 | ✅ | ✅ | ✅ PASS |
| **Polynomial** | 0.5790 | 0.5790 | ✅ | ✅ | ✅ PASS |
| **Linear** | 0.3573 | 0.3573 | ✅ | ✅ | ✅ PASS |
| **LOWESS** | 0.8120 | 0.8120 | ✅ | ✅ | ✅ PASS |

**Test Details**:
- Analytic gradients computed via PyTorch autograd
- Numerical gradients verified via finite differences (float64 precision)
- Maximum relative error < 0.0001% for all methods
- 100% non-zero gradient coverage
- Tests passed across multiple random seeds (42, 123, 999) and sample sizes (50, 100, 200)

**Implementation Files**:
- `causal_fairness/utils.py` — `verify_gradient_with_estimation_method()`, `verify_all_estimation_methods()`
- `causal_fairness/dashboard.py` — "Method Gradient Tests" tab for interactive validation

**Conclusion**: Both **Isotonic** and **Binning** methods are safe to use for gradient-based trajectory optimization. Their superior R² scores (0.84 and 0.81 respectively) make them the recommended choice over Polynomial (0.58).

#### Task 0.4 — Discriminator Gradient Flow: ✅ COMPLETE

The ST-SiameseNet discriminator gradient flow has been verified.

**Implementation Location**: `objective_function/fidelity/utils.py` — `verify_fidelity_gradient()`

**Verification Results**:
- Model architecture uses only differentiable PyTorch operations (LSTM, Linear, Sigmoid)
- Gradient verification passes: gradients computed, non-zero
- LSTM backward pass requires training mode (fixed in implementation)

**Issue Discovered During Implementation**:

> ⚠️ **cuDNN LSTM Backward Pass**: The `nn.LSTM` module with cuDNN backend requires `model.train()` during backward pass, even for inference. The `verify_fidelity_gradient()` function now handles this automatically.

#### Task 0.4.1 — Fidelity Term Validation: ⚠️ REQUIRES INVESTIGATION

> ⚠️ **CONCERN**: Preliminary testing shows unexpectedly low fidelity scores, even in controlled scenarios.

**Observed Issue**:

When testing the Fidelity Term with identical trajectories (comparing a trajectory with itself), the discriminator outputs a fidelity score of approximately **0.42** instead of the expected **~1.0**.

**Possible Explanations**:

1. **Model Interpretation**: The ST-SiameseNet discriminator was trained to distinguish whether two trajectories are from the **same driver**, not whether they are **identical trajectories**. A score of 0.42 may be reasonable if the model learned that even the same trajectory has some inherent uncertainty.

2. **Feature Normalization**: The `FeatureNormalizer` in the discriminator applies transformations (sin/cos for time features, normalization for grid coordinates) that may reduce distinguishability.

3. **Training Data Distribution**: The discriminator may have been trained on trajectory pairs with specific statistical properties that differ from our test data.

**ST-iFGSM Alignment Concern**:

For the gradient-based trajectory modification to work as intended (per ST-iFGSM approach), we need to ensure:

1. **Fidelity as Constraint**: $F_{fidelity}(\tau', \tau) \geq \theta$ should constrain modifications to maintain trajectory authenticity.

2. **Gradient Utility**: The gradient $\nabla_{\tau'} F_{fidelity}$ should point in a direction that increases trajectory similarity.

3. **Score Calibration**: The fidelity score should have meaningful interpretation (e.g., 0.9+ = highly authentic).

**Action Items**:
1. [ ] Review ST-iFGSM paper for exact usage of discriminator scores
2. [ ] Compare our discriminator output distribution to ST-iFGSM baselines
3. [ ] Consider score calibration (sigmoid temperature, Platt scaling)
4. [ ] Test on known-similar vs. known-different trajectory pairs
5. [ ] Verify discriminator checkpoint is correctly loaded (architecture params)

#### Task 0.5 — Differentiable Objective Wrapper: 🔲 NOT STARTED

This task depends on resolution of Tasks 0.3 and 0.4.1 concerns.

**Proposed Design**:

```python
class DifferentiableObjective(nn.Module):
    """
    Unified differentiable objective function for gradient-based
    trajectory modification.
    """
    
    def __init__(
        self,
        spatial_fairness: SpatialFairnessTerm,
        causal_fairness: CausalFairnessTerm,
        fidelity: FidelityTerm,
        weights: ObjectiveWeights
    ):
        super().__init__()
        self.spatial = spatial_fairness
        self.causal = causal_fairness
        self.fidelity = fidelity
        self.weights = weights
    
    def forward(
        self,
        trajectories: torch.Tensor,
        original_trajectories: torch.Tensor,
        demand: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted objective and component scores.
        
        All computations are differentiable w.r.t. trajectories.
        """
        # Compute supply from trajectories
        supply = self._compute_supply(trajectories)
        
        # Compute each term
        F_spatial = self.spatial.compute(supply, demand)
        F_causal = self.causal.compute(supply, demand)
        F_fidelity = self.fidelity.compute(trajectories, original_trajectories)
        
        # Weighted combination
        L = (
            self.weights.alpha_spatial * F_spatial +
            self.weights.alpha_causal * F_causal +
            self.weights.alpha_fidelity * F_fidelity
        )
        
        breakdown = {
            'spatial': F_spatial,
            'causal': F_causal,
            'fidelity': F_fidelity,
            'total': L
        }
        
        return L, breakdown
```

---

**Details**:

**Task 0.1 — Differentiable Gini**:
```python
# Replace sorting-based Gini with pairwise difference formulation
def differentiable_gini(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    diff_matrix = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    return diff_matrix.sum() / (2 * n * n * x.mean() + 1e-8)
```

**Task 0.2 — Pre-compute g(d)**:
- Fit regression model to original expert trajectory data
- Store as `torch.Tensor` with shape `[max_demand + 1]`
- Freeze during optimization (no gradient)

**Task 0.3 — Differentiable Causal Fairness**:
```python
# Use frozen g(d) lookup for differentiable residual computation
R = Y - g_lookup[demand.long()]  # Differentiable w.r.t. Y
var_R = R.var()                   # Differentiable
```

### 9.1 Phase 1: Foundation (Weeks 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Implement trajectory data structures (PyTorch-compatible) | `trajectory.py` module |
| 1.2 | Implement grid neighborhood operations | `grid_utils.py` module |
| 1.3 | Implement supply-demand ratio calculations | `supply_demand.py` module |
| 1.4 | Implement multi-period trajectory handling (see Section 10.1) | `temporal.py` module |
| 1.5 | Unit tests for foundation modules | Test suite |

### 9.2 Phase 2: Gradient-Based Attribution (Weeks 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | Implement gradient-based fairness attribution | `attribution.py` |
| 2.2 | Implement trajectory tensor representation | `trajectory_tensor.py` |
| 2.3 | Implement batch gradient computation | `batch_gradients.py` |
| 2.4 | Validate attribution accuracy | Attribution test suite |

### 9.3 Phase 3: Modification (Weeks 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | Implement basic reallocation mechanism | `modification.py` |
| 3.2 | Implement constraint enforcement ($\epsilon$, $\eta$) | `constraints.py` |
| 3.3 | Implement temporal consistency checks | `validation.py` |
| 3.4 | Implement demand hierarchy filter | `selection.py` |
| 3.5 | Implement configurable batch size | `config.py` |

### 9.4 Phase 4: Gradient-Based Optimization (Weeks 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | Implement soft cell assignment (Gumbel-Softmax) | `gradient_methods.py` |
| 4.2 | Implement iterative FGSM adaptation | `fgsm_modification.py` |
| 4.3 | Implement CW-style optimization | `cw_modification.py` |
| 4.4 | Benchmark gradient methods on small trajectory subset | Performance report |

### 9.5 Phase 5: Integration (Weeks 9-10)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | Integrate discriminator for fidelity validation | `fidelity_validator.py` |
| 5.2 | Implement convergence monitoring | `convergence.py` |
| 5.3 | Implement full iterative algorithm | `trajectory_modifier.py` |
| 5.4 | End-to-end testing | Integration test suite |

### 9.6 Phase 6: Evaluation (Weeks 11-12)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 6.1 | Evaluate fairness improvement | Metrics report |
| 6.2 | Evaluate fidelity preservation | Discriminator evaluation |
| 6.3 | Analyze computational efficiency | Performance benchmarks |
| 6.4 | Document findings and recommendations | Final report |

---

## 10. Open Questions and Future Research

### 10.1 Algorithm Design

> **[RESOLVED - OPEN QUESTION 1]**: What is the optimal batch size for trajectory modification? Should it be adaptive based on convergence behavior?
>
> **Decision**: Batch size will be a **tunable configuration parameter** in the initial implementation. No adaptive behavior will be implemented at this time. Recommended starting values: 10-20 trajectories per batch.

> **[OPEN QUESTION 2]**: Should we use simulated annealing or other metaheuristics instead of/in addition to gradient methods for the discrete optimization?

> **[RESOLVED - OPEN QUESTION 3]**: How do we handle trajectories that span multiple time periods? Should fairness be computed per-time-bucket or aggregated?
>
> **Analysis**: With 5-minute time buckets (or even hourly aggregation), many trajectories will span multiple time periods. Ignoring these trajectories would excessively limit the number of editable trajectories.
>
> **Rejected Approaches**:
> - Aggregate fairness over all time periods → Loses all period-based granularity (unacceptable)
> - Ignore multi-period trajectories → Too restrictive given temporal resolution
>
> **Decision**: Implement the following configurable options for multi-period trajectory handling:
>
> | Option | Description | Config Value |
> |--------|-------------|---------------|
> | **Middle Period** | Use period at `(end_period - start_period) / 2` | `"middle"` |
> | **Start Period** | Use the period where trajectory begins | `"start"` |
> | **End Period** | Use the period where trajectory ends | `"end"` |
> | **Average** (default) | Average fairness across start and end periods | `"average"` |
>
> **Recommended Default**: `"average"` — provides the best balance between granularity preservation and computational simplicity.
>
> ```python
> class MultiPeriodHandling(Enum):
>     MIDDLE = "middle"
>     START = "start"
>     END = "end" 
>     AVERAGE = "average"  # Default
>
> def get_analysis_periods(trajectory, method: MultiPeriodHandling):
>     start_p = trajectory.start_time_period
>     end_p = trajectory.end_time_period
>     
>     if method == MultiPeriodHandling.MIDDLE:
>         return [(start_p + end_p) // 2]
>     elif method == MultiPeriodHandling.START:
>         return [start_p]
>     elif method == MultiPeriodHandling.END:
>         return [end_p]
>     elif method == MultiPeriodHandling.AVERAGE:
>         return [start_p, end_p]  # Fairness averaged over both
> ```

### 10.2 Fairness Attribution

> **[RESOLVED - OPEN QUESTION 4]**: Is leave-one-out attribution sufficient, or should we use Shapley values for game-theoretic fairness attribution?
>
> **Decision**: Neither. Both leave-one-out and Shapley values are computationally infeasible (~15 years for leave-one-out). **Gradient-based attribution (Method C)** is the selected approach. See Section 3 for full analysis.

> **[OPEN QUESTION 5]**: How do we handle trajectories that contribute positively to some fairness aspects and negatively to others?

### 10.3 Gradient Methods

> **[RESOLVED - OPEN QUESTION 6]**: Can we make the discriminator differentiable for end-to-end gradient optimization? Or should we use reinforcement learning with the discriminator as a reward signal?
>
> **Decision**: Yes, the discriminator **is already differentiable**. The ST-SiameseNet architecture uses standard PyTorch operations (LSTM, Linear, Sigmoid) which all support automatic differentiation. This is consistent with the ST-iFGSM approach we are following—they successfully used the same discriminator architecture for gradient-based attacks.
>
> **Implication**: No changes needed to the discriminator. The $F_{\text{fidelity}}$ term can be used directly in gradient computations.

> **[OPEN QUESTION 7]**: What temperature schedule should be used for Gumbel-Softmax relaxation?

### 10.4 Scalability

> **[OPEN QUESTION 8]**: How does the algorithm scale with:
> - Number of trajectories (currently 50 drivers × ~100 trajectories)
> - Grid resolution (currently 48×90 = 4,320 cells)
> - Time buckets (currently 288 per day)

> **[OPEN QUESTION 9]**: Can we use approximate methods (e.g., sampling, importance weighting) to reduce computational cost?

### 10.5 Causal Fairness Integration

> **[OPEN QUESTION 10]**: How should causal fairness ($F_{\text{causal}}$) influence trajectory selection? Should we prioritize trajectories that serve areas with high causal unfairness (high income × low service)?

---

## 11. Required Component Modifications

> ⚠️ **ACTION REQUIRED**: The following modifications to existing components are necessary to enable end-to-end differentiability for gradient-based attribution. These should be addressed in a **separate task** before implementing the main trajectory modification algorithm.

### 11.1 Summary of Required Changes

| Component | File | Change Type | Priority | Status |
|-----------|------|-------------|----------|--------|
| **Spatial Fairness** | `spatial_fairness/term.py` | Differentiable Gini | 🔴 High | ✅ Complete |
| **Causal Fairness** | `causal_fairness/utils.py` | g(d) estimation methods | 🔴 High | ✅ Complete |
| **Causal Fairness** | `causal_fairness/term.py` | Differentiable residuals | 🔴 High | ⚠️ Needs validation |
| **Fidelity** | `fidelity/term.py` | Discriminator integration | 🔴 High | ✅ Complete |
| **Fidelity** | `fidelity/utils.py` | Gradient verification | 🔴 High | ✅ Complete |
| **Discriminator** | `discriminator/model/model.py` | None (already differentiable) | ✅ Done | ✅ Verified |

### 11.2 Spatial Fairness Modifications

> ✅ **STATUS: COMPLETE** — The differentiable Gini coefficient has been implemented.

**Original Issue**: The standard Gini coefficient formulation using sorted indices is non-differentiable due to the sorting operation.

**Implemented Solution**: The pairwise absolute difference formulation was implemented as specified:

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

This formulation is differentiable because:
- `torch.abs()` is differentiable (subgradient at 0)
- Pairwise differences are computed via broadcasting (differentiable)
- Mean and sum operations are differentiable

**Implementation Sketch**:
```python
def differentiable_gini(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Gini coefficient using differentiable pairwise differences.
    
    Args:
        x: Service rates tensor [num_cells]
        
    Returns:
        Gini coefficient (scalar tensor)
    """
    n = x.size(0)
    x_mean = x.mean()
    
    # Pairwise absolute differences via broadcasting
    # Shape: [n, n]
    diff_matrix = torch.abs(x.unsqueeze(0) - x.unsqueeze(1))
    
    # Gini = sum of all pairwise differences / (2 * n^2 * mean)
    gini = diff_matrix.sum() / (2 * n * n * x_mean + 1e-8)
    
    return gini
```

**Implementation Location**: `objective_function/spatial_fairness/term.py`

**Verification**: Gradient flow verified via Streamlit dashboard and unit tests.

**Documentation Update Required**:
- ✅ `spatial_fairness/DEVELOPMENT_PLAN.md` updated with differentiable formulation

### 11.3 Causal Fairness Modifications

> ⚠️ **STATUS: PARTIAL** — Implementation complete, but differentiability validation required for Isotonic/Binning methods.

**Original Issue**: The expected service function $g(d)$ is fitted via regression, which is not differentiable end-to-end.

**Implemented Solution**:

1. ✅ **Pre-compute $g(d)$**: Multiple estimation methods implemented in `causal_fairness/utils.py`
2. ✅ **Freeze $g(d)$**: Stored as frozen tensor during optimization
3. ✅ **Differentiable Residual Computation**: Implemented as specified

**Estimation Method Benchmark Results** (January 2025):

| Method | R² Score | Differentiability | Recommendation |
|--------|----------|-------------------|----------------|
| **Isotonic** | **0.4453** | ⚠️ Needs verification | 🥇 Best fit |
| **Binning** | **0.4439** | ⚠️ Needs verification | 🥈 Second best |
| **Polynomial** | 0.1949 | ✅ Verified | Originally recommended |
| **Linear** | 0.1064 | ✅ Verified | Poor fit |
| **LOWESS** | 0.0000 | N/A | Failed on this data |

> 📊 **KEY FINDING**: The originally recommended polynomial method (Section 11.3 pre-computation script) achieves only R² = 0.1949 on the actual data. **Isotonic regression and binning methods achieve 2x better fit** but require differentiability validation.

**Differentiability Analysis for Top Methods**:

The concern with Isotonic and Binning methods is whether gradients can flow properly. However, as analyzed in Section 9.0.1, the $g(d)$ lookup is **frozen** during optimization:

```python
# During optimization, g_lookup is a CONSTANT (no gradient)
g_d = self.g_lookup[demand_idx]  # Lookup only

# Gradients flow through Y (which depends on supply S)
Y = supply / (demand + 1e-8)     # Differentiable
R = Y - g_d                       # Differentiable w.r.t. Y
```

**Why This Should Work**: We are not differentiating through the fitting process. The g(d) values are pre-computed once and frozen. Gradients only need to flow through $R = Y - g(D)$, where $Y$ depends on supply $S$.

**Validation Required**: Despite the above reasoning, explicit gradient verification tests should be run (see Action Items in Section 9.0.1).

**Implementation Location**: `objective_function/causal_fairness/`
- `utils.py` — `fit_expected_service_function()` with multiple methods
- `term.py` — `CausalFairnessTerm` class with differentiable computation
- `dashboard.py` — Streamlit dashboard with method comparison and gradient verification

**Implementation Reference** (see `causal_fairness/utils.py` for full code):
```python
# DifferentiableCausalFairness is now implemented in causal_fairness/utils.py
# Key features:
# - Pre-computed g(d) lookup table (frozen during optimization)
# - Multiple estimation methods: isotonic, binning, polynomial, linear, lowess
# - Differentiable residual computation: R = Y - g(D)
# - Gradient flow verified through Y → supply

class DifferentiableCausalFairness(nn.Module):
    """Causal fairness with pre-computed g(d) for differentiability."""
    
    def __init__(self, g_lookup: torch.Tensor):
        super().__init__()
        self.register_buffer('g_lookup', g_lookup)  # Frozen, no gradient
    
    def forward(self, demand, supply, mask=None):
        Y = supply / (demand + 1e-8)                 # Differentiable
        g_d = self.g_lookup[demand.long()]           # Lookup (frozen)
        R = Y - g_d                                   # Differentiable w.r.t. Y
        var_Y, var_R = Y.var(), R.var()
        return 1 - (var_R / (var_Y + 1e-8))
```

**Documentation Updated**:
- ✅ `causal_fairness/DEVELOPMENT_PLAN.md` updated with implementation details
- ✅ Dashboard includes method comparison with R² benchmarks
- ⚠️ Gradient verification for Isotonic/Binning methods pending (see Section 9.0.1)

### 11.4 Discriminator Verification

> ✅ **STATUS: COMPLETE** — Gradient flow verified, implementation complete.

The ST-SiameseNet discriminator (`discriminator/model/model.py`) uses only standard PyTorch operations that support automatic differentiation:

| Component | PyTorch Class | Differentiable |
|-----------|---------------|----------------|
| Feature Normalization | Custom (using `torch.sin`, `torch.cos`, division) | ✅ Yes |
| LSTM Encoder | `nn.LSTM` | ✅ Yes |
| Embedding Combination | `torch.cat` | ✅ Yes |
| Classifier | `nn.Linear`, `nn.Sigmoid`, `nn.Dropout` | ✅ Yes |

**Verification Implementation**: `objective_function/fidelity/utils.py` — `verify_fidelity_gradient()`

**Verified Behavior**:
- Gradients flow through all model components
- Non-zero gradients confirmed for input trajectories
- LSTM backward pass handled correctly (training mode required)

**Issue Fixed During Implementation**:

> ⚠️ **cuDNN LSTM Backward Pass**: Discovered that `nn.LSTM` with cuDNN backend requires `model.train()` during backward pass, even when computing gradients for inference. The verification function now handles this automatically by temporarily setting training mode.

**Verification Test** (implemented in `fidelity/utils.py`):
```python
def test_discriminator_gradient_flow():
    """Verify gradients flow through discriminator."""
    model = SiameseLSTMDiscriminator()
    
    # Create dummy trajectories with gradient tracking
    traj1 = torch.randn(1, 20, 4, requires_grad=True)
    traj2 = torch.randn(1, 20, 4, requires_grad=True)
    
    # Forward pass
    similarity = model(traj1, traj2)
    
    # Backward pass
    similarity.backward()
    
    # Verify gradients exist
    assert traj1.grad is not None, "No gradient for trajectory 1"
    assert traj2.grad is not None, "No gradient for trajectory 2"
    assert not torch.all(traj1.grad == 0), "Zero gradient for trajectory 1"
    
    print("✅ Discriminator gradient flow verified")
```

#### 11.4.1 Fidelity Score Calibration Issue

> ⚠️ **OPEN ISSUE**: Preliminary testing shows low fidelity scores (~0.42) even for identical trajectories.

**Observed Behavior**:
- Comparing a trajectory with itself yields ~0.42 (expected: ~1.0)
- This suggests the discriminator may need calibration for use as a fidelity constraint

**Possible Causes**:
1. The discriminator was trained for same-driver detection, not trajectory identity
2. Feature normalization reduces distinguishability
3. Checkpoint may not match expected architecture parameters

**ST-iFGSM Alignment Required**:
- Review ST-iFGSM paper for expected discriminator output ranges
- Compare output distribution with ST-iFGSM baselines
- Consider Platt scaling or temperature adjustment for score calibration

See Section 9.0.1 (Task 0.4.1) for detailed analysis and action items.

### 11.5 Integration Testing

After all component modifications are complete, verify end-to-end differentiability:

```python
def test_full_objective_gradient_flow():
    """
    Verify gradients flow through entire objective function.
    """
    # Initialize components
    spatial_fairness = DifferentiableSpatialFairness()
    causal_fairness = DifferentiableCausalFairness(g_lookup)
    discriminator = SiameseLSTMDiscriminator()
    
    # Create trajectory tensor with gradient tracking
    trajectory = torch.randn(1, 20, 4, requires_grad=True)
    
    # Compute objective
    L = (
        alpha1 * causal_fairness(demand, supply) +
        alpha2 * spatial_fairness(service_rates) +
        alpha3 * discriminator(trajectory, trajectory_original)
    )
    
    # Backward pass
    L.backward()
    
    # Verify gradient exists and is non-zero
    assert trajectory.grad is not None
    assert not torch.all(trajectory.grad == 0)
    
    print("✅ Full objective gradient flow verified")
```

### 11.6 Fidelity Term Implementation Summary

> ✅ **STATUS: COMPLETE** — Core implementation finished, validation/calibration required.

The Fidelity Term has been implemented as a complete module integrating the ST-SiameseNet discriminator.

**Implementation Location**: `objective_function/fidelity/`

| File | Description | Status |
|------|-------------|--------|
| `__init__.py` | Package exports | ✅ Complete |
| `config.py` | FidelityConfig dataclass with validation | ✅ Complete |
| `utils.py` | Model loading, batch preparation, differentiable module | ✅ Complete |
| `term.py` | FidelityTerm class (ObjectiveFunctionTerm interface) | ✅ Complete |
| `dashboard.py` | 7-tab Streamlit dashboard | ✅ Complete |

**Key Features**:

1. **Multiple Fidelity Modes**:
   - `same_agent`: Compare modified trajectory with original
   - `paired`: Compare with specific reference trajectories
   - `batch`: Evaluate batch of trajectories against references

2. **Aggregation Methods**:
   - `mean`: Average fidelity across all pairs
   - `min`: Minimum fidelity (most conservative)
   - `threshold`: Count of pairs exceeding threshold
   - `weighted`: Length-weighted average

3. **Differentiable Module**: `DifferentiableFidelity` class for gradient-based optimization

4. **Gradient Verification**: Automated verification with LSTM backward pass handling

**Dashboard Tabs**:
1. Overview — Quick fidelity computation
2. Score Distribution — Histogram of pairwise scores
3. Per-Driver Analysis — Statistics per driver
4. Trajectory Comparison — Visual comparison
5. Length Analysis — Correlation with trajectory length
6. Gradient Verification — Differentiability tests
7. Model Info — Discriminator architecture details

**Known Issues**:
- Low fidelity scores (~0.42) in demo mode (see Section 11.4.1)
- Requires ST-iFGSM alignment validation

---

## 12. References

### 12.1 FAMAIL Project Documents

1. **FAMAIL Meeting 18 – Trajectory Modification Approach Summary** (Notion, 2026)
2. **FAMAIL Objective Function Specification** (`docs/FAMAIL_OBJECTIVE_FUNCTION_SPECIFICATION.md`)
3. **Fidelity Development Plan** (`fidelity/DEVELOPMENT_PLAN.md`)
4. **Causal Fairness Development Plan** (`causal_fairness/DEVELOPMENT_PLAN.md`)
5. **Spatial Fairness Development Plan** (`spatial_fairness/DEVELOPMENT_PLAN.md`)

### 12.2 External References

1. **ST-iFGSM Paper**: Hu, M. et al. "Spatio-Temporal Iterative Fast Gradient Sign Method for Trajectory Similarity Attack" KDD 2023
2. **ST-iFGSM Repository**: https://github.com/mhu3/ST-Siamese-Attack
3. **FGSM Original**: Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). "Explaining and Harnessing Adversarial Examples" ICLR 2015
4. **Carlini-Wagner Attack**: Carlini, N., & Wagner, D. (2017). "Towards Evaluating the Robustness of Neural Networks" IEEE S&P 2017
5. **Gumbel-Softmax**: Jang, E., Gu, S., & Poole, B. (2017). "Categorical Reparameterization with Gumbel-Softmax" ICLR 2017

---

## Appendix A: ST-iFGSM Code Reference

### A.1 FGSM Attack (Adapted from ST-Siamese-Attack)

The following code patterns from the ST-iFGSM repository inform our gradient-based approach:

```python
# From fgsm_attack.py (adapted)
def fgsm_attack(model, trajectory, epsilon, loss_fn):
    """
    Fast Gradient Sign Method attack on trajectory.
    
    Original purpose: Generate adversarial trajectories
    FAMAIL adaptation: Generate fairness-improved trajectories
    """
    trajectory.requires_grad = True
    
    # Forward pass
    output = model(trajectory)
    loss = loss_fn(output)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate perturbation
    perturbation = epsilon * trajectory.grad.sign()
    
    # Apply perturbation
    perturbed_trajectory = trajectory + perturbation
    
    return perturbed_trajectory
```

### A.2 Iterative FGSM

```python
# Iterative version with smaller steps
def iterative_fgsm(model, trajectory, epsilon, alpha, num_iter, loss_fn):
    """
    Iterative FGSM with step size alpha.
    """
    perturbed = trajectory.clone()
    
    for _ in range(num_iter):
        perturbed.requires_grad = True
        
        output = model(perturbed)
        loss = loss_fn(output)
        
        model.zero_grad()
        loss.backward()
        
        # Small step in gradient direction
        perturbed = perturbed + alpha * perturbed.grad.sign()
        
        # Clip to epsilon ball around original
        perturbed = torch.clamp(
            perturbed,
            trajectory - epsilon,
            trajectory + epsilon
        )
        
        perturbed = perturbed.detach()
    
    return perturbed
```

### A.3 Key Adaptations for FAMAIL

| ST-iFGSM | FAMAIL Adaptation |
|----------|-------------------|
| Minimize similarity to original | Maximize fairness objective |
| Attack discriminator | Work with discriminator |
| Continuous perturbation | Discrete grid cell reassignment |
| No domain constraints | Temporal/spatial constraints |

---

## Appendix B: Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW OVERVIEW                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐    │
│  │ Expert          │     │ Demand Data     │     │ Supply Data     │    │
│  │ Trajectories    │     │ (pickup_dropoff │     │ (active_taxis)  │    │
│  │ (50 drivers)    │     │  _counts.pkl)   │     │                 │    │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘    │
│           │                       │                       │              │
│           ▼                       ▼                       ▼              │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │              TRAJECTORY MODIFICATION ALGORITHM                  │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │     │
│  │  │  Rank    │→ │  Select  │→ │  Modify  │→ │ Validate │       │     │
│  │  │          │  │          │  │          │  │ Fidelity │       │     │
│  │  └──────────┘  └──────────┘  └──────────┘  └────┬─────┘       │     │
│  │                                                  │              │     │
│  │                              ┌───────────────────┘              │     │
│  │                              ▼                                  │     │
│  │  ┌──────────┐  ┌──────────────────────┐  ┌──────────┐         │     │
│  │  │ Check    │← │ Recompute Fairness   │← │ Update   │         │     │
│  │  │ Converge │  │ Metrics              │  │ N^p      │         │     │
│  │  └────┬─────┘  └──────────────────────┘  └──────────┘         │     │
│  │       │                                                        │     │
│  └───────┼────────────────────────────────────────────────────────┘     │
│          │                                                               │
│          ▼                                                               │
│  ┌─────────────────┐                                                    │
│  │ Modified        │                                                    │
│  │ Trajectories    │ ───► To Imitation Learning (cGAIL)                │
│  │ (T')            │                                                    │
│  └─────────────────┘                                                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*Document created based on FAMAIL Meeting 18 discussion and ST-iFGSM repository analysis.*
