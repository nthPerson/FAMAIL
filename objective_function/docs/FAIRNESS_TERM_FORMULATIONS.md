# Fairness Term Formulations for Per-Trajectory Gradient Optimization

**Version**: 1.0.0  
**Date**: 2026-01-15  
**Status**: Specification Document  
**Authors**: FAMAIL Research Team

---

## Table of Contents

1. [Introduction and Design Rationale](#1-introduction-and-design-rationale)
2. [Spatial Fairness Term Formulation](#2-spatial-fairness-term-formulation)
3. [Causal Fairness Term Formulation](#3-causal-fairness-term-formulation)
4. [Soft Cell Assignment for Differentiability](#4-soft-cell-assignment-for-differentiability)
5. [Integration: Per-Trajectory Attribution and Optimization](#5-integration-per-trajectory-attribution-and-optimization)
6. [Implementation Recommendations](#6-implementation-recommendations)
7. [Mathematical Coherence Validation](#7-mathematical-coherence-validation)
8. [Data Requirements](#8-data-requirements)
9. [References](#9-references)

---

## 1. Introduction and Design Rationale

### 1.1 The FAMAIL Optimization Challenge

The FAMAIL project aims to modify taxi trajectories to achieve both **fidelity** (realistic, discriminator-fooling trajectories) and **fairness** (equitable service distribution). The ST-iFGSM algorithm provides a gradient-based framework for trajectory modification, but requires that all objective function terms be differentiable with respect to individual trajectory coordinates.

The core optimization problem is:

$$
\min_{\tau'} \mathcal{L}(\tau') = \alpha_1 (1 - F_{\text{causal}}) + \alpha_2 (1 - F_{\text{spatial}}) + \alpha_3 F_{\text{fidelity}}
$$

where we minimize the complement of fairness terms (to maximize fairness) while maintaining fidelity.

### 1.2 Why New Formulations Are Needed

The original fairness metrics were designed for **evaluation** (measuring system-wide fairness) not **optimization** (guiding individual trajectory modifications). Key challenges include:

| Challenge | Description |
|-----------|-------------|
| **Discrete Cell Assignment** | Trajectories are assigned to grid cells discretely, breaking gradient flow |
| **Global Statistics** | Gini coefficient and R² are global metrics over all cells |
| **Attribution Gap** | How does modifying one trajectory affect global fairness? |

### 1.3 Selected Approaches

Based on the analysis in `PER_TRAJECTORY_GRADIENT_FORMULATION_ANALYSIS.md`, we selected:

| Term | Gradient Computation | Attribution/Ranking |
|------|---------------------|---------------------|
| **Spatial Fairness** | Option A: Direct Differentiable (full Gini) | Option D: Local Inequality Score (LIS) |
| **Causal Fairness** | Option A: Direct Differentiable (full R²) | Option C: Demand-Conditional Deviation (DCD) |
| **Discrete Problem** | Soft Cell Assignment (Gaussian softmax) | N/A |

### 1.4 Design Principles

Our formulation choices are guided by:

1. **Fidelity to Original Metrics**: Optimization should minimize actual Gini/R², not proxies
2. **Meaningful Gradients**: Gradients should indicate actionable directions for improvement
3. **Computational Tractability**: Must scale to thousands of trajectories
4. **Interpretability**: Per-trajectory scores should be understandable for analysis

### 1.5 Document Organization

This document is divided into focused sections to avoid length limits:

- **Sections 2-3**: Detailed formulations of each fairness term
- **Section 4**: Soft cell assignment mechanism
- **Section 5**: Integration into optimization loop
- **Section 6**: Implementation guidance
- **Section 7**: Mathematical validation

---

## 2. Spatial Fairness Term Formulation

### 2.1 Background: The Original Metric

The spatial fairness term measures **equality of taxi service distribution** using the Gini coefficient applied to service rates:

$$
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)
$$

where:
- $P$ = set of time periods
- $G_a^p$ = Gini coefficient of **Arrival Service Rates** (dropoffs) in period $p$
- $G_d^p$ = Gini coefficient of **Departure Service Rates** (pickups) in period $p$

**Service Rate Definitions**:

$$
\text{DSR}_c^p = \frac{O_c^p}{N_c^p \cdot T^p} \quad \text{(Departure Service Rate)}
$$

$$
\text{ASR}_c^p = \frac{D_c^p}{N_c^p \cdot T^p} \quad \text{(Arrival Service Rate)}
$$

where:
- $O_c^p$ = pickup (origin) count in cell $c$, period $p$
- $D_c^p$ = dropoff (destination) count in cell $c$, period $p$
- $N_c^p$ = active taxi count in cell $c$, period $p$
- $T^p$ = duration of period $p$ in days

### 2.2 The Differentiable Gini Formulation

The Gini coefficient has a **pairwise differentiable formulation**:

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

This is differentiable everywhere except where $x_i = x_j$ (measure-zero set).

**Gradient of Gini with respect to a single value $x_k$**:

$$
\frac{\partial G}{\partial x_k} = \frac{1}{2n^2 \bar{x}} \left[ 2\sum_{j \neq k} \text{sign}(x_k - x_j) - \frac{G \cdot n}{\bar{x}} \right]
$$

**Implementation Note**: The sign function has zero gradient at ties. In practice, we use a smooth approximation:

$$
\text{sign}_\epsilon(x) = \frac{x}{\sqrt{x^2 + \epsilon^2}}
$$

### 2.3 Chain Rule: From Trajectory to Spatial Fairness

For a trajectory $\tau$ with pickup at coordinates $(x_\tau, y_\tau)$, the gradient chain is:

```
∂F_spatial     ∂F_spatial   ∂G      ∂DSR_c    ∂O_c      ∂σ_c
─────────── = ────────── · ──── · ──────── · ───── · ─────────
∂(x_τ, y_τ)      ∂G        ∂DSR_c  ∂O_c      ∂σ_c   ∂(x_τ, y_τ)
```

Each term:

| Component | Formula | Description |
|-----------|---------|-------------|
| $\frac{\partial F_{\text{spatial}}}{\partial G}$ | $-\frac{1}{2\|P\|}$ | Fixed weight |
| $\frac{\partial G}{\partial \text{DSR}_c}$ | See §2.2 | Gini gradient per cell |
| $\frac{\partial \text{DSR}_c}{\partial O_c}$ | $\frac{1}{N_c \cdot T}$ | Service rate scaling |
| $\frac{\partial O_c}{\partial \sigma_c}$ | $1$ | Soft count aggregation |
| $\frac{\partial \sigma_c}{\partial (x_\tau, y_\tau)}$ | See §4 | Soft assignment gradient |

### 2.4 Local Inequality Score (LIS) for Attribution

While the full Gini is used for gradient computation, the **Local Inequality Score (LIS)** provides interpretable per-trajectory attribution:

$$
\text{LIS}_\tau^{\text{pickup}} = \left| \frac{O_c}{N_c \cdot T} - \overline{\text{DSR}} \right| = |\text{DSR}_c - \overline{\text{DSR}}|
$$

**Interpretation**:
- LIS measures how far the trajectory's pickup cell deviates from the mean service rate
- High LIS → trajectory contributes more to inequality
- Trajectories from over-served cells should move to under-served cells

**Combined LIS for a trajectory**:

$$
\text{LIS}_\tau = \frac{\text{LIS}_\tau^{\text{pickup}} + \text{LIS}_\tau^{\text{dropoff}}}{2\overline{\text{DSR}}}
$$

**Relationship to Global Metric**:

The sum of LIS values relates to Mean Absolute Deviation (MAD):

$$
\text{MAD}(\text{DSR}) = \frac{1}{n} \sum_c |\text{DSR}_c - \overline{\text{DSR}}|
$$

The Gini coefficient satisfies: $G \approx \frac{\text{MAD}}{\bar{x}}$ (exact for uniform distributions).

### 2.5 Pros and Cons

**Advantages of this approach**:

| Pro | Rationale |
|-----|-----------|
| ✅ **Exact Metric** | Optimizes actual Gini, not a proxy |
| ✅ **End-to-End Differentiable** | Automatic differentiation handles the chain rule |
| ✅ **Interpretable Attribution** | LIS explains why each trajectory matters |
| ✅ **Existing Implementation** | `DifferentiableSpatialFairness` class ready |

**Disadvantages and mitigations**:

| Con | Mitigation |
|-----|------------|
| ⚠️ **Computational Cost** | Batch processing, sparse updates |
| ⚠️ **Temperature Tuning** | Annealing schedule (see §4.4) |
| ⚠️ **Dense Gradients** | Focus on high-LIS trajectories |

### 2.6 Mathematical Formulation Summary

**For gradient-based optimization**:

$$
\boxed{
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P} \left( G(\text{ASR}^p) + G(\text{DSR}^p) \right)
}
$$

where the Gini uses soft counts:

$$
O_c^p = \sum_{\tau \in \mathcal{T}} \sigma_c^p\left(\mathbf{x}_\tau^{\text{pickup}}\right)
$$

**For trajectory ranking/selection**:

$$
\boxed{
\text{LIS}_\tau = \frac{1}{2\overline{\text{DSR}}} \left( |\text{DSR}_{c_\text{pickup}} - \overline{\text{DSR}}| + |\text{ASR}_{c_\text{dropoff}} - \overline{\text{ASR}}| \right)
}
$$

---

## 3. Causal Fairness Term Formulation

### 3.1 Background: The Original Metric

The causal fairness term measures whether taxi service is **explained by passenger demand** rather than other (potentially discriminatory) factors:

$$
F_{\text{causal}}^p = R^2 = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)}
$$

where:
- $Y_{c,p} = \frac{S_{c,p}}{D_{c,p}}$ = service ratio (supply/demand) in cell $c$, period $p$
- $g(d)$ = expected service ratio given demand $d$ (pre-fitted function)
- $R_{c,p} = Y_{c,p} - g(D_{c,p})$ = residual (unexplained variation)

**Overall Causal Fairness**:

$$
F_{\text{causal}} = \frac{1}{|P|} \sum_{p \in P} F_{\text{causal}}^p
$$

### 3.2 The R² Formulation

The coefficient of determination $R^2$ measures variance explained:

$$
R^2 = 1 - \frac{\sum_c (Y_c - g(D_c))^2}{\sum_c (Y_c - \bar{Y})^2}
$$

**Gradient with respect to service ratio $Y_k$**:

$$
\frac{\partial R^2}{\partial Y_k} = \frac{-2}{\text{Var}(Y)} \left[ (Y_k - g(D_k)) - R^2 (Y_k - \bar{Y}) \right]
$$

### 3.3 Chain Rule: From Trajectory to Causal Fairness

Trajectory modifications affect **demand** $D$ (pickup counts). The gradient path:

```
∂F_causal     ∂F_causal   ∂Y_c      ∂D_c      ∂σ_c
─────────── = ────────── · ───── · ────── · ─────────
∂(x_τ, y_τ)      ∂Y_c      ∂D_c    ∂σ_c   ∂(x_τ, y_τ)
```

**Key insight**: $Y_c = S_c / D_c$, so:

$$
\frac{\partial Y_c}{\partial D_c} = -\frac{S_c}{D_c^2}
$$

This means **increasing demand decreases the service ratio** (more riders per taxi).

### 3.4 The Critical g(d) Freeze

**CRITICAL**: The function $g(d)$ must be **frozen** during optimization.

**Why?**
1. $g(d)$ represents the "expected" relationship between demand and service
2. If we re-fit $g(d)$ after each modification, we would be "moving the goalposts"
3. The goal is to make the current data better fit the original expectation, not to change the expectation

**Implementation**: Pre-compute $g(d)$ from original (unmodified) data, store as a lookup table or frozen function.

### 3.5 Demand-Conditional Deviation (DCD) for Attribution

For trajectory ranking, we use the **Demand-Conditional Deviation (DCD)**:

$$
\text{DCD}_{c,p} = Y_{c,p} - g(D_{c,p}) = R_{c,p}
$$

This is simply the residual — how much the cell's service deviates from what demand predicts.

**Per-Trajectory DCD Score**:

$$
\text{DCD}_\tau = |R_{c_\text{pickup}, p_\text{pickup}}|
$$

**Interpretation**:
- High $|R_c|$ → cell has unexplained variation (unfair)
- Positive $R_c$ → over-served relative to demand
- Negative $R_c$ → under-served relative to demand

### 3.6 How Trajectory Modification Affects Causal Fairness

When we move a pickup from cell $a$ to cell $b$:

| Effect | Cell A (origin) | Cell B (destination) |
|--------|-----------------|----------------------|
| Demand change | $D_a \to D_a - 1$ | $D_b \to D_b + 1$ |
| Service ratio | $Y_a \uparrow$ (fewer riders per taxi) | $Y_b \downarrow$ (more riders per taxi) |
| Residual | $R_a$ changes based on $g(D_a - 1)$ | $R_b$ changes based on $g(D_b + 1)$ |

**Optimal modification direction**:
- Move pickups **from** cells where $R_c > 0$ (over-served)
- Move pickups **to** cells where $R_c < 0$ (under-served)

### 3.7 Pros and Cons

**Advantages**:

| Pro | Rationale |
|-----|-----------|
| ✅ **Exact R² Metric** | Optimizes actual causal fairness |
| ✅ **Differentiable** | Variance computation is smooth |
| ✅ **Frozen g(d)** | Clear optimization target |
| ✅ **DCD Interpretability** | Direct residual interpretation |

**Disadvantages**:

| Con | Mitigation |
|-----|------------|
| ⚠️ **Ignores Supply Side** | Supply treated as fixed for pickup modification |
| ⚠️ **Demand-Only Effect** | Full trajectory modification would also affect S |
| ⚠️ **g(d) Quality** | Pre-fitting must be robust (use binning or isotonic) |

### 3.8 Mathematical Formulation Summary

**For gradient-based optimization**:

$$
\boxed{
F_{\text{causal}} = \frac{1}{|P|} \sum_{p \in P} \left( 1 - \frac{\text{Var}(Y^p - g(D^p))}{\text{Var}(Y^p)} \right)
}
$$

where demand uses soft counts:

$$
D_c^p = D_c^{p,\text{base}} + \sum_{\tau \in \mathcal{T}_{\text{modifying}}} \sigma_c^p\left(\mathbf{x}_\tau^{\text{pickup}}\right)
$$

**For trajectory ranking/selection**:

$$
\boxed{
\text{DCD}_\tau = |Y_{c_\text{pickup}} - g(D_{c_\text{pickup}})| = |R_{c_\text{pickup}}|
}
$$

---

## 4. Soft Cell Assignment for Differentiability

### 4.1 The Discrete-to-Continuous Problem

Trajectories are assigned to discrete grid cells. For gradient-based optimization:

$$
O_c = \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{pickup}(\tau) = c]
$$

The indicator function $\mathbf{1}[\cdot]$ has **zero gradient everywhere** (except at discontinuities where it's undefined). This breaks the gradient chain.

**Solution**: Replace hard cell assignment with a **soft probability distribution** over cells.

### 4.2 Gaussian Soft Assignment Formulation

For a trajectory with continuous pickup coordinates $(\tilde{x}, \tilde{y})$, define the soft assignment to cell $c = (i, j)$:

$$
\boxed{
\sigma_c(\tilde{x}, \tilde{y}) = \frac{\exp\left(-\frac{(i - \tilde{x})^2 + (j - \tilde{y})^2}{2\tau^2}\right)}{\sum_{c' \in \mathcal{N}} \exp\left(-\frac{(i' - \tilde{x})^2 + (j' - \tilde{y})^2}{2\tau^2}\right)}
}
$$

where:
- $\tau$ = temperature parameter controlling distribution sharpness
- $\mathcal{N}$ = neighborhood of valid cells (constraint region)

**Temperature Behavior**:

| $\tau$ Value | Behavior | Use Case |
|--------------|----------|----------|
| $\tau \to 0$ | Hard assignment (one-hot) | Final discrete output |
| $\tau \approx 0.5$ | Smooth over ~4-9 cells | Good gradient flow |
| $\tau \to \infty$ | Uniform distribution | Exploration |

### 4.3 Gradient of Soft Assignment

The gradient of $\sigma_c$ with respect to coordinates enables end-to-end backpropagation:

$$
\frac{\partial \sigma_c}{\partial \tilde{x}} = \sigma_c \left[ \frac{\tilde{x} - i}{\tau^2} - \sum_{c'} \sigma_{c'} \frac{\tilde{x} - i'}{\tau^2} \right]
$$

**Intuition**: The gradient points toward cells with higher assignment probability, scaled by the temperature.

### 4.4 Neighborhood-Constrained Soft Assignment

In FAMAIL, modifications are bounded by a neighborhood constraint (L∞ ball):

$$
\sigma_c(\tilde{x}, \tilde{y}) = \begin{cases}
\frac{\exp(-d_c^2 / 2\tau^2)}{Z} & \text{if } c \in \mathcal{N}(c_0, k) \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $\mathcal{N}(c_0, k) = \{c : \|c - c_0\|_\infty \leq k\}$ is the $k$-hop neighborhood
- $c_0$ is the original cell
- $Z = \sum_{c' \in \mathcal{N}} \exp(-d_{c'}^2 / 2\tau^2)$ is the normalization constant

**Example**: For $k = 2$, a 5×5 neighborhood (25 cells) is considered.

### 4.5 Soft Count Aggregation

With soft assignments, pickup counts become differentiable:

$$
O_c^{\text{soft}} = O_c^{\text{fixed}} + \sum_{\tau \in \mathcal{T}_{\text{modify}}} \sigma_c(\mathbf{x}_\tau)
$$

where:
- $O_c^{\text{fixed}}$ = counts from trajectories not being modified
- $\mathcal{T}_{\text{modify}}$ = set of trajectories being optimized

### 4.6 Temperature Annealing Schedule

During optimization, anneal from soft to hard:

$$
\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}
$$

**Recommended values**:
- $\tau_{\max} = 1.0$ (soft, good gradients at start)
- $\tau_{\min} = 0.1$ (near-discrete at end)
- $T$ = total optimization steps

### 4.7 Why Soft Assignment Works

**Mathematical Justification**:

1. **Continuity**: As $\tau \to 0$, $\sigma_c \to \mathbf{1}[c = \text{argmin}_c d_c]$
2. **Gradient Information**: Non-zero gradients guide optimization toward discrete solution
3. **Convex Relaxation**: Soft assignment is a convex relaxation of the discrete problem

**Practical Validation**:
- Gradients verified numerically (see `verify_gradients()` in existing code)
- Final hard assignment achieved via temperature annealing or rounding

---

## 5. Integration: Per-Trajectory Attribution and Optimization

### 5.1 Two-Phase Approach

The optimization uses a **two-phase approach**:

| Phase | Purpose | Method |
|-------|---------|--------|
| **Phase 1: Attribution** | Identify high-impact trajectories | LIS + DCD scoring |
| **Phase 2: Optimization** | Modify selected trajectories | Gradient descent with soft assignment |

### 5.2 Per-Trajectory Attribution Pseudocode

```python
def compute_trajectory_attribution_scores(
    trajectories: Dict[str, Trajectory],
    pickup_counts: np.ndarray,       # Shape: [n_cells, n_periods]
    dropoff_counts: np.ndarray,      # Shape: [n_cells, n_periods]
    active_taxis: np.ndarray,        # Shape: [n_cells, n_periods]
    g_function: Callable,            # Frozen g(d)
    period_duration: float,          # Days per period
) -> Dict[str, Dict[str, float]]:
    """
    Compute LIS and DCD scores for all trajectories.
    
    Returns:
        Dict mapping trajectory_id to {
            'lis_pickup': float,      # Local Inequality Score (pickup)
            'lis_dropoff': float,     # Local Inequality Score (dropoff)
            'lis_combined': float,    # Combined LIS
            'dcd': float,             # Demand-Conditional Deviation
            'combined_score': float,  # Weighted combination
        }
    """
    scores = {}
    
    # Compute global statistics
    dsr = pickup_counts / (active_taxis * period_duration + EPS)  # [n_cells, n_periods]
    asr = dropoff_counts / (active_taxis * period_duration + EPS)
    
    mean_dsr = dsr.mean(axis=0)  # Per-period mean [n_periods]
    mean_asr = asr.mean(axis=0)
    
    # Compute service ratios and residuals
    demand = pickup_counts
    supply = active_taxis
    Y = supply / (demand + EPS)  # Service ratio
    expected_Y = g_function(demand)  # g(D)
    residuals = Y - expected_Y  # R = Y - g(D)
    
    for traj_id, traj in trajectories.items():
        c_pickup, p_pickup = get_cell_period(traj, 'pickup')
        c_dropoff, p_dropoff = get_cell_period(traj, 'dropoff')
        
        # Local Inequality Score
        lis_pickup = abs(dsr[c_pickup, p_pickup] - mean_dsr[p_pickup])
        lis_dropoff = abs(asr[c_dropoff, p_dropoff] - mean_asr[p_dropoff])
        lis_combined = (lis_pickup + lis_dropoff) / (2 * mean_dsr[p_pickup] + EPS)
        
        # Demand-Conditional Deviation
        dcd = abs(residuals[c_pickup, p_pickup])
        
        # Combined score (weights can be tuned)
        combined = ALPHA_SPATIAL * lis_combined + ALPHA_CAUSAL * dcd
        
        scores[traj_id] = {
            'lis_pickup': lis_pickup,
            'lis_dropoff': lis_dropoff,
            'lis_combined': lis_combined,
            'dcd': dcd,
            'combined_score': combined,
        }
    
    return scores
```

### 5.3 Optimization Loop Pseudocode (ST-iFGSM Style)

```python
def optimize_trajectory_fairness(
    tau: Trajectory,
    objective_fn: Callable,           # Computes combined fairness loss
    epsilon: float = 2.0,             # Neighborhood constraint (cells)
    alpha: float = 0.5,               # Step size
    num_iterations: int = 10,
    tau_max: float = 1.0,             # Initial temperature
    tau_min: float = 0.1,             # Final temperature
) -> Tuple[int, int]:
    """
    Optimize a single trajectory's pickup location for fairness.
    
    Args:
        tau: Trajectory to modify
        objective_fn: Function computing total loss (lower = better fairness)
        epsilon: L∞ bound on modification
        alpha: Gradient step size
        num_iterations: Optimization steps
        tau_max, tau_min: Temperature annealing bounds
        
    Returns:
        (new_cell_x, new_cell_y): Modified pickup cell
    """
    import torch
    
    # Initialize continuous location at original cell center
    original_cell = torch.tensor(tau.pickup_cell, dtype=torch.float32)
    location = original_cell.clone().requires_grad_(True)
    
    for iteration in range(num_iterations):
        # Temperature annealing
        progress = iteration / max(num_iterations - 1, 1)
        temperature = tau_max * (tau_min / tau_max) ** progress
        
        # Compute soft cell assignment
        soft_probs = soft_cell_assignment(
            location=location.unsqueeze(0),
            original_cell=original_cell.unsqueeze(0),
            neighborhood_size=int(2 * epsilon + 1),
            temperature=temperature,
        )  # Shape: [1, neighborhood_size, neighborhood_size]
        
        # Compute fairness loss (negated because higher fairness = better)
        loss = objective_fn(soft_probs, tau)
        
        # Backward pass
        loss.backward()
        
        # FGSM-style update (gradient sign)
        with torch.no_grad():
            grad_sign = location.grad.sign()
            location = location + alpha * grad_sign
            
            # Project to L∞ ball (neighborhood constraint)
            location = torch.clamp(
                location,
                original_cell - epsilon,
                original_cell + epsilon,
            )
            
            # Also clamp to grid bounds
            location = torch.clamp(
                location,
                torch.tensor([0.0, 0.0]),
                torch.tensor([GRID_DIMS[0] - 1, GRID_DIMS[1] - 1]),
            )
        
        # Reset gradient for next iteration
        location = location.detach().requires_grad_(True)
    
    # Discretize final location
    final_cell = location.round().long()
    return tuple(final_cell.tolist())
```

### 5.4 Combined Objective Function

```python
def compute_combined_fairness_loss(
    soft_probs: torch.Tensor,           # Soft assignment probabilities
    trajectory: Trajectory,
    base_pickup_counts: torch.Tensor,   # Counts without this trajectory
    base_dropoff_counts: torch.Tensor,
    active_taxis: torch.Tensor,
    g_function: Callable,
    alpha_spatial: float = 0.5,
    alpha_causal: float = 0.5,
) -> torch.Tensor:
    """
    Compute combined fairness loss for optimization.
    
    Loss = α_spatial * (1 - F_spatial) + α_causal * (1 - F_causal)
    
    Returns:
        Scalar loss tensor (lower = better fairness)
    """
    # Update soft pickup counts with this trajectory's contribution
    soft_pickup_counts = update_counts_with_soft_assignment(
        base_pickup_counts,
        soft_probs,
        trajectory.original_pickup_cell,
    )
    
    # Compute spatial fairness
    spatial_module = DifferentiableSpatialFairness(grid_dims=GRID_DIMS)
    dsr = soft_pickup_counts / (active_taxis * PERIOD_DURATION + EPS)
    asr = base_dropoff_counts / (active_taxis * PERIOD_DURATION + EPS)
    f_spatial = spatial_module.compute(dsr, asr)
    
    # Compute causal fairness
    causal_module = DifferentiableCausalFairness(
        frozen_demands=base_pickup_counts.detach().numpy(),
        g_function=g_function,
    )
    f_causal = causal_module.compute(
        supply=active_taxis,
        demand=soft_pickup_counts,
    )
    
    # Combined loss (minimize to maximize fairness)
    loss = alpha_spatial * (1.0 - f_spatial) + alpha_causal * (1.0 - f_causal)
    
    return loss
```

### 5.5 Batch Optimization Strategy

For efficiency, optimize trajectories in batches:

```python
def optimize_trajectory_batch(
    trajectories: List[Trajectory],
    batch_size: int = 32,
    **kwargs,
) -> Dict[str, Tuple[int, int]]:
    """
    Optimize multiple trajectories with periodic count updates.
    """
    # Sort by combined attribution score (highest first)
    sorted_trajs = sorted(
        trajectories,
        key=lambda t: t.attribution_score,
        reverse=True,
    )
    
    results = {}
    current_counts = base_counts.copy()
    
    for batch_start in range(0, len(sorted_trajs), batch_size):
        batch = sorted_trajs[batch_start:batch_start + batch_size]
        
        # Optimize batch in parallel (no interference within batch)
        for traj in batch:
            new_cell = optimize_trajectory_fairness(
                traj,
                base_pickup_counts=current_counts,
                **kwargs,
            )
            results[traj.id] = new_cell
        
        # Update counts after batch completes
        current_counts = update_counts_from_modifications(
            current_counts, batch, results
        )
    
    return results
```

---

## 6. Implementation Recommendations

### 6.1 Module Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                   FAIRNESS-AWARE TRAJECTORY OPTIMIZER                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    PHASE 1: ATTRIBUTION                             │  │
│  │                                                                     │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │  │
│  │  │ LIS Scorer   │   │ DCD Scorer   │   │ Combined Ranking     │    │  │
│  │  │ (Spatial)    │   │ (Causal)     │   │ & Selection          │    │  │
│  │  └──────────────┘   └──────────────┘   └──────────────────────┘    │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                │                                          │
│                                ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    PHASE 2: OPTIMIZATION                            │  │
│  │                                                                     │  │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │  │
│  │  │ Soft Cell    │──►│ Differentiable│──►│ Gradient-Based      │    │  │
│  │  │ Assignment   │   │ Fairness Terms│   │ Update (FGSM)       │    │  │
│  │  └──────────────┘   └──────────────┘   └──────────────────────┘    │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                │                                          │
│                                ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    PHASE 3: DISCRETIZATION                          │  │
│  │                                                                     │  │
│  │  Temperature Annealing → Hard Assignment → Validity Check          │  │
│  │                                                                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Key Implementation Details

#### 6.2.1 Soft Cell Assignment Module

```python
class SoftCellAssignment(torch.nn.Module):
    """Differentiable cell assignment for trajectory optimization."""
    
    def __init__(
        self,
        grid_dims: Tuple[int, int],
        neighborhood_size: int = 5,
        initial_temperature: float = 1.0,
    ):
        super().__init__()
        self.grid_dims = grid_dims
        self.k = (neighborhood_size - 1) // 2
        self.temperature = initial_temperature
        
        # Pre-compute neighborhood offsets
        offsets = torch.arange(-self.k, self.k + 1)
        self.register_buffer(
            'offset_grid',
            torch.stack(torch.meshgrid(offsets, offsets, indexing='ij'), dim=-1)
        )  # [neighborhood_size, neighborhood_size, 2]
    
    def forward(
        self,
        location: torch.Tensor,      # [batch, 2] continuous coordinates
        original_cell: torch.Tensor, # [batch, 2] original cell indices
    ) -> torch.Tensor:
        """
        Compute soft assignment over neighborhood.
        
        Returns:
            [batch, neighborhood_size, neighborhood_size] probabilities
        """
        batch_size = location.shape[0]
        ns = 2 * self.k + 1
        
        # Cell centers in neighborhood
        # [batch, ns, ns, 2]
        cell_centers = original_cell[:, None, None, :] + self.offset_grid
        
        # Squared distances from current location to cell centers
        # [batch, ns, ns]
        sq_dist = ((location[:, None, None, :] - cell_centers) ** 2).sum(dim=-1)
        
        # Soft assignment via softmax
        logits = -sq_dist / (2 * self.temperature ** 2)
        probs = torch.softmax(logits.view(batch_size, -1), dim=-1)
        
        return probs.view(batch_size, ns, ns)
    
    def set_temperature(self, temperature: float):
        """Update temperature for annealing."""
        self.temperature = temperature
```

#### 6.2.2 Gradient Verification

Always verify gradients numerically:

```python
def verify_fairness_gradients(
    objective_fn: Callable,
    test_location: torch.Tensor,
    eps: float = 1e-4,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Verify analytical gradients match numerical gradients.
    """
    test_location.requires_grad_(True)
    
    # Analytical gradient
    loss = objective_fn(test_location)
    loss.backward()
    analytical_grad = test_location.grad.clone()
    
    # Numerical gradient (central difference)
    numerical_grad = torch.zeros_like(test_location)
    for i in range(test_location.numel()):
        test_location_flat = test_location.view(-1)
        
        test_location_flat[i] += eps
        loss_plus = objective_fn(test_location_flat.view_as(test_location))
        
        test_location_flat[i] -= 2 * eps
        loss_minus = objective_fn(test_location_flat.view_as(test_location))
        
        numerical_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
        test_location_flat[i] += eps  # Restore
    
    # Compare
    max_rel_error = (
        (analytical_grad - numerical_grad).abs() / 
        (numerical_grad.abs() + 1e-8)
    ).max().item()
    
    return {
        'analytical_grad': analytical_grad,
        'numerical_grad': numerical_grad,
        'max_rel_error': max_rel_error,
        'gradients_match': max_rel_error < rtol,
    }
```

### 6.3 Hyperparameter Recommendations

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| `neighborhood_size` | 5 (for k=2) | Balances flexibility and constraint |
| `epsilon` (L∞ bound) | 2.0 cells | Matches neighborhood_size |
| `alpha` (step size) | 0.3-0.5 | FGSM-style, may need tuning |
| `num_iterations` | 5-10 | More for difficult cases |
| `tau_max` | 1.0 | Soft gradients initially |
| `tau_min` | 0.1 | Near-discrete at end |
| `batch_size` | 32-64 | Memory-compute tradeoff |
| `alpha_spatial` | 0.5 | Equal weighting default |
| `alpha_causal` | 0.5 | Equal weighting default |

### 6.4 Computational Considerations

**Memory**:
- Soft assignment: O(batch_size × neighborhood_size²)
- Gini gradient: O(n_cells²) per period — precompute if possible

**Speed**:
- Attribution (Phase 1): O(n_trajectories) — fully parallelizable
- Optimization (Phase 2): O(n_selected × iterations × n_cells)

**GPU Acceleration**:
- All operations are PyTorch-native
- Batch processing enables GPU parallelism

---

## 7. Mathematical Coherence Validation

### 7.1 Existing Implementation Verification

The formulations in this document are validated against the existing codebase:

| Component | Implementation | Location | Verified |
|-----------|---------------|----------|----------|
| Pairwise Gini (NumPy) | `compute_gini_pairwise()` | `spatial_fairness/utils.py` | ✅ |
| Pairwise Gini (Torch) | `compute_gini_torch()` | `spatial_fairness/utils.py` | ✅ |
| Causal R² (Torch) | `compute_causal_fairness_torch()` | `causal_fairness/utils.py` | ✅ |
| DifferentiableSpatialFairness | Class | `spatial_fairness/utils.py` | ✅ |
| DifferentiableCausalFairness | Class | `causal_fairness/utils.py` | ✅ |

The existing `verify_gini_gradient()` and `verify_causal_fairness_gradient()` functions confirm:
- Gradients flow correctly through both metrics
- No NaN or Inf values in gradients
- PyTorch and NumPy implementations produce consistent results

### 7.2 Gradient Consistency Check

To validate that our formulations work together:

**Test 1: Gradient Directions Are Meaningful**

For a trajectory in an over-served cell (high DSR, positive residual):
- Spatial gradient should point toward under-served cells
- Causal gradient should also point toward cells where adding demand reduces residual variance

**Verification**: Compute gradients for synthetic cases and verify directions match intuition.

**Test 2: Combined Objective Is Well-Behaved**

The combined loss should be:
- Continuous everywhere (soft assignment ensures this)
- Bounded in [0, 2] (since each term is in [0, 1])
- Non-degenerate (not flat) — gradients should be non-zero for unfair states

### 7.2 Metric Alignment Validation

After optimization with soft formulations, verify hard metrics improve:

```python
def validate_metric_alignment(
    original_trajectories: List[Trajectory],
    modified_trajectories: List[Trajectory],
    counts_data: Dict,
) -> Dict[str, Any]:
    """
    Verify that soft optimization improves hard fairness metrics.
    """
    # Compute hard metrics before modification
    original_spatial = compute_hard_spatial_fairness(original_trajectories, counts_data)
    original_causal = compute_hard_causal_fairness(original_trajectories, counts_data)
    
    # Compute hard metrics after modification  
    modified_spatial = compute_hard_spatial_fairness(modified_trajectories, counts_data)
    modified_causal = compute_hard_causal_fairness(modified_trajectories, counts_data)
    
    return {
        'spatial_improved': modified_spatial > original_spatial,
        'causal_improved': modified_causal > original_causal,
        'spatial_delta': modified_spatial - original_spatial,
        'causal_delta': modified_causal - original_causal,
    }
```

### 7.3 Attribution Score Validation

Verify that high-attribution trajectories, when modified, have larger impact:

```python
def validate_attribution_scores(
    trajectories: List[Trajectory],
    attribution_scores: Dict[str, float],
    modification_impacts: Dict[str, float],
) -> float:
    """
    Compute correlation between attribution scores and actual impacts.
    
    High correlation indicates attribution is predictive.
    """
    scores = [attribution_scores[t.id] for t in trajectories]
    impacts = [modification_impacts[t.id] for t in trajectories]
    
    correlation = np.corrcoef(scores, impacts)[0, 1]
    return correlation
```

### 7.4 Convergence Properties

**Expected behavior**:
1. Loss should decrease monotonically (on average)
2. Final discrete assignments should be stable (not oscillating)
3. Temperature annealing should smoothly transition from soft to hard

**Red flags**:
- Gradient explosion (add clipping)
- Oscillation at low temperature (reduce step size)
- Loss plateaus early (increase initial temperature)

---

## 8. Data Requirements

### 8.1 Required Datasets

| Dataset | Source | Usage |
|---------|--------|-------|
| `pickup_dropoff_counts.pkl` | Pre-processed | Baseline pickup/dropoff counts per cell/period |
| `all_trajs.pkl` | Pre-processed | Trajectory coordinates and timestamps |
| `active_taxis_*.pkl` | Pre-processed | Active taxi counts per cell/period |

### 8.2 Pre-Computed Artifacts

| Artifact | Computation | Storage |
|----------|-------------|---------|
| g(d) function | Isotonic regression or binning on original data | Pickle (function + lookup table) |
| Mean service rates | Average DSR/ASR per period | NumPy array |
| Pre-computed residuals | R = Y - g(D) per cell/period | NumPy array |
| LIS scores | Per-trajectory attribution | Dict[traj_id, float] |
| DCD scores | Per-trajectory attribution | Dict[traj_id, float] |

### 8.3 Data Alignment Requirements

- Grid dimensions must match across all datasets (e.g., 48×90)
- Period definitions must be consistent (hourly, daily, etc.)
- Temporal coverage must overlap between trajectory and count data

---

## 9. References

### 9.1 FAMAIL Project Documents

1. `PER_TRAJECTORY_GRADIENT_FORMULATION_ANALYSIS.md` — Original analysis and option comparison
2. `ST-iFGSM_ALGORITHM_REFERENCE.md` — Gradient-based trajectory modification algorithm
3. `FAMAIL_OBJECTIVE_FUNCTION_SPECIFICATION.md` — Complete objective function specification
4. `TRAJECTORY_MODIFICATION_ALGORITHM_DEVELOPMENT_PLAN.md` — Development roadmap

### 9.2 Academic References

1. **Gini Coefficient Differentiability**:
   - The pairwise formulation $G = \frac{\sum_{i,j}|x_i - x_j|}{2n^2\bar{x}}$ is differentiable with smooth sign approximation

2. **Soft Attention / Soft Assignment**:
   - Bahdanau et al. (2014) — Attention mechanisms
   - Jang et al. (2017) — "Categorical Reparameterization with Gumbel-Softmax" (ICLR)

3. **R² and Variance Gradients**:
   - Standard calculus; variance is a differentiable function of inputs

4. **Counterfactual Fairness**:
   - Kusner et al. (2017) — "Counterfactual Fairness" (NeurIPS)
   - The g(d) function captures the "fair" expectation given demand

### 9.3 Implementation References

- PyTorch Autograd: https://pytorch.org/docs/stable/autograd.html
- Numerical Gradient Verification: Standard practice in deep learning
- FGSM: Goodfellow et al. (2015) — "Explaining and Harnessing Adversarial Examples"

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $\tau$ | Trajectory |
| $c$ | Grid cell index |
| $p$ | Time period |
| $O_c^p$ | Pickup (origin) count in cell c, period p |
| $D_c^p$ | Dropoff (destination) count in cell c, period p |
| $N_c^p$ | Active taxi count in cell c, period p |
| $T^p$ | Duration of period p (days) |
| $\text{DSR}_c^p$ | Departure Service Rate |
| $\text{ASR}_c^p$ | Arrival Service Rate |
| $G$ | Gini coefficient |
| $Y_{c,p}$ | Service ratio = Supply/Demand |
| $g(d)$ | Expected service ratio given demand d |
| $R_{c,p}$ | Residual = $Y - g(D)$ |
| $\sigma_c$ | Soft cell assignment probability |
| $\tau$ (temperature) | Softmax temperature parameter |
| $\epsilon$ | Neighborhood constraint (L∞ bound) |
| $\alpha$ | Gradient step size |

---

## Appendix B: Complete Pseudocode

### B.1 Full Optimization Pipeline

```python
def famail_fairness_optimization_pipeline(
    trajectories: Dict[str, Trajectory],
    pickup_dropoff_counts: np.ndarray,
    active_taxis: np.ndarray,
    g_function: Callable,
    config: OptimizationConfig,
) -> Dict[str, Trajectory]:
    """
    Complete fairness-aware trajectory optimization pipeline.
    """
    # =========================================================
    # PHASE 1: ATTRIBUTION
    # =========================================================
    print("Phase 1: Computing attribution scores...")
    
    attribution_scores = compute_trajectory_attribution_scores(
        trajectories=trajectories,
        pickup_counts=pickup_dropoff_counts[..., 0],
        dropoff_counts=pickup_dropoff_counts[..., 1],
        active_taxis=active_taxis,
        g_function=g_function,
        period_duration=config.period_duration,
    )
    
    # Select top-k trajectories for modification
    sorted_trajs = sorted(
        attribution_scores.items(),
        key=lambda x: x[1]['combined_score'],
        reverse=True,
    )
    selected_ids = [tid for tid, _ in sorted_trajs[:config.num_trajectories_to_modify]]
    
    print(f"Selected {len(selected_ids)} trajectories for modification")
    
    # =========================================================
    # PHASE 2: OPTIMIZATION
    # =========================================================
    print("Phase 2: Optimizing selected trajectories...")
    
    # Initialize modules
    soft_assignment = SoftCellAssignment(
        grid_dims=config.grid_dims,
        neighborhood_size=config.neighborhood_size,
        initial_temperature=config.tau_max,
    )
    
    spatial_module = DifferentiableSpatialFairness(grid_dims=config.grid_dims)
    causal_module = DifferentiableCausalFairness(
        frozen_demands=pickup_dropoff_counts[..., 0],
        g_function=g_function,
    )
    
    # Prepare base counts (excluding selected trajectories)
    base_counts = compute_base_counts_excluding(
        pickup_dropoff_counts,
        [trajectories[tid] for tid in selected_ids],
    )
    
    modifications = {}
    
    for batch_start in range(0, len(selected_ids), config.batch_size):
        batch_ids = selected_ids[batch_start:batch_start + config.batch_size]
        
        for traj_id in batch_ids:
            traj = trajectories[traj_id]
            
            # Optimize this trajectory
            new_pickup_cell = optimize_trajectory_fairness(
                tau=traj,
                objective_fn=lambda probs: compute_combined_fairness_loss(
                    soft_probs=probs,
                    trajectory=traj,
                    base_pickup_counts=base_counts,
                    base_dropoff_counts=pickup_dropoff_counts[..., 1],
                    active_taxis=active_taxis,
                    g_function=g_function,
                    alpha_spatial=config.alpha_spatial,
                    alpha_causal=config.alpha_causal,
                ),
                epsilon=config.epsilon,
                alpha=config.step_size,
                num_iterations=config.num_iterations,
                tau_max=config.tau_max,
                tau_min=config.tau_min,
            )
            
            modifications[traj_id] = new_pickup_cell
        
        # Update base counts after batch
        base_counts = update_counts_from_modifications(
            base_counts,
            [trajectories[tid] for tid in batch_ids],
            modifications,
        )
    
    # =========================================================
    # PHASE 3: APPLY MODIFICATIONS
    # =========================================================
    print("Phase 3: Applying modifications...")
    
    modified_trajectories = {}
    for traj_id, traj in trajectories.items():
        if traj_id in modifications:
            modified_traj = traj.copy()
            modified_traj.pickup_cell = modifications[traj_id]
            modified_trajectories[traj_id] = modified_traj
        else:
            modified_trajectories[traj_id] = traj
    
    # =========================================================
    # VALIDATION
    # =========================================================
    print("Validating results...")
    
    validation = validate_metric_alignment(
        original_trajectories=list(trajectories.values()),
        modified_trajectories=list(modified_trajectories.values()),
        counts_data=pickup_dropoff_counts,
    )
    
    print(f"Spatial fairness improved: {validation['spatial_improved']}")
    print(f"Causal fairness improved: {validation['causal_improved']}")
    print(f"Spatial delta: {validation['spatial_delta']:+.4f}")
    print(f"Causal delta: {validation['causal_delta']:+.4f}")
    
    return modified_trajectories
```

---

*Document created for FAMAIL project. Last updated: January 15, 2026.*
