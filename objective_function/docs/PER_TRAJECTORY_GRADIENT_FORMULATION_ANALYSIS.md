# Per-Trajectory Gradient Formulation Analysis

**Version**: 1.0.0  
**Date**: 2026-01-15  
**Status**: Research Document  
**Authors**: FAMAIL Research Team

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Chain Rule Decomposition Framework](#2-chain-rule-decomposition-framework)
3. [Spatial Fairness Term Reformulations](#3-spatial-fairness-term-reformulations)
4. [Causal Fairness Term Reformulations](#4-causal-fairness-term-reformulations)
5. [Soft Cell Assignment for Differentiability](#5-soft-cell-assignment-for-differentiability)
6. [Implementation Recommendations](#6-implementation-recommendations)
7. [Data Requirements Summary](#7-data-requirements-summary)

---

## 1. Problem Statement

### 1.1 The Core Challenge

The current FAMAIL objective function formulation computes **global fairness metrics** over the entire trajectory set:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

For gradient-based trajectory modification (following the ST-iFGSM approach), we need to compute:

$$
\nabla_{\tau_i} \mathcal{L}
$$

where $\tau_i$ is a **single trajectory** being modified.

**The Problem**: The current formulations of $F_{\text{spatial}}$ and $F_{\text{causal}}$ are defined as global statistics (Gini coefficients, R² values) computed over **all cells in all time periods**. These global metrics don't naturally decompose to individual trajectory contributions, making gradient computation with respect to a single trajectory non-trivial.

### 1.2 What We Need

To use gradient-based optimization as in ST-iFGSM, we need formulations where:

1. **The objective function is differentiable** with respect to trajectory coordinates
2. **Gradients are meaningful** — they indicate how modifying a single trajectory affects global fairness
3. **Computation is tractable** — we can't recompute all fairness metrics from scratch for each gradient step

### 1.3 Key Insight from ST-iFGSM

ST-iFGSM succeeds because the discriminator loss is computed on a **per-trajectory-pair basis**:

$$
\ell(f(\tau, \tau'), y)
$$

This loss depends directly on the perturbed trajectory $\tau'$, making gradient computation straightforward. We need to find analogous formulations for fairness metrics.

---

## 2. Chain Rule Decomposition Framework

### 2.1 The Gradient Path

To compute $\nabla_{\tau} \mathcal{L}$, we apply the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \tau} = \frac{\partial \mathcal{L}}{\partial \text{(global metrics)}} \times \frac{\partial \text{(global metrics)}}{\partial \text{(cell counts)}} \times \frac{\partial \text{(cell counts)}}{\partial \tau}
$$

Let's analyze each component:

### 2.2 Component 1: How Trajectories Affect Cell Counts

A trajectory $\tau$ with pickup at cell $(x_p, y_p)$ and dropoff at cell $(x_d, y_d)$ contributes:

- **+1 to pickup count** at cell $(x_p, y_p)$ in period $p_{\text{pickup}}$
- **+1 to dropoff count** at cell $(x_d, y_d)$ in period $p_{\text{dropoff}}$

Let $O_c^p$ = pickup count and $D_c^p$ = dropoff count at cell $c$ in period $p$.

The trajectory-to-counts relationship is:

$$
O_c^p = \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{pickup}(\tau) = (c, p)]
$$

$$
D_c^p = \sum_{\tau \in \mathcal{T}} \mathbf{1}[\text{dropoff}(\tau) = (c, p)]
$$

**Problem**: The indicator function $\mathbf{1}[\cdot]$ is not differentiable.

**Solution**: Use **soft cell assignment** (see Section 5).

### 2.3 Component 2: How Cell Counts Affect Global Metrics

Once we have differentiable cell counts, we can trace gradients through:

- **Service rates**: $\text{DSR}_c^p = O_c^p / (N_c^p \cdot T)$
- **Gini coefficient**: $G = \frac{\sum_{i,j} |x_i - x_j|}{2n^2 \bar{x}}$ (already differentiable)
- **R² computation**: $R^2 = 1 - \text{Var}(R) / \text{Var}(Y)$ (already differentiable)

### 2.4 The Missing Piece: Trajectory-Level Formulation

The key insight is that **we don't need to reformulate the global metrics themselves**. Instead, we need to:

1. Express cell counts as differentiable functions of trajectory locations
2. Let automatic differentiation handle the chain rule

However, there are also **alternative formulations** that compute per-trajectory contributions more directly, which may offer computational and interpretability advantages.

---

## 3. Spatial Fairness Term Reformulations

### 3.1 Current Formulation Recap

$$
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)
$$

Where the Gini coefficient is:

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

Applied to service rates:
$$
\text{DSR}_c^p = \frac{O_c^p}{N_c^p \cdot T^p}, \quad \text{ASR}_c^p = \frac{D_c^p}{N_c^p \cdot T^p}
$$

### 3.2 Option A: Direct Differentiable Formulation (Chain Rule Approach)

**Concept**: Keep the global Gini formulation but make the entire computation graph differentiable from trajectory coordinates to final metric.

**Formulation**:

Let $\mathbf{p}_{\tau}^{\text{pickup}} = (x_\tau^p, y_\tau^p)$ be the soft pickup location of trajectory $\tau$ (a probability distribution over cells).

Then:
$$
O_c^p = \sum_{\tau \in \mathcal{T}} \sigma_c^p(\mathbf{p}_{\tau}^{\text{pickup}})
$$

Where $\sigma_c^p(\cdot)$ is a soft assignment function (e.g., softmax over squared distances).

**Gradient Computation**:
$$
\frac{\partial F_{\text{spatial}}}{\partial (x_\tau, y_\tau)} = \frac{\partial F_{\text{spatial}}}{\partial \mathbf{DSR}} \cdot \frac{\partial \mathbf{DSR}}{\partial O} \cdot \frac{\partial O}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial (x_\tau, y_\tau)}
$$

**Pros**:
- Exact correspondence to current metric definition
- Leverages existing differentiable Gini implementation
- End-to-end automatic differentiation

**Cons**:
- Computational cost: Full Gini computation for each gradient step
- Soft assignment requires careful temperature tuning
- All trajectories contribute to gradient (dense computation)

**Data Requirements**:
- `pickup_dropoff_counts.pkl` for baseline counts
- `all_trajs.pkl` for trajectory coordinates
- `active_taxis_*.pkl` for $N_c^p$ normalization

---

### 3.3 Option B: Per-Trajectory Marginal Contribution (Approximate)

**Concept**: Approximate the marginal contribution of each trajectory to Gini without full recomputation.

**Formulation**:

The Gini coefficient can be written as:
$$
G = \frac{1}{2n^2\bar{x}} \sum_{i,j} |x_i - x_j|
$$

When trajectory $\tau$ moves a pickup from cell $a$ to cell $b$, the change in pickup counts is:
- $\Delta O_a = -1$
- $\Delta O_b = +1$

The first-order change in Gini is:
$$
\Delta G \approx \frac{\partial G}{\partial O_a} \cdot (-1) + \frac{\partial G}{\partial O_b} \cdot (+1)
$$

The partial derivative of Gini with respect to count $O_c$ (affecting service rate $x_c$) is:

$$
\frac{\partial G}{\partial O_c} = \frac{\partial G}{\partial x_c} \cdot \frac{\partial x_c}{\partial O_c} = \frac{\partial G}{\partial x_c} \cdot \frac{1}{N_c \cdot T}
$$

Where (from the pairwise Gini formula):
$$
\frac{\partial G}{\partial x_c} = \frac{1}{2n^2\bar{x}} \left[ 2\sum_{j \neq c} \text{sign}(x_c - x_j) - \frac{\sum_{i,j}|x_i - x_j|}{n\bar{x}} \right]
$$

**Per-Trajectory Gradient**:

For trajectory $\tau$ with pickup at cell $c$ in period $p$:
$$
\nabla_{\text{pickup}} G^p = \frac{\partial G^p}{\partial x_c^p} \cdot \frac{1}{N_c^p \cdot T^p}
$$

This tells us: *"If this trajectory's pickup moved away from cell $c$, how much would Gini change?"*

**Pros**:
- Explicit per-trajectory gradient interpretation
- Pre-computable gradients w.r.t. cells (sparse update)
- No soft assignment needed for gradient computation (only for optimization)

**Cons**:
- First-order approximation (may be inaccurate for large changes)
- Requires updating cell gradients after each batch of modifications
- Sign function has zero gradient at ties

**Data Requirements**:
- Pre-computed service rates per cell/period
- Pre-computed Gini partial derivatives per cell

---

### 3.4 Option C: Variance-Based Reformulation

**Concept**: Reformulate spatial fairness using variance instead of Gini, which has simpler gradients.

**Mathematical Relationship**:

The Gini coefficient is related to the variance through:
$$
G = \frac{\sigma}{2\mu} \cdot C
$$

Where $C$ is a constant depending on the distribution shape.

For computational simplicity, we can define:
$$
F_{\text{spatial}}^{*} = 1 - \frac{\text{CV}(\text{DSR})}{2}
$$

Where $\text{CV} = \sigma / \mu$ is the coefficient of variation.

**Gradient Derivation**:

$$
\text{CV}^2 = \frac{\sum_c (x_c - \bar{x})^2 / n}{\bar{x}^2}
$$

$$
\frac{\partial \text{CV}^2}{\partial x_c} = \frac{2(x_c - \bar{x})}{n\bar{x}^2} - \frac{2\text{CV}^2}{n\bar{x}}
$$

**For trajectory $\tau$ affecting cell $c$**:
$$
\frac{\partial F_{\text{spatial}}^{*}}{\partial O_c} = -\frac{1}{2\cdot\text{CV}} \cdot \frac{\partial \text{CV}^2}{\partial x_c} \cdot \frac{1}{N_c \cdot T}
$$

**Pros**:
- Simpler gradients than Gini
- Well-behaved everywhere (no sign function)
- Faster computation

**Cons**:
- Not exactly equivalent to Gini (though highly correlated)
- May require justification for metric change

**Data Requirements**: Same as Option A

---

### 3.5 Option D: Local Inequality Score (Novel Formulation)

**Concept**: Define a per-trajectory "inequality contribution" that sums to global inequality.

**Formulation**:

Define the **Local Inequality Score (LIS)** for trajectory $\tau$ with pickup in cell $c$:

$$
\text{LIS}_{\tau}^{\text{pickup}} = \left| \frac{O_c}{N_c \cdot T} - \bar{\text{DSR}} \right|
$$

This measures how much the trajectory's pickup cell deviates from the mean service rate.

**Global Connection**:
$$
\text{MAD}(\text{DSR}) = \frac{1}{n} \sum_c \left| \text{DSR}_c - \bar{\text{DSR}} \right|
$$

The Mean Absolute Deviation is related to Gini: $G \approx \text{MAD} / \bar{x}$ (under certain conditions).

**Per-Trajectory Spatial Fairness**:
$$
F_{\text{spatial}}^{\tau} = 1 - \frac{\text{LIS}_{\tau}^{\text{pickup}} + \text{LIS}_{\tau}^{\text{dropoff}}}{2\bar{\text{DSR}}}
$$

**Optimization Interpretation**: 
- Trajectories with pickups in **over-served cells** (high $\text{DSR}_c$) have negative LIS contribution
- Moving such pickups to **under-served cells** improves fairness
- The gradient naturally points toward cells with below-average service

**Gradient**:
$$
\nabla_{\text{cell}} \text{LIS}_\tau = \begin{cases}
+\text{sign}(\text{DSR}_c - \bar{\text{DSR}}) / (N_c \cdot T) & \text{if trajectory pickup in cell } c \\
-\text{sign}(\text{DSR}_c - \bar{\text{DSR}}) / (n \cdot N_c \cdot T) & \text{(mean adjustment for all cells)}
\end{cases}
$$

**Pros**:
- Intuitive per-trajectory interpretation
- Directly actionable: tells us to move from over-served to under-served
- Computationally efficient

**Cons**:
- Not exactly Gini (uses MAD)
- Requires validation that optimizing LIS improves Gini

**Data Requirements**:
- Mean service rate $\bar{\text{DSR}}$ (pre-computable)
- Per-cell service rates (from `pickup_dropoff_counts.pkl` + `active_taxis`)

---

### 3.6 Spatial Fairness Recommendation

**Recommended Approach**: **Option A (Direct Differentiable) + Option D (LIS for Attribution)**

**Rationale**:
1. Use **Option A** for the actual gradient computation during optimization — this maintains fidelity to the original Gini-based metric
2. Use **Option D (LIS)** for trajectory selection/ranking — this provides interpretable per-trajectory fairness scores

**Implementation Strategy**:
1. Implement soft cell assignment (Section 5) to make trajectory-to-count mapping differentiable
2. Use existing `DifferentiableSpatialFairness` class with soft counts as input
3. Pre-compute LIS for all trajectories to identify candidates for modification
4. During optimization, compute gradients through the full Gini formula

---

## 4. Causal Fairness Term Reformulations

### 4.1 Current Formulation Recap

$$
F_{\text{causal}}^p = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)} = 1 - \frac{\text{Var}(Y - g(D))}{\text{Var}(Y)}
$$

Where:
- $Y_{c,p} = S_{c,p} / D_{c,p}$ (service ratio: supply/demand)
- $S_{c,p}$ = active taxi count (supply) in cell $c$, period $p$
- $D_{c,p}$ = pickup count (demand) in cell $c$, period $p$
- $g(D)$ = expected service ratio given demand (pre-fitted)
- $R_{c,p} = Y_{c,p} - g(D_{c,p})$ (residual)

### 4.2 The Gradient Path for Causal Fairness

**Key Observation**: Trajectory modifications affect **demand** ($D$) directly (via pickup counts) and affect **supply** ($S$) indirectly (via taxi presence).

For a trajectory $\tau$:
- Modifying pickup location changes $D_c$ for the affected cells
- The taxi's GPS trace through the city affects $S_c$ for all cells visited

**Simplification**: Focus on the demand effect, as it's directly tied to the pickup location.

### 4.3 Option A: Direct Differentiable Formulation

**Concept**: Express the R² computation with soft demand counts.

**Formulation**:

Let $D_c^p = D_c^{p,\text{base}} + \sum_\tau \sigma_c^p(\mathbf{p}_\tau^{\text{pickup}})$

Where $D_c^{p,\text{base}}$ is the fixed demand from unmodified trajectories.

Then:
$$
Y_c^p = \frac{S_c^p}{D_c^p + \epsilon}
$$

$$
R_c^p = Y_c^p - g(D_c^p)
$$

$$
F_{\text{causal}}^p = 1 - \frac{\text{Var}(R^p)}{\text{Var}(Y^p)}
$$

**Gradient**:
$$
\frac{\partial F_{\text{causal}}}{\partial D_c} = \frac{\partial F_{\text{causal}}}{\partial Y_c} \cdot \frac{\partial Y_c}{\partial D_c} + \frac{\partial F_{\text{causal}}}{\partial R_c} \cdot \frac{\partial R_c}{\partial D_c}
$$

Note: $g(D_c)$ is frozen (pre-computed), so:
$$
\frac{\partial R_c}{\partial D_c} = \frac{\partial Y_c}{\partial D_c} = -\frac{S_c}{D_c^2}
$$

**Pros**:
- Exact correspondence to current metric
- Uses existing differentiable implementation
- Handles the pre-computed $g(d)$ correctly

**Cons**:
- Gradient w.r.t. demand is typically negative (increasing demand decreases $Y$)
- Supply effect is harder to model (requires full trajectory simulation)

**Data Requirements**:
- `pickup_dropoff_counts.pkl` for demand
- `active_taxis_*.pkl` for supply
- Pre-fitted $g(d)$ function

---

### 4.4 Option B: Residual-Based Per-Trajectory Score

**Concept**: Define a per-trajectory contribution to the residual variance.

**Formulation**:

For trajectory $\tau$ with pickup in cell $c$, period $p$:

$$
\text{ResidualContribution}_\tau = (R_c^p)^2 - \bar{R^2}
$$

Where $\bar{R^2} = \frac{1}{n}\sum_c (R_c^p)^2$ is the mean squared residual.

**Interpretation**: 
- Positive value: Cell $c$ has larger-than-average residual (unexplained variance)
- Negative value: Cell $c$ has smaller-than-average residual

**Per-Trajectory Causal Fairness Impact**:
$$
\Delta F_{\text{causal}}^\tau \propto -\frac{\partial \text{Var}(R)}{\partial D_c} \cdot \frac{1}{\text{Var}(Y)}
$$

A trajectory should move its pickup from a cell with high $|R_c|$ to a cell with low $|R_c|$ to improve causal fairness.

**Pros**:
- Interpretable per-trajectory contribution
- Identifies cells where service ratio is "unexplained by demand"

**Cons**:
- Doesn't capture the effect of changing demand on $g(D)$ evaluation
- First-order approximation

**Data Requirements**:
- Pre-computed residuals per cell
- Variance statistics

---

### 4.5 Option C: Demand-Conditional Fairness Score

**Concept**: Define fairness in terms of whether each cell receives service proportional to its demand.

**Formulation**:

Define the **Demand-Conditional Deviation (DCD)**:
$$
\text{DCD}_c^p = Y_c^p - g(D_c^p) = R_c^p
$$

The causal fairness term is essentially:
$$
F_{\text{causal}} \propto 1 - \frac{\sum_c (\text{DCD}_c)^2}{\sum_c (Y_c - \bar{Y})^2}
$$

**Per-Trajectory Formulation**:

For trajectory $\tau$ with pickup in cell $c$:
$$
\text{CausalImpact}_\tau = |\text{DCD}_c^p|
$$

This measures: *"How much does this trajectory's pickup cell deviate from demand-expected service?"*

**Gradient**:

Moving a pickup from cell $a$ to cell $b$:
- $D_a$ decreases by 1 → $Y_a = S_a/D_a$ increases → $R_a$ changes
- $D_b$ increases by 1 → $Y_b = S_b/D_b$ decreases → $R_b$ changes

The optimal move is from a cell where the residual is large and positive to a cell where adding demand would reduce total residual variance.

**Pros**:
- Direct connection to causal fairness definition
- Identifies cells where intervention is most needed

**Cons**:
- Ignores supply side effects
- Assumes $g(D)$ remains valid after modifications

---

### 4.6 Option D: Supply-Demand Ratio Targeting

**Concept**: Instead of minimizing residual variance, target a uniform supply-demand ratio.

**Formulation**:

In a perfectly causally fair system, $Y_c = g(D_c)$ for all cells. The ideal state is:
$$
\frac{S_c}{D_c} = g(D_c) \quad \forall c
$$

**Per-Trajectory Target**:

For trajectory $\tau$ with pickup in cell $c$:
$$
\text{TargetGap}_\tau = \left| \frac{S_c}{D_c} - g(D_c) \right|
$$

**Optimization Objective**:
$$
\min_{\tau'} \sum_\tau \text{TargetGap}_{\tau'}
$$

**Gradient**:
$$
\frac{\partial \text{TargetGap}}{\partial D_c} = -\frac{S_c}{D_c^2} \cdot \text{sign}\left(\frac{S_c}{D_c} - g(D_c)\right)
$$

**Interpretation**:
- If $Y_c > g(D_c)$ (over-served given demand): Increasing $D_c$ helps (add pickups)
- If $Y_c < g(D_c)$ (under-served given demand): Decreasing $D_c$ helps (remove pickups)

**Pros**:
- Clear optimization target
- Gradient has intuitive meaning
- Doesn't require variance computation

**Cons**:
- Not exactly R² (targets mean, not variance)
- May conflict with spatial fairness goals

---

### 4.7 Causal Fairness Recommendation

**Recommended Approach**: **Option A (Direct Differentiable) + Option C (DCD for Attribution)**

**Rationale**:
1. Use **Option A** for actual gradient computation — maintains fidelity to R² metric
2. Use **Option C (DCD)** for trajectory selection — identifies cells with high unexplained variance

**Key Implementation Notes**:
1. The $g(d)$ function must be **frozen** during optimization (already implemented)
2. Gradients flow through $Y = S/D$ to demand $D$
3. Supply $S$ is treated as fixed during pickup/dropoff modification (but would change for full trajectory edits)

---

## 5. Soft Cell Assignment for Differentiability

### 5.1 The Discrete-to-Continuous Problem

Trajectories have discrete pickup/dropoff locations (grid cells). For gradient-based optimization:
- We need to compute $\partial \mathcal{L} / \partial (x, y)$ where $(x, y)$ is the cell location
- Discrete cell assignments have zero gradients almost everywhere

**Solution**: Relax discrete cell assignment to a probability distribution over cells.

### 5.2 Gaussian Soft Assignment

**Formulation**:

For a pickup location at continuous coordinates $(\tilde{x}, \tilde{y})$, the soft assignment to cell $c = (i, j)$ is:

$$
\sigma_c(\tilde{x}, \tilde{y}) = \frac{\exp\left(-\frac{(i - \tilde{x})^2 + (j - \tilde{y})^2}{2\tau^2}\right)}{\sum_{c'} \exp\left(-\frac{(i' - \tilde{x})^2 + (j' - \tilde{y})^2}{2\tau^2}\right)}
$$

Where $\tau$ is the temperature parameter controlling softness.

**Interpretation**:
- $\tau \to 0$: Hard assignment (one-hot)
- $\tau \to \infty$: Uniform distribution
- $\tau \approx 0.5$: Smooth distribution over ~4-9 nearby cells

### 5.3 Gumbel-Softmax for Discrete Sampling

For training/optimization that requires discrete decisions with gradient flow:

$$
\sigma_c = \frac{\exp((z_c + g_c) / \tau)}{\sum_{c'} \exp((z_{c'} + g_{c'}) / \tau)}
$$

Where:
- $z_c$ = logit for cell $c$ (learned or computed from location)
- $g_c \sim \text{Gumbel}(0, 1)$ = Gumbel noise for exploration
- $\tau$ = temperature

**Straight-Through Estimator**: Use hard assignment in forward pass, soft gradients in backward pass.

### 5.4 Neighborhood-Constrained Soft Assignment

For FAMAIL, modifications are constrained to a neighborhood. The soft assignment should respect this:

$$
\sigma_c(\tilde{x}, \tilde{y}) = \begin{cases}
\frac{\exp(-d_c^2 / 2\tau^2)}{\sum_{c' \in \mathcal{N}} \exp(-d_{c'}^2 / 2\tau^2)} & \text{if } c \in \mathcal{N}(c_{\text{original}}, k) \\
0 & \text{otherwise}
\end{cases}
$$

Where $\mathcal{N}(c_0, k) = \{c : \|c - c_0\|_\infty \leq k\}$ is the $k$-neighborhood.

### 5.5 Implementation Sketch

```python
import torch
import torch.nn.functional as F

def soft_cell_assignment(
    location: torch.Tensor,  # Shape: [batch, 2] - continuous (x, y)
    original_cell: torch.Tensor,  # Shape: [batch, 2] - original cell
    neighborhood_size: int = 5,  # 5x5 neighborhood
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Compute soft assignment over neighborhood cells.
    
    Returns:
        Tensor of shape [batch, neighborhood_size, neighborhood_size]
        representing probability distribution over cells.
    """
    k = (neighborhood_size - 1) // 2  # e.g., k=2 for 5x5
    
    # Create neighborhood grid offsets
    offsets = torch.arange(-k, k + 1, device=location.device)
    grid_x, grid_y = torch.meshgrid(offsets, offsets, indexing='ij')
    
    # Compute cell centers in neighborhood
    # Shape: [batch, neighborhood_size, neighborhood_size, 2]
    cell_centers = original_cell[:, None, None, :] + torch.stack([grid_x, grid_y], dim=-1)
    
    # Compute squared distances
    # Shape: [batch, neighborhood_size, neighborhood_size]
    sq_distances = ((location[:, None, None, :] - cell_centers) ** 2).sum(dim=-1)
    
    # Soft assignment via softmax
    logits = -sq_distances / (2 * temperature ** 2)
    probs = F.softmax(logits.view(-1, neighborhood_size * neighborhood_size), dim=-1)
    
    return probs.view(-1, neighborhood_size, neighborhood_size)


def compute_soft_counts(
    trajectory_probs: torch.Tensor,  # [num_traj, neighborhood_size, neighborhood_size]
    original_cells: torch.Tensor,  # [num_traj, 2]
    grid_dims: tuple,  # (48, 90)
) -> torch.Tensor:
    """
    Aggregate soft assignments into grid-level counts.
    
    Returns:
        Tensor of shape [grid_dims[0], grid_dims[1]] with soft counts.
    """
    counts = torch.zeros(grid_dims, device=trajectory_probs.device)
    k = (trajectory_probs.shape[1] - 1) // 2
    
    for traj_idx in range(trajectory_probs.shape[0]):
        ox, oy = original_cells[traj_idx]
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ci, cj = int(ox + di), int(oy + dj)
                if 0 <= ci < grid_dims[0] and 0 <= cj < grid_dims[1]:
                    prob = trajectory_probs[traj_idx, di + k, dj + k]
                    counts[ci, cj] += prob
    
    return counts
```

### 5.6 Temperature Annealing

During optimization, anneal temperature from soft to hard:

$$
\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}
$$

- Start with $\tau_{\max} \approx 1.0$ (soft, good gradients)
- End with $\tau_{\min} \approx 0.1$ (near-discrete assignments)

---

## 6. Implementation Recommendations

### 6.1 Recommended Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     GRADIENT-BASED TRAJECTORY OPTIMIZATION                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │ Trajectory τ    │ ──► │ Soft Cell       │ ──► │ Soft Pickup/Dropoff │   │
│  │ (x, y, t)       │     │ Assignment σ(·) │     │ Counts              │   │
│  └─────────────────┘     └─────────────────┘     └──────────┬──────────┘   │
│                                                             │              │
│                          ┌──────────────────────────────────┘              │
│                          ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     DIFFERENTIABLE OBJECTIVE TERMS                  │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │   │
│  │  │ Spatial Fairness│  │ Causal Fairness │  │ Fidelity            │  │   │
│  │  │ (Gini)          │  │ (R²)            │  │ (Discriminator)     │  │   │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘  │   │
│  │           │                    │                      │             │   │
│  │           └────────────────────┴──────────────────────┘             │   │
│  │                                │                                    │   │
│  └────────────────────────────────┼────────────────────────────────────┘   │
│                                   ▼                                        │
│                    ┌───────────────────────────┐                           │
│                    │ Combined Loss L           │                           │
│                    │ = α₁F_causal + α₂F_spatial│                           │
│                    │   + α₃F_fidelity          │                           │
│                    └─────────────┬─────────────┘                           │
│                                  │                                         │
│                                  ▼                                         │
│                    ┌───────────────────────────┐                           │
│                    │ ∇_τ L (backpropagation)   │                           │
│                    └─────────────┬─────────────┘                           │
│                                  │                                         │
│                                  ▼                                         │
│                    ┌───────────────────────────┐                           │
│                    │ Update τ (FGSM-style)     │                           │
│                    │ τ' = τ + α·sign(∇L)       │                           │
│                    └───────────────────────────┘                           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Per-Trajectory Attribution (Pre-Optimization)

Before optimization, compute per-trajectory scores for ranking/selection:

```python
def compute_trajectory_fairness_scores(trajectories, counts, active_taxis):
    """
    Compute per-trajectory fairness impact scores.
    
    Returns dict mapping trajectory_id to:
    - spatial_impact: LIS score (how much this traj contributes to inequality)
    - causal_impact: DCD score (how much this traj's cell deviates from expected)
    - combined_score: weighted combination
    """
    scores = {}
    
    # Compute global statistics
    dsr = compute_service_rates(counts, active_taxis)
    mean_dsr = dsr.mean()
    
    residuals = compute_residuals(counts, active_taxis, g_function)
    
    for traj_id, traj in trajectories.items():
        pickup_cell = get_pickup_cell(traj)
        
        # Local Inequality Score
        lis = abs(dsr[pickup_cell] - mean_dsr) / mean_dsr
        
        # Demand-Conditional Deviation
        dcd = abs(residuals[pickup_cell])
        
        scores[traj_id] = {
            'spatial_impact': lis,
            'causal_impact': dcd,
            'combined': alpha_spatial * lis + alpha_causal * dcd
        }
    
    return scores
```

### 6.3 Optimization Loop (ST-iFGSM Inspired)

```python
def optimize_trajectory(
    tau: Trajectory,
    objective_fn: DifferentiableObjective,
    epsilon: float,  # L∞ bound (neighborhood size)
    alpha: float,  # step size
    num_iterations: int,
    temperature_schedule: Callable,
):
    """
    Optimize a single trajectory using gradient-based perturbation.
    """
    # Initialize soft location at original cell
    location = torch.tensor(tau.pickup_cell, dtype=torch.float32, requires_grad=True)
    original_cell = tau.pickup_cell.clone()
    
    for iteration in range(num_iterations):
        # Current temperature
        tau_temp = temperature_schedule(iteration)
        
        # Soft cell assignment
        probs = soft_cell_assignment(location, original_cell, temperature=tau_temp)
        
        # Compute objective (higher is better for fairness)
        loss = -objective_fn(probs)  # Negate to maximize fairness
        
        # Backward pass
        loss.backward()
        
        # FGSM-style update
        with torch.no_grad():
            grad_sign = location.grad.sign()
            location += alpha * grad_sign
            
            # Project to epsilon-ball (neighborhood constraint)
            location = torch.clamp(
                location,
                original_cell - epsilon,
                original_cell + epsilon
            )
            
            # Reset gradient
            location.grad.zero_()
    
    # Discretize final location
    final_cell = location.round().long()
    return final_cell
```

### 6.4 Batch Optimization Considerations

When optimizing multiple trajectories:

1. **Sequential**: Optimize one trajectory at a time, update counts, repeat
2. **Batch**: Optimize a batch together, then update counts
3. **Parallel with Interference Estimation**: Estimate gradient interference and adjust

**Recommendation**: Start with batch size 1 (sequential) for correctness, then scale up with interference monitoring.

---

## 7. Data Requirements Summary

### 7.1 For Spatial Fairness

| Data | Source | Usage |
|------|--------|-------|
| Pickup counts $O_c^p$ | `pickup_dropoff_counts.pkl` | Base counts for service rate |
| Dropoff counts $D_c^p$ | `pickup_dropoff_counts.pkl` | Base counts for service rate |
| Active taxis $N_c^p$ | `active_taxis_5x5_hourly.pkl` | Normalization |
| Trajectory locations | `all_trajs.pkl` | Coordinates to optimize |

### 7.2 For Causal Fairness

| Data | Source | Usage |
|------|--------|-------|
| Demand $D_c^p$ | `pickup_dropoff_counts.pkl` | Pickup counts as demand proxy |
| Supply $S_c^p$ | `active_taxis_5x5_hourly.pkl` | Active taxi counts |
| $g(d)$ function | Pre-fitted (frozen) | Expected service lookup |
| Trajectory locations | `all_trajs.pkl` | Coordinates to optimize |

### 7.3 New Data to Generate

| Data | Generation Method | Purpose |
|------|-------------------|---------|
| Pre-computed LIS scores | Compute from existing data | Trajectory selection |
| Pre-computed DCD scores | Compute from existing data | Trajectory selection |
| Cell-level Gini gradients | Derive from Gini formula | Efficient gradient lookup |

---

## 8. Summary and Next Steps

### 8.1 Key Findings

1. **The current formulations CAN be made differentiable** with respect to individual trajectories via soft cell assignment
2. **Soft assignment** bridges discrete cell locations and continuous optimization
3. **Per-trajectory attribution scores** (LIS for spatial, DCD for causal) enable trajectory selection
4. **The ST-iFGSM framework** translates well to fairness optimization with appropriate formulation

### 8.2 Recommended Implementation Order

1. **Implement soft cell assignment** (Section 5.5)
2. **Integrate with existing differentiable modules** (update `DifferentiableSpatialFairness`, `DifferentiableCausalFairness`)
3. **Implement per-trajectory scoring** (LIS, DCD)
4. **Build optimization loop** following ST-iFGSM structure
5. **Validate** on small trajectory subsets before full-scale

### 8.3 Open Questions for Future Investigation

1. **Temperature scheduling**: Optimal annealing strategy for soft→hard assignment
2. **Batch effects**: How to handle gradient interference when modifying multiple trajectories
3. **Supply-side effects**: Should we model how trajectory path changes affect active_taxis counts?
4. **Metric alignment**: Validate that optimizing soft formulations improves hard metrics

---

## References

1. ST-iFGSM Algorithm Reference (`ST-iFGSM_ALGORITHM_REFERENCE.md`)
2. FAMAIL Objective Function Specification (`FAMAIL_OBJECTIVE_FUNCTION_SPECIFICATION.md`)
3. Trajectory Modification Algorithm Development Plan (`TRAJECTORY_MODIFICATION_ALGORITHM_DEVELOPMENT_PLAN.md`)
4. Gumbel-Softmax: Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017)
5. Differentiable Sorting/Ranking: Cuturi et al., "Differentiable Ranking and Sorting using Optimal Transport" (NeurIPS 2019)

---

*Document created for FAMAIL project. Last updated: January 15, 2026.*
