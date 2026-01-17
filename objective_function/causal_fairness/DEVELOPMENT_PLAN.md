# Causal Fairness Term ($F_{\text{causal}}$) Development Plan

## Document Metadata

| Property | Value |
|----------|-------|
| **Term Name** | Causal Fairness |
| **Symbol** | $F_{\text{causal}}$ |
| **Version** | 1.3.0 |
| **Last Updated** | 2026-01 |
| **Status** | Implementation Complete |
| **Author** | FAMAIL Research Team |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
   - 2.1 [Core Formula](#21-core-formula)
   - 2.2 [Component Definitions](#22-component-definitions)
   - 2.3 [Derivation and Justification](#23-derivation-and-justification)
   - 2.4 [Differentiability Requirements](#24-differentiability-requirements) ✨ **NEW**
3. [Literature and References](#3-literature-and-references)
4. [Data Requirements](#4-data-requirements)
5. [Implementation Plan](#5-implementation-plan)
6. [Configuration Parameters](#6-configuration-parameters)
7. [Testing Strategy](#7-testing-strategy)
8. [Expected Challenges](#8-expected-challenges)
9. [Development Milestones](#9-development-milestones)
10. [Appendix](#10-appendix)

---

## 1. Overview

### 1.1 Purpose and Definition

The **Causal Fairness Term** ($F_{\text{causal}}$) quantifies the degree to which taxi service supply is explained by passenger demand alone, rather than by other contextual factors that may represent unfair biases (e.g., neighborhood characteristics, time-based discrimination).

**Core Principle**: In a causally fair system, the service supply-to-demand ratio should be consistent across all locations when controlling for demand. If two areas have the same demand, they should receive the same supply, regardless of their other characteristics.

**Causal Interpretation**:
- **Legitimate factor**: Demand ($D$) — it's fair for service to vary with demand
- **Potentially unfair factor**: Context ($C$) — service should NOT vary with location/time beyond demand
- **Outcome**: Service ratio ($Y = \text{Supply}/\text{Demand}$)

### 1.2 Role in Objective Function

The causal fairness term is one of two fairness components in the FAMAIL objective function:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

- **Weight**: $\alpha_1$ (typically 0.33 of total weight)
- **Optimization Direction**: **Maximize** (higher values = more causally fair)
- **Value Range**: [0, 1]
  - $F_{\text{causal}} = 1$: Service perfectly explained by demand (no contextual bias)
  - $F_{\text{causal}} = 0$: Service independent of demand (maximum unfairness)

### 1.3 Relationship to Other Terms

| Related Term | Relationship |
|--------------|-------------|
| $F_{\text{spatial}}$ | Complementary; spatial measures distribution equality, causal measures demand alignment |
| $F_{\text{fidelity}}$ | May trade off; aligning supply to demand might require significant trajectory changes |

### 1.4 Key Insights

**Why Causal Fairness Matters**:

1. **Beyond equality**: Equal distribution (spatial fairness) isn't always fair—high-demand areas should receive more service
2. **Demand legitimacy**: Demand differences are legitimate reasons for service variation
3. **Contextual bias**: Service differences beyond demand may reflect discrimination
4. **Actionable**: Identifies specific areas/times with supply-demand mismatch

**Example**:
- Area A: High demand (100 requests), receives 80 taxis (ratio: 0.8)
- Area B: Low demand (10 requests), receives 8 taxis (ratio: 0.8)

This is **causally fair** (same ratio) even though Area A gets more service (spatially unequal).

---

## 2. Mathematical Formulation

### 2.1 Core Formula

The causal fairness term is computed as the **coefficient of determination** ($R^2$) measuring how much of the variance in service ratio is explained by demand:

$$
F_{\text{causal}} = \frac{1}{|P|} \sum_{p \in P} F_{\text{causal}}^p
$$

Where for each period $p$:

$$
F_{\text{causal}}^p = \frac{\text{Var}_p(g(D_{i,p}))}{\text{Var}_p(Y_{i,p})} = 1 - \frac{\text{Var}_p(R_{i,p})}{\text{Var}_p(Y_{i,p})}
$$

### 2.2 Component Definitions

#### 2.2.1 Demand

For each grid cell $i$ and time period $p$:

$$
D_{i,p} = \text{pickup\_count}_{i,p}
$$

**Interpretation**: Demand is proxied by the number of pickup requests (passengers seeking service).

**Source**: `pickup_dropoff_counts.pkl`

#### 2.2.2 Supply

Supply is the count of active taxis in the neighborhood surrounding each grid cell:

$$
S_{i,p} = N^p_i = \text{active\_taxi\_count}_{i,p}
$$

Where:
- $N^p_i$ = number of unique taxis that had at least one GPS reading in the $(2k+1) \times (2k+1)$ neighborhood centered on cell $i$ during period $p$
- Default: $k=2$ (5×5 neighborhood)

**Interpretation**: Supply represents the number of taxis available to serve the area during the time period. This is a direct measure of taxi availability, not just traffic volume.

**Source**: `active_taxis/output/active_taxis_5x5_hourly.pkl` (or other active_taxis output files)

**Why Active Taxis Instead of Traffic Volume?**

Previously, we considered using `latest_volume_pickups.pkl` or `latest_traffic.pkl` for supply data. However, these datasets were aggregated from a larger pool of drivers (>50) than our current study set (50 drivers). Using them would create a mismatch between the supply data and our trajectory data.

The `active_taxis` dataset is generated directly from the same 50-driver GPS data, ensuring consistency:
- Same drivers as in `all_trajs.pkl`
- Same quantization (grid cells, time buckets, days)
- Direct count of available taxis rather than derived traffic metrics

#### 2.2.3 Service Ratio

The service ratio (supply-to-demand ratio) for each cell-period:

$$
Y_{i,p} = \frac{S_{i,p}}{D_{i,p}} \quad \text{for } D_{i,p} > 0
$$

**Interpretation**: How much supply is available per unit of demand. Higher = better service.

**Note**: Cells with zero demand ($D_{i,p} = 0$) are excluded from analysis.

#### 2.2.4 Expected Service Function

The function $g(d)$ represents the expected service ratio given only the demand level:

$$
g(d) = \mathbb{E}[Y \mid D = d]
$$

This can be estimated using several methods:

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Binning** | Group by demand bins, compute mean Y | Simple, interpretable | Sensitive to bin choice |
| **Linear Regression** | Fit $Y \sim \beta_0 + \beta_1 D$ | Fast, smooth | May not capture nonlinearity |
| **Polynomial** | Fit $Y \sim \sum_k \beta_k D^k$ | Captures curvature | Overfitting risk |
| **LOESS/LOWESS** | Local polynomial smoothing | Flexible | Computationally expensive |
| **Isotonic Regression** | Monotonic fitting | Respects monotonicity | May be step-like |

**Default**: To be determined after data visualization (see Section 2.4).

#### 2.2.4.1 Choosing the Estimation Method: Visualization-First Approach

Before selecting the estimation method for $g(d)$, we will **visualize the relationship between demand ($D$) and service ratio ($Y$)** to understand the data characteristics. This visualization should be created early in development to inform our choice.

**Required Visualization: Demand vs. Service Ratio Scatter Plot**

Create a scatter plot with:
- **X-axis**: Demand ($D_{i,p}$) — pickup counts
- **Y-axis**: Service Ratio ($Y_{i,p} = S_{i,p} / D_{i,p}$)
- **Points**: One per (cell, period) combination with positive demand
- **Overlay**: Fitted curves for each candidate estimation method

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

def visualize_demand_service_relationship(
    demands: np.ndarray,
    ratios: np.ndarray,
    output_path: str = "g_estimation_comparison.png"
):
    """
    Visualize D vs Y relationship and compare estimation methods.
    
    Args:
        demands: Array of demand values (pickup counts)
        ratios: Array of service ratios (Y = S/D)
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Scatter plot in all panels
    for ax in axes.flat:
        ax.scatter(demands, ratios, alpha=0.1, s=1, c='gray', label='Data')
        ax.set_xlabel('Demand (Pickup Count)')
        ax.set_ylabel('Service Ratio (Y = S/D)')
    
    # Sort for line plots
    sort_idx = np.argsort(demands)
    D_sorted = demands[sort_idx]
    Y_sorted = ratios[sort_idx]
    
    # 1. Binning (top-left)
    n_bins = 10
    bin_means, bin_edges, _ = binned_statistic(demands, ratios, statistic='mean', bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axes[0, 0].plot(bin_centers, bin_means, 'ro-', linewidth=2, markersize=8, label='Binning (n=10)')
    axes[0, 0].set_title('Binning Method')
    axes[0, 0].legend()
    
    # 2. Linear Regression (top-right)
    lr = LinearRegression()
    lr.fit(demands.reshape(-1, 1), ratios)
    D_pred = np.linspace(demands.min(), demands.max(), 100).reshape(-1, 1)
    Y_pred_lr = lr.predict(D_pred)
    axes[0, 1].plot(D_pred, Y_pred_lr, 'b-', linewidth=2, label='Linear Regression')
    axes[0, 1].set_title(f'Linear Regression (R²={lr.score(demands.reshape(-1,1), ratios):.3f})')
    axes[0, 1].legend()
    
    # 3. Polynomial (degree 2 and 3) (bottom-left)
    for degree, color in [(2, 'green'), (3, 'purple')]:
        poly = PolynomialFeatures(degree=degree)
        D_poly = poly.fit_transform(demands.reshape(-1, 1))
        lr_poly = LinearRegression()
        lr_poly.fit(D_poly, ratios)
        D_pred_poly = poly.transform(D_pred)
        Y_pred_poly = lr_poly.predict(D_pred_poly)
        axes[1, 0].plot(D_pred, Y_pred_poly, color=color, linewidth=2, label=f'Polynomial (d={degree})')
    axes[1, 0].set_title('Polynomial Regression')
    axes[1, 0].legend()
    
    # 4. LOWESS and Isotonic (bottom-right)
    lowess_result = lowess(Y_sorted, D_sorted, frac=0.1)
    axes[1, 1].plot(lowess_result[:, 0], lowess_result[:, 1], 'orange', linewidth=2, label='LOWESS')
    
    iso = IsotonicRegression(increasing=False)  # Expect decreasing Y with increasing D
    Y_iso = iso.fit_transform(D_sorted, Y_sorted)
    axes[1, 1].plot(D_sorted, Y_iso, 'cyan', linewidth=2, label='Isotonic')
    axes[1, 1].set_title('Non-parametric Methods')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    
    return fig
```

**Interpretation Guidelines:**

| Observed Pattern | Recommended Method | Rationale |
|------------------|-------------------|------------|
| **Clear linear trend** | Linear Regression | Simple, interpretable, avoids overfitting |
| **Monotonic but curved** | Isotonic Regression | Respects monotonicity without assuming functional form |
| **Smooth nonlinear curve** | Polynomial (degree 2-3) | Captures curvature with few parameters |
| **Highly variable, noisy** | Binning | Robust to noise, no assumptions about shape |
| **Complex local patterns** | LOWESS | Most flexible, adapts to local structure |
| **Uncertain / mixed signals** | Binning | Safe default, easy to interpret |

**Key Questions to Answer from Visualization:**

1. **Is there a clear relationship?** If the scatter shows no pattern (random cloud), causal fairness will be low regardless of method.

2. **Is the relationship monotonic?** If higher demand consistently leads to lower (or higher) service ratios, consider isotonic regression.

3. **Is there obvious curvature?** If the relationship curves (e.g., diminishing returns), polynomial may be appropriate.

4. **How much variance is there?** High variance at each demand level suggests binning may be most robust.

5. **Are there outliers?** LOWESS and binning are more robust to outliers than regression.

**Action Item**: Generate this visualization before finalizing the implementation. The choice will be documented in the configuration and code comments.

#### 2.2.5 Residual (Unexplained Variation)

The residual captures the portion of service not explained by demand:

$$
R_{i,p} = Y_{i,p} - g(D_{i,p})
$$

**Interpretation**:
- $R_{i,p} > 0$: Cell $i$ receives MORE service than expected given its demand
- $R_{i,p} < 0$: Cell $i$ receives LESS service than expected (potentially unfair)
- $R_{i,p} = 0$: Service perfectly matches demand expectation

#### 2.2.6 Per-Period Causal Fairness

Using the $R^2$ formulation:

$$
F_{\text{causal}}^p = \frac{\text{Var}_p(g(D_{i,p}))}{\text{Var}_p(Y_{i,p})}
$$

Equivalently:

$$
F_{\text{causal}}^p = 1 - \frac{\text{Var}_p(R_{i,p})}{\text{Var}_p(Y_{i,p})} = 1 - \frac{\sum_{i \in \mathcal{I}_p} R_{i,p}^2 / |\mathcal{I}_p|}{\text{Var}_p(Y_{i,p})}
$$

Where $\mathcal{I}_p = \{i : D_{i,p} > 0\}$ is the set of cells with positive demand.

### 2.3 Derivation and Justification

**Why $R^2$ (Coefficient of Determination)?**

1. **Variance decomposition**: Naturally decomposes variance into explained and unexplained
2. **Bounded**: Always in [0, 1] (with proper handling)
3. **Interpretable**: "Proportion of variance explained by demand"
4. **Standard metric**: Widely understood in statistics and causal inference

**Causal Justification**:

Using the potential outcomes framework:
- $Y(d)$ = service ratio if demand were $d$
- Fair system: $Y(d)$ is same for all cells with demand $d$
- Unfair system: $Y(d)$ varies by cell characteristics

The residual $R_{i,p}$ captures variation not attributable to demand, which may reflect unfair treatment.

**Alternative Formulations Considered**:

1. **MSE-based**: $F = -\text{MSE}(R)$ — unbounded, less interpretable
2. **Correlation**: $F = \text{Corr}(S, D)$ — doesn't capture the ratio relationship
3. **KL-divergence**: More complex, harder to compute gradients

### 2.4 Differentiability Requirements

For gradient-based trajectory modification (see [TRAJECTORY_MODIFICATION_DEVELOPMENT_PLAN.md](../docs/TRAJECTORY_MODIFICATION_DEVELOPMENT_PLAN.md)), the causal fairness term **must be end-to-end differentiable**. This section specifies the differentiability approach.

#### 2.4.1 Rationale

The Trajectory Modification Algorithm uses gradient-based attribution (Method C) to identify which spatial modifications will most improve fairness. This requires:

1. **Complete gradient flow** from the objective function back to trajectory coordinates
2. **Differentiable operations** in all intermediate computations
3. **Frozen auxiliary components** that don't participate in optimization

#### 2.4.2 Pre-Computed Frozen $g(d)$ Lookup Table

**Critical Design Decision**: The expected service function $g(d)$ **must be pre-computed and frozen** before optimization.

**Why freeze $g(d)$?**

- During trajectory modification, we only want gradients with respect to the **service ratio** $Y_{i,p}$
- The function $g(d)$ represents the *expected* service-to-demand relationship learned from the original data
- If $g(d)$ were re-estimated during optimization, it would adapt to the modified trajectories, defeating the purpose
- Freezing $g(d)$ ensures residuals measure deviation from the *original* expected relationship

**Implementation Approach**:

1. **Pre-computation Phase** (before optimization):
   - Load original demand/supply data
   - Fit $g(d)$ using chosen estimation method (binning, polynomial, etc.)
   - Convert to lookup table: $g_{\text{lookup}}[d] = g(d)$ for $d \in \{0, 1, ..., d_{\max}\}$
   - Store as a frozen `torch.Tensor` buffer

2. **Optimization Phase**:
   - Given modified trajectories → compute modified $S'_{i,p}$
   - Look up $\hat{Y}_{i,p} = g_{\text{lookup}}[D_{i,p}]$ (frozen, no gradient)
   - Compute $Y'_{i,p} = S'_{i,p} / D_{i,p}$ (requires gradient through $S'$)
   - Compute residual $R'_{i,p} = Y'_{i,p} - \hat{Y}_{i,p}$ (gradient flows through $Y'$)

#### 2.4.3 Differentiable R² Computation

The variance-based $R^2$ formula is naturally differentiable:

$$
F_{\text{causal}}^p = 1 - \frac{\text{Var}_p(R_{i,p})}{\text{Var}_p(Y_{i,p})}
$$

Both variance computations are differentiable:

$$
\text{Var}(X) = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = \frac{1}{n}\sum_i x_i^2 - \left(\frac{1}{n}\sum_i x_i\right)^2
$$

**PyTorch Implementation**:

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple


class DifferentiableCausalFairness(nn.Module):
    """
    Differentiable causal fairness term for gradient-based optimization.
    
    The g(d) lookup table is frozen (no gradient). Gradients flow through
    the service ratio Y = S/D back to the supply term S.
    """
    
    def __init__(
        self,
        g_lookup: torch.Tensor,
        min_demand: int = 1,
        eps: float = 1e-8
    ):
        """
        Initialize with pre-computed g(d) lookup table.
        
        Args:
            g_lookup: 1D tensor of shape (max_demand + 1,) where
                      g_lookup[d] = E[Y | D = d]
            min_demand: Minimum demand threshold
            eps: Numerical stability constant
        """
        super().__init__()
        
        # Freeze g_lookup - no gradients flow through it
        self.register_buffer('g_lookup', g_lookup)
        self.min_demand = min_demand
        self.eps = eps
    
    def forward(
        self,
        supply: torch.Tensor,
        demand: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute differentiable causal fairness.
        
        Args:
            supply: Tensor of supply values S_{i,p}, shape (n_cells,)
            demand: Tensor of demand values D_{i,p}, shape (n_cells,)
        
        Returns:
            Causal fairness score in [0, 1]
        """
        # Filter cells with sufficient demand
        mask = demand >= self.min_demand
        S = supply[mask]
        D = demand[mask]
        
        if S.numel() == 0:
            return torch.tensor(0.0, device=supply.device)
        
        # Compute service ratio Y = S/D (gradient flows through S)
        Y = S / (D.float() + self.eps)
        
        # Look up expected Y from frozen g(d)
        # Clamp indices to valid range
        d_indices = D.long().clamp(0, self.g_lookup.size(0) - 1)
        Y_expected = self.g_lookup[d_indices]  # No gradient here
        
        # Compute residual (gradient flows through Y)
        R = Y - Y_expected
        
        # Compute variances
        var_Y = self._differentiable_var(Y)
        var_R = self._differentiable_var(R)
        
        # R² = 1 - Var(R) / Var(Y)
        if var_Y < self.eps:
            return torch.tensor(1.0, device=supply.device)
        
        r_squared = 1.0 - var_R / (var_Y + self.eps)
        
        return torch.clamp(r_squared, 0.0, 1.0)
    
    def _differentiable_var(self, x: torch.Tensor) -> torch.Tensor:
        """Compute variance in a differentiable way."""
        mean_x = x.mean()
        return ((x - mean_x) ** 2).mean()
    
    @staticmethod
    def create_g_lookup(
        demands: torch.Tensor,
        ratios: torch.Tensor,
        method: str = 'binning',
        n_bins: int = 10
    ) -> torch.Tensor:
        """
        Pre-compute g(d) lookup table from original data.
        
        Call this BEFORE optimization to create the frozen lookup.
        
        Args:
            demands: Original demand values
            ratios: Original service ratios Y = S/D
            method: Estimation method ('binning', 'polynomial')
            n_bins: Number of bins for binning method
        
        Returns:
            Lookup tensor g_lookup where g_lookup[d] = E[Y | D = d]
        """
        max_demand = int(demands.max().item()) + 1
        g_lookup = torch.zeros(max_demand)
        
        if method == 'binning':
            # Quantile-based bins
            bin_edges = torch.quantile(
                demands.float(),
                torch.linspace(0, 1, n_bins + 1)
            )
            bin_edges = torch.unique(bin_edges)
            
            for d in range(max_demand):
                # Find which bin d falls into
                bin_idx = torch.searchsorted(bin_edges, float(d))
                bin_idx = torch.clamp(bin_idx, 0, len(bin_edges) - 1)
                
                # Mean ratio for this bin
                in_bin = (demands >= bin_edges[bin_idx - 1]) & \
                         (demands < bin_edges[bin_idx])
                if in_bin.any():
                    g_lookup[d] = ratios[in_bin].mean()
        
        elif method == 'polynomial':
            # Fit polynomial regression
            import numpy as np
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            D_np = demands.numpy().reshape(-1, 1)
            Y_np = ratios.numpy()
            
            poly = PolynomialFeatures(degree=2)
            D_poly = poly.fit_transform(D_np)
            
            model = LinearRegression().fit(D_poly, Y_np)
            
            # Fill lookup table
            for d in range(max_demand):
                d_poly = poly.transform([[d]])
                g_lookup[d] = torch.tensor(model.predict(d_poly)[0])
        
        return g_lookup
```

#### 2.4.4 Gradient Flow Diagram

```
                     ┌─────────────────────────────────────────────────┐
                     │         CAUSAL FAIRNESS GRADIENT FLOW           │
                     └─────────────────────────────────────────────────┘

    Trajectory Coordinates (x, y)
              │
              │ ∂L/∂coord (backprop)
              ▼
    ┌─────────────────┐
    │ Grid Cell Count │ (how many trajectories pass through each cell)
    │   (Soft Binning)│
    └────────┬────────┘
              │
              │ ∂supply/∂coord
              ▼
    ┌─────────────────┐
    │    Supply S     │ = count of active taxis
    └────────┬────────┘
              │
              │ ∂Y/∂S = 1/D
              ▼
    ┌─────────────────┐     ┌─────────────────────┐
    │   Ratio Y=S/D   │ ◄───│ Demand D (fixed)    │ No gradient
    └────────┬────────┘     └─────────────────────┘
              │
              │ ∂R/∂Y = 1
              ▼
    ┌─────────────────┐     ┌─────────────────────┐
    │ Residual R=Y-ĝ  │ ◄───│ ĝ = g_lookup[D]    │ FROZEN (no grad)
    └────────┬────────┘     │ (pre-computed)      │
              │              └─────────────────────┘
              │
              │ ∂Var(R)/∂R, ∂Var(Y)/∂Y
              ▼
    ┌─────────────────┐
    │  R² = 1 - Var(R)│ / Var(Y)
    └────────┬────────┘
              │
              │ ∂F_causal/∂R²
              ▼
    ┌─────────────────┐
    │  F_causal       │ = mean(R² per period)
    └─────────────────┘

    Key:
    ───── Gradient flows through
    ◄──── Dependency (no gradient for frozen components)
```

#### 2.4.5 Implementation Checklist

- [ ] Pre-compute $g(d)$ lookup table before optimization starts
- [ ] Register lookup as buffer (`register_buffer`) to prevent gradient computation
- [ ] Use `torch.clamp` instead of `np.clip` for differentiability
- [ ] Ensure demand values are detached (no gradient)
- [ ] Add numerical stability epsilon to divisions
- [ ] Verify gradient flow with `torch.autograd.grad()`

#### 2.4.6 Gradient Verification

```python
def verify_causal_fairness_gradients():
    """
    Test that gradients flow correctly through causal fairness.
    """
    import torch
    
    # Create synthetic data
    torch.manual_seed(42)
    n_cells = 100
    
    # Supply (what we're optimizing)
    supply = torch.randn(n_cells, requires_grad=True) * 10 + 50
    supply = torch.abs(supply)  # Ensure positive
    
    # Demand (fixed)
    demand = torch.randint(1, 20, (n_cells,))
    
    # Create frozen g(d) lookup
    g_lookup = torch.ones(50) * 5.0  # Simple constant g(d) = 5
    
    # Compute causal fairness
    model = DifferentiableCausalFairness(g_lookup, min_demand=1)
    f_causal = model(supply, demand)
    
    # Check gradient exists
    f_causal.backward()
    
    assert supply.grad is not None, "No gradient for supply!"
    assert not torch.isnan(supply.grad).any(), "NaN gradients!"
    assert not torch.isinf(supply.grad).any(), "Inf gradients!"
    
    print(f"✓ Causal fairness: {f_causal.item():.4f}")
    print(f"✓ Gradient shape: {supply.grad.shape}")
    print(f"✓ Gradient mean: {supply.grad.mean().item():.6f}")
    print(f"✓ Gradient std: {supply.grad.std().item():.6f}")
    
    return True


if __name__ == "__main__":
    verify_causal_fairness_gradients()
```

### 2.5 Soft Cell Assignment (Full Differentiable Pipeline)

> **STATUS: IMPLEMENTED** — `objective_function/causal_fairness/utils.py`

To enable end-to-end gradient flow from raw trajectory coordinates to causal fairness, we extend the differentiable causal fairness with soft cell assignment.

#### 2.5.1 The Soft Assignment Approach

Instead of hard cell assignment (`cell = (int(x), int(y))`), distribute trajectory points across cells using a Gaussian softmax (see [spatial_fairness/DEVELOPMENT_PLAN.md](../spatial_fairness/DEVELOPMENT_PLAN.md) Section 2.5 for full details):

$$
\sigma_c(x, y) = \frac{\exp\left(-\frac{d^2_c(x,y)}{2\tau^2}\right)}{\sum_{c' \in \mathcal{N}} \exp\left(-\frac{d^2_{c'}(x,y)}{2\tau^2}\right)}
$$

#### 2.5.2 Full Differentiable Pipeline

```python
class DifferentiableCausalFairnessWithSoftCounts(nn.Module):
    """
    Complete differentiable causal fairness with soft cell assignment.
    
    Enables gradient flow from F_causal back to raw trajectory coordinates.
    """
    
    def __init__(self, g_function, grid_shape=(48, 90), 
                 neighborhood_size=5, temperature=1.0):
        super().__init__()
        self.g_function = g_function  # Pre-fitted, FROZEN
        self.soft_assign = SoftCellAssignment(
            grid_shape=grid_shape,
            neighborhood_size=neighborhood_size,
            temperature=temperature
        )
        self.grid_shape = grid_shape
    
    def forward(self, pickup_coords, supply_tensor):
        """
        Compute differentiable causal fairness.
        
        Args:
            pickup_coords: Tensor (N, 2) - pickup (x, y) coordinates
            supply_tensor: Tensor (48, 90) - supply per cell (fixed)
        
        Returns:
            F_causal: Scalar R² score ∈ [0, 1] (higher = more fair)
        """
        # Soft demand counts (DIFFERENTIABLE)
        demand = self.soft_assign(pickup_coords).flatten()  # (4320,)
        
        # Compute Y = supply / demand
        supply = supply_tensor.flatten()
        mask = demand > 0.1  # Minimum demand threshold
        
        if mask.sum() < 2:
            return torch.tensor(0.0, device=pickup_coords.device)
        
        Y = supply[mask] / (demand[mask] + 1e-8)
        
        # g(d) is FROZEN - no gradient
        with torch.no_grad():
            g_d = self.g_function(demand[mask].detach().cpu().numpy())
        g_d = torch.tensor(g_d, device=Y.device, dtype=Y.dtype)
        
        # Residuals (differentiable w.r.t. Y)
        R = Y - g_d
        
        # R² computation
        var_Y = Y.var() + 1e-8
        var_R = R.var()
        r_squared = 1 - (var_R / var_Y)
        
        return torch.clamp(r_squared, 0.0, 1.0)
```

**Location**: `objective_function/causal_fairness/utils.py`

#### 2.5.3 Key Design Decision: Frozen g(d)

The expected service function $g(d)$ is **pre-computed and frozen**:

```python
# g(d) values are pre-computed and FROZEN (no gradient)
# Gradients flow through: pickup_loc → soft_assign → demand → Y = S/D → R² 
with torch.no_grad():
    g_d = self.g_function(demand.detach().numpy())
```

**Rationale**: If $g(d)$ were also optimized, the model could trivially achieve $R^2 = 1$ by making $g(d) = Y$ everywhere, defeating the metric's purpose.

#### 2.5.4 g(d) Estimation Method Benchmark

| Method | R² Score | Differentiability | Recommendation |
|--------|----------|-------------------|----------------|
| **Isotonic** | **0.4453** | ✅ Compatible | 🥇 Best fit |
| **Binning** | **0.4439** | ✅ Compatible | 🥈 Alternative |
| Polynomial | 0.1949 | ✅ Compatible | Poor fit |
| Linear | 0.1064 | ✅ Compatible | Poor fit |

**Note**: All methods are "compatible" because $g(d)$ is frozen—the estimation method's differentiability is irrelevant.

**Recommended Default**: Isotonic regression (best fit on real data).

#### 2.5.5 Demand-Conditional Deviation (DCD) Attribution

For trajectory-level attribution, compute each trajectory's contribution to unexplained variance:

$$
\text{DCD}_\tau = \sum_{c \in \mathcal{C}} \sigma_c(\tau) \cdot R_c^2
$$

Where $R_c = Y_c - g(D_c)$ is the squared residual for cell $c$.

**Interpretation**: Higher DCD means the trajectory contributes more to cells with large deviations from expected service levels.

```python
def compute_demand_conditional_deviation(
    trajectory_coords: torch.Tensor,
    residuals_squared: torch.Tensor,
    soft_assign: SoftCellAssignment
) -> torch.Tensor:
    """
    Compute DCD attribution for a trajectory.
    
    Args:
        trajectory_coords: Tensor (T, 2) - trajectory (x, y) positions
        residuals_squared: Tensor (48, 90) - R_c² per cell
        soft_assign: Soft cell assignment module
    
    Returns:
        dcd: Scalar attribution score
    """
    soft_counts = soft_assign(trajectory_coords)
    dcd = (soft_counts * residuals_squared).sum()
    return dcd
```

**Location**: `objective_function/causal_fairness/utils.py::compute_demand_conditional_deviation()`

#### 2.5.6 End-to-End Gradient Flow Verification

```python
def verify_causal_end_to_end():
    """Verify gradients flow from F_causal to trajectory coordinates."""
    # Create model with dummy g(d)
    g_func = lambda d: np.ones_like(d) * 0.5  # Constant g(d) = 0.5
    model = DifferentiableCausalFairnessWithSoftCounts(g_func)
    
    # Create trajectory with gradient tracking
    pickups = torch.tensor([[10.5, 20.3], [15.2, 45.8]], requires_grad=True)
    supply = torch.rand(48, 90)  # Fixed supply
    
    # Forward pass
    F_causal = model(pickups, supply)
    
    # Backward pass
    F_causal.backward()
    
    # Verify gradients exist
    assert pickups.grad is not None, "No gradient for pickup coordinates"
    assert not torch.all(pickups.grad == 0), "Zero gradient"
    
    print(f"F_causal = {F_causal.item():.4f}")
    print(f"∂F/∂pickups = {pickups.grad}")
    print("✅ End-to-end gradient verification passed")
```

---

## 3. Literature and References

### 3.1 Primary Sources

#### 3.1.1 FAMAIL Project Documentation

The causal fairness formulation is developed specifically for FAMAIL based on:
- Notion project documentation on causal fairness
- Counterfactual fairness literature adaptation

#### 3.1.2 Related Papers

**Counterfactual Fairness**:
- Kusner et al. (2017): "Counterfactual Fairness" — foundational framework
- Chiappa (2019): "Path-Specific Counterfactual Fairness"

**Fairness in Transportation**:
- Ge et al. (2016): "Racial and Gender Discrimination in Transportation Network Companies"
- Brown (2018): "Ridehail Revolution: Ridehail Travel and Equity in Los Angeles"

### 3.2 Theoretical Foundation

**Causal Inference Background**:

The causal fairness term is based on the idea of separating:
- **Direct effect** of demand on service (legitimate)
- **Indirect/spurious effects** through context (potentially unfair)

Using Pearl's causal framework:
- Demand ($D$) → Service ($Y$): Direct path (fair)
- Context ($C$) → Service ($Y$): Alternative path (potentially unfair)

**Identification Assumption**:
We assume that conditioning on demand ($D$) removes confounding:

$$
Y \perp C \mid D \quad \text{(in a fair system)}
$$

Violations of this indicate unfairness.

### 3.3 Related Work in FAMAIL

| Component | Relationship |
|-----------|-------------|
| Spatial Fairness | Uses same pickup/dropoff data but measures distribution equality |
| ST-iFGSM | Causal fairness provides gradients for trajectory optimization |
| Discriminator | Maintains authenticity while improving causal fairness |

---

## 4. Data Requirements

### 4.1 Required Datasets

#### 4.1.1 Demand Data: `pickup_dropoff_counts.pkl`

**Location**: `FAMAIL/source_data/pickup_dropoff_counts.pkl`

**Structure**:
```python
{
    (x_grid, y_grid, time_bucket, day_of_week): [pickup_count, dropoff_count],
}
```

**Fields Used**:
| Field | Usage |
|-------|-------|
| `pickup_count` (index 0) | Demand proxy ($D_{i,p}$) |
| Key components | Spatiotemporal indexing |

#### 4.1.2 Supply Data: `active_taxis` Output

**Location**: `FAMAIL/active_taxis/output/active_taxis_5x5_hourly.pkl` (or similar)

**Important**: The supply data must come from the `active_taxis` tool, NOT from `latest_volume_pickups.pkl` or `latest_traffic.pkl`. Those datasets were aggregated from a larger driver pool than our 50-driver study set.

**Structure**:
```python
{
    (x_grid, y_grid, period_key...): active_taxi_count,
    # For hourly:
    (1, 1, 0, 1): 15,  # Cell (1,1), hour 0, Monday: 15 active taxis
    # For time_bucket:
    (1, 1, 144, 3): 12,  # Cell (1,1), bucket 144, Wednesday: 12 active taxis
}
```

**Fields Used**:
| Field | Usage |
|-------|-------|
| `active_taxi_count` (value) | Supply proxy ($S_{i,p} = N^p_i$) |
| Key components | Spatiotemporal indexing |

**Generation**: Use the `active_taxis` tool with matching configuration:
```python
from active_taxis import ActiveTaxisConfig, generate_active_taxi_counts

config = ActiveTaxisConfig(
    neighborhood_size=2,      # 5×5 neighborhood (must match causal fairness config)
    period_type='hourly',     # Or 'time_bucket' for finer granularity
    test_mode=False,
)
counts, stats = generate_active_taxi_counts(config)
```

#### 4.1.3 Data Alignment Requirements

| Property | Demand Data | Supply Data | Notes |
|----------|-------------|-------------|-------|
| Grid dimensions | 48 × 90 | 48 × 90 | Must match |
| Coordinate indexing | 1-indexed | 1-indexed | Both use same convention |
| Days | Monday (1) – Saturday (6) | Monday (1) – Saturday (6) | Sunday excluded |
| Time resolution | 5-min buckets (288/day) | Configurable | May need aggregation |

**Note on Temporal Alignment**: The `pickup_dropoff_counts` uses 5-minute time buckets, while `active_taxis` may be generated at hourly or daily granularity. If resolutions differ, aggregate the demand data to match the supply data resolution before computing service ratios.

### 4.2 Data Preprocessing

#### 4.2.1 Loading and Aligning Data

```python
def load_causal_fairness_data(
    pickup_counts_path: str,
    active_taxis_path: str
) -> Tuple[Dict, Dict]:
    """
    Load and align demand and supply data.
    
    Args:
        pickup_counts_path: Path to pickup_dropoff_counts.pkl
        active_taxis_path: Path to active_taxis output file
    
    Returns:
        (demand_data, supply_data) dictionaries with aligned keys
    """
    import pickle
    
    with open(pickup_counts_path, 'rb') as f:
        pickup_data = pickle.load(f)
    
    with open(active_taxis_path, 'rb') as f:
        active_taxis_data = pickle.load(f)
    
    # Handle active_taxis output format (may contain metadata)
    if isinstance(active_taxis_data, tuple):
        supply = active_taxis_data[0]  # First element is the counts dict
    else:
        supply = active_taxis_data
    
    # Extract demand (pickup counts)
    demand = {key: val[0] for key, val in pickup_data.items()}
    
    return demand, supply
```

#### 4.2.2 Temporal Aggregation (if needed)

If supply data is at a coarser resolution (e.g., hourly) than demand data (5-min buckets), aggregate demand:

```python
def aggregate_demand_to_hourly(
    demand: Dict[Tuple, int]
) -> Dict[Tuple, int]:
    """
    Aggregate 5-minute demand data to hourly.
    
    Args:
        demand: Dict[(x, y, time_bucket, day)] -> pickup_count
        
    Returns:
        Dict[(x, y, hour, day)] -> total_pickup_count
    """
    from collections import defaultdict
    
    hourly_demand = defaultdict(int)
    
    for (x, y, time_bucket, day), count in demand.items():
        hour = (time_bucket - 1) // 12  # Convert bucket (1-288) to hour (0-23)
        hourly_demand[(x, y, hour, day)] += count
    
    return dict(hourly_demand)
```

#### 4.2.3 Computing Service Ratios

```python
def compute_service_ratios(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, int],
    min_demand: int = 1
) -> Dict[Tuple, float]:
    """
    Compute service ratio Y = S/D for each cell-period.
    
    Note: Supply data from active_taxis already includes neighborhood
    aggregation, so no additional aggregation is needed here.
    
    Args:
        demand: Demand (pickup counts) per cell-period
        supply: Supply (active taxi counts) per cell-period
        min_demand: Minimum demand threshold
    
    Returns:
        Service ratios for cells with sufficient demand
    """
    ratios = {}
    
    for key, d in demand.items():
        if d >= min_demand:
            s = supply.get(key, 0)
            ratios[key] = s / d
    
    return ratios
```
        demand: Demand (pickup counts) per cell-period
        supply: Supply (aggregated traffic volume) per cell-period
        min_demand: Minimum demand threshold
    
    Returns:
        Service ratios for cells with sufficient demand
    """
    ratios = {}
    
    for key, d in demand.items():
        if d >= min_demand:
            s = supply.get(key, 0)
            ratios[key] = s / d
    
    return ratios
```

### 4.3 Data Validation

```python
def validate_causal_fairness_data(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, int]
) -> List[str]:
    """
    Validate data for causal fairness computation.
    
    Args:
        demand: Pickup counts from pickup_dropoff_counts.pkl
        supply: Active taxi counts from active_taxis output
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check non-empty
    if len(demand) == 0:
        errors.append("Demand data is empty")
    if len(supply) == 0:
        errors.append("Supply data is empty")
    
    # Check overlap
    demand_keys = set(demand.keys())
    supply_keys = set(supply.keys())
    overlap = demand_keys.intersection(supply_keys)
    
    if len(overlap) == 0:
        errors.append("No overlapping keys between demand and supply")
    elif len(overlap) < 0.5 * len(demand_keys):
        errors.append(f"Low overlap: {len(overlap)} / {len(demand_keys)} keys")
    
    # Check for negative values
    neg_demand = sum(1 for v in demand.values() if v < 0)
    neg_supply = sum(1 for v in supply.values() if v < 0)
    
    if neg_demand > 0:
        errors.append(f"{neg_demand} negative demand values")
    if neg_supply > 0:
        errors.append(f"{neg_supply} negative supply values")
    
    return errors
```

---

## 5. Implementation Plan

### 5.1 Algorithm Steps

```
ALGORITHM: Compute Causal Fairness Term
═══════════════════════════════════════════════════════════════════════

INPUT:
  - demand: Dict[(x, y, time, day)] → pickup_count
  - supply: Dict[(x, y, time, day)] → active_taxi_count (from active_taxis tool)
  - config: CausalFairnessConfig
    - grid_dims: (48, 90)
    - neighborhood_size: 2 (for reference; aggregation done in active_taxis)
    - estimation_method: to be determined from visualization
    - min_demand: 1
    - period_type: "hourly"

OUTPUT:
  - F_causal: float ∈ [0, 1] (higher = more fair)

STEPS:

1. DATA PREPARATION
   ─────────────────────────────────
   1.1 Aggregate supply over neighborhoods
   1.2 Compute service ratios Y = S/D for cells with D > 0
   1.3 Group by period according to period_type

2. ESTIMATE g(d) FUNCTION
   ─────────────────────────────────
   2.1 Collect all (D, Y) pairs across all cells and periods
   2.2 If estimation_method == "binning":
       - Create demand bins (e.g., quantile-based)
       - For each bin, compute mean Y
       - g(d) = mean Y of bin containing d
   2.3 If estimation_method == "regression":
       - Fit Y ~ polynomial(D)
       - g(d) = model.predict(d)

3. COMPUTE PER-PERIOD CAUSAL FAIRNESS
   ─────────────────────────────────
   3.1 For each period p:
       3.1.1 Get all (D_ip, Y_ip) pairs for period p
       3.1.2 Compute predicted Y: Ŷ_ip = g(D_ip)
       3.1.3 Compute Var_p(Y) = variance of Y_ip values
       3.1.4 Compute Var_p(Ŷ) = variance of Ŷ_ip values
       3.1.5 F_causal_p = Var_p(Ŷ) / Var_p(Y)  [clipped to [0, 1]]

4. AGGREGATE
   ─────────────────────────────────
   4.1 F_causal = mean(F_causal_p for all periods p)

5. RETURN F_causal
```

### 5.2 Pseudocode

```
function compute_causal_fairness(demand, supply, config):
    # Step 1: Align temporal resolution if needed
    if supply uses hourly and demand uses time_bucket:
        demand = aggregate_to_hourly(demand)
    
    # Note: No neighborhood aggregation needed here -
    # supply from active_taxis already includes it
    
    # Compute service ratios
    ratios = {}
    for key in demand:
        if demand[key] >= config.min_demand:
            ratios[key] = supply.get(key, 0) / demand[key]
    
    # Collect (D, Y, period) observations
    observations = []
    for key, Y in ratios.items():
        D = demand[key]
        period = extract_period(key, config.period_type)
        observations.append((D, Y, period))
    
    # Step 2: Estimate g(d)
    all_D = [obs[0] for obs in observations]
    all_Y = [obs[1] for obs in observations]
    g = estimate_g_function(all_D, all_Y, config.estimation_method)
    
    # Step 3: Compute per-period R²
    periods = unique([obs[2] for obs in observations])
    F_causal_periods = []
    
    for p in periods:
        period_obs = [(D, Y) for (D, Y, per) in observations if per == p]
        
        if len(period_obs) > 1:
            D_vals = [obs[0] for obs in period_obs]
            Y_vals = [obs[1] for obs in period_obs]
            Y_pred = [g(d) for d in D_vals]
            
            var_Y = variance(Y_vals)
            var_pred = variance(Y_pred)
            
            if var_Y > 0:
                r_squared = var_pred / var_Y
                F_causal_periods.append(clip(r_squared, 0, 1))
    
    # Step 4: Aggregate
    F_causal = mean(F_causal_periods) if F_causal_periods else 0.0
    
    return F_causal
```

### 5.3 Python Implementation Outline

```python
# File: objective_function/causal_fairness/term.py

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Callable, Literal
import numpy as np
from scipy.stats import binned_statistic
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from objective_function.base import ObjectiveFunctionTerm, TermMetadata, TermConfig


@dataclass
class CausalFairnessConfig(TermConfig):
    """Configuration for causal fairness term."""
    grid_dims: Tuple[int, int] = (48, 90)
    neighborhood_size: int = 2  # k for (2k+1)×(2k+1) window (5×5 default)
    estimation_method: str = "binning"  # "binning", "regression", "polynomial", "isotonic", "lowess"
    n_bins: int = 10
    poly_degree: int = 2
    min_demand: int = 1
    period_type: str = "hourly"  # "time_bucket", "hourly", "daily"


class CausalFairnessTerm(ObjectiveFunctionTerm):
    """
    Causal Fairness term based on R² of demand-explained service variance.
    
    Measures how much of the variation in service is explained by demand.
    Higher values indicate more fair (demand-based) service allocation.
    
    Supply data comes from the active_taxis tool, which pre-computes
    neighborhood-aggregated taxi counts for consistency with our 50-driver dataset.
    """
    
    def _build_metadata(self) -> TermMetadata:
        return TermMetadata(
            name="causal_fairness",
            display_name="Causal Fairness",
            version="1.1.0",
            description="R²-based measure of demand-explained service variation",
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,
            required_data=["pickup_dropoff_counts", "active_taxis"],
            optional_data=[],
            author="FAMAIL Team",
            last_updated="2026-01-12"
        )
    
    def _validate_config(self) -> None:
        valid_methods = ["binning", "regression", "polynomial", "isotonic", "lowess"]
        if self.config.estimation_method not in valid_methods:
            raise ValueError(f"Invalid estimation_method: {self.config.estimation_method}")
        if self.config.neighborhood_size < 0:
            raise ValueError("neighborhood_size must be non-negative")
        if self.config.min_demand < 1:
            raise ValueError("min_demand must be at least 1")
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """Compute causal fairness value."""
        # Get demand and supply data
        demand = self._extract_demand(auxiliary_data)
        supply = self._extract_supply(auxiliary_data)  # From active_taxis
        
        # Align temporal resolution if needed
        demand = self._align_temporal_resolution(demand, supply)
        
        # Compute service ratios (no neighborhood aggregation needed -
        # active_taxis output already includes it)
        ratios = self._compute_ratios(demand, supply)
        
        # Compute causal fairness
        return self._compute_r_squared(demand, ratios)
    
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute with detailed breakdown."""
        # Returns per-period R², g function parameters, residual statistics
        pass
    
    def _estimate_g_function(
        self,
        demands: np.ndarray,
        ratios: np.ndarray
    ) -> Callable:
        """Estimate g(d) = E[Y | D = d]."""
        if self.config.estimation_method == "binning":
            return self._estimate_binning(demands, ratios)
        elif self.config.estimation_method == "polynomial":
            return self._estimate_polynomial(demands, ratios)
        else:
            return self._estimate_linear(demands, ratios)
    
    def _estimate_binning(
        self,
        demands: np.ndarray,
        ratios: np.ndarray
    ) -> Callable:
        """Estimate g using binning approach."""
        # Create quantile-based bins
        percentiles = np.linspace(0, 100, self.config.n_bins + 1)
        bin_edges = np.percentile(demands, percentiles)
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        # Compute mean ratio per bin
        bin_means, _, _ = binned_statistic(
            demands, ratios, statistic='mean', bins=bin_edges
        )
        
        def g(d):
            d_arr = np.atleast_1d(d)
            indices = np.digitize(d_arr, bin_edges) - 1
            indices = np.clip(indices, 0, len(bin_means) - 1)
            result = np.array([
                bin_means[i] if not np.isnan(bin_means[i]) else 0.0
                for i in indices
            ])
            return result if len(result) > 1 else result[0]
        
        return g
    
    def _compute_r_squared(
        self,
        demand: Dict[Tuple, int],
        ratios: Dict[Tuple, float]
    ) -> float:
        """Compute R² across periods."""
        # Collect observations
        observations = []
        for key, Y in ratios.items():
            D = demand.get(key, 0)
            if D >= self.config.min_demand:
                period = self._extract_period(key)
                observations.append((D, Y, period))
        
        if len(observations) == 0:
            return 0.0
        
        # Estimate g
        demands = np.array([obs[0] for obs in observations])
        ratios_arr = np.array([obs[1] for obs in observations])
        g = self._estimate_g_function(demands, ratios_arr)
        
        # Compute per-period R²
        periods = list(set(obs[2] for obs in observations))
        r_squared_periods = []
        
        for p in periods:
            mask = np.array([obs[2] == p for obs in observations])
            p_demands = demands[mask]
            p_ratios = ratios_arr[mask]
            
            if len(p_ratios) > 1:
                p_predicted = g(p_demands)
                
                var_Y = np.var(p_ratios)
                if var_Y > 0:
                    var_pred = np.var(p_predicted)
                    r_sq = np.clip(var_pred / var_Y, 0.0, 1.0)
                    r_squared_periods.append(r_sq)
        
        return np.mean(r_squared_periods) if r_squared_periods else 0.0
```

### 5.4 Computational Considerations

#### 5.4.1 Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Neighborhood aggregation | $O(|K| \cdot (2k+1)^2)$ |
| Service ratio computation | $O(|K|)$ |
| g(d) estimation (binning) | $O(|K| \log |K|)$ (sorting) |
| g(d) estimation (regression) | $O(|K| \cdot d^2)$ where $d$ = degree |
| Per-period R² | $O(|P| \cdot |K|/|P|)$ = $O(|K|)$ |
| **Total** | $O(|K| \cdot (2k+1)^2 + |K| \log |K|)$ |

With $|K| \approx 234,000$ and $k=1$: manageable computational cost.

#### 5.4.2 Memory Considerations

- Service ratios: $O(|K|)$ — same as input size
- g function lookup: $O(n_{bins})$ or $O(d)$ for regression
- Per-period arrays: $O(\max_p |K_p|)$

#### 5.4.3 Numerical Stability

```python
def safe_r_squared(var_predicted: float, var_actual: float) -> float:
    """Compute R² with numerical stability."""
    if var_actual < 1e-10:  # Near-zero variance
        return 1.0 if var_predicted < 1e-10 else 0.0
    
    r_sq = var_predicted / var_actual
    return np.clip(r_sq, 0.0, 1.0)
```

---

## 6. Configuration Parameters

### 6.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_dims` | Tuple[int, int] | Spatial grid dimensions |
| `active_taxis_path` | str | Path to active_taxis output file |

### 6.2 Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neighborhood_size` | int | 2 | $k$ for $(2k+1) \times (2k+1)$ aggregation (must match active_taxis config) |
| `estimation_method` | str | TBD | Method for estimating g(d) — to be determined from visualization |
| `n_bins` | int | 10 | Number of bins for binning method |
| `poly_degree` | int | 2 | Degree for polynomial regression |
| `min_demand` | int | 1 | Minimum demand to include cell |
| `period_type` | str | "hourly" | Temporal aggregation granularity |
| `weight` | float | 0.33 | Weight in objective function ($\alpha_1$) |

### 6.3 Default Values and Rationale

```python
DEFAULT_CONFIG = CausalFairnessConfig(
    # Spatial configuration
    grid_dims=(48, 90),       # Shenzhen grid
    neighborhood_size=2,      # 5×5 window (matches active_taxis default)
    
    # g(d) estimation — to be finalized after visualization
    estimation_method="binning",  # Placeholder; will update after analyzing D vs Y plot
    n_bins=10,                    # Sufficient resolution
    poly_degree=2,                # If using polynomial
    
    # Filtering
    min_demand=1,             # Include all cells with any demand
    
    # Aggregation
    period_type="hourly",     # Matches typical active_taxis output
    weight=0.33,              # Equal weight with spatial and fidelity terms
)
```

**Rationale**:

- `neighborhood_size=2`: 5×5 window is the project standard; must match the active_taxis configuration used to generate supply data
- `estimation_method="binning"`: Default placeholder; **the final choice will be made after visualizing the demand vs. service ratio relationship** (see Section 2.2.4.1)
- `period_type="hourly"`: Hourly aggregation balances temporal granularity with computational efficiency and matches recommended active_taxis output
- `n_bins=10`: Provides granularity while maintaining stable estimates per bin

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### 7.1.1 g(d) Estimation Tests

```python
class TestGEstimation:
    """Test g(d) estimation methods."""
    
    def test_binning_basic(self):
        """Test binning estimation."""
        demands = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ratios = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Inverse relationship
        
        term = CausalFairnessTerm(CausalFairnessConfig(n_bins=5))
        g = term._estimate_binning(demands, ratios)
        
        # g should capture decreasing trend
        assert g(1) > g(10)
    
    def test_constant_ratio(self):
        """Constant Y → g(d) = constant."""
        demands = np.array([1, 2, 3, 4, 5])
        ratios = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        
        term = CausalFairnessTerm(CausalFairnessConfig())
        g = term._estimate_binning(demands, ratios)
        
        assert abs(g(1) - 5.0) < 0.1
        assert abs(g(5) - 5.0) < 0.1
    
    def test_linear_relationship(self):
        """Linear Y = D → g should capture this."""
        demands = np.array([1, 2, 3, 4, 5])
        ratios = demands.astype(float)  # Y = D
        
        term = CausalFairnessTerm(CausalFairnessConfig(estimation_method="polynomial"))
        g = term._estimate_polynomial(demands, ratios)
        
        # Should predict accurately
        assert abs(g(3) - 3.0) < 0.5
```

#### 7.1.2 Causal Fairness Term Tests

```python
class TestCausalFairnessTerm:
    """Test causal fairness term computation."""
    
    @pytest.fixture
    def config(self):
        return CausalFairnessConfig(
            grid_dims=(4, 4),
            neighborhood_size=0,  # No aggregation for testing
            period_type="daily"
        )
    
    def test_perfect_fairness(self, config):
        """Y perfectly explained by D → F_causal = 1.0."""
        # Create data where Y = D (perfect demand-supply match)
        demand = {}
        supply = {}
        
        for x in range(4):
            for y in range(4):
                d = (x + 1) * (y + 1)  # Varying demand
                demand[(x, y, 1, 1)] = d
                supply[(x, y, 1, 1)] = d * 2  # Supply = 2 * Demand (constant ratio)
        
        term = CausalFairnessTerm(config)
        result = term._compute_r_squared(demand, supply)
        
        # All variance explained → R² = 1
        assert abs(result - 1.0) < 0.05
    
    def test_no_relationship(self, config):
        """Y independent of D → F_causal ≈ 0."""
        # Create data where Y is random (no relationship to D)
        np.random.seed(42)
        
        demand = {}
        supply = {}
        
        for x in range(4):
            for y in range(4):
                demand[(x, y, 1, 1)] = (x + 1) * (y + 1)
                supply[(x, y, 1, 1)] = np.random.randint(1, 100)
        
        term = CausalFairnessTerm(config)
        result = term._compute_r_squared(demand, supply)
        
        # Very little variance explained
        assert result < 0.5
    
    def test_output_range(self, config):
        """Output always in [0, 1]."""
        np.random.seed(42)
        
        demand = {
            (x, y, 1, 1): np.random.randint(1, 50)
            for x in range(4) for y in range(4)
        }
        supply = {
            (x, y, 1, 1): np.random.randint(1, 100)
            for x in range(4) for y in range(4)
        }
        
        term = CausalFairnessTerm(config)
        result = term._compute_r_squared(demand, supply)
        
        assert 0.0 <= result <= 1.0
```

### 7.2 Integration Tests

```python
class TestCausalFairnessIntegration:
    """Integration tests with real data."""
    
    def test_with_real_data(self):
        """Test with actual FAMAIL datasets."""
        import pickle
        
        # Load demand data
        with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
            pickup_data = pickle.load(f)
        
        # Load supply data from active_taxis (5x5 neighborhood, hourly)
        with open('active_taxis/output/active_taxis_5x5_hourly.pkl', 'rb') as f:
            active_taxis_data = pickle.load(f)
        
        # Extract counts from active_taxis output (may be tuple with metadata)
        if isinstance(active_taxis_data, tuple):
            supply = active_taxis_data[0]
        else:
            supply = active_taxis_data
        
        # Aggregate demand to hourly to match supply resolution
        from collections import defaultdict
        hourly_demand = defaultdict(int)
        for (x, y, time_bucket, day), val in pickup_data.items():
            hour = (time_bucket - 1) // 12
            hourly_demand[(x, y, hour, day)] += val[0]
        demand = dict(hourly_demand)
        
        config = CausalFairnessConfig(
            neighborhood_size=2,  # 5×5, matching active_taxis config
            period_type="hourly"
        )
        term = CausalFairnessTerm(config)
        
        result = term._compute_r_squared(demand, supply)
        
        # Should be valid
        assert 0.0 <= result <= 1.0
        
        # Real data should show some relationship (not zero)
        assert result > 0.1
        
        # But not perfect (not 1.0)
        assert result < 0.95
```

### 7.3 Validation with Real Data

**Expected Behavior**:

1. **Moderate R²**: Real data should show 0.3-0.7 (some but not perfect demand-supply alignment)
2. **Temporal variation**: R² may vary by time of day
3. **Improvement potential**: Edited trajectories should show higher R²

---

## 8. Expected Challenges

### 8.1 Known Difficulties

#### 8.1.1 Sparse Overlap

**Challenge**: Demand and supply data may have different coverage.

**Impact**: Missing values for one dataset reduce analyzable cells.

**Mitigation**:
- Document coverage statistics
- Use intersection of available keys
- Consider imputation for missing supply values

#### 8.1.2 Extreme Ratios

**Challenge**: Very low demand cells have unstable ratios.

**Impact**: $Y = S/D$ can be very large when $D$ is small.

**Mitigation**:
- `min_demand` threshold
- Robust statistics (median instead of mean in g estimation)
- Winsorization of extreme ratios

#### 8.1.3 Non-Linear Relationships

**Challenge**: True relationship between D and Y may be complex.

**Impact**: Simple g estimation may not capture true pattern.

**Mitigation**:
- Multiple estimation methods available
- Increase number of bins
- Visual validation of g curve

### 8.2 Mitigation Strategies

| Challenge | Strategy | Implementation |
|-----------|----------|----------------|
| Sparse overlap | Coverage statistics | Log overlap percentage in diagnostics |
| Extreme ratios | Filtering & robust stats | `min_demand` parameter, median option |
| Non-linearity | Flexible estimation | Multiple `estimation_method` options |
| Numerical issues | Safe division | `safe_r_squared()` function |

---

## 9. Development Milestones

### 9.1 Phase 1: Core Implementation (Week 1-2)

- [ ] **M1.1**: Set up directory structure
- [ ] **M1.2**: Implement `CausalFairnessConfig` dataclass
- [ ] **M1.3**: Implement neighborhood aggregation
- [ ] **M1.4**: Implement g(d) estimation (binning)
- [ ] **M1.5**: Implement R² computation
- [ ] **M1.6**: Implement main `compute()` method

**Deliverables**:
- Working `CausalFairnessTerm` class
- Basic unit tests passing

### 9.2 Phase 2: Testing and Validation (Week 2-3)

- [ ] **M2.1**: Complete unit test suite
- [ ] **M2.2**: Integration tests with real data
- [ ] **M2.3**: Implement alternative g estimation methods
- [ ] **M2.4**: Implement `compute_with_breakdown()`
- [ ] **M2.5**: Validate interpretation with known scenarios

**Deliverables**:
- >90% test coverage
- Validation report

### 9.3 Phase 3: Integration (Week 3-4)

- [ ] **M3.1**: Integrate with base interface
- [ ] **M3.2**: Test gradient computation (if differentiable)
- [ ] **M3.3**: Test with combined objective function
- [ ] **M3.4**: Documentation and code review
- [ ] **M3.5**: Performance optimization

**Deliverables**:
- Integration-ready module
- Complete documentation

---

## 10. Appendix

### 10.1 Code Snippets

#### 10.1.1 Complete g(d) Estimation Suite

```python
from typing import Callable, Literal
import numpy as np
from scipy.stats import binned_statistic
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

def estimate_g_function(
    demands: np.ndarray,
    ratios: np.ndarray,
    method: Literal["binning", "linear", "polynomial", "isotonic"] = "binning",
    n_bins: int = 10,
    poly_degree: int = 2
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Estimate g(d) = E[Y | D = d] using specified method.
    
    Args:
        demands: Array of demand values
        ratios: Array of service ratios
        method: Estimation method
        n_bins: Number of bins for binning
        poly_degree: Degree for polynomial
    
    Returns:
        Function g(d) that predicts expected ratio given demand
    """
    if method == "binning":
        # Quantile-based binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.unique(np.percentile(demands, percentiles))
        
        bin_means, _, _ = binned_statistic(
            demands, ratios, statistic='mean', bins=bin_edges
        )
        
        def g(d):
            d_arr = np.atleast_1d(d).astype(float)
            idx = np.digitize(d_arr, bin_edges) - 1
            idx = np.clip(idx, 0, len(bin_means) - 1)
            result = np.where(
                np.isnan(bin_means[idx]),
                0.0,
                bin_means[idx]
            )
            return result
        
        return g
    
    elif method == "linear":
        model = LinearRegression()
        model.fit(demands.reshape(-1, 1), ratios)
        
        def g(d):
            return model.predict(np.atleast_1d(d).reshape(-1, 1))
        
        return g
    
    elif method == "polynomial":
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(demands.reshape(-1, 1))
        model = LinearRegression()
        model.fit(X_poly, ratios)
        
        def g(d):
            d_arr = np.atleast_1d(d).reshape(-1, 1)
            return model.predict(poly.transform(d_arr))
        
        return g
    
    elif method == "isotonic":
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(demands, ratios)
        
        def g(d):
            return model.predict(np.atleast_1d(d))
        
        return g
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 10.2 Sample Data

```python
# Expected data format for causal fairness
sample_demand = {
    # High demand area (downtown) - hourly aggregation
    (24, 45, 12, 1): 50,  # 50 pickup requests at hour 12, Monday
    (24, 46, 12, 1): 45,
    
    # Medium demand area
    (10, 30, 12, 1): 15,
    
    # Low demand area
    (2, 85, 12, 1): 2,
}

sample_supply = {
    # Corresponding supply from active_taxis (5×5 neighborhood counts)
    (24, 45, 12, 1): 25,  # 25 active taxis, Ratio: 0.5
    (24, 46, 12, 1): 23,  # 23 active taxis, Ratio: ~0.5
    
    (10, 30, 12, 1): 8,   # 8 active taxis, Ratio: ~0.5 (fair: same ratio)
    
    (2, 85, 12, 1): 1,    # 1 active taxi, Ratio: 0.5 (fair in this example)
}

# In a fair system: all ratios should be similar given similar demand
# If the low-demand area received fewer taxis per demand unit, that would be unfair
```

### 10.3 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial development plan |
| 1.1.0 | 2026-01-12 | Updated supply data source from `latest_volume_pickups.pkl` to `active_taxis` output; changed default neighborhood to 5×5 (k=2); added visualization-first approach for g(d) estimation method selection |
| 1.3.0 | 2026-01 | Added Soft Cell Assignment (Section 2.5), full differentiable pipeline, g(d) estimation benchmark results, DCD attribution; status updated to "Implementation Complete" |

---

*This document serves as the comprehensive development guide for the Causal Fairness term. All implementation should follow the specifications outlined here.*
