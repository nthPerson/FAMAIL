# FAMAIL Objective Function: Comprehensive Implementation Specification

## Document Purpose

This document provides a complete technical specification for implementing the FAMAIL (Fairness-Aware Multi-Agent Imitation Learning) trajectory editing objective function. It is designed to guide coding agents and developers through the implementation of each component, including mathematical formulations, data requirements, algorithms, and code structure.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Objective Function Overview](#2-objective-function-overview)
3. [Spatial Fairness Term ($F_{\text{spatial}}$)](#3-spatial-fairness-term-f_spatial)
4. [Causal Fairness Term ($F_{\text{causal}}$)](#4-causal-fairness-term-f_causal)
5. [Trajectory Fidelity Term ($F_{\text{fidelity}}$)](#5-trajectory-fidelity-term-f_fidelity)
6. [Trajectory Modification Algorithm](#6-trajectory-modification-algorithm)
7. [Constraints](#7-constraints)
8. [Data Sources and Structure](#8-data-sources-and-structure)
9. [Implementation Architecture](#9-implementation-architecture)
10. [Testing and Validation](#10-testing-and-validation)
11. [References](#11-references)

---

## 1. Project Overview

### 1.1 Research Context

The FAMAIL project addresses fairness in urban taxi services by developing trajectory editing techniques that can modify expert driver trajectories to improve fairness metrics while maintaining trajectory authenticity. The edited trajectories are then used to train imitation learning models that produce fairer driver policies.

### 1.2 Problem Statement

Taxi services in cities like Shenzhen exhibit spatial inequality—certain areas receive disproportionately more or less service relative to demand. This inequality may correlate with socioeconomic factors (e.g., income levels of neighborhoods). FAMAIL aims to:

1. **Quantify unfairness** in taxi service distribution using spatial and causal fairness metrics
2. **Edit expert trajectories** to reduce unfairness while preserving trajectory authenticity
3. **Train fairer policies** using the edited trajectories in an imitation learning framework

### 1.3 Study Area and Data Context

- **Geographic Area**: Shenzhen, China
- **Time Period**: July 2016 (with supplementary data from August-September 2016)
- **Spatial Resolution**: 48×90 grid (approximately 0.01° × 0.01° per cell)
- **Temporal Resolution**: 288 time buckets per day (5-minute intervals)
- **Fleet Size**: 50 expert drivers (subset of 17,877 total taxis)

---

## 2. Objective Function Overview

### 2.1 Weighted Multi-Objective Formulation

The FAMAIL objective function is formulated as a weighted sum of fairness and fidelity components:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}}
$$

Where:
- $\alpha_i \in \mathbb{R}$ are tunable weights (hyperparameters)
- Each $F_*$ term is computed over the trajectory set $\mathcal{T}' = \{\tau'_1, \tau'_2, ..., \tau'_N\}$

### 2.2 Optimization Direction

**Important**: Following the ST-iFGSM framework, the FAMAIL objective function is designed to be **maximized**. Each fairness term is formulated such that:
- **Higher values = more fair/better**
- The optimization problem is to **maximize** $\mathcal{L}$

$$
\max_{\mathcal{T}'} \mathcal{L} = \max_{\mathcal{T}'} \left( \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} \right)
$$

All $\alpha_i$ coefficients are positive, and each $F_*$ term is defined so that higher values represent better outcomes (greater fairness, higher fidelity).

### 2.3 Subject to Constraints

The optimization is subject to the following constraints (detailed in Section 7):

1. **Subtle Edits Constraint**: $\|\tau' - \tau\|_\infty \leq \epsilon \quad \forall \tau \in \mathcal{T}$
2. **Limited Modifications Constraint**: $\|\tau' - \tau\|_0 \leq \eta \quad \forall \tau \in \mathcal{T}$
3. **Dataset-Level Constraint**: $\|\mathcal{T}' - \mathcal{T}\|_0 \leq \zeta$
4. **Authenticity Constraint**: $\text{Discriminator\_confidence}(\tau') \geq \theta$

---

## 3. Spatial Fairness Term ($F_{\text{spatial}}$)

### 3.1 Conceptual Foundation

The spatial fairness term measures equality in taxi service distribution across geographic regions. It is based on the work of Su et al. (2018) who studied spatial inequality in taxi services using service rates and Gini coefficients.

**Key Insight**: Perfect spatial fairness ($F_{\text{spatial}} = 1$) means all grid cells receive equal taxi service rates. Lower values (approaching 0) indicate greater inequality in service distribution.

### 3.2 Mathematical Formulation

#### 3.2.1 Service Rate Definitions

For each grid cell $i$ and time period $p$:

**Arrival Service Rate (ASR)** - measures drop-off frequency:
$$
ASR_i^p = \frac{D_i^p}{N^p \cdot T^p}
$$

**Departure Service Rate (DSR)** - measures pickup frequency:
$$
DSR_i^p = \frac{O_i^p}{N^p \cdot T^p}
$$

Where:
- $D_i^p$ = number of drop-offs (trip destinations/arrivals) in cell $i$ during period $p$
- $O_i^p$ = number of pickups (trip origins/departures) in cell $i$ during period $p$
- $N^p$ = number of active taxis during period $p$
- $T^p$ = number of days (or temporal units) in period $p$

#### 3.2.2 Gini Coefficient Calculation

The Gini coefficient quantifies inequality in a distribution:

$$
G = 1 + \frac{1}{n} - \frac{2}{n^2 \bar{x}} \sum_{i=1}^{n} (n - i + 1) \cdot x_{(i)}
$$

Where:
- $x_{(i)}$ = the $i$-th smallest value in the sorted list $\{x_1, x_2, ..., x_n\}$
- $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$ = mean of all values
- $n$ = number of grid cells

**Properties**:
- $G = 0$: Perfect equality (all cells have identical service rates)
- $G = 1$: Maximum inequality (all service concentrated in one cell)

#### 3.2.3 Per-Period Spatial Fairness

For each time period $p$, compute separate Gini coefficients for arrivals and departures:

$$
G_a^p = \text{Gini}(\{ASR_i^p : i \in \mathcal{I}\})
$$
$$
G_d^p = \text{Gini}(\{DSR_i^p : i \in \mathcal{I}\})
$$

Where $\mathcal{I}$ is the set of all valid grid cells.

Since the Gini coefficient measures inequality (where $G=0$ means perfect equality and $G=1$ means maximum inequality), we define the per-period spatial fairness score as the complement:
$$
F_{\text{spatial}}^p = 1 - \frac{1}{2}(G_a^p + G_d^p)
$$

This formulation ensures that **higher values indicate greater fairness** ($F_{\text{spatial}}^p = 1$ means perfect equality, $F_{\text{spatial}}^p = 0$ means maximum inequality).

#### 3.2.4 Aggregated Spatial Fairness Term

The final spatial fairness term aggregates across all time periods:

$$
F_{\text{spatial}} = \frac{1}{|P|} \sum_{p \in P} F_{\text{spatial}}^p = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)
$$

This formulation yields a **fairness score** where:
- $F_{\text{spatial}} = 1$ indicates perfect spatial equality across all periods
- $F_{\text{spatial}} = 0$ indicates maximum spatial inequality
- **Higher values = more fair**

The term can be maximized directly in the objective function.

### 3.3 Implementation Algorithm

```
Algorithm: Compute Spatial Fairness Term
────────────────────────────────────────────────────────────────────────

Input:
  - trajectories: Dict[driver_id → List[trajectory]]
  - grid_dims: Tuple[int, int] = (48, 90)  # (x_grid, y_grid)
  - period_definition: str ∈ {"hourly", "daily", "weekly", "monthly"}
  - num_taxis: int

Output:
  - F_spatial: float (fairness score ∈ [0, 1], higher = more fair)

Procedure:
  1. Initialize counts:
     pickup_counts[period][cell] = 0
     dropoff_counts[period][cell] = 0

  2. For each driver in trajectories:
       For each trajectory of driver:
         Extract pickup events (passenger_indicator: 0 → 1)
         Extract dropoff events (passenger_indicator: 1 → 0)
         
         For each pickup at (x, y, t):
           period = assign_period(t, period_definition)
           cell = (x, y)
           pickup_counts[period][cell] += 1
         
         For each dropoff at (x, y, t):
           period = assign_period(t, period_definition)
           cell = (x, y)
           dropoff_counts[period][cell] += 1

  3. For each period p in observed_periods:
       T_p = compute_duration(p, period_definition)
       N_p = num_taxis (or count active taxis in period)
       
       ASR_list = []
       DSR_list = []
       
       For each cell in all_cells:
         ASR_i = dropoff_counts[p][cell] / (N_p * T_p)
         DSR_i = pickup_counts[p][cell] / (N_p * T_p)
         ASR_list.append(ASR_i)
         DSR_list.append(DSR_i)
       
       G_a[p] = compute_gini(ASR_list)
       G_d[p] = compute_gini(DSR_list)

  4. F_spatial = 1 - (1 / (2 * |P|)) * Σ_p (G_a[p] + G_d[p])
  
  5. Return F_spatial
```

### 3.4 Python Implementation Reference

```python
def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a distribution of values.
    
    Args:
        values: Array of non-negative values (e.g., service rates)
    
    Returns:
        Gini coefficient in [0, 1]
    """
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    mean_value = np.mean(sorted_values)
    
    # Compute weighted sum
    weights = np.arange(n, 0, -1)  # [n, n-1, ..., 2, 1]
    weighted_sum = np.sum(weights * sorted_values)
    
    gini = 1 + (1/n) - (2 / (n**2 * mean_value)) * weighted_sum
    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


def compute_spatial_fairness(
    pickup_counts: Dict[Tuple, int],
    dropoff_counts: Dict[Tuple, int],
    grid_dims: Tuple[int, int],
    num_taxis: int,
    num_days: float
) -> float:
    """
    Compute the spatial fairness term (Gini-based).
    
    Args:
        pickup_counts: Dict[(x, y, period)] → count
        dropoff_counts: Dict[(x, y, period)] → count
        grid_dims: (n_x, n_y) grid dimensions
        num_taxis: Number of active taxis
        num_days: Number of days in the period
    
    Returns:
        F_spatial: Fairness score in [0, 1] (higher = more fair)
    """
    # Extract unique periods
    periods = set()
    for key in pickup_counts.keys():
        periods.add(key[2] if len(key) > 2 else 0)
    
    gini_arrivals = []
    gini_departures = []
    
    for period in periods:
        # Compute ASR and DSR for each cell
        asr_values = []
        dsr_values = []
        
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                dropoffs = dropoff_counts.get((x, y, period), 0)
                pickups = pickup_counts.get((x, y, period), 0)
                
                asr = dropoffs / (num_taxis * num_days)
                dsr = pickups / (num_taxis * num_days)
                
                asr_values.append(asr)
                dsr_values.append(dsr)
        
        gini_arrivals.append(compute_gini_coefficient(np.array(asr_values)))
        gini_departures.append(compute_gini_coefficient(np.array(dsr_values)))
    
    # Aggregate across periods (complement of average Gini)
    avg_gini = 0.5 * (np.mean(gini_arrivals) + np.mean(gini_departures))
    F_spatial = 1.0 - avg_gini
    return F_spatial
```

### 3.5 Data Requirements

| Data Source | Fields Used | Purpose |
|-------------|-------------|---------|
| `all_trajs.pkl` | `x_grid`, `y_grid`, `time_bucket`, `action_code` | Extract trajectory points |
| `pickup_dropoff_counts.pkl` | `(x, y, time, day)` → `[pickups, dropoffs]` | Pre-computed event counts |

### 3.6 Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `period_definition` | Temporal granularity for aggregation | "hourly", "daily", "monthly" |
| `grid_dims` | Spatial grid dimensions | (48, 90) |
| `include_zero_cells` | Whether to include cells with zero activity | True/False |

### 3.7 Differentiable Formulation

> **IMPLEMENTED**: The following differentiable implementation enables gradient-based trajectory optimization.

#### 3.7.1 Pairwise Gini with Soft Counts

The differentiable spatial fairness term uses **pairwise Gini** formulation:

$$
G_{\text{pairwise}} = \frac{\sum_{i}\sum_{j}|x_i - x_j|}{2n^2\bar{x}}
$$

This formulation avoids the non-differentiable sorting operation in the standard Gini formula.

#### 3.7.2 Soft Count Aggregation

Instead of hard assignment to cells, trajectory points contribute soft counts:

$$
\text{soft\_count}_c = \sum_{i \in \text{events}} \sigma_c(x_i, y_i)
$$

Where $\sigma_c(x, y)$ is the soft cell assignment probability (see Section 6.5.1).

#### 3.7.3 PyTorch Implementation

```python
class DifferentiableSpatialFairnessWithSoftCounts(nn.Module):
    """Fully differentiable spatial fairness with soft cell assignment."""
    
    def __init__(self, grid_shape=(48, 90), neighborhood_size=5, temperature=1.0):
        super().__init__()
        self.soft_assign = SoftCellAssignment(
            grid_shape=grid_shape,
            neighborhood_size=neighborhood_size,
            temperature=temperature
        )
        self.grid_shape = grid_shape
    
    def forward(self, pickup_coords, dropoff_coords):
        """
        Compute differentiable spatial fairness from trajectory coordinates.
        
        Args:
            pickup_coords: Tensor of shape (N, 2) - pickup (x, y) coordinates
            dropoff_coords: Tensor of shape (M, 2) - dropoff (x, y) coordinates
        
        Returns:
            F_spatial: Scalar fairness score ∈ [0, 1] (higher = more fair)
        """
        # Compute soft counts (differentiable)
        pickup_soft_counts = self.soft_assign(pickup_coords)   # (48, 90)
        dropoff_soft_counts = self.soft_assign(dropoff_coords) # (48, 90)
        
        # Flatten for Gini computation
        dsr = pickup_soft_counts.flatten()  # Departure service rates
        asr = dropoff_soft_counts.flatten() # Arrival service rates
        
        # Pairwise Gini (differentiable)
        gini_d = self._pairwise_gini(dsr)
        gini_a = self._pairwise_gini(asr)
        
        # Fairness = 1 - average Gini
        F_spatial = 1.0 - 0.5 * (gini_d + gini_a)
        return F_spatial
    
    def _pairwise_gini(self, values):
        """Compute Gini coefficient using pairwise formula (differentiable)."""
        n = values.numel()
        if n <= 1:
            return torch.tensor(0.0, device=values.device)
        
        mean_val = values.mean() + 1e-8
        diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
        gini = diff_matrix.sum() / (2 * n * n * mean_val)
        return gini
```

**Implementation Location**: `objective_function/spatial_fairness/utils.py`

#### 3.7.4 Local Inequality Score (LIS) Attribution

For trajectory-level attribution, compute each trajectory's contribution to inequality:

$$
\text{LIS}_\tau = \sum_{c \in \mathcal{C}} \sigma_c(\tau) \cdot |x_c - \bar{x}|
$$

Where $\sigma_c(\tau)$ is the soft contribution of trajectory $\tau$ to cell $c$.

**Implementation**: `objective_function/spatial_fairness/utils.py::compute_local_inequality_score()`

---

## 4. Causal Fairness Term ($F_{\text{causal}}$)

### 4.1 Conceptual Foundation

The causal fairness term quantifies how well service supply matches demand across different spatial/temporal contexts. The key insight is that in a fair system, service differences should be explainable by **demand** alone, not by other contextual factors (e.g., neighborhood characteristics, time of day).

**Causal Model**:
- Let $Y$ = service supply-demand ratio (outcome)
- Let $D$ = demand (legitimate explanatory factor)
- Let $C$ = spatial/temporal context (potentially unfair factor)

The causal fairness term measures the proportion of service variation explained by demand:
$$
Y_{i,p} = \underbrace{g(D_{i,p})}_{\text{demand-explained component}} + \underbrace{(Y_{i,p} - g(D_{i,p}))}_{\text{unexplained residual}}
$$

**Higher causal fairness** means that demand better explains service distribution (less contextual bias).

### 4.2 Mathematical Formulation

#### 4.2.1 Supply-Demand Definitions

For each grid cell $i$ and time period $p$:

**Demand**:
$$
D_{i,p} = \text{number\_of\_pickups}_{i,p}
$$

**Supply** (with neighborhood aggregation):
$$
S_{i,p} = \sum_{j \in \mathcal{N}_k(i)} \text{traffic\_volume}_{j,p}
$$

Where $\mathcal{N}_k(i)$ is the $k \times k$ neighborhood centered on cell $i$.

**Service Supply-Demand Ratio**:
$$
Y_{i,p} = \frac{S_{i,p}}{D_{i,p}} \quad \text{for } D_{i,p} > 0
$$

#### 4.2.2 Expected Service Function

The function $g(d) = \mathbb{E}[Y \mid D = d]$ represents the expected service ratio given only the demand level. This can be estimated via:

1. **Binning Approach**: Group observations by demand bins and compute mean $Y$ per bin
2. **Regression Approach**: Fit $Y \sim f(D)$ using linear, polynomial, or non-parametric regression
3. **Smoothing Approach**: Use LOESS, splines, or kernel smoothing

#### 4.2.3 Unfair Residual

The unfair residual for each cell-period:
$$
R_{i,p}^{\text{(unfair)}} = Y_{i,p} - g(D_{i,p})
$$

#### 4.2.4 Per-Period Causal Fairness

For each time period $p$, we use a variance-based formulation (coefficient of determination or $R^2$):
$$
F_{\text{causal}}^p = \frac{\text{Var}_p(g(D_{i,p}))}{\text{Var}_p(Y_{i,p})} = 1 - \frac{\text{Var}_p(R_{i,p})}{\text{Var}_p(Y_{i,p})}
$$

Where:
- $\mathcal{I}_p = \{i : D_{i,p} > 0\}$ is the set of cells with positive demand in period $p$
- $R_{i,p} = Y_{i,p} - g(D_{i,p})$ is the residual (unexplained component)
- $\text{Var}_p(\cdot)$ denotes variance across cells in period $p$

This represents the **proportion of variance in service explained by demand alone**:
- $F_{\text{causal}}^p = 1$: Service perfectly matches demand (perfectly fair)
- $F_{\text{causal}}^p = 0$: Service is independent of demand (maximally unfair)
- **Higher values = more fair** (demand better explains service distribution)

#### 4.2.5 Aggregated Causal Fairness Term

$$
F_{\text{causal}} = \frac{1}{|P|} \sum_{p \in P} F_{\text{causal}}^p
$$

### 4.3 Implementation Algorithm

```
Algorithm: Compute Causal Fairness Term
────────────────────────────────────────────────────────────────────────

Input:
  - pickup_counts: Dict[(x, y, period)] → int
  - traffic_volume: Dict[(x, y, period)] → int
  - neighborhood_size: int = 1 (for k×k = 3×3 neighborhood)
  - estimation_method: str ∈ {"binning", "regression", "loess"}

Output:
  - F_causal: float (fairness score ∈ [0, 1], higher = more fair)

Procedure:
  1. DATA PREPARATION
     For each cell i and period p where pickup_counts[(i, p)] > 0:
       D_ip = pickup_counts[(i, p)]
       S_ip = sum(traffic_volume[(j, p)] for j in neighborhood(i))
       Y_ip = S_ip / D_ip
       Store (D_ip, Y_ip, period=p)

  2. ESTIMATE g(d)
     Collect all (D, Y) pairs across cells and periods
     
     If estimation_method == "binning":
       Create bins: [1], [2-5], [6-10], [11-20], [21+]
       For each bin b:
         g_values[b] = mean(Y for (D, Y) if D in bin b)
       g(d) = lookup bin containing d → return mean
     
     Elif estimation_method == "regression":
       Fit model: Y ~ β₀ + β₁·D + β₂·D²  (or other form)
       g(d) = model.predict(d)
     
     Elif estimation_method == "loess":
       Fit LOESS smoother to (D, Y) pairs
       g(d) = loess.predict(d)

  3. COMPUTE RESIDUALS
     For each (D_ip, Y_ip, p):
       R_ip = Y_ip - g(D_ip)

  4. COMPUTE PER-PERIOD CAUSAL FAIRNESS (R²)
     For each period p:
       I_p = cells with D > 0 in period p
       Compute Var_p(Y) = variance of Y_ip over cells in I_p
       Compute Var_p(g(D)) = variance of g(D_ip) over cells in I_p
       F_causal_p = Var_p(g(D)) / Var_p(Y)
       
       // Alternative: F_causal_p = 1 - (Var_p(R) / Var_p(Y))

  5. AGGREGATE
     F_causal = (1/|P|) * Σ_p F_causal_p

  6. Return F_causal
```

### 4.4 Python Implementation Reference

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import binned_statistic
import numpy as np
from typing import Dict, Tuple, List, Literal

def estimate_g_function(
    demands: np.ndarray,
    service_ratios: np.ndarray,
    method: Literal["binning", "regression", "polynomial"] = "binning",
    n_bins: int = 10,
    poly_degree: int = 2
) -> callable:
    """
    Estimate g(d) = E[Y | D = d] using specified method.
    
    Args:
        demands: Array of demand values D
        service_ratios: Array of service ratios Y
        method: Estimation method
        n_bins: Number of bins for binning method
        poly_degree: Polynomial degree for regression
    
    Returns:
        Function g(d) that predicts expected Y given D
    """
    if method == "binning":
        # Create bins based on demand quantiles
        bin_edges = np.percentile(demands, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # Remove duplicates
        
        bin_means, _, bin_indices = binned_statistic(
            demands, service_ratios, statistic='mean', bins=bin_edges
        )
        
        def g(d):
            idx = np.digitize(d, bin_edges) - 1
            idx = np.clip(idx, 0, len(bin_means) - 1)
            return bin_means[idx] if not np.isnan(bin_means[idx]) else 0
        
        return np.vectorize(g)
    
    elif method == "polynomial":
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(demands.reshape(-1, 1))
        model = LinearRegression().fit(X_poly, service_ratios)
        
        def g(d):
            d_array = np.atleast_1d(d).reshape(-1, 1)
            return model.predict(poly.transform(d_array))
        
        return g
    
    else:  # Linear regression
        model = LinearRegression().fit(demands.reshape(-1, 1), service_ratios)
        
        def g(d):
            return model.predict(np.atleast_1d(d).reshape(-1, 1))
        
        return g


def compute_causal_fairness(
    pickup_counts: Dict[Tuple, int],
    traffic_volume: Dict[Tuple, int],
    grid_dims: Tuple[int, int],
    neighborhood_size: int = 1,
    g_estimation_method: str = "binning"
) -> float:
    """
    Compute the causal fairness term (R² - proportion of variance explained by demand).
    
    Args:
        pickup_counts: Dict[(x, y, time, day)] → pickup count
        traffic_volume: Dict[(x, y, time, day)] → traffic volume
        grid_dims: (n_x, n_y) grid dimensions
        neighborhood_size: k for (2k+1)×(2k+1) neighborhood
        g_estimation_method: Method for estimating g(d)
    
    Returns:
        F_causal: Causal fairness score ∈ [0, 1] (higher = more fair)
    """
    # Collect all (D, Y, period) observations
    observations = []
    
    for key, demand in pickup_counts.items():
        if demand <= 0:
            continue
        
        x, y = key[0], key[1]
        period = (key[2], key[3]) if len(key) >= 4 else key[2]
        
        # Compute supply from neighborhood
        supply = 0
        for dx in range(-neighborhood_size, neighborhood_size + 1):
            for dy in range(-neighborhood_size, neighborhood_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_dims[0] and 0 <= ny < grid_dims[1]:
                    neighbor_key = (nx, ny) + key[2:]
                    supply += traffic_volume.get(neighbor_key, 0)
        
        if supply > 0:
            Y = supply / demand
            observations.append((demand, Y, period))
    
    if len(observations) == 0:
        return 0.0
    
    # Extract arrays
    demands = np.array([obs[0] for obs in observations])
    service_ratios = np.array([obs[1] for obs in observations])
    periods = [obs[2] for obs in observations]
    
    # Estimate g(d)
    g = estimate_g_function(demands, service_ratios, method=g_estimation_method)
    
    # Compute predicted values
    expected_Y = g(demands)
    
    # Group by period and compute per-period R²
    unique_periods = list(set(periods))
    F_causal_periods = []
    
    for p in unique_periods:
        mask = np.array([periods[i] == p for i in range(len(periods))])
        period_Y = service_ratios[mask]
        period_expected_Y = expected_Y[mask]
        
        if len(period_Y) > 1:
            # Compute R² = Var(predicted) / Var(actual)
            var_Y = np.var(period_Y)
            if var_Y > 0:
                var_predicted = np.var(period_expected_Y)
                r_squared = var_predicted / var_Y
                F_causal_periods.append(np.clip(r_squared, 0.0, 1.0))
    
    # Aggregate across periods
    F_causal = np.mean(F_causal_periods) if F_causal_periods else 0.0
    return F_causal
```

### 4.5 Data Requirements

| Data Source | Fields Used | Purpose |
|-------------|-------------|---------|
| `pickup_dropoff_counts.pkl` | `(x, y, time, day)` → `[pickups, dropoffs]` | Demand ($D_{i,p}$) |
| `latest_volume_pickups.pkl` | `(x, y, time, day)` → `[pickups, volume]` | Traffic volume for supply |

### 4.6 Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `neighborhood_size` | $k$ for $(2k+1) \times (2k+1)$ neighborhood | 1 (3×3), 2 (5×5) |
| `g_estimation_method` | Method for estimating $g(d)$ | "binning", "regression", "polynomial" |
| `n_bins` | Number of bins for binning method | 5, 10, 20 |
| `poly_degree` | Polynomial degree for regression | 1, 2, 3 |
| `min_demand_threshold` | Minimum demand to include cell | 1, 5 |

### 4.7 Validation

To validate the causal fairness term:

1. **Sanity Check 1**: If supply is allocated exactly proportional to demand, $F_{\text{causal}} \approx 1$ (perfect fairness)
2. **Sanity Check 2**: If supply is randomly distributed regardless of demand, $F_{\text{causal}} \approx 0$ (no relationship)
3. **Sanity Check 3**: The term should increase when trajectories are edited to better match supply to demand
4. **Bounds Check**: $F_{\text{causal}} \in [0, 1]$ always (it's an $R^2$ value)

### 4.8 Differentiable Formulation

> **IMPLEMENTED**: The following differentiable implementation enables gradient-based trajectory optimization.

#### 4.8.1 Frozen g(d) Design

The expected service function $g(d) = \mathbb{E}[Y \mid D = d]$ is **frozen** during optimization:

$$
\mathcal{L}_{\text{causal}} = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)} \quad \text{where } R = Y - g(D)
$$

**Critical Design Decision**: $g(d)$ is computed from the **original** (unmodified) demand distribution and remains fixed. Gradients flow through $Y$ (supply/demand ratio) but NOT through $g(d)$.

**Rationale**: If $g(d)$ were also optimized, the optimizer could trivially achieve $R^2 = 1$ by making $g(d) = Y$ everywhere, defeating the purpose of the metric.

#### 4.8.2 Soft Demand Aggregation

Demand counts use soft cell assignment for differentiability:

$$
D_c = \sum_{i \in \text{pickups}} \sigma_c(x_i, y_i)
$$

#### 4.8.3 PyTorch Implementation

```python
class DifferentiableCausalFairnessWithSoftCounts(nn.Module):
    """Fully differentiable causal fairness with soft cell assignment."""
    
    def __init__(self, g_function, grid_shape=(48, 90), 
                 neighborhood_size=5, temperature=1.0):
        super().__init__()
        self.g_function = g_function  # Pre-fitted, frozen
        self.soft_assign = SoftCellAssignment(
            grid_shape=grid_shape,
            neighborhood_size=neighborhood_size,
            temperature=temperature
        )
    
    def forward(self, pickup_coords, supply_tensor):
        """
        Compute differentiable causal fairness.
        
        Args:
            pickup_coords: Tensor (N, 2) - pickup (x, y) coordinates
            supply_tensor: Tensor (48, 90) - supply per cell
        
        Returns:
            F_causal: Scalar R² score ∈ [0, 1] (higher = more fair)
        """
        # Soft demand counts (differentiable)
        demand = self.soft_assign(pickup_coords).flatten()  # (4320,)
        
        # Compute Y = supply / demand
        supply = supply_tensor.flatten()
        mask = demand > 0.1  # Minimum demand threshold
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

**Implementation Location**: `objective_function/causal_fairness/utils.py`

#### 4.8.4 g(d) Estimation Methods

| Method | R² Score | Differentiability | Recommendation |
|--------|----------|-------------------|----------------|
| **Isotonic** | 0.4453 | ✅ Compatible | 🥇 Best fit |
| **Binning** | 0.4439 | ✅ Compatible | 🥈 Alternative |
| Polynomial | 0.1949 | ✅ Compatible | Poor fit |
| Linear | 0.1064 | ✅ Compatible | Poor fit |

**Note**: All methods are "compatible" because $g(d)$ is frozen—differentiability of the estimation method itself is not required.

#### 4.8.5 Demand-Conditional Deviation (DCD) Attribution

For trajectory-level attribution, compute each trajectory's contribution to unexplained variance:

$$
\text{DCD}_\tau = \sum_{c \in \mathcal{C}} \sigma_c(\tau) \cdot R_c^2
$$

Where $R_c = Y_c - g(D_c)$ is the residual for cell $c$.

**Implementation**: `objective_function/causal_fairness/utils.py::compute_demand_conditional_deviation()`

---

## 5. Trajectory Fidelity Term ($F_{\text{fidelity}}$)

### 5.1 Conceptual Foundation

The fidelity term ensures edited trajectories remain realistic and indistinguishable from authentic expert driver trajectories. This prevents the optimizer from making arbitrary edits that improve fairness metrics but produce implausible trajectories.

### 5.2 Components

#### 5.2.1 Distributional Fidelity (KL Divergence)

Measure the difference between original and edited trajectory distributions:

$$
F_{\text{KL}} = D_{KL}(P_{\text{original}} \| P_{\text{edited}})
$$

Where $P$ represents the empirical distribution of trajectory features (e.g., action sequences, state transitions).

#### 5.2.2 Discriminator-Based Fidelity

Use the trained ST-SiameseNet discriminator to assess trajectory authenticity:

$$
F_{\text{discriminator}} = 1 - \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \text{Discriminator}(\tau', \tau_{\text{ref}})
$$

Where the discriminator outputs the probability that $\tau'$ belongs to the same driver as $\tau_{\text{ref}}$.

### 5.3 Implementation

The discriminator is already implemented in `discriminator/model/`. Integration:

```python
from discriminator.model import load_model_from_checkpoint

def compute_fidelity_score(
    original_trajectories: List[np.ndarray],
    edited_trajectories: List[np.ndarray],
    discriminator_path: str
) -> float:
    """
    Compute trajectory fidelity using the discriminator.
    
    Args:
        original_trajectories: List of original trajectory arrays
        edited_trajectories: List of edited trajectory arrays
        discriminator_path: Path to trained discriminator checkpoint
    
    Returns:
        F_fidelity: Fidelity score (higher = more authentic)
    """
    model, config = load_model_from_checkpoint(discriminator_path)
    model.eval()
    
    scores = []
    for orig, edited in zip(original_trajectories, edited_trajectories):
        # Discriminator expects paired trajectories
        with torch.no_grad():
            score = model(orig, edited)
            scores.append(score.item())
    
    return np.mean(scores)
```

---

## 6. Trajectory Modification Algorithm

### 6.1 Overview

The FAMAIL trajectory modification algorithm is the core procedure for improving fairness in taxi service distribution. It treats each taxi trajectory as a data sample contributing to overall fairness, identifies trajectories that cause the most unfairness, and adjusts them in a controlled manner.

**Key Insight**: Some trajectories disproportionately contribute to unfair service distribution, analogous to how certain training examples contribute high loss in machine learning. The algorithm finds and modifies these "worst-offending" trajectories.

### 6.2 Algorithm Steps

```
ALGORITHM: FAMAIL Fair Trajectory Modification
═══════════════════════════════════════════════════════════════════════

INPUT:
  - 𝒯: Original expert trajectory set
  - discriminator: Trained ST-SiameseNet model
  - config:
    - neighborhood_size: n (for n×n grid neighborhood constraint)
    - max_iterations: int
    - convergence_threshold: float

OUTPUT:
  - 𝒯': Modified trajectory set with improved fairness

STEPS:

1. RANK TRAJECTORIES BY FAIRNESS IMPACT
   Calculate each trajectory's influence on global fairness using
   spatial and causal fairness metrics.

2. IDENTIFY WORST-OFFENDING TRAJECTORIES
   Apply "demand hierarchy" filter: consider high-demand areas first,
   then select lowest-fairness trajectories within those areas.

3. MODIFY SELECTED TRAJECTORIES
   For each selected trajectory:
   - Identify well-served (fair) and underserved (unfair) regions nearby
   - Reallocate pickups/drop-offs from well-served to underserved
     regions within n×n neighborhood
   - Keep overall route structure similar

4. VALIDATE TRAJECTORY FIDELITY
   Run discriminator on each edited trajectory. If score < threshold,
   reject edit or reduce edit magnitude.

5. RECOMPUTE FAIRNESS METRICS
   Update active taxi count distribution (N^p), recompute F_spatial
   and F_causal with the new trajectory set.

6. ITERATE IF NECESSARY
   Re-rank trajectories and make further adjustments until fairness
   metrics reach acceptable levels or no further gains can be made.

RETURN 𝒯'
```

### 6.3 Key Constraints

**Local Boundedness**: All edits must stay within the original vicinity (±n grid cells):
$$
\forall s' \in \tau', \exists s \in \tau : \|s' - s\|_\infty \leq n
$$

**Trajectory Editing vs. Generation**: The algorithm focuses on small adjustments to existing routes, not generating new trajectories. Issues beyond the edit threshold are noted for future work.

**Efficiency Optimization**: The algorithm holds $N^p$ (active taxi counts) constant within iterations, only recalculating after modifying a batch of worst-offending trajectories.

### 6.4 Causal Fairness Integration

| Fairness Type | Measures | Role in Algorithm |
|---------------|----------|-------------------|
| **Spatial Fairness** | **What** the service distribution is | Identifies geographic inequality |
| **Causal Fairness** | **Why** disparities exist | Guides intelligent trajectory selection |

By combining demand estimates with fairness metrics, the algorithm shifts service from high-demand/well-served areas to high-demand/poorly-served areas, addressing root causes of unfairness.

### 6.5 Gradient-Based Trajectory Optimization

> **IMPLEMENTED**: The following differentiable formulations enable gradient-based trajectory optimization.

Rather than using heuristic modification rules, FAMAIL supports **gradient-based optimization** where trajectory modifications are guided by $\nabla_{\tau} \mathcal{L}$—the gradient of the objective function with respect to trajectory coordinates.

#### 6.5.1 Soft Cell Assignment (Key Innovation)

Traditional grid assignment uses hard cell boundaries, which are non-differentiable. The soft cell assignment module enables gradient flow by computing **probabilistic assignments** using a Gaussian softmax:

$$
\sigma_c(x, y) = \frac{\exp\left(-\frac{d^2_c(x,y)}{2\tau^2}\right)}{\sum_{c' \in \mathcal{N}} \exp\left(-\frac{d^2_{c'}(x,y)}{2\tau^2}\right)}
$$

Where:
- $d_c(x, y)$ = Euclidean distance from $(x, y)$ to center of cell $c$
- $\tau$ = Temperature parameter (controls assignment "softness")
- $\mathcal{N}$ = Local neighborhood of cells (default: 5×5)

**Temperature Annealing Schedule**:
- Training: $\tau = 1.0$ (soft, exploratory)
- Inference: $\tau \to 0.1$ (sharp, deterministic-like)

**Implementation**: `objective_function/spatial_fairness/soft_cell_assignment.py`

#### 6.5.2 Differentiable Pipeline

The full differentiable pipeline for a single trajectory point $(x, y)$:

```
Trajectory Point (x, y) with requires_grad=True
        ↓
   Soft Cell Assignment σ_c(x, y)
        ↓
   Soft Counts: count_c = Σ_i σ_c(x_i, y_i)
        ↓
   ┌─────────────────┬───────────────────┐
   ↓                 ↓                   ↓
Spatial Fairness  Causal Fairness    Fidelity
(Pairwise Gini)   (R² with g(d))   (Discriminator)
   ↓                 ↓                   ↓
   └─────────────────┴───────────────────┘
                     ↓
              Weighted Sum: L
                     ↓
              L.backward()
                     ↓
              ∂L/∂x, ∂L/∂y (trajectory gradients)
```

#### 6.5.3 Gradient-Based Modification Algorithm

```
Algorithm: Gradient-Based Trajectory Modification
═══════════════════════════════════════════════════════════════════════

INPUT:
  - τ: Trajectory tensor (requires_grad=True)
  - objective: Differentiable FAMAIL objective function
  - ε: Maximum perturbation (L∞ constraint)
  - η: Learning rate
  - max_iterations: Maximum optimization steps

OUTPUT:
  - τ': Modified trajectory with improved fairness

STEPS:

1. INITIALIZE
   τ' = τ.clone().requires_grad_(True)
   τ_original = τ.detach()

2. FOR iteration in 1..max_iterations:
   
   a. FORWARD PASS
      L = objective(τ')  # Compute objective
   
   b. BACKWARD PASS
      L.backward()       # Compute gradients
      grad = τ'.grad
   
   c. GRADIENT UPDATE (maximize objective)
      τ' = τ' + η * grad.sign()  # FGSM-style step
   
   d. PROJECT TO CONSTRAINTS
      # L∞ constraint: stay within ε of original
      τ' = torch.clamp(τ', τ_original - ε, τ_original + ε)
      
      # Grid bounds constraint
      τ' = torch.clamp(τ', min=0, max=grid_dims)
   
   e. CLEAR GRADIENTS
      τ'.grad.zero_()
   
   f. CHECK CONVERGENCE
      if |grad| < threshold: break

3. RETURN τ'
```

This follows the ST-iFGSM (Spatial-Temporal Iterative FGSM) approach from the reference paper.

---

## 7. Constraints

### 7.1 Subtle Edits Constraint ($L_\infty$ Bound)

$$
\|\tau' - \tau\|_\infty \leq \epsilon \quad \forall \tau \in \mathcal{T}
$$

**Interpretation**: The maximum change to any single GPS point is bounded by $\epsilon$.

**Implementation**:
```python
def check_linf_constraint(
    original: np.ndarray,
    edited: np.ndarray,
    epsilon: float
) -> bool:
    """Check if L-infinity constraint is satisfied."""
    max_change = np.max(np.abs(edited - original))
    return max_change <= epsilon
```

**Typical Values**: $\epsilon \in [0.001, 0.01]$ (in normalized coordinates or grid units)

### 7.2 Limited Modifications Constraint ($L_0$ Bound)

$$
\|\tau' - \tau\|_0 \leq \eta \quad \forall \tau \in \mathcal{T}
$$

**Interpretation**: At most $\eta$ trajectory points can be modified per trajectory.

**Implementation**:
```python
def check_l0_constraint(
    original: np.ndarray,
    edited: np.ndarray,
    eta: int
) -> bool:
    """Check if L0 constraint is satisfied."""
    num_changes = np.sum(np.any(original != edited, axis=-1))
    return num_changes <= eta
```

**Typical Values**: $\eta \in [1, 10]$ points per trajectory

### 7.3 Dataset-Level Constraint

$$
\|\mathcal{T}' - \mathcal{T}\|_0 \leq \zeta
$$

**Interpretation**: At most $\zeta$ trajectories in the entire dataset can be modified.

### 7.4 Authenticity Constraint

$$
\text{Discriminator\_confidence}(\tau') \geq \theta
$$

**Interpretation**: Edited trajectories must pass the discriminator with confidence at least $\theta$.

**Typical Values**: $\theta \geq 0.5$

---

## 8. Data Sources and Structure

### 8.1 Primary Datasets

| Dataset | Path | Description |
|---------|------|-------------|
| `all_trajs.pkl` | `source_data/` | Expert driver trajectories (50 drivers) |
| `pickup_dropoff_counts.pkl` | `source_data/` | Aggregated pickup/dropoff counts |
| `latest_traffic.pkl` | `source_data/` | Traffic speed and waiting time |
| `latest_volume_pickups.pkl` | `source_data/` | Pickup and traffic volume |

### 8.2 State Space Definition

| Dimension | Field | Range | Description |
|-----------|-------|-------|-------------|
| Spatial X | `x_grid` | [0, 47] | Longitude grid index |
| Spatial Y | `y_grid` | [0, 89] | Latitude grid index |
| Temporal | `time_bucket` | [0, 287] | 5-minute interval |
| Day | `day_of_week` | [1, 5] | Day of week (Mon-Fri only) |

### 8.3 Data Loading Patterns

```python
import pickle
from pathlib import Path

def load_famail_data(data_dir: str):
    """Load all FAMAIL datasets."""
    data_path = Path(data_dir)
    
    with open(data_path / "all_trajs.pkl", "rb") as f:
        trajectories = pickle.load(f)
    
    with open(data_path / "pickup_dropoff_counts.pkl", "rb") as f:
        pickup_dropoff = pickle.load(f)
    
    with open(data_path / "latest_traffic.pkl", "rb") as f:
        traffic = pickle.load(f)
    
    with open(data_path / "latest_volume_pickups.pkl", "rb") as f:
        volume_pickups = pickle.load(f)
    
    return {
        "trajectories": trajectories,
        "pickup_dropoff": pickup_dropoff,
        "traffic": traffic,
        "volume_pickups": volume_pickups
    }
```

---

## 9. Implementation Architecture

### 9.1 Recommended Module Structure

```
objective_function/
├── __init__.py
├── base.py                      # Base classes and interfaces
├── config.py                    # Configuration and hyperparameters
├── spatial_fairness/
│   ├── __init__.py
│   ├── base.py                  # SpatialFairnessBase class
│   ├── utils.py                 # Gini, differentiable implementations
│   ├── soft_cell_assignment.py  # SoftCellAssignment module ✨ NEW
│   └── DEVELOPMENT_PLAN.md
├── causal_fairness/
│   ├── __init__.py
│   ├── base.py                  # CausalFairnessBase class
│   ├── utils.py                 # R², g(d) estimation, differentiable implementations
│   └── DEVELOPMENT_PLAN.md
├── fidelity/
│   ├── __init__.py
│   ├── base.py                  # FidelityBase class
│   ├── utils.py                 # Discriminator gradient verification
│   └── DEVELOPMENT_PLAN.md
├── quality/                     # (Merged into Fidelity)
├── docs/
│   ├── FAMAIL_OBJECTIVE_FUNCTION_SPECIFICATION.md
│   ├── FAIRNESS_TERM_FORMULATIONS.md
│   └── TRAJECTORY_MODIFICATION_ALGORITHM_DEVELOPMENT_PLAN.md
├── dashboards/
│   └── objective_function_dashboard.py  # Streamlit visualization
└── tests/
    ├── test_spatial_fairness.py
    ├── test_causal_fairness.py
    ├── test_soft_cell_assignment.py  # ✨ NEW
    └── test_integration.py
```

### 9.2 Key Classes (Updated for Differentiability)

| Class | Module | Purpose |
|-------|--------|---------|
| `SoftCellAssignment` | `spatial_fairness/soft_cell_assignment.py` | Gaussian softmax cell assignment |
| `DifferentiableSpatialFairnessWithSoftCounts` | `spatial_fairness/utils.py` | Full differentiable spatial pipeline |
| `DifferentiableCausalFairnessWithSoftCounts` | `causal_fairness/utils.py` | Full differentiable causal pipeline |
| `compute_local_inequality_score()` | `spatial_fairness/utils.py` | LIS trajectory attribution |
| `compute_demand_conditional_deviation()` | `causal_fairness/utils.py` | DCD trajectory attribution |

### 9.2 Main Objective Function Class

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class ObjectiveFunctionConfig:
    """Configuration for the FAMAIL objective function."""
    alpha_causal: float = 0.33
    alpha_spatial: float = 0.33
    alpha_fidelity: float = 0.34
    
    # Spatial fairness parameters
    period_definition: str = "daily"
    
    # Causal fairness parameters
    neighborhood_size: int = 1
    g_estimation_method: str = "binning"
    
    # Constraint parameters
    epsilon_linf: float = 0.01
    eta_l0: int = 5
    zeta_dataset: int = 100
    theta_discriminator: float = 0.5


class FAMAILObjectiveFunction:
    """
    FAMAIL Trajectory Editing Objective Function.
    
    Computes the weighted multi-objective loss for trajectory editing:
    L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity
    """
    
    def __init__(self, config: ObjectiveFunctionConfig):
        self.config = config
        self._spatial_module = SpatialFairnessModule(config)
        self._causal_module = CausalFairnessModule(config)
        self._fidelity_module = FidelityModule(config)
    
    def compute(
        self,
        original_trajectories: Dict[str, Any],
        edited_trajectories: Dict[str, Any],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute all objective function terms.
        
        Returns:
            Dictionary with individual terms and total loss
        """
        F_spatial = self._spatial_module.compute(edited_trajectories, auxiliary_data)
        F_causal = self._causal_module.compute(edited_trajectories, auxiliary_data)
        F_fidelity = self._fidelity_module.compute(
            original_trajectories, edited_trajectories
        )
        
        # All terms are fairness/fidelity scores (higher = better)
        # Objective is maximized
        total_objective = (
            self.config.alpha_causal * F_causal +
            self.config.alpha_spatial * F_spatial +
            self.config.alpha_fidelity * F_fidelity
        )
        
        return {
            "F_causal": F_causal,
            "F_spatial": F_spatial,
            "F_fidelity": F_fidelity,
            "total_objective": total_objective
        }
    
    def check_constraints(
        self,
        original_trajectories: Dict[str, Any],
        edited_trajectories: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check all constraints."""
        return {
            "linf_satisfied": self._check_linf(original_trajectories, edited_trajectories),
            "l0_satisfied": self._check_l0(original_trajectories, edited_trajectories),
            "dataset_l0_satisfied": self._check_dataset_l0(
                original_trajectories, edited_trajectories
            ),
            "discriminator_satisfied": self._check_discriminator(edited_trajectories)
        }
```

---

## 10. Testing and Validation

### 10.1 Unit Tests for Individual Terms

Each term should have unit tests verifying:

1. **Edge Cases**: Empty inputs, single-cell grids, uniform distributions
2. **Known Values**: Synthetic data with known expected outputs
3. **Monotonicity**: Term values change in expected direction with changes to inputs
4. **Bounds**: Outputs are within expected ranges (e.g., Gini ∈ [0, 1])

### 10.2 Integration Tests

Test the complete objective function:

```python
def test_objective_function_smoke():
    """Basic smoke test for objective function."""
    config = ObjectiveFunctionConfig()
    objective = FAMAILObjectiveFunction(config)
    
    # Load sample data
    data = load_famail_data("source_data/")
    
    # Compute baseline (no edits)
    result = objective.compute(
        original_trajectories=data["trajectories"],
        edited_trajectories=data["trajectories"],
        auxiliary_data=data
    )
    
    # Verify all terms are computed
    assert "F_causal" in result
    assert "F_spatial" in result
    assert "F_fidelity" in result
    assert "total_objective" in result
    
    # Verify values are reasonable (all fairness/fidelity scores in [0, 1])
    assert 0 <= result["F_spatial"] <= 1
    assert 0 <= result["F_causal"] <= 1
    assert 0 <= result["F_fidelity"] <= 1
    assert result["total_objective"] >= 0  # Sum of positive terms
```

### 10.3 Baseline Metrics Experiment

Before implementing trajectory editing, compute baseline fairness metrics on the original dataset:

```python
def compute_baseline_metrics():
    """Compute fairness metrics on original (unedited) trajectories."""
    data = load_famail_data("source_data/")
    config = ObjectiveFunctionConfig()
    
    spatial_fairness = compute_spatial_fairness(
        pickup_counts=extract_pickup_counts(data["trajectories"]),
        dropoff_counts=extract_dropoff_counts(data["trajectories"]),
        grid_dims=(48, 90),
        num_taxis=50,
        num_days=30
    )
    
    causal_fairness = compute_causal_fairness(
        pickup_counts=data["pickup_dropoff"],
        traffic_volume=extract_volume(data["volume_pickups"]),
        grid_dims=(48, 90)
    )
    
    print(f"Baseline Spatial Fairness (Gini): {spatial_fairness:.4f}")
    print(f"Baseline Causal Fairness: {causal_fairness:.4f}")
    
    return {
        "spatial_fairness": spatial_fairness,
        "causal_fairness": causal_fairness
    }
```

---

## 11. References

### 11.1 Primary Sources

1. **Su et al. (2018)**: "Uncovering Spatial Inequality in Taxi Services" - Foundation for spatial fairness metrics
   - Location: `objective_function/spatial_fairness/Uncovering_Spatial_Inequality_in_Taxi_Services__Su.pdf`

2. **KDD Fairness MAIL Paper (WIP)**: Research paper for the FAMAIL project
   - Location: `objective_function/docs/KDD_Fairness_MAIL.pdf`

3. **ST-iFGSM Paper**: "Enhancing Robustness of Human Mobility Signature Identification Model via Spatial-Temporal Iterative FGSM"
   - Foundation for trajectory editing algorithm
   - URL: https://dl.acm.org/doi/10.1145/3580305.3599377

4. **cGAIL Paper**: "Conditional Generative Adversarial Imitation Learning—An Application in Taxi Drivers' Strategy Learning"
   - Foundation for the imitation learning framework and dataset structure

### 11.2 Fairness Literature

- Causal Fairness Analysis: https://causalai.net/r90.pdf
- Fairlearn User Guide: https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html

### 11.3 Data Dictionaries

- `data_dictionary/dictionaries/all_trajs_data_dictionary.md`
- `data_dictionary/dictionaries/pickup_dropoff_counts_data_dictionary.md`
- `data_dictionary/dictionaries/latest_traffic_data_dictionary.md`
- `data_dictionary/dictionaries/latest_volume_pickups_data_dictionary.md`

---

## Appendix A: Quick Reference - Key Equations

### Spatial Fairness (Complement of Average Gini)

$$
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P}(G_a^p + G_d^p)
$$

**Gini Coefficient (Standard - Non-Differentiable):**
$$
G = 1 + \frac{1}{n} - \frac{2}{n^2 \bar{x}} \sum_{i=1}^{n}(n-i+1) \cdot x_{(i)}
$$

**Gini Coefficient (Pairwise - Differentiable):**
$$
G_{\text{pairwise}} = \frac{\sum_{i}\sum_{j}|x_i - x_j|}{2n^2\bar{x}}
$$

**Interpretation:** $F_{\text{spatial}} = 1$ is perfect equality, $F_{\text{spatial}} = 0$ is maximum inequality.

### Causal Fairness (R² - Variance Explained by Demand)

$$
F_{\text{causal}} = \frac{1}{|P|} \sum_{p \in P} F_{\text{causal}}^p = \frac{1}{|P|} \sum_{p \in P} \frac{\text{Var}_p(g(D_{i,p}))}{\text{Var}_p(Y_{i,p})}
$$

**Equivalent Residual Form (Used in Differentiable Implementation):**
$$
F_{\text{causal}} = 1 - \frac{\text{Var}(R)}{\text{Var}(Y)} \quad \text{where } R = Y - g(D)
$$

**Interpretation:** $F_{\text{causal}} = 1$ means service perfectly matches demand, $F_{\text{causal}} = 0$ means no relationship.

### Soft Cell Assignment (Gaussian Softmax)

$$
\sigma_c(x, y) = \frac{\exp\left(-\frac{d^2_c(x,y)}{2\tau^2}\right)}{\sum_{c' \in \mathcal{N}} \exp\left(-\frac{d^2_{c'}(x,y)}{2\tau^2}\right)}
$$

**Temperature Schedule:** $\tau = 1.0$ (training) → $\tau = 0.1$ (inference)

### Objective Function (Maximization)

$$
\max_{\mathcal{T}'} \mathcal{L} = \max_{\mathcal{T}'} \left( \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} \right)
$$

**All terms are fairness/fidelity scores where higher values are better.**

### Gradient-Based Trajectory Update (FGSM-style)

$$
\tau' = \tau + \eta \cdot \text{sign}(\nabla_\tau \mathcal{L})
$$

Subject to: $\|\tau' - \tau\|_\infty \leq \epsilon$

---

## Appendix B: Configuration Template

```yaml
# FAMAIL Objective Function Configuration
objective_function:
  # Term weights (must sum to 1.0)
  weights:
    alpha_causal: 0.33
    alpha_spatial: 0.33
    alpha_fidelity: 0.34
  
  # Spatial fairness parameters
  spatial:
    period_definition: "daily"  # "hourly", "daily", "weekly", "monthly"
    include_zero_cells: true
  
  # Causal fairness parameters
  causal:
    neighborhood_size: 1  # k for (2k+1)×(2k+1) neighborhood
    g_estimation_method: "binning"  # "binning", "regression", "polynomial"
    n_bins: 10
    min_demand_threshold: 1
  
  # Constraint parameters
  constraints:
    epsilon_linf: 0.01
    eta_l0: 5
    zeta_dataset: 100
    theta_discriminator: 0.5
  
  # Data paths
  data:
    trajectories: "source_data/all_trajs.pkl"
    pickup_dropoff: "source_data/pickup_dropoff_counts.pkl"
    traffic: "source_data/latest_traffic.pkl"
    volume_pickups: "source_data/latest_volume_pickups.pkl"
  
  # Grid parameters
  grid:
    x_dim: 48
    y_dim: 90
    time_buckets: 288
```

---

*Document Version: 1.2*  
*Last Updated: January 2026*  
*Author: FAMAIL Research Team*

---

## Revision History

| Version | Date | Changes |
|---------|------|----------|
| 1.0 | 2026-01 | Initial comprehensive specification |
| 1.1 | 2026-01-12 | Removed Quality Term (overlap with Fidelity); added Trajectory Modification Algorithm section |
| 1.2 | 2026-01 | Added differentiable formulations with soft cell assignment for gradient-based optimization |
