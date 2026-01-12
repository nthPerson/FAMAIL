# Causal Fairness Term ($F_{\text{causal}}$) Development Plan

## Document Metadata

| Property | Value |
|----------|-------|
| **Term Name** | Causal Fairness |
| **Symbol** | $F_{\text{causal}}$ |
| **Version** | 1.0.0 |
| **Last Updated** | 2026-01-09 |
| **Status** | Development Planning |
| **Author** | FAMAIL Research Team |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
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

**Source**: `pickup_dropoff_counts.pkl` or `latest_volume_pickups.pkl`

#### 2.2.2 Supply

Supply is computed using neighborhood aggregation to capture available taxi capacity:

$$
S_{i,p} = \sum_{j \in \mathcal{N}_k(i)} \text{traffic\_volume}_{j,p}
$$

Where:
- $\mathcal{N}_k(i)$ = $(2k+1) \times (2k+1)$ neighborhood centered on cell $i$
- Default: $k=1$ (3×3 neighborhood)

**Interpretation**: Supply is proxied by total taxi traffic volume in the vicinity.

**Source**: `latest_volume_pickups.pkl` (index 1) or `latest_traffic.pkl`

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

**Default**: Binning with 10 quantile-based bins.

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

#### 4.1.2 Supply Data: `latest_volume_pickups.pkl`

**Location**: `FAMAIL/source_data/latest_volume_pickups.pkl`

**Structure**:
```python
{
    (x_grid, y_grid, time_bucket, day_of_week): [pickup_count, traffic_volume],
}
```

**Fields Used**:
| Field | Usage |
|-------|-------|
| `traffic_volume` (index 1) | Supply proxy ($S_{i,p}$, with neighborhood aggregation) |

#### 4.1.3 Alternative: `latest_traffic.pkl`

**Location**: `FAMAIL/source_data/latest_traffic.pkl`

Can be used if `latest_volume_pickups.pkl` is unavailable.

**Structure**:
```python
{
    (x_grid, y_grid, time_bucket, day_of_week): [speed, wait_time, volume],
}
```

### 4.2 Data Preprocessing

#### 4.2.1 Loading and Aligning Data

```python
def load_causal_fairness_data(
    pickup_counts_path: str,
    volume_pickups_path: str
) -> Tuple[Dict, Dict]:
    """
    Load and align demand and supply data.
    
    Returns:
        (demand_data, supply_data) dictionaries with aligned keys
    """
    import pickle
    
    with open(pickup_counts_path, 'rb') as f:
        pickup_data = pickle.load(f)
    
    with open(volume_pickups_path, 'rb') as f:
        volume_data = pickle.load(f)
    
    # Extract demand (pickup counts)
    demand = {key: val[0] for key, val in pickup_data.items()}
    
    # Extract supply (traffic volume)
    supply = {key: val[1] for key, val in volume_data.items()}
    
    return demand, supply
```

#### 4.2.2 Neighborhood Aggregation for Supply

```python
def aggregate_supply_neighborhood(
    supply: Dict[Tuple, int],
    grid_dims: Tuple[int, int],
    neighborhood_size: int = 1
) -> Dict[Tuple, float]:
    """
    Aggregate supply over neighborhood for each cell.
    
    Args:
        supply: Raw supply data (cell → volume)
        grid_dims: (x_max, y_max)
        neighborhood_size: k for (2k+1)×(2k+1) window
    
    Returns:
        Aggregated supply per cell
    """
    aggregated = {}
    
    # Get unique time-day combinations
    time_days = set((key[2], key[3]) for key in supply.keys())
    
    for t, d in time_days:
        for x in range(grid_dims[0]):
            for y in range(grid_dims[1]):
                total_supply = 0
                
                for dx in range(-neighborhood_size, neighborhood_size + 1):
                    for dy in range(-neighborhood_size, neighborhood_size + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < grid_dims[0] and 0 <= ny < grid_dims[1]:
                            key = (nx, ny, t, d)
                            total_supply += supply.get(key, 0)
                
                aggregated[(x, y, t, d)] = total_supply
    
    return aggregated
```

#### 4.2.3 Computing Service Ratios

```python
def compute_service_ratios(
    demand: Dict[Tuple, int],
    supply: Dict[Tuple, float],
    min_demand: int = 1
) -> Dict[Tuple, float]:
    """
    Compute service ratio Y = S/D for each cell-period.
    
    Args:
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
    supply: Dict[Tuple, float]
) -> List[str]:
    """
    Validate data for causal fairness computation.
    
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
  - supply: Dict[(x, y, time, day)] → traffic_volume
  - config: CausalFairnessConfig
    - grid_dims: (48, 90)
    - neighborhood_size: 1
    - estimation_method: "binning"
    - min_demand: 1
    - period_type: "time_bucket"

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
    # Step 1: Aggregate supply
    agg_supply = aggregate_neighborhood(supply, config.neighborhood_size)
    
    # Compute service ratios
    ratios = {}
    for key in demand:
        if demand[key] >= config.min_demand:
            ratios[key] = agg_supply.get(key, 0) / demand[key]
    
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
    neighborhood_size: int = 1  # k for (2k+1)×(2k+1) window
    estimation_method: str = "binning"  # "binning", "regression", "polynomial"
    n_bins: int = 10
    poly_degree: int = 2
    min_demand: int = 1
    period_type: str = "time_bucket"  # "time_bucket", "hourly", "daily"


class CausalFairnessTerm(ObjectiveFunctionTerm):
    """
    Causal Fairness term based on R² of demand-explained service variance.
    
    Measures how much of the variation in service is explained by demand.
    Higher values indicate more fair (demand-based) service allocation.
    """
    
    def _build_metadata(self) -> TermMetadata:
        return TermMetadata(
            name="causal_fairness",
            display_name="Causal Fairness",
            version="1.0.0",
            description="R²-based measure of demand-explained service variation",
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,
            required_data=["pickup_dropoff_counts", "latest_volume_pickups"],
            optional_data=["latest_traffic"],
            author="FAMAIL Team",
            last_updated="2026-01-09"
        )
    
    def _validate_config(self) -> None:
        if self.config.estimation_method not in ["binning", "regression", "polynomial"]:
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
        supply = self._extract_supply(auxiliary_data)
        
        # Aggregate supply over neighborhood
        agg_supply = self._aggregate_neighborhood(supply)
        
        # Compute service ratios
        ratios = self._compute_ratios(demand, agg_supply)
        
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

### 6.2 Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `neighborhood_size` | int | 1 | $k$ for $(2k+1) \times (2k+1)$ aggregation |
| `estimation_method` | str | "binning" | Method for estimating g(d) |
| `n_bins` | int | 10 | Number of bins for binning method |
| `poly_degree` | int | 2 | Degree for polynomial regression |
| `min_demand` | int | 1 | Minimum demand to include cell |
| `period_type` | str | "time_bucket" | Temporal aggregation granularity |
| `weight` | float | 1.0 | Weight in objective function ($\alpha_1$) |

### 6.3 Default Values and Rationale

```python
DEFAULT_CONFIG = CausalFairnessConfig(
    # Spatial configuration
    grid_dims=(48, 90),       # Shenzhen grid
    neighborhood_size=1,      # 3×3 window (balance local vs. regional)
    
    # g(d) estimation
    estimation_method="binning",  # Simple, interpretable
    n_bins=10,                    # Sufficient resolution
    poly_degree=2,                # If using polynomial
    
    # Filtering
    min_demand=1,             # Include all cells with any demand
    
    # Aggregation
    period_type="time_bucket",  # Finest granularity
    weight=1.0,
)
```

**Rationale**:

- `neighborhood_size=1`: Captures local supply availability without excessive smoothing
- `estimation_method="binning"`: Robust to non-linear relationships, easy to interpret
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
        
        with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
            pickup_data = pickle.load(f)
        
        with open('source_data/latest_volume_pickups.pkl', 'rb') as f:
            volume_data = pickle.load(f)
        
        demand = {k: v[0] for k, v in pickup_data.items()}
        supply = {k: v[1] for k, v in volume_data.items()}
        
        config = CausalFairnessConfig()
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
    # High demand area (downtown)
    (24, 45, 144, 1): 50,  # 50 pickup requests
    (24, 46, 144, 1): 45,
    
    # Medium demand area
    (10, 30, 144, 1): 15,
    
    # Low demand area
    (2, 85, 144, 1): 2,
}

sample_supply = {
    # Corresponding supply (traffic volume)
    (24, 45, 144, 1): 500,  # Ratio: 10
    (24, 46, 144, 1): 450,  # Ratio: 10
    
    (10, 30, 144, 1): 150,  # Ratio: 10 (fair: same ratio)
    
    (2, 85, 144, 1): 10,    # Ratio: 5 (unfair: lower ratio)
}

# In a fair system: all ratios should be similar given similar demand
# The low-demand area receives less service per demand unit (unfair)
```

### 10.3 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial development plan |

---

*This document serves as the comprehensive development guide for the Causal Fairness term. All implementation should follow the specifications outlined here.*
