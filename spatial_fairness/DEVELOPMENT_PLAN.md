# Spatial Fairness Term ($F_{\text{spatial}}$) Development Plan

## Document Metadata

| Property | Value |
|----------|-------|
| **Term Name** | Spatial Fairness |
| **Symbol** | $F_{\text{spatial}}$ |
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

The **Spatial Fairness Term** ($F_{\text{spatial}}$) quantifies the degree of equality in taxi service distribution across geographic regions within the study area (Shenzhen, China). It measures whether taxi services are distributed equitably across all grid cells, or whether certain areas receive disproportionately more or less service.

**Core Principle**: In a perfectly spatially fair system, every geographic region would receive taxi service proportional to some baseline expectation. The spatial fairness term uses the **Gini coefficient** to measure deviation from perfect equality.

### 1.2 Role in Objective Function

The spatial fairness term is one of two fairness components in the FAMAIL objective function:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} + \alpha_4 F_{\text{quality}}
$$

- **Weight**: $\alpha_2$ (typically 0.25-0.5 of total weight)
- **Optimization Direction**: **Maximize** (higher values = more spatially fair)
- **Value Range**: [0, 1]
  - $F_{\text{spatial}} = 1$: Perfect equality (all cells have identical service rates)
  - $F_{\text{spatial}} = 0$: Maximum inequality (all service concentrated in one cell)

### 1.3 Relationship to Other Terms

| Related Term | Relationship |
|--------------|-------------|
| $F_{\text{causal}}$ | Complementary fairness measure; spatial fairness addresses distribution, causal fairness addresses demand-service alignment |
| $F_{\text{fidelity}}$ | May trade off; improving spatial fairness could reduce trajectory authenticity |
| $F_{\text{quality}}$ | Generally independent; quality measures different aspects of trajectories |

### 1.4 Key Insights from Literature

From Su et al. (2018):
- Taxi services in Shenzhen exhibit significant spatial inequality
- The Gini coefficient effectively captures this inequality
- Arrival Service Rate (ASR) and Departure Service Rate (DSR) are appropriate metrics
- Inequality varies by time of day and day of week

---

## 2. Mathematical Formulation

### 2.1 Core Formula

The spatial fairness term is computed as the **complement of the average Gini coefficient** across time periods:

$$
F_{\text{spatial}} = 1 - \frac{1}{2|P|} \sum_{p \in P} (G_a^p + G_d^p)
$$

Where:
- $P$ = set of all time periods
- $G_a^p$ = Gini coefficient of Arrival Service Rates in period $p$
- $G_d^p$ = Gini coefficient of Departure Service Rates in period $p$

### 2.2 Component Definitions

#### 2.2.1 Service Rate Metrics

**Arrival Service Rate (ASR)** - measures dropoff frequency per cell:

$$
ASR_i^p = \frac{D_i^p}{N^p \cdot T^p}
$$

**Departure Service Rate (DSR)** - measures pickup frequency per cell:

$$
DSR_i^p = \frac{O_i^p}{N^p \cdot T^p}
$$

| Symbol | Definition | Source |
|--------|------------|--------|
| $D_i^p$ | Number of dropoffs in cell $i$ during period $p$ | `pickup_dropoff_counts.pkl` (index 1) |
| $O_i^p$ | Number of pickups in cell $i$ during period $p$ | `pickup_dropoff_counts.pkl` (index 0) |
| $N^p$ | Number of active taxis during period $p$ | Configuration (default: 50) |
| $T^p$ | Duration of period $p$ in days | Computed from period definition |
| $i$ | Grid cell index | $(x, y) \in [0,47] \times [0,89]$ |
| $p$ | Time period | Depends on period definition |

#### 2.2.2 Gini Coefficient

The Gini coefficient quantifies inequality in a distribution:

$$
G = 1 + \frac{1}{n} - \frac{2}{n^2 \bar{x}} \sum_{i=1}^{n} (n - i + 1) \cdot x_{(i)}
$$

Where:
- $x_{(i)}$ = the $i$-th smallest value in the sorted distribution
- $\bar{x}$ = mean of all values
- $n$ = number of observations (grid cells)

**Properties**:
- $G = 0$: Perfect equality
- $G = 1$: Maximum inequality
- Non-negative values only

**Alternative Formulation** (numerically equivalent):

$$
G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |x_i - x_j|}{2n^2 \bar{x}}
$$

#### 2.2.3 Per-Period Spatial Fairness

For each time period $p$:

$$
F_{\text{spatial}}^p = 1 - \frac{1}{2}(G_a^p + G_d^p)
$$

This averages the arrival and departure Gini coefficients to capture both aspects of service inequality.

### 2.3 Derivation and Justification

**Why the Gini Coefficient?**

1. **Well-established metric**: Widely used in economics to measure income inequality
2. **Scale-invariant**: Works regardless of absolute magnitudes
3. **Interpretable**: Easy to understand (0 = equality, 1 = inequality)
4. **Bounded**: Always in [0, 1]
5. **Prior use in transportation**: Su et al. (2018) validated its applicability to taxi services

**Why separate ASR and DSR?**

- Pickups (DSR) reflect where passengers request service (demand)
- Dropoffs (ASR) reflect where passengers are taken (supply response)
- Both are important for comprehensive fairness assessment
- Averaging them provides a balanced view

**Why use the complement (1 - Gini)?**

- FAMAIL uses a maximization objective
- All terms should be higher = better
- Complement converts inequality measure to equality measure

---

## 3. Literature and References

### 3.1 Primary Source

**Su, L., Yan, Z., & Cao, J. (2018). Uncovering Spatial Inequality in Taxi Services in the Context of a Subsidy War among E-Hailing Apps.**

**Key Contributions**:
- Introduced service rate metrics (ASR, DSR) for taxi service analysis
- Applied Gini coefficient to quantify spatial inequality
- Analyzed Shenzhen taxi data (same city as FAMAIL)
- Demonstrated temporal variation in inequality patterns

**Relevant Findings**:
- Spatial inequality is significant in urban taxi services
- Inequality varies by time of day (peak vs. off-peak)
- Some areas consistently underserved regardless of time
- E-hailing apps may exacerbate inequality

**Location in Repository**: `FAMAIL/objective_function/spatial_fairness/Uncovering_Spatial_Inequality_in_Taxi_Services__Su.pdf`

### 3.2 Theoretical Foundation

**Gini Coefficient Origins**:
- Corrado Gini (1912): "Variabilità e mutabilità"
- Originally developed for income inequality measurement
- Extended to various fields including transportation

**Key Properties**:
- Lorenz curve relationship: $G = 1 - 2 \times \text{Area under Lorenz curve}$
- Satisfies Pigou-Dalton transfer principle
- Mean-independent (scale-invariant)

### 3.3 Related Work

| Reference | Relevance |
|-----------|-----------|
| Banerjee et al. (2020) | Fairness in ride-hailing platforms |
| Lesmana et al. (2019) | Geographic equity in transportation |
| Bertsimas et al. (2011) | Equity vs. efficiency trade-offs |

---

## 4. Data Requirements

### 4.1 Required Datasets

#### 4.1.1 Primary: `pickup_dropoff_counts.pkl`

**Location**: `FAMAIL/source_data/pickup_dropoff_counts.pkl`

**Structure**:
```python
{
    (x_grid, y_grid, time_bucket, day_of_week): [pickup_count, dropoff_count],
    # Example: (4, 22, 201, 2): [20, 14]
}
```

**Key Fields**:
| Field | Range | Description |
|-------|-------|-------------|
| `x_grid` | [0, 47] | Grid x-coordinate (48 cells) |
| `y_grid` | [0, 89] | Grid y-coordinate (90 cells) |
| `time_bucket` | [1, 288] | 5-minute time bucket (1-indexed) |
| `day_of_week` | [1, 6] | Monday=1 through Saturday=6 |
| `pickup_count` | ≥0 | Number of pickups at this key |
| `dropoff_count` | ≥0 | Number of dropoffs at this key |

**Coverage**: ~234,000 non-zero keys (3.1% of state space)

#### 4.1.2 Alternative: `all_trajs.pkl`

**Location**: `FAMAIL/source_data/all_trajs.pkl`

Can be used to infer pickup events only (dataset contains only passenger-seeking trajectories). (`x_grid`, `y_grid`) of last state in trajectory can be treated as pickup event location. Can be used if computing on modified trajectories (only the `all_trajs.pkl` trajectory data will be modified).

**Usage**: Extract state information where action implies pickup.

### 4.2 Data Preprocessing

#### 4.2.1 From `pickup_dropoff_counts.pkl`

```python
def load_pickup_dropoff_data(filepath: str) -> Dict[Tuple, List[int]]:
    """Load and return pickup/dropoff counts."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def extract_counts_by_period(
    data: Dict[Tuple, List[int]],
    period_type: str = "time_bucket"
) -> Tuple[Dict, Dict]:
    """
    Extract pickup and dropoff counts grouped by period.
    
    Args:
        data: Raw pickup_dropoff_counts data
        period_type: One of "time_bucket", "hourly", "daily"
    
    Returns:
        (pickup_counts, dropoff_counts) dictionaries
    """
    pickups = {}
    dropoffs = {}
    
    for key, counts in data.items():
        x, y, time, day = key
        
        # Determine period based on period_type
        if period_type == "time_bucket":
            period = (time, day)  # Each 5-min bucket is a period
        elif period_type == "hourly":
            hour = (time - 1) // 12  # 0-23
            period = (hour, day)
        elif period_type == "daily":
            period = day
        
        cell = (x, y)
        
        # Aggregate
        if (cell, period) not in pickups:
            pickups[(cell, period)] = 0
            dropoffs[(cell, period)] = 0
        
        pickups[(cell, period)] += counts[0]
        dropoffs[(cell, period)] += counts[1]
    
    return pickups, dropoffs
```

#### 4.2.2 From Trajectory Data

If computing on modified trajectories:

```python
def extract_events_from_trajectories(
    trajectories: Dict[str, List[List[List[float]]]]
) -> Tuple[Dict, Dict]:
    """
    Extract pickup and dropoff events from trajectory data.
    
    Pickup: Transition from empty to occupied
    Dropoff: Transition from occupied to empty
    
    Note: Requires inferring passenger status from trajectory patterns
    or having explicit passenger indicator.
    """
    pickups = {}
    dropoffs = {}
    
    for driver_id, driver_trajs in trajectories.items():
        for traj in driver_trajs:
            for i in range(1, len(traj)):
                prev_state = traj[i-1]
                curr_state = traj[i]
                
                # Extract location and time
                x, y = int(curr_state[0]), int(curr_state[1])
                time = int(curr_state[2])
                day = int(curr_state[3])
                
                # Infer event type from action or state change
                # This depends on how the trajectory encodes events
                # Simplified example:
                action = int(curr_state[125])
                
                # Key for aggregation
                key = ((x, y), (time, day))
                
                # Aggregate based on inferred event type
                # (Actual logic depends on trajectory encoding)
    
    return pickups, dropoffs
```

### 4.3 Data Validation

Before computing spatial fairness, validate:

```python
def validate_spatial_fairness_data(data: Dict[Tuple, List[int]]) -> List[str]:
    """
    Validate pickup/dropoff counts data.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check data is not empty
    if len(data) == 0:
        errors.append("Data is empty")
        return errors
    
    # Check key format
    sample_key = list(data.keys())[0]
    if len(sample_key) != 4:
        errors.append(f"Expected 4-element keys, got {len(sample_key)}")
    
    # Check value format
    sample_value = list(data.values())[0]
    if len(sample_value) != 2:
        errors.append(f"Expected 2-element values, got {len(sample_value)}")
    
    # Check coordinate ranges
    x_vals = [k[0] for k in data.keys()]
    y_vals = [k[1] for k in data.keys()]
    
    if min(x_vals) < 0 or max(x_vals) > 47:
        errors.append(f"x_grid out of range: [{min(x_vals)}, {max(x_vals)}]")
    
    if min(y_vals) < 0 or max(y_vals) > 89:
        errors.append(f"y_grid out of range: [{min(y_vals)}, {max(y_vals)}]")
    
    # Check for negative counts
    neg_pickups = sum(1 for v in data.values() if v[0] < 0)
    neg_dropoffs = sum(1 for v in data.values() if v[1] < 0)
    
    if neg_pickups > 0:
        errors.append(f"{neg_pickups} entries have negative pickup counts")
    if neg_dropoffs > 0:
        errors.append(f"{neg_dropoffs} entries have negative dropoff counts")
    
    return errors
```

---

## 5. Implementation Plan

### 5.1 Algorithm Steps

```
ALGORITHM: Compute Spatial Fairness Term
═══════════════════════════════════════════════════════════════════════

INPUT:
  - data: pickup_dropoff_counts dictionary
  - config: SpatialFairnessConfig
    - grid_dims: (48, 90)
    - num_taxis: 50
    - num_days: 21.0 (for July 2016)
    - period_type: "time_bucket" | "hourly" | "daily"
    - include_zero_cells: True

OUTPUT:
  - F_spatial: float ∈ [0, 1] (higher = more fair)

STEPS:

1. DATA LOADING AND PREPROCESSING
   ─────────────────────────────────
   1.1 Load pickup_dropoff_counts.pkl
   1.2 Parse keys into (cell, period) format
   1.3 Aggregate counts by period according to period_type

2. SERVICE RATE COMPUTATION
   ─────────────────────────────────
   2.1 For each period p:
       2.1.1 Compute T_p = duration in days
       2.1.2 For each cell i in grid:
           - Get O_ip = pickup count (or 0 if missing)
           - Get D_ip = dropoff count (or 0 if missing)
           - Compute DSR_ip = O_ip / (N * T_p)
           - Compute ASR_ip = D_ip / (N * T_p)
       2.1.3 Collect all DSR values → DSR_list[p]
       2.1.4 Collect all ASR values → ASR_list[p]

3. GINI COEFFICIENT COMPUTATION
   ─────────────────────────────────
   3.1 For each period p:
       3.1.1 G_d[p] = compute_gini(DSR_list[p])
       3.1.2 G_a[p] = compute_gini(ASR_list[p])

4. AGGREGATION
   ─────────────────────────────────
   4.1 avg_gini = 0.5 * (mean(G_a) + mean(G_d))
   4.2 F_spatial = 1 - avg_gini

5. RETURN F_spatial
```

### 5.2 Pseudocode

```
function compute_spatial_fairness(data, config):
    # Step 1: Extract counts by period
    pickups, dropoffs = extract_counts_by_period(data, config.period_type)
    
    # Step 2: Get unique periods
    periods = unique_periods(pickups)
    
    # Step 3: Compute Gini coefficients per period
    gini_arrivals = []
    gini_departures = []
    
    for p in periods:
        # Compute service rates for all cells
        asr_values = []
        dsr_values = []
        
        T_p = compute_period_duration(p, config.period_type)
        
        for x in range(config.grid_dims[0]):
            for y in range(config.grid_dims[1]):
                cell = (x, y)
                
                pickups_ip = lookupPickups(pickups, cell, p)
                dropoffs_ip = lookupDropoffs(dropoffs, cell, p)
                
                dsr = pickups_ip / (config.num_taxis * T_p)
                asr = dropoffs_ip / (config.num_taxis * T_p)
                
                if config.include_zero_cells or (dsr > 0 or asr > 0):
                    dsr_values.append(dsr)
                    asr_values.append(asr)
        
        # Compute Gini for this period
        gini_arrivals.append(compute_gini(asr_values))
        gini_departures.append(compute_gini(dsr_values))
    
    # Step 4: Aggregate
    avg_gini = 0.5 * (mean(gini_arrivals) + mean(gini_departures))
    F_spatial = 1.0 - avg_gini
    
    return F_spatial
```

### 5.3 Python Implementation Outline

```python
# File: objective_function/spatial_fairness/term.py

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np

from objective_function.base import ObjectiveFunctionTerm, TermMetadata, TermConfig


@dataclass
class SpatialFairnessConfig(TermConfig):
    """Configuration for spatial fairness term."""
    period_type: str = "time_bucket"  # "time_bucket", "hourly", "daily"
    grid_dims: Tuple[int, int] = (48, 90)
    num_taxis: int = 50
    num_days: float = 21.0
    include_zero_cells: bool = True


class SpatialFairnessTerm(ObjectiveFunctionTerm):
    """
    Spatial Fairness term based on Gini coefficient of service rates.
    
    Measures equality of taxi service distribution across geographic regions.
    Higher values indicate more equal distribution (more fair).
    """
    
    def _build_metadata(self) -> TermMetadata:
        return TermMetadata(
            name="spatial_fairness",
            display_name="Spatial Fairness",
            version="1.0.0",
            description="Gini-based measure of service distribution equality",
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=True,  # Gini is piecewise differentiable
            required_data=["pickup_dropoff_counts"],
            optional_data=["all_trajs"],
            author="FAMAIL Team",
            last_updated="2026-01-09"
        )
    
    def _validate_config(self) -> None:
        if self.config.period_type not in ["time_bucket", "hourly", "daily"]:
            raise ValueError(f"Invalid period_type: {self.config.period_type}")
        if self.config.num_taxis <= 0:
            raise ValueError("num_taxis must be positive")
        if self.config.num_days <= 0:
            raise ValueError("num_days must be positive")
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """Compute spatial fairness value."""
        # Get pickup/dropoff counts
        if 'pickup_dropoff_counts' in auxiliary_data:
            data = auxiliary_data['pickup_dropoff_counts']
        else:
            # Extract from trajectories if needed
            data = self._extract_counts_from_trajectories(trajectories)
        
        # Compute and return
        return self._compute_from_counts(data)
    
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute with detailed breakdown."""
        # Implementation with per-period breakdown
        # Returns gini coefficients per period, statistics, etc.
        pass
    
    def _compute_from_counts(self, data: Dict[Tuple, List[int]]) -> float:
        """Core computation from pickup/dropoff counts."""
        # Extract by period
        pickups, dropoffs = self._extract_by_period(data)
        
        # Get unique periods
        periods = set()
        for key in pickups.keys():
            periods.add(key[1])  # (cell, period) format
        
        # Compute Gini per period
        gini_arrivals = []
        gini_departures = []
        
        for p in periods:
            asr_values = self._compute_service_rates(dropoffs, p, is_arrivals=True)
            dsr_values = self._compute_service_rates(pickups, p, is_arrivals=False)
            
            gini_arrivals.append(self._compute_gini(asr_values))
            gini_departures.append(self._compute_gini(dsr_values))
        
        # Aggregate
        avg_gini = 0.5 * (np.mean(gini_arrivals) + np.mean(gini_departures))
        return 1.0 - avg_gini
    
    @staticmethod
    def _compute_gini(values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        n = len(values)
        if n == 0 or np.sum(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        mean_value = np.mean(sorted_values)
        
        weights = np.arange(n, 0, -1)
        weighted_sum = np.sum(weights * sorted_values)
        
        gini = 1 + (1/n) - (2 / (n**2 * mean_value)) * weighted_sum
        return max(0.0, min(1.0, gini))
```

### 5.4 Computational Considerations

#### 5.4.1 Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Data loading | $O(K)$ where $K$ = number of keys |
| Period aggregation | $O(K)$ |
| Service rate computation | $O(|P| \cdot |G|)$ where $|G|$ = grid size |
| Gini coefficient (per period) | $O(|G| \log |G|)$ (sorting) |
| **Total** | $O(K + |P| \cdot |G| \log |G|)$ |

For FAMAIL data: $K \approx 234,000$, $|P| \leq 1728$ (288 × 6), $|G| = 4320$ (48 × 90)

#### 5.4.2 Memory Considerations

- Keep only aggregated counts in memory, not raw data
- Stream processing possible for very large datasets
- Service rate arrays: $O(|G|)$ per period

#### 5.4.3 Optimization Opportunities

1. **Vectorization**: Use NumPy for batch operations
2. **Caching**: Cache Gini coefficients for unchanged periods
3. **Sparse representation**: Skip zero-only periods
4. **Parallelization**: Compute periods in parallel

---

## 6. Configuration Parameters

### 6.1 Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid_dims` | Tuple[int, int] | Spatial grid dimensions (default: (48, 90)) |
| `num_taxis` | int | Number of active taxis (default: 50) |

### 6.2 Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period_type` | str | "time_bucket" | Temporal aggregation granularity |
| `num_days` | float | 21.0 | Number of days in dataset |
| `include_zero_cells` | bool | True | Include cells with zero activity |
| `weight` | float | 1.0 | Weight in objective function ($\alpha_2$) |
| `cache_intermediate` | bool | True | Cache Gini coefficients |
| `verbose` | bool | False | Print debug information |

### 6.3 Default Values and Rationale

```python
DEFAULT_CONFIG = SpatialFairnessConfig(
    # Spatial dimensions
    grid_dims=(48, 90),       # Matches Shenzhen grid (0.01° resolution)
    
    # Fleet parameters
    num_taxis=50,             # 50 expert drivers in dataset
    num_days=21.0,            # July 2016 weekdays (excludes Sundays)
    
    # Temporal aggregation
    period_type="time_bucket", # Finest granularity (288 periods/day)
    
    # Computation options
    include_zero_cells=True,   # Include inactive cells for complete picture
    weight=1.0,                # Equal weight (adjusted during integration)
    cache_intermediate=True,   # Improve performance for repeated calls
    verbose=False,
)
```

**Rationale for Defaults**:

- `period_type="time_bucket"`: Captures temporal variation in inequality
- `include_zero_cells=True`: Zero-activity cells contribute to inequality measure
- `num_days=21.0`: July 2016 has 21 non-Sunday days in the dataset

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### 7.1.1 Gini Coefficient Tests

```python
class TestGiniCoefficient:
    """Test Gini coefficient computation."""
    
    def test_perfect_equality(self):
        """All values equal → Gini = 0."""
        values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert SpatialFairnessTerm._compute_gini(values) == 0.0
    
    def test_maximum_inequality(self):
        """One value, rest zero → Gini ≈ 1."""
        values = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        gini = SpatialFairnessTerm._compute_gini(values)
        assert gini > 0.8  # Should be close to 1
    
    def test_moderate_inequality(self):
        """Unequal distribution → 0 < Gini < 1."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini = SpatialFairnessTerm._compute_gini(values)
        assert 0.0 < gini < 1.0
    
    def test_empty_array(self):
        """Empty array → Gini = 0."""
        values = np.array([])
        assert SpatialFairnessTerm._compute_gini(values) == 0.0
    
    def test_all_zeros(self):
        """All zeros → Gini = 0 (no inequality among zeros)."""
        values = np.array([0.0, 0.0, 0.0])
        assert SpatialFairnessTerm._compute_gini(values) == 0.0
    
    def test_single_value(self):
        """Single value → Gini = 0."""
        values = np.array([5.0])
        assert SpatialFairnessTerm._compute_gini(values) == 0.0
    
    def test_known_value(self):
        """Test against known Gini coefficient."""
        # Income distribution: [1, 2, 3, 4, 5] has Gini ≈ 0.267
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        gini = SpatialFairnessTerm._compute_gini(values)
        assert abs(gini - 0.267) < 0.05
```

#### 7.1.2 Spatial Fairness Term Tests

```python
class TestSpatialFairnessTerm:
    """Test spatial fairness term computation."""
    
    @pytest.fixture
    def config(self):
        return SpatialFairnessConfig(
            grid_dims=(4, 4),  # Small grid for testing
            num_taxis=1,
            num_days=1.0,
            period_type="daily"
        )
    
    def test_perfect_equality(self, config):
        """Uniform distribution → F_spatial = 1.0."""
        # All cells have same count
        data = {
            (x, y, 1, 1): [10, 10]
            for x in range(4) for y in range(4)
        }
        term = SpatialFairnessTerm(config)
        result = term._compute_from_counts(data)
        assert abs(result - 1.0) < 0.01
    
    def test_maximum_inequality(self, config):
        """All activity in one cell → F_spatial ≈ 0."""
        data = {(0, 0, 1, 1): [100, 100]}
        for x in range(4):
            for y in range(4):
                if (x, y) != (0, 0):
                    data[(x, y, 1, 1)] = [0, 0]
        
        term = SpatialFairnessTerm(config)
        result = term._compute_from_counts(data)
        assert result < 0.2
    
    def test_output_range(self, config):
        """Output always in [0, 1]."""
        # Random data
        import random
        data = {
            (x, y, 1, 1): [random.randint(0, 100), random.randint(0, 100)]
            for x in range(4) for y in range(4)
        }
        term = SpatialFairnessTerm(config)
        result = term._compute_from_counts(data)
        assert 0.0 <= result <= 1.0
    
    def test_determinism(self, config):
        """Same input → same output."""
        data = {
            (x, y, 1, 1): [x + y, x * y]
            for x in range(4) for y in range(4)
        }
        term = SpatialFairnessTerm(config)
        result1 = term._compute_from_counts(data)
        result2 = term._compute_from_counts(data)
        assert result1 == result2
```

### 7.2 Integration Tests

```python
class TestSpatialFairnessIntegration:
    """Integration tests with real data."""
    
    def test_with_real_data(self):
        """Test with actual pickup_dropoff_counts.pkl."""
        import pickle
        with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
            data = pickle.load(f)
        
        config = SpatialFairnessConfig()
        term = SpatialFairnessTerm(config)
        
        # Validate input
        is_valid, errors = term.validate_input({}, {'pickup_dropoff_counts': data})
        assert is_valid, f"Validation errors: {errors}"
        
        # Compute
        result = term._compute_from_counts(data)
        
        # Check reasonable range
        assert 0.0 <= result <= 1.0
        
        # In real data, expect some inequality (not perfect)
        assert result < 0.95
        # But not extreme inequality either
        assert result > 0.2
```

### 7.3 Validation with Real Data

**Expected Behavior**:

Based on Su et al. (2018) findings, Shenzhen taxi data should exhibit:

1. **Moderate inequality**: $F_{\text{spatial}}$ between 0.4 and 0.8
2. **Temporal variation**: Higher inequality during peak hours
3. **Pickup vs. Dropoff difference**: Slight difference between $G_a$ and $G_d$

**Validation Script**:

```python
def validate_spatial_fairness_with_literature():
    """Validate against Su et al. (2018) findings."""
    import pickle
    
    # Load data
    with open('source_data/pickup_dropoff_counts.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Compute per-hour Gini coefficients
    hourly_gini = {}
    for hour in range(24):
        time_buckets = range(hour * 12 + 1, (hour + 1) * 12 + 1)
        # ... aggregate and compute Gini
    
    # Verify temporal patterns match literature
    # Peak hours should have higher Gini (more inequality)
    peak_hours = [8, 9, 17, 18]  # Morning and evening rush
    off_peak = [2, 3, 4, 5]      # Late night
    
    avg_peak_gini = np.mean([hourly_gini[h] for h in peak_hours])
    avg_offpeak_gini = np.mean([hourly_gini[h] for h in off_peak])
    
    # Peak hours typically show more inequality
    print(f"Peak hour avg Gini: {avg_peak_gini:.3f}")
    print(f"Off-peak avg Gini: {avg_offpeak_gini:.3f}")
```

---

## 8. Expected Challenges

### 8.1 Known Difficulties

#### 8.1.1 Sparse Data Handling

**Challenge**: Many cells have zero activity in any given time period.

**Impact**: Gini coefficient interpretation changes with many zeros.

**Mitigation**:
- Option to include/exclude zero cells (`include_zero_cells`)
- Document the effect on interpretation
- Consider alternative metrics for sparse data

#### 8.1.2 Temporal Aggregation Choice

**Challenge**: Choice of period granularity affects results.

**Impact**: Too fine → noisy; too coarse → loses temporal patterns.

**Mitigation**:
- Support multiple granularities
- Document trade-offs
- Recommend "hourly" as balanced default

#### 8.1.3 Edge Effects

**Challenge**: Grid boundary cells may have different characteristics.

**Impact**: Boundary cells may appear underserved due to data truncation.

**Mitigation**:
- Consider excluding boundary cells
- Document limitation
- Validate against known geographic patterns

### 8.2 Mitigation Strategies

| Challenge | Strategy | Implementation |
|-----------|----------|----------------|
| Sparse data | Configurable zero-cell handling | `include_zero_cells` parameter |
| Period choice | Multi-granularity support | `period_type` parameter |
| Edge effects | Boundary cell handling | Future: `exclude_boundary_cells` option |
| Interpretation | Detailed breakdown | `compute_with_breakdown()` method |
| Performance | Caching and vectorization | NumPy, intermediate caching |

---

## 9. Development Milestones

### 9.1 Phase 1: Core Implementation (Week 1-2)

- [ ] **M1.1**: Set up directory structure and package
- [ ] **M1.2**: Implement `SpatialFairnessConfig` dataclass
- [ ] **M1.3**: Implement `_compute_gini()` helper function
- [ ] **M1.4**: Implement `_extract_by_period()` method
- [ ] **M1.5**: Implement `compute()` method
- [ ] **M1.6**: Implement `_build_metadata()` and `_validate_config()`

**Deliverables**:
- Working `SpatialFairnessTerm` class
- Passes basic unit tests
- Documentation complete

### 9.2 Phase 2: Testing and Validation (Week 2-3)

- [ ] **M2.1**: Complete unit test suite
- [ ] **M2.2**: Integration tests with real data
- [ ] **M2.3**: Validate against literature (Su et al.)
- [ ] **M2.4**: Implement `compute_with_breakdown()`
- [ ] **M2.5**: Performance benchmarking

**Deliverables**:
- >90% test coverage
- Validation report
- Performance baseline

### 9.3 Phase 3: Integration (Week 3-4)

- [ ] **M3.1**: Integrate with base `ObjectiveFunctionTerm` interface
- [ ] **M3.2**: Test with combined objective function
- [ ] **M3.3**: Document API and usage
- [ ] **M3.4**: Code review and cleanup
- [ ] **M3.5**: Final documentation

**Deliverables**:
- Integration-ready module
- Complete documentation
- Code review approved

---

## 10. Appendix

### 10.1 Code Snippets

#### 10.1.1 Complete Gini Coefficient Implementation

```python
import numpy as np
from typing import Union

def compute_gini_coefficient(
    values: Union[np.ndarray, list],
    handle_zeros: str = "include"
) -> float:
    """
    Compute the Gini coefficient for a distribution.
    
    Args:
        values: Array of non-negative values
        handle_zeros: How to handle zeros
            - "include": Include zeros in calculation
            - "exclude": Remove zeros before calculation
            - "replace": Replace zeros with small epsilon
    
    Returns:
        Gini coefficient in [0, 1]
        
    Example:
        >>> compute_gini_coefficient([1, 1, 1, 1])
        0.0
        >>> compute_gini_coefficient([0, 0, 0, 1])
        0.75
    """
    values = np.asarray(values, dtype=np.float64)
    
    # Handle zeros
    if handle_zeros == "exclude":
        values = values[values > 0]
    elif handle_zeros == "replace":
        values = np.where(values == 0, 1e-10, values)
    
    n = len(values)
    if n == 0:
        return 0.0
    
    total = np.sum(values)
    if total == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    mean_value = total / n
    
    # Weighted sum: sum of (n - i + 1) * x_(i) for i = 1 to n
    # Using descending weights [n, n-1, ..., 2, 1]
    weights = np.arange(n, 0, -1)
    weighted_sum = np.dot(weights, sorted_values)
    
    # Gini formula
    gini = 1.0 + (1.0 / n) - (2.0 / (n * n * mean_value)) * weighted_sum
    
    # Clamp to valid range (numerical precision)
    return max(0.0, min(1.0, gini))
```

### 10.2 Sample Data

#### 10.2.1 Expected Data Format

```python
# Sample pickup_dropoff_counts.pkl structure
sample_data = {
    # High activity area (downtown)
    (24, 45, 144, 1): [50, 45],   # Monday noon
    (24, 45, 144, 2): [52, 48],   # Tuesday noon
    
    # Medium activity area
    (10, 30, 144, 1): [15, 12],
    (10, 30, 144, 2): [18, 14],
    
    # Low activity area (outskirts)
    (2, 85, 144, 1): [2, 1],
    (2, 85, 144, 2): [1, 2],
    
    # Zero activity (excluded area)
    (0, 0, 144, 1): [0, 0],
}
```

### 10.3 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial development plan |
| 1.0.1 | 2026-01-10 | Clarify that only pickups can be inferred from `all_trajs.pkl` data |

---

*This document serves as the comprehensive development guide for the Spatial Fairness term. All implementation should follow the specifications outlined here.*
