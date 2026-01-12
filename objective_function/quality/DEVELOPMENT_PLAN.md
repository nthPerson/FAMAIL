# Quality Term ($F_{\text{quality}}$) Development Plan

> ⚠️ **DEPRECATED**: This term has been eliminated from the FAMAIL objective function as of January 12, 2026.
> See [DEPRECATED.md](./DEPRECATED.md) for rationale and migration notes.
> The Trajectory Fidelity Term ($F_{\text{fidelity}}$) now handles trajectory realism validation.

---

## Document Metadata

| Property | Value |
|----------|-------|
| **Term Name** | Trajectory Quality |
| **Symbol** | $F_{\text{quality}}$ |
| **Version** | 1.0.0 |
| **Last Updated** | 2026-01-09 |
| **Status** | ~~Development Planning~~ **DEPRECATED** |
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

The **Quality Term** ($F_{\text{quality}}$) measures the operational quality and efficiency of edited trajectories from a taxi driver perspective. While fidelity measures whether trajectories look realistic, quality measures whether they represent good taxi driving behavior.

**Core Principle**: High-quality trajectories should exhibit characteristics of successful taxi operations:
- Efficient routing (minimal unnecessary travel)
- Appropriate passenger acquisition patterns
- Reasonable temporal efficiency
- Spatial coherence and continuity

### 1.2 Role in Objective Function

The quality term complements fidelity in the FAMAIL objective function:

$$
\mathcal{L} = \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} + \alpha_4 F_{\text{quality}}
$$

- **Weight**: $\alpha_4$ (typically 0.1-0.3 of total weight)
- **Optimization Direction**: **Maximize** (higher values = better quality)
- **Value Range**: [0, 1]
  - $F_{\text{quality}} = 1$: Optimal taxi driving behavior
  - $F_{\text{quality}} = 0$: Inefficient/poor driving behavior

### 1.3 Relationship to Other Terms

| Related Term | Relationship |
|--------------|-------------|
| $F_{\text{fidelity}}$ | **Complementary**: Fidelity = looks real, Quality = is good |
| $F_{\text{spatial}}$, $F_{\text{causal}}$ | **Trade-off**: Fairness may require suboptimal routes |
| cGAIL | **Aligned**: Quality metrics match imitation learning rewards |

### 1.4 Key Insights

**Why Quality Matters**:

1. **Efficiency**: Better trajectories → more efficient taxi service
2. **Realism**: Expert drivers exhibit quality characteristics
3. **Generalization**: Quality trajectories generalize better to new scenarios
4. **Balance**: Prevents fairness optimization from degrading service

**Example Quality Metrics**:
- Trip completion rate (passenger acquired → dropped off)
- Deadhead distance (travel without passenger)
- Temporal efficiency (time per trip)
- Route directness

---

## 2. Mathematical Formulation

### 2.1 Core Formula

The quality term is a weighted combination of multiple quality metrics:

$$
F_{\text{quality}} = \sum_{k=1}^{K} \beta_k Q_k
$$

Where:
- $Q_k$ = individual quality metric (each normalized to [0, 1])
- $\beta_k$ = weight for metric $k$ ($\sum_k \beta_k = 1$)

### 2.2 Component Definitions

#### 2.2.1 Spatial Continuity ($Q_{\text{continuity}}$)

Measures whether trajectories follow physically possible paths:

$$
Q_{\text{continuity}} = 1 - \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \frac{\text{discontinuities}(\tau')}{|\tau'| - 1}
$$

Where:
- $\text{discontinuities}(\tau')$ = number of "jumps" (non-adjacent grid transitions)

**Definition of Discontinuity**:
A state transition $s_t \to s_{t+1}$ is discontinuous if:
$$
|x_{t+1} - x_t| > 1 \quad \text{OR} \quad |y_{t+1} - y_t| > 1
$$

**Interpretation**:
- $Q_{\text{continuity}} = 1$: All transitions are to adjacent cells (valid movement)
- $Q_{\text{continuity}} = 0$: All transitions are "teleportation" (invalid)

#### 2.2.2 Route Efficiency ($Q_{\text{efficiency}}$)

Measures how directly trajectories connect origin to destination:

$$
Q_{\text{efficiency}} = \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \frac{d_{\text{manhattan}}(s_0, s_T)}{|\tau'|}
$$

Where:
- $d_{\text{manhattan}}(s_0, s_T)$ = Manhattan distance from start to end
- $|\tau'|$ = trajectory length (number of states)

**Interpretation**:
- Higher values indicate more direct routes
- Normalized by trajectory length to account for detours

**Note**: This is a simplified efficiency metric. More sophisticated versions could consider:
- Road network constraints
- Traffic conditions
- Pickup/dropoff necessity

#### 2.2.3 Temporal Consistency ($Q_{\text{temporal}}$)

Measures whether time progresses appropriately:

$$
Q_{\text{temporal}} = 1 - \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \frac{\text{time\_violations}(\tau')}{|\tau'| - 1}
$$

Where:
- $\text{time\_violations}(\tau')$ = count of backward time jumps or unrealistic gaps

**Definition of Time Violation**:
A transition is a time violation if:
$$
t_{t+1} < t_t \quad \text{(backward)} \quad \text{OR} \quad t_{t+1} - t_t > \delta_{\max} \quad \text{(gap)}
$$

Where $\delta_{\max}$ = maximum allowed time gap (e.g., 30 time buckets = 2.5 hours).

#### 2.2.4 Action Consistency ($Q_{\text{action}}$)

Measures whether actions match state transitions:

$$
Q_{\text{action}} = \frac{1}{|\mathcal{T}'|} \sum_{\tau' \in \mathcal{T}'} \frac{\text{consistent\_actions}(\tau')}{|\tau'| - 1}
$$

An action is consistent if the action code matches the actual state change:

| Action Code | Expected Δx | Expected Δy |
|-------------|-------------|-------------|
| 0 (North) | 0 | +1 |
| 1 (NE) | +1 | +1 |
| 2 (East) | +1 | 0 |
| 3 (SE) | +1 | -1 |
| 4 (South) | 0 | -1 |
| 5 (SW) | -1 | -1 |
| 6 (West) | -1 | 0 |
| 7 (NW) | -1 | +1 |
| 8 (Stay) | 0 | 0 |

#### 2.2.5 Feature Validity ($Q_{\text{features}}$)

Measures whether contextual features are within valid ranges:

$$
Q_{\text{features}} = 1 - \frac{1}{|\mathcal{T}'| \cdot M} \sum_{\tau' \in \mathcal{T}'} \sum_{s \in \tau'} \text{invalid\_features}(s)
$$

Where:
- $M$ = number of validated features
- $\text{invalid\_features}(s)$ = count of features outside valid ranges

**Valid Feature Ranges**:
| Feature | Valid Range |
|---------|-------------|
| `x_grid` | [0, 47] |
| `y_grid` | [0, 89] |
| `time_bucket` | [0, 287] |
| `day_index` | [0, 6] |
| Normalized features | [-5, 5] (typical, based on normalization) |

### 2.3 Aggregated Quality Formula

Combining all components with default weights:

$$
F_{\text{quality}} = \beta_1 Q_{\text{continuity}} + \beta_2 Q_{\text{efficiency}} + \beta_3 Q_{\text{temporal}} + \beta_4 Q_{\text{action}} + \beta_5 Q_{\text{features}}
$$

Default weights: $\beta = [0.3, 0.2, 0.2, 0.2, 0.1]$

### 2.4 Derivation and Justification

**Why Multiple Metrics?**

1. **Comprehensive**: Captures different aspects of quality
2. **Robust**: Single metric can be gamed; multiple are harder
3. **Interpretable**: Each metric has clear meaning
4. **Configurable**: Weights can be adjusted for different priorities

**Priority Ranking**:
1. **Continuity** (0.3): Most fundamental—trajectories must be physically possible
2. **Temporal** (0.2): Time must progress correctly
3. **Action** (0.2): Actions should match movements
4. **Efficiency** (0.2): Routes should be reasonably direct
5. **Features** (0.1): Features should be valid

---

## 3. Literature and References

### 3.1 Primary Sources

#### 3.1.1 Trajectory Quality in Imitation Learning

**Ho & Ermon (2016)**: Generative Adversarial Imitation Learning
- Quality as discriminator objective
- Trajectory realism requirements

#### 3.1.2 Taxi Trajectory Analysis

**Yuan et al. (2013)**: T-Drive: Driving Directions Based on Taxi Trajectories
- Route efficiency metrics
- Trajectory quality for navigation

### 3.2 Theoretical Foundation

**Valid Trajectory Properties**:
1. **Markov property**: Next state depends on current state
2. **Physical constraints**: Movement limited to adjacent cells
3. **Temporal monotonicity**: Time progresses forward
4. **Action-state consistency**: Actions produce correct state changes

### 3.3 Related Work in FAMAIL

| Component | Relationship |
|-----------|-------------|
| cGAIL | Uses reward signals related to trajectory quality |
| Discriminator | Implicitly captures some quality aspects |
| State representation | 126-element vector defines quality constraints |

---

## 4. Data Requirements

### 4.1 Required Datasets

#### 4.1.1 Trajectory Data: `all_trajs.pkl`

**Location**: `FAMAIL/source_data/all_trajs.pkl`

**Usage**: Both original and edited trajectories must be evaluated.

**State Vector Fields for Quality**:
| Index | Field | Quality Check |
|-------|-------|--------------|
| 0 | `x_grid` | Range, continuity |
| 1 | `y_grid` | Range, continuity |
| 2 | `time_bucket` | Range, progression |
| 3 | `day_index` | Range |
| 125 | `action_code` | Consistency with movement |

### 4.2 Data Preprocessing

#### 4.2.1 Extracting State Transitions

```python
def extract_transitions(trajectory: List[List[float]]) -> List[Dict]:
    """
    Extract state transitions from trajectory.
    
    Returns:
        List of transition dictionaries with before/after states
    """
    transitions = []
    
    for i in range(len(trajectory) - 1):
        prev_state = trajectory[i]
        curr_state = trajectory[i + 1]
        
        transitions.append({
            'prev_x': int(prev_state[0]),
            'prev_y': int(prev_state[1]),
            'prev_time': int(prev_state[2]),
            'prev_day': int(prev_state[3]),
            'curr_x': int(curr_state[0]),
            'curr_y': int(curr_state[1]),
            'curr_time': int(curr_state[2]),
            'curr_day': int(curr_state[3]),
            'action': int(prev_state[125]),
        })
    
    return transitions
```

### 4.3 Data Validation

```python
def validate_trajectory_structure(
    trajectory: List[List[float]]
) -> List[str]:
    """
    Validate basic trajectory structure.
    
    Returns:
        List of validation errors
    """
    errors = []
    
    if len(trajectory) < 2:
        errors.append("Trajectory too short (< 2 states)")
    
    for i, state in enumerate(trajectory):
        if len(state) != 126:
            errors.append(f"State {i} has {len(state)} features, expected 126")
    
    return errors
```

---

## 5. Implementation Plan

### 5.1 Algorithm Steps

```
ALGORITHM: Compute Trajectory Quality Term
═══════════════════════════════════════════════════════════════════════

INPUT:
  - trajectories: Dict[driver_id → List[trajectory]]
  - config: QualityConfig
    - weights: Dict[str → float]
    - max_time_gap: int
    - validate_features: bool

OUTPUT:
  - F_quality: float ∈ [0, 1] (higher = better quality)

STEPS:

1. INITIALIZE METRICS
   ─────────────────────────────────
   For each metric k:
     metric_values[k] = []

2. COMPUTE PER-TRAJECTORY METRICS
   ─────────────────────────────────
   For each trajectory τ':
     2.1 Compute continuity score
     2.2 Compute efficiency score
     2.3 Compute temporal consistency score
     2.4 Compute action consistency score
     2.5 Compute feature validity score
     
     Store all scores

3. AGGREGATE METRICS
   ─────────────────────────────────
   For each metric k:
     Q_k = mean(metric_values[k])

4. COMBINE WITH WEIGHTS
   ─────────────────────────────────
   F_quality = Σ_k β_k * Q_k

5. RETURN F_quality
```

### 5.2 Pseudocode

```
function compute_quality(trajectories, config):
    all_continuity = []
    all_efficiency = []
    all_temporal = []
    all_action = []
    all_features = []
    
    for driver_id, driver_trajs in trajectories:
        for traj in driver_trajs:
            transitions = extract_transitions(traj)
            
            # Continuity
            discontinuities = count_discontinuities(transitions)
            continuity = 1 - (discontinuities / max(1, len(transitions)))
            all_continuity.append(continuity)
            
            # Efficiency
            if len(traj) > 1:
                start = (traj[0][0], traj[0][1])
                end = (traj[-1][0], traj[-1][1])
                manhattan = abs(end[0] - start[0]) + abs(end[1] - start[1])
                efficiency = manhattan / len(traj)
                all_efficiency.append(efficiency)
            
            # Temporal
            time_violations = count_time_violations(transitions, config.max_time_gap)
            temporal = 1 - (time_violations / max(1, len(transitions)))
            all_temporal.append(temporal)
            
            # Action consistency
            action_matches = count_action_matches(transitions)
            action = action_matches / max(1, len(transitions))
            all_action.append(action)
            
            # Feature validity
            if config.validate_features:
                invalid = count_invalid_features(traj)
                features = 1 - (invalid / (len(traj) * 126))
                all_features.append(features)
    
    # Aggregate
    Q = {
        'continuity': mean(all_continuity),
        'efficiency': min(1, mean(all_efficiency)),  # Cap at 1
        'temporal': mean(all_temporal),
        'action': mean(all_action),
        'features': mean(all_features) if all_features else 1.0,
    }
    
    # Weighted combination
    F_quality = sum(config.weights[k] * Q[k] for k in Q)
    
    return F_quality
```

### 5.3 Python Implementation Outline

```python
# File: objective_function/quality/term.py

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import numpy as np

from objective_function.base import ObjectiveFunctionTerm, TermMetadata, TermConfig


@dataclass
class QualityConfig(TermConfig):
    """Configuration for quality term."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'continuity': 0.3,
        'efficiency': 0.2,
        'temporal': 0.2,
        'action': 0.2,
        'features': 0.1,
    })
    max_time_gap: int = 30  # Time buckets (2.5 hours)
    validate_features: bool = True
    grid_dims: Tuple[int, int] = (48, 90)
    time_buckets: int = 288


class QualityTerm(ObjectiveFunctionTerm):
    """
    Trajectory Quality term measuring operational characteristics.
    
    Evaluates spatial continuity, temporal consistency, route efficiency,
    action-state consistency, and feature validity.
    """
    
    # Action code to delta mapping
    ACTION_DELTAS = {
        0: (0, 1),    # North
        1: (1, 1),    # NE
        2: (1, 0),    # East
        3: (1, -1),   # SE
        4: (0, -1),   # South
        5: (-1, -1),  # SW
        6: (-1, 0),   # West
        7: (-1, 1),   # NW
        8: (0, 0),    # Stay
        9: (0, 0),    # Stay (alternate)
    }
    
    def _build_metadata(self) -> TermMetadata:
        return TermMetadata(
            name="quality",
            display_name="Trajectory Quality",
            version="1.0.0",
            description="Multi-metric measure of trajectory operational quality",
            value_range=(0.0, 1.0),
            higher_is_better=True,
            is_differentiable=False,  # Discrete metrics
            required_data=[],
            optional_data=["all_trajs"],
            author="FAMAIL Team",
            last_updated="2026-01-09"
        )
    
    def _validate_config(self) -> None:
        # Check weights sum to ~1
        weight_sum = sum(self.config.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1, got {weight_sum}")
        
        if self.config.max_time_gap <= 0:
            raise ValueError("max_time_gap must be positive")
    
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """Compute quality value."""
        metrics = self._compute_all_metrics(trajectories)
        
        # Weighted combination
        quality = sum(
            self.config.weights.get(k, 0) * v
            for k, v in metrics.items()
        )
        
        return float(np.clip(quality, 0.0, 1.0))
    
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute with detailed breakdown."""
        metrics = self._compute_all_metrics(trajectories)
        
        return {
            'value': self.compute(trajectories, auxiliary_data),
            'components': metrics,
            'statistics': self._compute_statistics(trajectories),
            'diagnostics': {}
        }
    
    def _compute_all_metrics(
        self,
        trajectories: Dict[str, List[List[List[float]]]]
    ) -> Dict[str, float]:
        """Compute all quality metrics."""
        continuity_scores = []
        efficiency_scores = []
        temporal_scores = []
        action_scores = []
        feature_scores = []
        
        for driver_trajs in trajectories.values():
            for traj in driver_trajs:
                if len(traj) < 2:
                    continue
                
                # Continuity
                cont = self._compute_continuity(traj)
                continuity_scores.append(cont)
                
                # Efficiency
                eff = self._compute_efficiency(traj)
                efficiency_scores.append(eff)
                
                # Temporal
                temp = self._compute_temporal(traj)
                temporal_scores.append(temp)
                
                # Action
                act = self._compute_action_consistency(traj)
                action_scores.append(act)
                
                # Features
                if self.config.validate_features:
                    feat = self._compute_feature_validity(traj)
                    feature_scores.append(feat)
        
        return {
            'continuity': np.mean(continuity_scores) if continuity_scores else 1.0,
            'efficiency': min(1.0, np.mean(efficiency_scores)) if efficiency_scores else 1.0,
            'temporal': np.mean(temporal_scores) if temporal_scores else 1.0,
            'action': np.mean(action_scores) if action_scores else 1.0,
            'features': np.mean(feature_scores) if feature_scores else 1.0,
        }
    
    def _compute_continuity(self, traj: List[List[float]]) -> float:
        """Compute spatial continuity score."""
        discontinuities = 0
        
        for i in range(len(traj) - 1):
            x1, y1 = int(traj[i][0]), int(traj[i][1])
            x2, y2 = int(traj[i+1][0]), int(traj[i+1][1])
            
            if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
                discontinuities += 1
        
        return 1.0 - discontinuities / max(1, len(traj) - 1)
    
    def _compute_efficiency(self, traj: List[List[float]]) -> float:
        """Compute route efficiency score."""
        start_x, start_y = int(traj[0][0]), int(traj[0][1])
        end_x, end_y = int(traj[-1][0]), int(traj[-1][1])
        
        manhattan = abs(end_x - start_x) + abs(end_y - start_y)
        
        # Efficiency = direct distance / actual distance
        # Capped at 1.0 (can't be more efficient than direct)
        return min(1.0, manhattan / len(traj)) if len(traj) > 0 else 1.0
    
    def _compute_temporal(self, traj: List[List[float]]) -> float:
        """Compute temporal consistency score."""
        violations = 0
        
        for i in range(len(traj) - 1):
            t1 = int(traj[i][2])
            t2 = int(traj[i+1][2])
            d1 = int(traj[i][3])
            d2 = int(traj[i+1][3])
            
            # Check backward time (within same day)
            if d1 == d2 and t2 < t1:
                violations += 1
            
            # Check excessive gap
            if d1 == d2:
                gap = t2 - t1
                if gap > self.config.max_time_gap:
                    violations += 1
        
        return 1.0 - violations / max(1, len(traj) - 1)
    
    def _compute_action_consistency(self, traj: List[List[float]]) -> float:
        """Compute action-state consistency score."""
        matches = 0
        
        for i in range(len(traj) - 1):
            x1, y1 = int(traj[i][0]), int(traj[i][1])
            x2, y2 = int(traj[i+1][0]), int(traj[i+1][1])
            action = int(traj[i][125])
            
            expected_dx, expected_dy = self.ACTION_DELTAS.get(action, (0, 0))
            actual_dx, actual_dy = x2 - x1, y2 - y1
            
            if actual_dx == expected_dx and actual_dy == expected_dy:
                matches += 1
        
        return matches / max(1, len(traj) - 1)
    
    def _compute_feature_validity(self, traj: List[List[float]]) -> float:
        """Compute feature validity score."""
        invalid_count = 0
        total_checks = 0
        
        for state in traj:
            # Grid coordinates
            if not (0 <= state[0] < self.config.grid_dims[0]):
                invalid_count += 1
            total_checks += 1
            
            if not (0 <= state[1] < self.config.grid_dims[1]):
                invalid_count += 1
            total_checks += 1
            
            # Time bucket
            if not (0 <= state[2] < self.config.time_buckets):
                invalid_count += 1
            total_checks += 1
            
            # Day index
            if not (0 <= state[3] <= 6):
                invalid_count += 1
            total_checks += 1
            
            # Action code
            if not (0 <= state[125] <= 9):
                invalid_count += 1
            total_checks += 1
        
        return 1.0 - invalid_count / max(1, total_checks)
```

### 5.4 Computational Considerations

#### 5.4.1 Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Per-trajectory metrics | $O(T)$ where $T$ = trajectory length |
| All trajectories | $O(N \cdot \bar{T})$ where $N$ = number of trajectories |

For FAMAIL: ~5000 trajectories × ~100 avg length = ~500,000 operations (fast).

#### 5.4.2 Memory Considerations

- Minimal memory: processes one trajectory at a time
- Can be parallelized across trajectories

---

## 6. Configuration Parameters

### 6.1 Required Parameters

None required—all have sensible defaults.

### 6.2 Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | Dict[str, float] | See below | Metric weights |
| `max_time_gap` | int | 30 | Max allowed time gap (buckets) |
| `validate_features` | bool | True | Check feature validity |
| `grid_dims` | Tuple[int, int] | (48, 90) | Grid dimensions |
| `time_buckets` | int | 288 | Number of time buckets |
| `weight` | float | 1.0 | Weight in objective function ($\alpha_4$) |

**Default Weights**:
```python
{
    'continuity': 0.3,   # Most important
    'efficiency': 0.2,
    'temporal': 0.2,
    'action': 0.2,
    'features': 0.1,     # Least important (usually valid)
}
```

### 6.3 Default Values and Rationale

```python
DEFAULT_CONFIG = QualityConfig(
    weights={
        'continuity': 0.3,  # Fundamental physical constraint
        'efficiency': 0.2,  # Important for taxi operations
        'temporal': 0.2,    # Time must progress correctly
        'action': 0.2,      # Actions should match movements
        'features': 0.1,    # Usually valid; sanity check
    },
    max_time_gap=30,        # 2.5 hours = reasonable for breaks
    validate_features=True,
    grid_dims=(48, 90),
    time_buckets=288,
)
```

**Rationale**:

- `continuity` highest weight: Invalid spatial transitions are most problematic
- `efficiency` moderate: Important but fairness may require detours
- `max_time_gap=30`: Allows for driver breaks while catching anomalies

---

## 7. Testing Strategy

### 7.1 Unit Tests

#### 7.1.1 Continuity Tests

```python
class TestContinuity:
    """Test spatial continuity computation."""
    
    def test_perfect_continuity(self):
        """Adjacent-only transitions → continuity = 1.0."""
        traj = [
            [5, 5, 0, 0] + [0.0] * 121 + [8],  # Stay
            [5, 6, 1, 0] + [0.0] * 121 + [0],  # North
            [6, 6, 2, 0] + [0.0] * 121 + [2],  # East
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_continuity(traj)
        assert result == 1.0
    
    def test_teleportation(self):
        """Large jump → low continuity."""
        traj = [
            [0, 0, 0, 0] + [0.0] * 121 + [0],
            [10, 10, 1, 0] + [0.0] * 121 + [0],  # Jump!
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_continuity(traj)
        assert result == 0.0
```

#### 7.1.2 Efficiency Tests

```python
class TestEfficiency:
    """Test route efficiency computation."""
    
    def test_direct_route(self):
        """Direct path → efficiency ~ 1.0."""
        # 5-step journey from (0,0) to (4,0)
        traj = [
            [i, 0, i, 0] + [0.0] * 121 + [2]  # East each step
            for i in range(5)
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_efficiency(traj)
        # Manhattan: 4, Length: 5, Ratio: 0.8
        assert abs(result - 0.8) < 0.01
    
    def test_round_trip(self):
        """Return to start → efficiency = 0."""
        traj = [
            [5, 5, 0, 0] + [0.0] * 121 + [0],
            [5, 6, 1, 0] + [0.0] * 121 + [4],
            [5, 5, 2, 0] + [0.0] * 121 + [0],  # Back to start
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_efficiency(traj)
        assert result == 0.0
```

#### 7.1.3 Action Consistency Tests

```python
class TestActionConsistency:
    """Test action-state consistency."""
    
    def test_matching_actions(self):
        """Actions match movements → action score = 1.0."""
        traj = [
            [5, 5, 0, 0] + [0.0] * 121 + [0],  # Action: North
            [5, 6, 1, 0] + [0.0] * 121 + [2],  # Move North, Action: East
            [6, 6, 2, 0] + [0.0] * 121 + [0],  # Move East
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_action_consistency(traj)
        assert result == 1.0
    
    def test_mismatched_actions(self):
        """Actions don't match → action score = 0."""
        traj = [
            [5, 5, 0, 0] + [0.0] * 121 + [4],  # Action: South (wrong)
            [5, 6, 1, 0] + [0.0] * 121 + [0],  # Moved North
        ]
        
        term = QualityTerm(QualityConfig())
        result = term._compute_action_consistency(traj)
        assert result == 0.0
```

### 7.2 Integration Tests

```python
class TestQualityIntegration:
    """Integration tests with real data."""
    
    def test_with_real_trajectories(self):
        """Test with actual all_trajs.pkl data."""
        import pickle
        
        with open('source_data/all_trajs.pkl', 'rb') as f:
            all_trajs = pickle.load(f)
        
        # Convert format
        trajectories = {
            str(k): v for k, v in all_trajs.items()
        }
        
        config = QualityConfig()
        term = QualityTerm(config)
        
        result = term.compute(trajectories, {})
        
        # Real data should have high quality
        assert result > 0.7
        assert result <= 1.0
    
    def test_breakdown(self):
        """Test compute_with_breakdown provides metrics."""
        import pickle
        
        with open('source_data/all_trajs.pkl', 'rb') as f:
            all_trajs = pickle.load(f)
        
        trajectories = {
            str(k): v[:3] for k, v in list(all_trajs.items())[:3]
        }
        
        config = QualityConfig()
        term = QualityTerm(config)
        
        result = term.compute_with_breakdown(trajectories, {})
        
        assert 'value' in result
        assert 'components' in result
        assert 'continuity' in result['components']
```

### 7.3 Validation with Real Data

**Expected Behavior**:

1. **Original trajectories**: Quality > 0.8 (expert drivers)
2. **Random modifications**: Quality drops significantly
3. **Reasonable edits**: Quality remains > 0.6

---

## 8. Expected Challenges

### 8.1 Known Difficulties

#### 8.1.1 Metric Sensitivity

**Challenge**: Some metrics may be overly sensitive to minor issues.

**Impact**: Small edits could cause large quality drops.

**Mitigation**:
- Configurable weights
- Tolerance thresholds for each metric
- Aggregation smooths individual outliers

#### 8.1.2 Trade-off with Fairness

**Challenge**: Improving fairness may require quality sacrifices.

**Impact**: Cannot optimize both fully.

**Mitigation**:
- Explicit weight balancing
- Constraint-based quality floors
- Multi-objective optimization visualization

### 8.2 Mitigation Strategies

| Challenge | Strategy | Implementation |
|-----------|----------|----------------|
| Metric sensitivity | Tolerances | Allow small violations |
| Trade-offs | Weight tuning | Experiment with $\alpha$ values |
| Edge cases | Robust aggregation | Median instead of mean option |

---

## 9. Development Milestones

### 9.1 Phase 1: Core Implementation (Week 1-2)

- [ ] **M1.1**: Set up directory structure
- [ ] **M1.2**: Implement `QualityConfig` dataclass
- [ ] **M1.3**: Implement continuity metric
- [ ] **M1.4**: Implement efficiency metric
- [ ] **M1.5**: Implement temporal metric
- [ ] **M1.6**: Implement action consistency metric
- [ ] **M1.7**: Implement feature validity metric
- [ ] **M1.8**: Implement aggregation and `compute()`

**Deliverables**:
- Working `QualityTerm` class
- All metrics implemented

### 9.2 Phase 2: Testing and Validation (Week 2-3)

- [ ] **M2.1**: Unit tests for each metric
- [ ] **M2.2**: Integration tests with real data
- [ ] **M2.3**: Validate metric interpretations
- [ ] **M2.4**: Implement `compute_with_breakdown()`
- [ ] **M2.5**: Performance optimization

**Deliverables**:
- Comprehensive test suite
- Validated metrics

### 9.3 Phase 3: Integration (Week 3-4)

- [ ] **M3.1**: Integrate with objective function
- [ ] **M3.2**: Test weight combinations
- [ ] **M3.3**: Document API and usage
- [ ] **M3.4**: Code review

**Deliverables**:
- Integration-ready module
- Complete documentation

---

## 10. Appendix

### 10.1 Action Code Reference

| Code | Direction | Δx | Δy |
|------|-----------|-----|-----|
| 0 | North | 0 | +1 |
| 1 | Northeast | +1 | +1 |
| 2 | East | +1 | 0 |
| 3 | Southeast | +1 | -1 |
| 4 | South | 0 | -1 |
| 5 | Southwest | -1 | -1 |
| 6 | West | -1 | 0 |
| 7 | Northwest | -1 | +1 |
| 8 | Stay | 0 | 0 |
| 9 | Stay (alt) | 0 | 0 |

### 10.2 Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-09 | Initial development plan |

---

*This document serves as the comprehensive development guide for the Trajectory Quality term. All implementation should follow the specifications outlined here.*
