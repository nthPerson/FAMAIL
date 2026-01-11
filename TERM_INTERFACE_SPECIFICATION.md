# FAMAIL Objective Function Term Interface Specification

## Document Purpose

This document defines the **standard interface and template** that all objective function terms in the FAMAIL project must implement. A consistent interface ensures:

1. **Modularity**: Terms can be developed, tested, and debugged independently
2. **Composability**: Terms integrate seamlessly into the complete objective function
3. **Maintainability**: Researchers can easily understand, modify, and extend individual terms
4. **Reproducibility**: Clear interfaces enable other researchers to replicate and build upon this work

---

## Table of Contents

1. [Interface Overview](#1-interface-overview)
2. [Abstract Term Class](#2-abstract-term-class)
3. [Required Methods](#3-required-methods)
4. [Data Contracts](#4-data-contracts)
5. [Configuration Patterns](#5-configuration-patterns)
6. [Testing Interface](#6-testing-interface)
7. [Documentation Template](#7-documentation-template)
8. [File Organization](#8-file-organization)
9. [Implementation Checklist](#9-implementation-checklist)

---

## 1. Interface Overview

### 1.1 Design Philosophy

Each objective function term is encapsulated as an independent module that:

- **Accepts standardized inputs**: Trajectory data, configuration parameters, and auxiliary datasets
- **Produces standardized outputs**: A scalar value in [0, 1] where higher values indicate better outcomes
- **Is differentiable** (where applicable): Supports gradient computation for optimization
- **Is deterministic**: Same inputs always produce the same outputs
- **Is self-contained**: Does not depend on the internal state of other terms

### 1.2 Term Categories

| Category | Terms | Optimization Direction |
|----------|-------|----------------------|
| **Fairness Terms** | $F_{\text{spatial}}$, $F_{\text{causal}}$ | Maximize (higher = more fair) |
| **Quality Terms** | $F_{\text{fidelity}}$, $F_{\text{quality}}$ | Maximize (higher = better quality) |

### 1.3 Common Properties

All terms share these properties:

```python
@dataclass
class TermMetadata:
    """Metadata describing an objective function term."""
    name: str                          # e.g., "spatial_fairness"
    display_name: str                  # e.g., "Spatial Fairness"
    version: str                       # Semantic version, e.g., "1.0.0"
    description: str                   # Brief description of what the term measures
    value_range: Tuple[float, float]   # Expected output range, typically (0.0, 1.0)
    higher_is_better: bool             # True for all FAMAIL terms
    is_differentiable: bool            # Whether gradients can be computed
    required_data: List[str]           # List of required dataset names
    optional_data: List[str]           # List of optional dataset names
    author: str                        # Primary developer/researcher
    last_updated: str                  # ISO date of last modification
```

---

## 2. Abstract Term Class

### 2.1 Base Class Definition

All objective function terms must inherit from the following abstract base class:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

class ObjectiveFunctionTerm(ABC):
    """
    Abstract base class for all FAMAIL objective function terms.
    
    All terms must implement this interface to ensure consistent behavior
    across the objective function framework.
    """
    
    def __init__(self, config: 'TermConfig'):
        """
        Initialize the term with configuration parameters.
        
        Args:
            config: Configuration object containing term-specific parameters
        """
        self.config = config
        self._validate_config()
        self._metadata = self._build_metadata()
    
    @abstractmethod
    def _build_metadata(self) -> 'TermMetadata':
        """Build and return the term's metadata."""
        pass
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def compute(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> float:
        """
        Compute the term value for the given trajectories.
        
        Args:
            trajectories: Dictionary mapping driver_id to list of trajectories,
                         where each trajectory is a list of states,
                         and each state is a list of features
            auxiliary_data: Dictionary containing additional datasets
                           (e.g., pickup_dropoff_counts, traffic data)
        
        Returns:
            Term value in [0, 1], where higher values indicate better outcomes
        """
        pass
    
    @abstractmethod
    def compute_with_breakdown(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute the term value with detailed breakdown for debugging/analysis.
        
        Args:
            trajectories: Trajectory data (same as compute())
            auxiliary_data: Additional datasets (same as compute())
        
        Returns:
            Dictionary containing:
                - 'value': float - the final term value
                - 'components': Dict - intermediate values used in computation
                - 'statistics': Dict - summary statistics for analysis
                - 'diagnostics': Dict - debugging information
        """
        pass
    
    def compute_gradient(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Compute the gradient of the term with respect to trajectory modifications.
        
        Default implementation returns None (term is not differentiable).
        Override this method for differentiable terms.
        
        Args:
            trajectories: Trajectory data
            auxiliary_data: Additional datasets
        
        Returns:
            Gradient array or None if not differentiable
        """
        return None
    
    @property
    def metadata(self) -> 'TermMetadata':
        """Return the term's metadata."""
        return self._metadata
    
    @property
    def name(self) -> str:
        """Return the term's internal name."""
        return self._metadata.name
    
    def validate_input(
        self,
        trajectories: Dict[str, List[List[List[float]]]],
        auxiliary_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the input data meets the term's requirements.
        
        Args:
            trajectories: Trajectory data to validate
            auxiliary_data: Additional datasets to validate
        
        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []
        
        # Check required data
        for required in self._metadata.required_data:
            if required not in auxiliary_data:
                errors.append(f"Missing required dataset: {required}")
        
        # Check trajectory format
        if not isinstance(trajectories, dict):
            errors.append("Trajectories must be a dictionary")
        
        return len(errors) == 0, errors
```

### 2.2 Configuration Base Class

```python
@dataclass
class TermConfig:
    """
    Base configuration class for objective function terms.
    
    Subclass this for term-specific configuration parameters.
    """
    # Common parameters
    enabled: bool = True                    # Whether this term is active
    weight: float = 1.0                     # Weight in objective function (α_i)
    
    # Data source paths
    raw_data_dir: str = "raw_data"          # Directory containing raw GPS data
    source_data_dir: str = "source_data"   # Directory containing processed data
    
    # Computation parameters
    cache_intermediate: bool = True         # Whether to cache intermediate results
    verbose: bool = False                   # Print debug information
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.weight < 0:
            raise ValueError("Weight must be non-negative")
```

---

## 3. Required Methods

### 3.1 Method Specifications

| Method | Required | Description |
|--------|----------|-------------|
| `__init__(config)` | Yes | Initialize with configuration |
| `_build_metadata()` | Yes | Return term metadata |
| `_validate_config()` | Yes | Validate configuration parameters |
| `compute(trajectories, auxiliary_data)` | Yes | Compute term value |
| `compute_with_breakdown(...)` | Yes | Compute with detailed breakdown |
| `compute_gradient(...)` | No | Compute gradient (if differentiable) |
| `validate_input(...)` | No | Validate input data (default provided) |

### 3.2 Return Value Specifications

#### 3.2.1 `compute()` Return Value

```python
# Must return a float in [0, 1]
# Higher values indicate better outcomes (more fair, higher quality)
return 0.75  # Example: 75% fairness score
```

#### 3.2.2 `compute_with_breakdown()` Return Value

```python
return {
    'value': 0.75,                    # Same as compute() output
    'components': {
        'per_period_values': [...],   # Term-specific intermediate values
        'gini_coefficients': [...],   # Example for spatial fairness
    },
    'statistics': {
        'mean': 0.75,
        'std': 0.05,
        'min': 0.65,
        'max': 0.85,
        'n_observations': 1000,
    },
    'diagnostics': {
        'computation_time_ms': 150.5,
        'cache_hit': False,
        'warnings': [],
    }
}
```

---

## 4. Data Contracts

### 4.1 Trajectory Data Format

The primary input to all terms is trajectory data in the following format:

```python
trajectories: Dict[str, List[List[List[float]]]]
# Structure:
# {
#     driver_id: [                    # str or int key
#         trajectory_0: [             # List of states
#             state_0: [x, y, time, day, ..., action],  # 126 elements
#             state_1: [x, y, time, day, ..., action],
#             ...
#         ],
#         trajectory_1: [...],
#         ...
#     ],
#     ...
# }
```

### 4.2 State Vector Schema

Each state is a 126-element list (per `all_trajs.pkl` format):

| Index | Field | Type | Description |
|-------|-------|------|-------------|
| 0 | `x_grid` | int | Grid x-coordinate (longitude) |
| 1 | `y_grid` | int | Grid y-coordinate (latitude) |
| 2 | `time_bucket` | int | Time bucket [0, 287] |
| 3 | `day_index` | int | Day of week |
| 4-24 | `poi_distances` | float | Manhattan distances to 21 POIs |
| 25-49 | `pickup_count_norm` | float | Normalized pickup counts (5×5) |
| 50-74 | `traffic_volume_norm` | float | Normalized traffic volumes (5×5) |
| 75-99 | `traffic_speed_norm` | float | Normalized traffic speeds (5×5) |
| 100-124 | `traffic_wait_norm` | float | Normalized waiting times (5×5) |
| 125 | `action_code` | int | Movement action [0-9] |

### 4.3 Auxiliary Data Contracts

```python
auxiliary_data: Dict[str, Any] = {
    # Pickup/Dropoff counts (required for spatial/causal fairness)
    'pickup_dropoff_counts': {
        (x, y, time, day): [pickup_count, dropoff_count],
        # Keys: (int, int, int, int) → Values: [int, int]
    },
    
    # Traffic data (required for causal fairness)
    'latest_traffic': {
        (x, y, time, day): [speed, wait_time, volume],
        # Keys: (int, int, int, int) → Values: [float, float, float]
    },
    
    # Volume/pickup data (alternative source)
    'latest_volume_pickups': {
        (x, y, time, day): [pickup_count, traffic_volume],
        # Keys: (int, int, int, int) → Values: [int, int]
    },
    
    # Grid configuration
    'grid_config': {
        'x_dim': 48,
        'y_dim': 90,
        'time_buckets': 288,
        'days': 6,
    },
}
```

---

## 5. Configuration Patterns

### 5.1 Term-Specific Configuration

Each term extends the base configuration:

```python
@dataclass
class SpatialFairnessConfig(TermConfig):
    """Configuration for spatial fairness term."""
    
    # Period aggregation
    period_type: str = "time_bucket"  # "time_bucket", "hourly", "daily"
    
    # Gini coefficient options
    include_zero_cells: bool = True   # Include cells with zero activity
    
    # Normalization
    num_taxis: int = 50               # Number of active taxis
    num_days: float = 21.0            # Days in dataset
    

@dataclass
class CausalFairnessConfig(TermConfig):
    """Configuration for causal fairness term."""
    
    # g(d) estimation method
    estimation_method: str = "binning"  # "binning", "regression", "loess"
    n_bins: int = 10                    # For binning method
    poly_degree: int = 2                # For polynomial regression
    
    # Neighborhood aggregation
    neighborhood_size: int = 1          # k for (2k+1)×(2k+1) window
    
    # Filtering
    min_demand_threshold: int = 1       # Minimum demand to include cell
```

### 5.2 Configuration Validation

```python
def _validate_config(self) -> None:
    """Validate spatial fairness configuration."""
    if self.config.period_type not in ["time_bucket", "hourly", "daily"]:
        raise ValueError(f"Invalid period_type: {self.config.period_type}")
    
    if self.config.num_taxis <= 0:
        raise ValueError("num_taxis must be positive")
    
    if self.config.num_days <= 0:
        raise ValueError("num_days must be positive")
```

---

## 6. Testing Interface

### 6.1 Required Test Cases

Every term implementation must include tests for:

1. **Boundary Cases**: Perfect fairness (return 1.0), maximum unfairness (return 0.0)
2. **Input Validation**: Correct error handling for invalid inputs
3. **Determinism**: Same inputs produce same outputs
4. **Value Range**: Output is always in [0, 1]
5. **Consistency**: Results are reasonable for known scenarios

### 6.2 Test Template

```python
import pytest
from objective_function.term_name import TermClass, TermConfig

class TestTermName:
    """Test suite for [TermName] objective function term."""
    
    @pytest.fixture
    def default_config(self):
        """Create default configuration for testing."""
        return TermConfig()
    
    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectory data for testing."""
        # Return minimal valid trajectory structure
        return {...}
    
    @pytest.fixture
    def sample_auxiliary_data(self):
        """Create sample auxiliary data for testing."""
        return {...}
    
    def test_output_range(self, default_config, sample_trajectories, sample_auxiliary_data):
        """Test that output is in valid range [0, 1]."""
        term = TermClass(default_config)
        result = term.compute(sample_trajectories, sample_auxiliary_data)
        assert 0.0 <= result <= 1.0, f"Value {result} outside valid range"
    
    def test_perfect_fairness(self, default_config):
        """Test that perfectly fair distribution returns 1.0."""
        term = TermClass(default_config)
        # Create perfectly fair data...
        result = term.compute(perfect_data, aux_data)
        assert abs(result - 1.0) < 0.01, "Perfect fairness should return ~1.0"
    
    def test_maximum_unfairness(self, default_config):
        """Test that maximally unfair distribution returns ~0.0."""
        term = TermClass(default_config)
        # Create maximally unfair data...
        result = term.compute(unfair_data, aux_data)
        assert result < 0.1, "Maximum unfairness should return ~0.0"
    
    def test_determinism(self, default_config, sample_trajectories, sample_auxiliary_data):
        """Test that same inputs produce same outputs."""
        term = TermClass(default_config)
        result1 = term.compute(sample_trajectories, sample_auxiliary_data)
        result2 = term.compute(sample_trajectories, sample_auxiliary_data)
        assert result1 == result2, "Results should be deterministic"
    
    def test_input_validation(self, default_config):
        """Test input validation catches errors."""
        term = TermClass(default_config)
        is_valid, errors = term.validate_input({}, {})
        assert not is_valid
        assert len(errors) > 0
    
    def test_metadata(self, default_config):
        """Test that metadata is properly defined."""
        term = TermClass(default_config)
        assert term.metadata.name is not None
        assert term.metadata.value_range == (0.0, 1.0)
        assert term.metadata.higher_is_better is True
```

---

## 7. Documentation Template

### 7.1 Development Plan Document Structure

Each term's development plan document should follow this structure:

```markdown
# [Term Name] Development Plan

## 1. Overview
   - 1.1 Purpose and Definition
   - 1.2 Role in Objective Function
   - 1.3 Relationship to Other Terms

## 2. Mathematical Formulation
   - 2.1 Core Formula
   - 2.2 Component Definitions
   - 2.3 Derivation/Justification

## 3. Literature and References
   - 3.1 Primary Sources
   - 3.2 Theoretical Foundation
   - 3.3 Related Work

## 4. Data Requirements
   - 4.1 Required Datasets
   - 4.2 Data Preprocessing
   - 4.3 Data Validation

## 5. Implementation Plan
   - 5.1 Algorithm Steps
   - 5.2 Pseudocode
   - 5.3 Python Implementation Outline
   - 5.4 Computational Considerations

## 6. Configuration Parameters
   - 6.1 Required Parameters
   - 6.2 Optional Parameters
   - 6.3 Default Values

## 7. Testing Strategy
   - 7.1 Unit Tests
   - 7.2 Integration Tests
   - 7.3 Validation with Real Data

## 8. Expected Challenges
   - 8.1 Known Difficulties
   - 8.2 Mitigation Strategies

## 9. Development Milestones
   - 9.1 Phase 1: Core Implementation
   - 9.2 Phase 2: Testing and Validation
   - 9.3 Phase 3: Integration

## 10. Appendix
    - 10.1 Code Snippets
    - 10.2 Sample Data
    - 10.3 Revision History
```

---

## 8. File Organization

### 8.1 Directory Structure

```
objective_function/
├── __init__.py                         # Package exports
├── base.py                             # Base classes and interfaces
├── config.py                           # Configuration dataclasses
├── TERM_INTERFACE_SPECIFICATION.md     # This document
├── INTEGRATION_DEVELOPMENT_PLAN.md     # Integration guide
├── docs/
│   └── FAMAIL_OBJECTIVE_FUNCTION_SPECIFICATION.md  # Existing spec
├── spatial_fairness/
│   ├── __init__.py
│   ├── term.py                         # SpatialFairnessTerm implementation
│   ├── config.py                       # SpatialFairnessConfig
│   ├── utils.py                        # Helper functions (Gini, etc.)
│   ├── DEVELOPMENT_PLAN.md             # Development plan document
│   └── tests/
│       ├── __init__.py
│       ├── test_term.py
│       └── test_utils.py
├── causal_fairness/
│   ├── __init__.py
│   ├── term.py
│   ├── config.py
│   ├── utils.py
│   ├── DEVELOPMENT_PLAN.md
│   └── tests/
│       └── ...
├── fidelity/
│   ├── __init__.py
│   ├── term.py
│   ├── config.py
│   ├── DEVELOPMENT_PLAN.md
│   └── tests/
│       └── ...
├── quality/
│   ├── __init__.py
│   ├── term.py
│   ├── config.py
│   ├── DEVELOPMENT_PLAN.md
│   └── tests/
│       └── ...
└── combined/
    ├── __init__.py
    ├── objective.py                     # Combined objective function
    ├── optimizer.py                     # ST-iFGSM-based optimizer
    └── tests/
        └── ...
```

### 8.2 Module Exports

```python
# objective_function/__init__.py

from .base import ObjectiveFunctionTerm, TermMetadata, TermConfig
from .spatial_fairness import SpatialFairnessTerm, SpatialFairnessConfig
from .causal_fairness import CausalFairnessTerm, CausalFairnessConfig
from .fidelity import FidelityTerm, FidelityConfig
from .quality import QualityTerm, QualityConfig
from .combined import FAMAILObjectiveFunction

__all__ = [
    'ObjectiveFunctionTerm',
    'TermMetadata',
    'TermConfig',
    'SpatialFairnessTerm',
    'SpatialFairnessConfig',
    'CausalFairnessTerm',
    'CausalFairnessConfig',
    'FidelityTerm',
    'FidelityConfig',
    'QualityTerm',
    'QualityConfig',
    'FAMAILObjectiveFunction',
]
```

---

## 9. Implementation Checklist

### 9.1 Pre-Implementation Checklist

- [ ] Read and understand the Term Interface Specification (this document)
- [ ] Review the mathematical formulation in the development plan
- [ ] Identify all required and optional datasets
- [ ] Create the term-specific configuration dataclass
- [ ] Set up the directory structure

### 9.2 Implementation Checklist

- [ ] Create `__init__.py` with appropriate exports
- [ ] Implement base class (`term.py`)
  - [ ] Inherit from `ObjectiveFunctionTerm`
  - [ ] Implement `_build_metadata()`
  - [ ] Implement `_validate_config()`
  - [ ] Implement `compute()`
  - [ ] Implement `compute_with_breakdown()`
  - [ ] Implement `compute_gradient()` (if differentiable)
- [ ] Create configuration class (`config.py`)
- [ ] Implement helper functions (`utils.py`)
- [ ] Write comprehensive docstrings

### 9.3 Testing Checklist

- [ ] Unit tests for each helper function
- [ ] Test output range [0, 1]
- [ ] Test boundary cases (perfect fairness, maximum unfairness)
- [ ] Test determinism
- [ ] Test input validation
- [ ] Test with real data samples
- [ ] Test integration with other terms

### 9.4 Documentation Checklist

- [ ] Complete development plan document
- [ ] Inline code documentation (docstrings)
- [ ] Usage examples
- [ ] Known limitations documented

---

## Appendix A: Quick Reference

### A.1 Term Value Interpretation

| Value | Interpretation |
|-------|---------------|
| 1.0 | Perfect (completely fair/high quality) |
| 0.8-1.0 | Excellent |
| 0.6-0.8 | Good |
| 0.4-0.6 | Moderate |
| 0.2-0.4 | Poor |
| 0.0-0.2 | Very poor (highly unfair/low quality) |

### A.2 Objective Function Summary

$$
\max_{\mathcal{T}'} \mathcal{L} = \max_{\mathcal{T}'} \left( \alpha_1 F_{\text{causal}} + \alpha_2 F_{\text{spatial}} + \alpha_3 F_{\text{fidelity}} + \alpha_4 F_{\text{quality}} \right)
$$

All terms: Higher = Better, Range = [0, 1]

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-09 | Robert Ashe (FAMAIL Team) | Initial specification |

---

*This document serves as the canonical reference for implementing objective function terms in the FAMAIL project. All developers should refer to this document before implementing or modifying any term.*
