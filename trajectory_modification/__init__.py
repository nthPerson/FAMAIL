"""
FAMAIL Trajectory Modification Module.

Implements the Fairness-Aware Trajectory Editing Algorithm (Modified ST-iFGSM)
for improving spatial and causal fairness in taxi service distribution.

Components:
- Trajectory: Core trajectory representation
- TrajectoryModifier: ST-iFGSM based trajectory modification
- DiscriminatorAdapter: Wrapper for trained discriminator model
- FAMAILObjective: Combined objective function (F_spatial + F_causal + F_fidelity)
- GlobalMetrics: System-wide fairness metrics tracking
- DataBundle: Consolidated data loading utilities
"""

from .trajectory import Trajectory, TrajectoryState
from .modifier import TrajectoryModifier, ModificationResult, ModificationHistory
from .discriminator_adapter import DiscriminatorAdapter
from .objective import (
    FAMAILObjective,
    MissingDataError,
    MissingComponentError,
    InsufficientDataError,
)
from .metrics import GlobalMetrics, FairnessSnapshot
from .data_loader import (
    DataBundle,
    TrajectoryLoader,
    PickupDropoffLoader,
    ActiveTaxisLoader,
    GFunctionLoader,
)

__all__ = [
    # Core data structures
    'Trajectory',
    'TrajectoryState',
    # Modification
    'TrajectoryModifier',
    'ModificationResult',
    'ModificationHistory',
    # Discriminator
    'DiscriminatorAdapter',
    # Objective function
    'FAMAILObjective',
    'MissingDataError',
    'MissingComponentError',
    'InsufficientDataError',
    # Metrics
    'GlobalMetrics',
    'FairnessSnapshot',
    # Data loading
    'DataBundle',
    'TrajectoryLoader',
    'PickupDropoffLoader',
    'ActiveTaxisLoader',
    'GFunctionLoader',
]
