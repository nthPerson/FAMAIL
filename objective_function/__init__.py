"""
FAMAIL Objective Function Package.

This package provides the objective function terms for the FAMAIL
(Fairness-Aware Multi-Agent Imitation Learning) system.

Objective Function:
    L(T') = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity

All terms are designed to be maximized, with values in [0, 1] where
higher values indicate better outcomes.

Available Terms:
    - spatial_fairness: Gini-based measure of service distribution equality
    - causal_fairness: R²-based measure of demand-service alignment (planned)
    - fidelity: Discriminator-based trajectory realism (planned)
"""

from .base import (
    ObjectiveFunctionTerm,
    TermMetadata,
    TermConfig,
    TrajectoryData,
    AuxiliaryData,
)

# Import available terms
from .spatial_fairness import (
    SpatialFairnessTerm,
    SpatialFairnessConfig,
)


__all__ = [
    # Base classes
    'ObjectiveFunctionTerm',
    'TermMetadata',
    'TermConfig',
    'TrajectoryData',
    'AuxiliaryData',
    
    # Spatial Fairness
    'SpatialFairnessTerm',
    'SpatialFairnessConfig',
]

__version__ = '0.1.0'
