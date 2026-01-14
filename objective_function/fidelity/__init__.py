"""
Trajectory Fidelity Term for FAMAIL Objective Function.

This module implements the Trajectory Fidelity term (F_fidelity) which measures
how authentic edited trajectories remain compared to genuine expert driver
trajectories using a pre-trained ST-SiameseNet discriminator.

The fidelity term ensures that while improving fairness, the modified trajectories
still resemble real taxi driver behavior and are physically realistic.

Mathematical Formulation:
    F_fidelity = (1/|T'|) * Σ_{τ' ∈ T'} Discriminator(τ', τ_ref)
    
where:
    - T' = set of edited trajectories
    - τ_ref = reference trajectory (original or paired expert trajectory)
    - Discriminator outputs probability that the pair is from the same agent

Value Range: [0, 1]
    - F_fidelity = 1: Perfectly authentic (indistinguishable from original)
    - F_fidelity = 0: Completely artificial (easily detected as fake)

Components:
    - FidelityConfig: Configuration dataclass with validation
    - FidelityTerm: Main objective function term implementation
    - DifferentiableFidelity: PyTorch module for gradient-based optimization

Usage:
    >>> from objective_function.fidelity import FidelityTerm, FidelityConfig
    >>> config = FidelityConfig(checkpoint_path="path/to/best.pt")
    >>> term = FidelityTerm(config)
    >>> result = term.compute(edited_trajectories, original_trajectories)
"""

from .config import (
    FidelityConfig,
    DEFAULT_CONFIG,
    HIGH_THRESHOLD_CONFIG,
)
from .term import FidelityTerm

__all__ = [
    "FidelityConfig",
    "FidelityTerm",
    "DEFAULT_CONFIG",
    "HIGH_THRESHOLD_CONFIG",
]
