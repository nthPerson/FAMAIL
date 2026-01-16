"""
Soft Cell Assignment Module for FAMAIL Objective Function.

This module provides differentiable soft cell assignment functionality
for gradient-based trajectory optimization. It bridges the gap between
discrete cell assignments and continuous optimization.

Key Components:
    - SoftCellAssignment: PyTorch module for differentiable assignment
    - soft_cell_assignment(): Functional implementation
    - compute_soft_counts(): Aggregate soft assignments to grid counts

Mathematical Formulation:
    σ_c(x, y) = exp(-d_c²/(2τ²)) / Σ exp(-d_c'²/(2τ²))
    
    where:
    - (x, y) = continuous trajectory location
    - c = grid cell
    - d_c = distance from (x, y) to cell c center
    - τ = temperature parameter
    
Usage:
    >>> from objective_function.soft_cell_assignment import SoftCellAssignment
    >>> 
    >>> # Create module
    >>> soft_assign = SoftCellAssignment(
    ...     grid_dims=(48, 90),
    ...     neighborhood_size=5,
    ...     initial_temperature=1.0,
    ... )
    >>> 
    >>> # Compute soft assignment for a batch of locations
    >>> locations = torch.tensor([[24.5, 45.3], [10.2, 80.1]])
    >>> original_cells = torch.tensor([[24, 45], [10, 80]])
    >>> probs = soft_assign(locations, original_cells)
    >>> # probs shape: [2, 5, 5] - probabilities over 5x5 neighborhoods

Reference:
    FAIRNESS_TERM_FORMULATIONS.md Section 4
"""

from .module import (
    SoftCellAssignment,
    soft_cell_assignment,
    compute_soft_counts,
    update_counts_with_soft_assignment,
)

from .verification import (
    verify_soft_assignment_gradients,
    verify_end_to_end_gradients,
    create_soft_assignment_verification_report,
)

__all__ = [
    # Main module
    'SoftCellAssignment',
    
    # Functional API
    'soft_cell_assignment',
    'compute_soft_counts',
    'update_counts_with_soft_assignment',
    
    # Verification
    'verify_soft_assignment_gradients',
    'verify_end_to_end_gradients',
    'create_soft_assignment_verification_report',
]
