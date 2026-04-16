"""Pooled (cell, t) fairness metrics."""

from famail_temporal.fairness.spatial import (
    pairwise_gini,
    compute_fspatial,
)
from famail_temporal.fairness.causal import (
    compute_fcausal,
    per_unit_attribution,
    per_unit_attribution_signed,
)
from famail_temporal.fairness.hat_matrices import (
    precompute_hat_matrices,
    compute_fcausal_torch,
    hat_matrices_to_torch,
)
from famail_temporal.fairness.g0_power_basis import (
    G0Function,
    build_power_basis_features,
    fit as fit_g0,
)

__all__ = [
    "pairwise_gini", "compute_fspatial",
    "compute_fcausal", "per_unit_attribution", "per_unit_attribution_signed",
    "precompute_hat_matrices", "compute_fcausal_torch", "hat_matrices_to_torch",
    "G0Function", "build_power_basis_features", "fit_g0",
]
