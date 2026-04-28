"""Pooled (cell, t) fairness metrics + canonical attribution functions.

The two ``per_cell_fairness_attribution_*`` functions are the SINGLE
canonical decomposition for each metric — used by both the trajectory-
modification algorithm and the fairness-attribution export tool. See
``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`` for the
formulation, sign convention, and sum invariants (each sums to its F).
"""

from famail_temporal.fairness.spatial import (
    pairwise_gini,
    compute_fspatial,
    per_cell_fairness_attribution_spatial,
)
from famail_temporal.fairness.causal import (
    compute_fcausal,
    compute_fcausal_from_compact,
    per_cell_fairness_attribution_causal,
)
from famail_temporal.fairness.hat_matrices import (
    precompute_hat_matrices,
    compute_fcausal_torch,
    compute_fcausal_compact,
    hat_matrices_to_torch,
    apply_i_minus_h,
)
from famail_temporal.fairness.g0_power_basis import (
    G0Function,
    build_power_basis_features,
    fit as fit_g0,
)

__all__ = [
    "pairwise_gini", "compute_fspatial",
    "per_cell_fairness_attribution_spatial",
    "compute_fcausal", "compute_fcausal_from_compact",
    "per_cell_fairness_attribution_causal",
    "precompute_hat_matrices", "compute_fcausal_torch",
    "compute_fcausal_compact", "hat_matrices_to_torch", "apply_i_minus_h",
    "G0Function", "build_power_basis_features", "fit_g0",
]
