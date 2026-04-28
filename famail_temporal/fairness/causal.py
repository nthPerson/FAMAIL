"""
Pooled Option B F_causal + per-cell attribution.

F_causal = R'(I-H_demo)R / R'MR  where R = Y - g_0(D), Y = S/D.

The single canonical per-cell decomposition is the 1/N-shifted form
that sums to F_causal — see ``per_cell_fairness_attribution_causal``
below and ``docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`` for the
formulation and rationale.
"""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config
from famail_temporal.fairness.hat_matrices import (
    apply_i_minus_h,
    compute_fcausal_compact,
    compute_fcausal_torch,
)


def compute_fcausal(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N -> Y -> R -> F_causal. Legacy dense form.

    Retained for small-N debug / test paths where the dense (I-H, M)
    matrices are explicitly constructed. Production code should use
    ``compute_fcausal_from_compact`` instead.
    """
    if not torch.isfinite(g0_D_N).all():
        raise ValueError(
            "g0_D_N contains non-finite values; check that g_0 is evaluated on "
            "in-range demand values."
        )
    D = torch.clamp(demand_N, min=config.DEMAND_FLOOR)
    Y = supply_N / D
    R = Y - g0_D_N
    f_causal = compute_fcausal_torch(R, I_minus_H_demo, M)
    debug = {
        'Y_min': float(Y.min().detach()),
        'Y_max': float(Y.max().detach()),
        'R_min': float(R.min().detach()),
        'R_max': float(R.max().detach()),
        'f_causal': float(f_causal.detach()),
    }
    return f_causal, debug


def compute_fcausal_from_compact(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N -> Y -> R -> F_causal (compact FWL form).

    Same mathematics as ``compute_fcausal`` but uses ``X_demo`` and
    ``XtX_inv`` (O(Np) memory) instead of the dense ``I_minus_H_demo``,
    ``M`` matrices (O(N²) memory). Required at production scale where
    N exceeds ``_DENSE_MATERIALIZATION_MAX_N`` (see hat_matrices.py).
    """
    if not torch.isfinite(g0_D_N).all():
        raise ValueError(
            "g0_D_N contains non-finite values; check that g_0 is evaluated on "
            "in-range demand values."
        )
    D = torch.clamp(demand_N, min=config.DEMAND_FLOOR)
    Y = supply_N / D
    R = Y - g0_D_N
    f_causal = compute_fcausal_compact(R, X_demo, XtX_inv)
    debug = {
        'Y_min': float(Y.min().detach()),
        'Y_max': float(Y.max().detach()),
        'R_min': float(R.min().detach()),
        'R_max': float(R.max().detach()),
        'f_causal': float(f_causal.detach()),
    }
    return f_causal, debug


def per_cell_fairness_attribution_causal(
    R: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> torch.Tensor:
    """Canonical per-cell decomposition of F_causal.

    Returns a 1-D tensor of length N (active cells) where
    ``Σᵢ result_i = F_causal`` and the sign convention is

        positive  → cell contributes more than 1/N baseline to fairness;
                    in the F_causal sense this means demographics explain
                    LESS than baseline of the cell's residual variance
        ≈ 0       → cell at the negative-fair / anti-fair boundary
        negative  → cell drags fairness below baseline; demographics
                    explain MORE than baseline of the cell's residual
                    (priority for modification)

    Formulation (1/N-shifted r²_demo decomposition):

        αᵢ = 1/N − ((MR)ᵢ² − ((I−H)R)ᵢ²) / R'MR

    where ``(MR)ᵢ²`` is the cell's centered residual squared and
    ``((I−H)R)ᵢ²`` is the cell's residual squared after demographic
    regression. The difference (always summing to ``r²_demo = 1 −
    F_causal``) is the cell's contribution to demographic-explained
    variance; subtracting from 1/N flips it to the "fairness contribution"
    side. See ``docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``.

    This is the single canonical causal-fairness attribution function in
    the codebase. The trajectory-modification algorithm and the
    fairness-attribution export tool both call it — there is no parallel
    "unfairness" variant.

    Implemented via the compact FWL form (X_demo, XtX_inv) so it works
    at any N including production scale (N ≈ 35,000 at T=24).

    Args:
        R: 1-D residual tensor (length N). Typically Y - g_0(D) on the
           active-unit subset.
        X_demo: (N, p+1) design matrix [1 | standardized(demographics)].
        XtX_inv: (p+1, p+1) inverse of X'X, precomputed.

    Returns:
        1-D tensor of length N. Sum equals F_causal (within EPS).
    """
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D; got shape {tuple(R.shape)}")
    if X_demo.ndim != 2:
        raise ValueError(f"X_demo must be 2-D; got shape {tuple(X_demo.shape)}")
    if XtX_inv.ndim != 2:
        raise ValueError(
            f"XtX_inv must be 2-D; got shape {tuple(XtX_inv.shape)}"
        )
    n = R.shape[0]
    if X_demo.shape[0] != n:
        raise ValueError(
            f"X_demo.shape[0]={X_demo.shape[0]} but R has length {n}"
        )

    with torch.no_grad():
        MR = R - R.mean()                       # centered residual
        IHR = apply_i_minus_h(R, X_demo, XtX_inv)  # post-demographic residual
        ss_tot_vec = MR ** 2                    # always >= 0
        ss_res_vec = IHR ** 2                   # always >= 0
        ss_explained_vec = ss_tot_vec - ss_res_vec  # signed
        ss_tot_scalar = ss_tot_vec.sum() + config.EPS
        unfairness_contrib = ss_explained_vec / ss_tot_scalar  # sums to 1 - F_causal
        return (1.0 / n) - unfairness_contrib    # sums to F_causal
