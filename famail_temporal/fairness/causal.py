"""
Pooled Option B F_causal + per-unit attribution.

F_causal = R'(I-H_demo)R / R'MR  where R = Y - g_0(D), Y = S/D.

Per-unit attribution decomposes `1 - F_causal = r^2_demo`:
    attribution_i = ((MR)_i^2 - ((I-H)R)_i^2) / R'MR
    sum_i attribution_i == 1 - F_causal

Note: individual per_unit_attribution[i] values CAN be negative when
((I-H)R)_i^2 > (MR)_i^2 — at those units demographics actively misalign
with R. The SUM over all i is always non-negative and equals 1 - F_causal.

Two API forms are provided:
- **Compact** (production): ``compute_fcausal_compact_wrapper``,
  ``per_unit_attribution_compact``, ``per_unit_attribution_signed_compact``
  take ``X_demo`` and ``XtX_inv`` and compute in O(Np + p²) memory. This
  is required at production scale (N ≥ 10,000).
- **Legacy dense** (small-N only): the classic ``compute_fcausal``,
  ``per_unit_attribution``, ``per_unit_attribution_signed`` take the
  N×N ``I_minus_H_demo`` and ``M`` matrices. Retained for small-N tests
  and back-compatible call sites.
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


# ---------------------------------------------------------------------------
# Legacy dense API — retained for small-N tests
# ---------------------------------------------------------------------------

def compute_fcausal(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N -> Y -> R -> F_causal. Legacy dense form."""
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


def per_unit_attribution(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Per-unit contribution to demographic-explained variance. Legacy dense form."""
    with torch.no_grad():
        MR = M @ R
        IHR = I_minus_H_demo @ R
        ss_tot_vec = MR ** 2
        ss_res_vec = IHR ** 2
        ss_explained_vec = ss_tot_vec - ss_res_vec
        ss_tot_scalar = ss_tot_vec.sum() + config.EPS
        return ss_explained_vec / ss_tot_scalar


def per_unit_attribution_signed(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Signed attribution: magnitude from per_unit_attribution, sign from (HR)_i.

    HR = H @ R is the projection of R onto the demographic span. Sign interpretation:
    - sign(HR)_i > 0: demographics predict Y_i > g_0(D_i) — relative over-service
    - sign(HR)_i < 0: demographics predict Y_i < g_0(D_i) — relative under-service

    Both over- and under-service contribute positive MAGNITUDE to the attribution
    sum (= 1 - F_causal). The sign labels direction, not intensity.
    """
    with torch.no_grad():
        HR = R - I_minus_H_demo @ R
        magnitudes = per_unit_attribution(R, I_minus_H_demo, M)
        return magnitudes * torch.sign(HR)


# ---------------------------------------------------------------------------
# Compact API — required at production scale (N ≥ 10,000)
# ---------------------------------------------------------------------------

def compute_fcausal_from_compact(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N -> Y -> R -> F_causal (compact FWL form).

    Same mathematics as ``compute_fcausal`` above, but uses ``X_demo`` and
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


def per_unit_attribution_from_compact(
    R: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> torch.Tensor:
    """Per-unit contribution to demographic-explained variance (compact form).

    Identity used:
        MR_i = R_i - mean(R)               (no N×N needed)
        (I-H)R = R - X(XᵀX)⁻¹(XᵀR)         (no N×N needed)
        attribution_i = ((MR)_i^2 - ((I-H)R)_i^2) / Σ (MR)_j^2
    """
    with torch.no_grad():
        MR = R - R.mean()
        IHR = apply_i_minus_h(R, X_demo, XtX_inv)
        ss_tot_vec = MR ** 2
        ss_res_vec = IHR ** 2
        ss_explained_vec = ss_tot_vec - ss_res_vec
        ss_tot_scalar = ss_tot_vec.sum() + config.EPS
        return ss_explained_vec / ss_tot_scalar


def per_unit_attribution_signed_from_compact(
    R: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> torch.Tensor:
    """Signed attribution (compact form). Sign from HR = R − (I−H)R."""
    with torch.no_grad():
        IHR = apply_i_minus_h(R, X_demo, XtX_inv)
        HR = R - IHR
        magnitudes = per_unit_attribution_from_compact(R, X_demo, XtX_inv)
        return magnitudes * torch.sign(HR)
