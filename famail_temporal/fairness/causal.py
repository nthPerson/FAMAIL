"""
Pooled Option B F_causal + per-unit attribution.

F_causal = R'(I-H_demo)R / R'MR  where R = Y - g_0(D), Y = S/D.

Per-unit attribution decomposes `1 - F_causal = r^2_demo`:
    attribution_i = ((MR)_i^2 - ((I-H)R)_i^2) / R'MR
    sum_i attribution_i == 1 - F_causal
"""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config
from famail_temporal.fairness.hat_matrices import compute_fcausal_torch


def compute_fcausal(
    demand_N: torch.Tensor,
    supply_N: torch.Tensor,
    g0_D_N: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Gradient flow: demand_N -> Y -> R -> F_causal."""
    D = torch.clamp(demand_N, min=config.DEMAND_FLOOR)
    Y = supply_N / (D + config.EPS)
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
    """Per-unit contribution to demographic-explained variance."""
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
    """Signed attribution: sign of (HR) indicates under/over-service."""
    with torch.no_grad():
        HR = R - I_minus_H_demo @ R
        magnitudes = per_unit_attribution(R, I_minus_H_demo, M)
        return magnitudes * torch.sign(HR)
