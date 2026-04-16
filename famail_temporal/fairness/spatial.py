"""Pooled spatial fairness: one Gini over all active (cell, t) units."""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config


def pairwise_gini(values: torch.Tensor) -> torch.Tensor:
    """Differentiable pairwise Gini: G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)."""
    n = values.numel()
    if n <= 1:
        return torch.tensor(0.0, device=values.device)
    mean_val = values.mean() + config.EPS
    diff = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
    gini = diff.sum() / (2 * n * n * mean_val)
    return torch.clamp(gini, 0.0, 1.0)


def compute_fspatial(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """F_spatial = 1 - 0.5*(Gini(DSR) + Gini(ASR)).

    Args:
        pickup_N:      1-D tensor of pickup counts per active (cell, t) unit.
        dropoff_N:     1-D tensor of dropoff counts per active (cell, t) unit.
        active_taxis_N: 1-D tensor of active taxi supply per active (cell, t) unit.

    Returns:
        (f_spatial, debug) where f_spatial is a scalar tensor in [0, 1] and
        debug is a dict with keys 'gini_dsr' and 'gini_asr'.

    Raises:
        ValueError: if inputs are not 1-D, have different shapes, or contain
                    negative values.
    """
    # --- Input validation ---
    if pickup_N.dim() != 1 or dropoff_N.dim() != 1 or active_taxis_N.dim() != 1:
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must be 1-D tensors; "
            f"got shapes {pickup_N.shape}, {dropoff_N.shape}, {active_taxis_N.shape}."
        )
    if not (pickup_N.shape == dropoff_N.shape == active_taxis_N.shape):
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must have the same shape; "
            f"got {pickup_N.shape}, {dropoff_N.shape}, {active_taxis_N.shape}."
        )
    if (pickup_N < 0).any() or (dropoff_N < 0).any() or (active_taxis_N < 0).any():
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must not contain negative values."
        )

    # --- Compute DSR and ASR ---
    dsr = pickup_N / (active_taxis_N + config.EPS)
    asr = dropoff_N / (active_taxis_N + config.EPS)

    # --- Gini coefficients ---
    gini_dsr = pairwise_gini(dsr)
    gini_asr = pairwise_gini(asr)

    f_spatial = 1.0 - 0.5 * (gini_dsr + gini_asr)

    debug = {
        'gini_dsr': float(gini_dsr.detach()),
        'gini_asr': float(gini_asr.detach()),
    }
    return f_spatial, debug
