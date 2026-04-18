"""Pooled spatial fairness: one Gini over all active (cell, t) units."""

from __future__ import annotations
from typing import Tuple

import torch

from famail_temporal import config


def per_unit_gini_decomposition(values: torch.Tensor) -> torch.Tensor:
    """Row-sum decomposition of the pairwise Gini on an N-vector.

    For each i in [0, N):  contrib_i = sum_j |x_i - x_j| / (2 * N^2 * mean(x))
    so that sum_i contrib_i == pairwise_gini(values) exactly (modulo float
    precision). Callers are responsible for passing only the active-unit
    subset - this function operates on 1-D N-vectors with no mask handling.
    """
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)
    mean_val = values.mean() + config.EPS
    diff = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))  # (N, N)
    row_sums = diff.sum(dim=1)                                    # (N,)
    return row_sums / (2 * n * n * mean_val)


def pairwise_gini(values: torch.Tensor) -> torch.Tensor:
    """Differentiable pairwise Gini.

    Implemented as sum(per_unit_gini_decomposition(values)) so the per-unit
    decomposition and the aggregate stay numerically linked by construction.
    Clamped to [0, 1] to guard against float drift at the upper boundary.
    """
    gini = per_unit_gini_decomposition(values).sum()
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


def compute_spatial_attribution(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> dict:
    """Per-unit spatial attribution (3 N-vector channels).

    Returns:
        dict with keys 'gini_decomp_dsr', 'gini_decomp_asr', 'spatial_attr'.
        spatial_attr = 0.5 * (gini_decomp_dsr + gini_decomp_asr), so that
        sum(spatial_attr) == 1 - F_spatial (same canonical decomposition
        consumed by the fairness-aware grid).

    Input validation mirrors compute_fspatial (1-D, matching shapes, non-negative).
    """
    if pickup_N.dim() != 1 or dropoff_N.dim() != 1 or active_taxis_N.dim() != 1:
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must be 1-D tensors."
        )
    if not (pickup_N.shape == dropoff_N.shape == active_taxis_N.shape):
        raise ValueError("pickup_N, dropoff_N, and active_taxis_N must have the same shape.")
    if (pickup_N < 0).any() or (dropoff_N < 0).any() or (active_taxis_N < 0).any():
        raise ValueError("pickup_N, dropoff_N, and active_taxis_N must not contain negatives.")

    dsr = pickup_N / (active_taxis_N + config.EPS)
    asr = dropoff_N / (active_taxis_N + config.EPS)
    gini_decomp_dsr = per_unit_gini_decomposition(dsr)
    gini_decomp_asr = per_unit_gini_decomposition(asr)
    spatial_attr = 0.5 * (gini_decomp_dsr + gini_decomp_asr)
    return {
        "gini_decomp_dsr": gini_decomp_dsr,
        "gini_decomp_asr": gini_decomp_asr,
        "spatial_attr": spatial_attr,
    }
