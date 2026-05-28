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

    Implementation: sorted-order identity. For sorted values x_(1) <= ... <= x_(n),
    the row-sum can be written as
        Σⱼ |x_(k) - x_(j)|  =  (2k - n) · x_(k)  -  2·C_(k)  +  S_total
    where C_(k) is the cumulative sum and S_total = Σ x_(k). This avoids the
    O(N²) pairwise materialization (which at N=34,524 is a 4.77 GB allocation
    done twice per modifier iteration) in favor of O(N log N) work and O(N)
    memory. ``torch.sort`` and ``torch.scatter`` are both autograd-registered;
    the gradient routes back via the sort permutation and is bit-equivalent
    (modulo float-summation-order) to the pairwise form.
    """
    n = values.numel()
    if n <= 1:
        return torch.zeros_like(values)

    # Compute reductions in float64 to keep the metric's noise floor well
    # below any reasonable convergence tolerance. Cast back to caller dtype
    # on return so type signatures (e.g. build_fairness_grid → float32
    # storage) stay unchanged.
    orig_dtype = values.dtype
    values_64 = values.to(torch.float64)
    mean_val = values_64.mean() + config.EPS

    xs, sort_idx = torch.sort(values_64)
    cumsum = torch.cumsum(xs, dim=0)
    s_total = xs.sum()
    k = torch.arange(
        1, n + 1, dtype=xs.dtype, device=xs.device,
    )
    # Σⱼ |x_(k) − x_(j)|  =  (2k − n)·x_(k)  −  2·C_(k)  +  S_total
    row_sums_sorted = (2 * k - n) * xs - 2 * cumsum + s_total
    contrib_sorted = row_sums_sorted / (2 * n * n * mean_val)
    # Unsort back to original positions; ``scatter`` is differentiable
    out_64 = torch.empty_like(values_64).scatter_(0, sort_idx, contrib_sorted)
    return out_64.to(orig_dtype)


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


def per_cell_fairness_attribution_spatial(
    pickup_N: torch.Tensor,
    dropoff_N: torch.Tensor,
    active_taxis_N: torch.Tensor,
) -> torch.Tensor:
    """Canonical per-cell decomposition of F_spatial.

    Returns a 1-D tensor of length N (active cells) where
    ``Σᵢ result_i = F_spatial`` and the sign convention is

        positive  → cell contributes more than 1/N baseline to fairness
        ≈ 0       → cell at the negative-fair / anti-fair boundary
        negative  → cell drags fairness below baseline (priority for modification)

    Formulation (1/N-shifted Gini decomposition):

        αᵢ = 1/N − 0.5·(gini_decomp_DSR_i + gini_decomp_ASR_i)

    where each ``gini_decomp_*_i`` is the cell's contribution to the Gini
    coefficient on the corresponding ratio (DSR = pickups/active_taxis,
    ASR = dropoffs/active_taxis). The 1/N baseline is the uniform
    "fairness mass per cell" prior — see
    ``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``.

    This is the single canonical spatial-fairness attribution function in
    the codebase. The trajectory-modification algorithm and the
    fairness-attribution export tool both call it — there is no parallel
    "unfairness" variant.

    Args:
        pickup_N: 1-D tensor of pickup counts per active (cell, t).
        dropoff_N: 1-D tensor of dropoff counts per active (cell, t).
        active_taxis_N: 1-D tensor of active taxi supply per active (cell, t).

    Returns:
        1-D tensor of length N. Sum equals F_spatial.

    Raises:
        ValueError: on shape / dimension / negative-value violations.
    """
    if pickup_N.dim() != 1 or dropoff_N.dim() != 1 or active_taxis_N.dim() != 1:
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must be 1-D tensors."
        )
    if not (pickup_N.shape == dropoff_N.shape == active_taxis_N.shape):
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must have the same shape."
        )
    if (pickup_N < 0).any() or (dropoff_N < 0).any() or (active_taxis_N < 0).any():
        raise ValueError(
            "pickup_N, dropoff_N, and active_taxis_N must not contain negatives."
        )

    n = pickup_N.numel()
    dsr = pickup_N / (active_taxis_N + config.EPS)
    asr = dropoff_N / (active_taxis_N + config.EPS)
    gini_decomp_dsr = per_unit_gini_decomposition(dsr)
    gini_decomp_asr = per_unit_gini_decomposition(asr)
    unfairness_contrib = 0.5 * (gini_decomp_dsr + gini_decomp_asr)
    return (1.0 / n) - unfairness_contrib
