"""Fairness-aware state-space grid builder.

Produces a (grid_x, grid_y, T, 4) tensor whose channels are:
    0: αᵢ_spatial         (sums to F_spatial; canonical 1/N-shifted decomp)
    1: αᵢ_causal          (sums to F_causal; canonical 1/N-shifted decomp)
    2: gini_decomp_dsr    (sums to Gini(DSR); per-cell Gini contribution)
    3: gini_decomp_asr    (sums to Gini(ASR); per-cell Gini contribution)

Channels 0 and 1 are the canonical fairness attributions (positive = cell
contributes more than 1/N baseline to fairness; negative = drags below).
Channels 2 and 3 are the underlying Gini decompositions retained for
diagnostic purposes (sum to the Gini coefficient = unfairness side).

Inactive units are NaN on all channels. See
``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`` for the
formulation of the per-cell fairness attribution.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.spatial import (
    per_cell_fairness_attribution_spatial,
    per_unit_gini_decomposition,
)
from famail_temporal.fairness.causal import per_cell_fairness_attribution_causal
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch


def build_fairness_grid(
    bundle: DataBundle,
    pickup_3d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the (grid_x, grid_y, T, 4) fairness-aware grid.

    Args:
        bundle: DataBundle - provides dropoff_3d, active_taxis_3d, mask_3d,
                hat_matrices, g0_func.
        pickup_3d: Optional override for the pickup tensor. If None, uses
                   bundle.pickup_3d (the before-state). For the after-state
                   grid, pass TrajectoryModifier.current_pickup_3d().

    Returns:
        (grid_x, grid_y, T, 4) float32 ndarray. Inactive cells are NaN on
        all 4 channels.
    """
    if pickup_3d is None:
        pickup_3d = bundle.pickup_3d
    if pickup_3d.shape != bundle.pickup_3d.shape:
        raise ValueError(
            f"pickup_3d shape {pickup_3d.shape} != bundle.pickup_3d shape "
            f"{bundle.pickup_3d.shape}"
        )

    if not bundle.mask_3d.any():
        raise ValueError(
            "bundle.mask_3d has no active units — cannot build a fairness grid "
            "with zero units. Check the preprocess cache and active-mask thresholds."
        )

    mask = bundle.mask_3d
    pickup_N = torch.from_numpy(pickup_3d[mask]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()

    # Channel 0 — spatial fairness attribution (sums to F_spatial).
    spatial_attr = per_cell_fairness_attribution_spatial(
        pickup_N, dropoff_N, active_N,
    ).detach().numpy()

    # Channels 2, 3 — diagnostic Gini decompositions (sum to Gini = unfair).
    dsr = pickup_N / (active_N + config.EPS)
    asr = dropoff_N / (active_N + config.EPS)
    gini_dsr = per_unit_gini_decomposition(dsr).detach().numpy()
    gini_asr = per_unit_gini_decomposition(asr).detach().numpy()

    # Channel 1 — causal fairness attribution (sums to F_causal).
    D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
    Y = active_N / D_clamped
    g0_D = torch.from_numpy(
        np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32)
    )
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    causal_attr = per_cell_fairness_attribution_causal(
        R, tensors["X_demo"], tensors["XtX_inv"],
    ).detach().numpy()

    # Scatter back with NaN on inactive cells. Shape is derived from the
    # bundle directly so we don't depend on a separately-sourced config.T —
    # pickup_3d.shape is the single authority for the artifact's geometry.
    grid = np.full(bundle.pickup_3d.shape + (4,), np.nan, dtype=np.float32)
    ix_x, ix_y, ix_t = np.where(mask)
    grid[ix_x, ix_y, ix_t, 0] = spatial_attr
    grid[ix_x, ix_y, ix_t, 1] = causal_attr
    grid[ix_x, ix_y, ix_t, 2] = gini_dsr
    grid[ix_x, ix_y, ix_t, 3] = gini_asr
    return grid
