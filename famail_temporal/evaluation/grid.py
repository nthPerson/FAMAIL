"""Fairness-aware state-space grid builder.

Produces a (grid_x, grid_y, T, 4) tensor whose channels are:
    0: spatial_attr       (sums to 1 - F_spatial)
    1: causal_attr        (sums to 1 - F_causal)
    2: gini_decomp_dsr    (sums to Gini(DSR))
    3: gini_decomp_asr    (sums to Gini(ASR))

Inactive units are NaN on all channels.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.spatial import compute_spatial_attribution
from famail_temporal.fairness.causal import per_unit_attribution_from_compact
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

    # Project 3D -> N in canonical order (numpy boolean indexing iterates
    # in C order, matching UnitIndexMap's cell-major/time-within-cell ordering).
    pickup_N = torch.from_numpy(pickup_3d[mask]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()

    # Channels 0, 2, 3 - spatial attribution.
    sp = compute_spatial_attribution(pickup_N, dropoff_N, active_N)
    spatial_attr = sp["spatial_attr"].detach().numpy()
    gini_dsr = sp["gini_decomp_dsr"].detach().numpy()
    gini_asr = sp["gini_decomp_asr"].detach().numpy()

    # Channel 1 - causal attribution (sums to 1 - F_causal).
    D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
    Y = active_N / D_clamped
    g0_D = torch.from_numpy(
        np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32)
    )
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    causal_attr = per_unit_attribution_from_compact(
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
