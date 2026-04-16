"""
Attribution pipeline: per-unit fairness contribution -> per-trajectory ranking.

Per-unit attribution comes from fairness.causal.per_unit_attribution, which
decomposes 1 - F_causal into per-unit contributions summing to r^2_demo.

Each trajectory inherits the attribution of its pickup's (cell, t) unit.
Trajectories in inactive units get score 0 and are excluded from modification.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.causal import per_unit_attribution, per_unit_attribution_signed
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.utils.trajectory import Trajectory


def compute_per_unit_attribution(
    bundle: DataBundle,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unsigned and signed attribution over active units.

    Returns:
        (unsigned, signed) each shape (N,) -- per-unit attribution scores.
        Unsigned scores sum to 1 - F_causal (the load-bearing invariant).
        Signed scores indicate direction (over/under-service).
    """
    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D_np = bundle.g0_func(D.numpy())
    g0_D = torch.from_numpy(np.asarray(g0_D_np, dtype=np.float32))
    R = Y - g0_D

    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    IH = tensors['I_minus_H_demo']
    M = tensors['M']
    unsigned = per_unit_attribution(R, IH, M).numpy()
    signed = per_unit_attribution_signed(R, IH, M).numpy()
    return unsigned, signed


def rank_trajectories(
    trajectories: List[Trajectory],
    unit_attribution: np.ndarray,
    unit_map: UnitIndexMap,
) -> List[Tuple[int, float]]:
    """Map each trajectory's pickup (cell, t) -> attribution score.

    Returns [(trajectory_idx, score), ...] sorted descending.
    Trajectories in inactive units get score 0 (placed at the end).
    """
    gy = config.GRID_DIMS[1]
    scored = []
    for i, traj in enumerate(trajectories):
        cx, cy = traj.pickup_cell
        time_bucket = traj.pickup_state.time_bucket
        hour = time_bucket_to_hour(time_bucket)
        t_block = hour_to_block_index(hour)
        flat_cell = cx * gy + cy
        unit_idx = unit_map.from_cell_time(flat_cell, t_block)
        score = float(unit_attribution[unit_idx]) if unit_idx >= 0 else 0.0
        scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def select_top_k(
    scored: List[Tuple[int, float]], k: int,
) -> List[int]:
    """Return indices of the top-k trajectories with strictly positive scores."""
    return [idx for idx, score in scored[:k] if score > 0]
