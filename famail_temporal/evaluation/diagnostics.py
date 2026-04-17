"""Tier C global gradient sensitivity field."""

from __future__ import annotations
import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.causal import compute_fcausal
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fairness.spatial import compute_fspatial


def compute_gradient_sensitivity(
    bundle: DataBundle,
    pickup_3d: np.ndarray,
) -> np.ndarray:
    """Global dF/dp sensitivity grid of shape (gx, gy, T, 2).

    Channels:
        0: d F_spatial / d pickup[x, y, t]
        1: d F_causal  / d pickup[x, y, t]

    Inactive cells are NaN. Fidelity channel omitted - F_fidelity is
    per-trajectory and has no global per-cell gradient.
    """
    mask = bundle.mask_3d
    mask_t = torch.from_numpy(mask)
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[mask]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[mask]).float()

    # Channel 0 - F_spatial
    pickup_tensor_a = torch.from_numpy(pickup_3d.copy()).float().requires_grad_(True)
    pickup_N = pickup_tensor_a[mask_t]
    f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_N)
    grad_sp = torch.autograd.grad(f_spatial, pickup_tensor_a)[0].detach().numpy()

    # Channel 1 - F_causal
    pickup_tensor_b = torch.from_numpy(pickup_3d.copy()).float().requires_grad_(True)
    pickup_N_b = pickup_tensor_b[mask_t]
    D_clamped = torch.clamp(pickup_N_b, min=config.DEMAND_FLOOR)
    with torch.no_grad():
        g0_D = torch.from_numpy(
            np.asarray(bundle.g0_func(D_clamped.detach().numpy()), dtype=np.float32),
        )
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal, _ = compute_fcausal(
        demand_N=pickup_N_b, supply_N=active_N,
        g0_D_N=g0_D,
        I_minus_H_demo=tensors["I_minus_H_demo"], M=tensors["M"],
    )
    grad_ca = torch.autograd.grad(f_causal, pickup_tensor_b)[0].detach().numpy()

    gx, gy = bundle.pickup_3d.shape[:2]
    sens = np.full((gx, gy, config.T, 2), np.nan, dtype=np.float32)
    sens[..., 0][mask] = grad_sp[mask]
    sens[..., 1][mask] = grad_ca[mask]
    return sens
