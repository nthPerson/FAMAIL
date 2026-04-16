"""
Differentiable soft cell assignment.

For a continuous pickup location (x, y) in R^2, produces a probability
distribution over a (2k+1) x (2k+1) neighborhood centered at floor((x, y)).
Temperature controls sharpness (tau=1 soft, tau=0.1 near-hard).

This is the mechanism that enables gradient-based trajectory modification:
gradients from the fairness objective flow back through the soft probability
distribution to the continuous pickup coordinates.

Ported from objective_function/soft_cell_assignment/module.py.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from famail_temporal import config


class SoftCellAssignment(nn.Module):
    def __init__(
        self,
        grid_dims: Tuple[int, int] = config.GRID_DIMS,
        neighborhood_size: int = config.SOFT_NEIGHBORHOOD_SIZE,
        initial_temperature: float = config.TAU_MAX,
    ):
        super().__init__()
        assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"
        self.grid_dims = grid_dims
        self.k = neighborhood_size // 2
        self.register_buffer(
            "temperature",
            torch.tensor(float(initial_temperature)),
        )

    def forward(self, loc: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        """Compute soft probability over the (2k+1) x (2k+1) neighborhood.

        loc: (batch, 2) continuous coordinates, may require_grad.
        cell: (batch, 2) integer cell coordinates (float tensor).
        returns: (batch, 2k+1, 2k+1) probability distribution.

        The probability distribution is produced by:
        1. Computing the squared distance from loc to each neighborhood cell center
        2. Applying a Gaussian kernel: logit_ij = -dist_sq_ij / temperature
        3. Softmax over the neighborhood -> probability distribution

        Gradient flows from probs -> logits -> dist_sq -> loc.
        """
        batch_size = loc.shape[0]
        k = self.k
        ns = 2 * k + 1

        offsets = torch.arange(-k, k + 1, device=loc.device, dtype=loc.dtype)
        dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
        rel = torch.stack([dx, dy], dim=-1)  # (ns, ns, 2)

        abs_cells = cell.unsqueeze(1).unsqueeze(2) + rel.unsqueeze(0)  # (B, ns, ns, 2)
        loc_exp = loc.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, 2)

        # Distance from loc to cell center (center = integer coord + 0.5)
        dist_sq = ((abs_cells + 0.5) - loc_exp).pow(2).sum(dim=-1)  # (B, ns, ns)

        logits = -dist_sq / (self.temperature + config.EPS)
        logits_flat = logits.view(batch_size, -1)
        probs_flat = torch.softmax(logits_flat, dim=-1)
        probs = probs_flat.view(batch_size, ns, ns)
        return probs

    def set_temperature(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError(f"Temperature must be > 0, got {tau}")
        self.temperature = torch.tensor(float(tau), device=self.temperature.device)

    def get_annealed_temperature(
        self, iteration: int, total_iterations: int,
        tau_max: float = config.TAU_MAX, tau_min: float = config.TAU_MIN,
    ) -> float:
        """Exponential annealing: tau_t = tau_max * (tau_min/tau_max)^(t/T)."""
        if total_iterations <= 1:
            return tau_min
        progress = iteration / (total_iterations - 1)
        return tau_max * (tau_min / tau_max) ** progress


def inject_soft_counts_into_3d(
    base_counts_3d: torch.Tensor,
    probs_2d: torch.Tensor,
    cell_xy: Tuple[int, int],
    t_block: int,
    k: int,
    pickup_mass: float,
) -> torch.Tensor:
    """Inject probs_2d * pickup_mass into base_counts_3d at slice t_block.

    Uses the delta-tensor pattern for autograd-safe gradient flow:
        delta = zeros_like(base)
        delta[:, :, t_block] = scatter(probs * pickup_mass)
        return base + delta

    The delta-tensor pattern is preferred over clone-then-in-place because
    it unambiguously preserves the computational graph from probs_2d through
    to the returned tensor. In-place ops on a clone can subtly break autograd
    under certain PyTorch versions.

    Only cells in the (2k+1, 2k+1) neighborhood of cell_xy in slice t_block
    are modified. Cells outside the grid bounds are silently skipped.

    Args:
        base_counts_3d: (grid_x, grid_y, T) float32 — the background counts
            (without the current trajectory's contribution)
        probs_2d: (2k+1, 2k+1) — soft assignment probabilities from SoftCellAssignment
        cell_xy: (cx, cy) — the original pickup cell (center of the neighborhood)
        t_block: time block index for this trajectory's pickup
        k: neighborhood half-width (k from SoftCellAssignment)
        pickup_mass: mass to inject (1 / (n_hours_per_block * n_days) for mean-hourly)

    Returns:
        (grid_x, grid_y, T) tensor = base_counts_3d + delta, where delta has
        gradient flow through probs_2d * pickup_mass at the t_block slice.
    """
    gx, gy, t_total = base_counts_3d.shape
    assert probs_2d.shape == (2 * k + 1, 2 * k + 1), (
        f"probs_2d shape {probs_2d.shape} != expected ({2*k+1}, {2*k+1})"
    )
    assert 0 <= t_block < t_total, f"t_block {t_block} out of range [0, {t_total})"

    delta = torch.zeros_like(base_counts_3d)
    cx, cy = cell_xy
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            ni, nj = cx + di, cy + dj
            if 0 <= ni < gx and 0 <= nj < gy:
                delta[ni, nj, t_block] = (
                    delta[ni, nj, t_block]
                    + probs_2d[di + k, dj + k] * pickup_mass
                )
    return base_counts_3d + delta
