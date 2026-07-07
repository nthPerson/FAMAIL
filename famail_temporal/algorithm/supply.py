"""
Delta-supply math for supply-lift editing.

Differentiable (soft) and discrete (hard) computation of supply changes
when moving pickups between cells.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PRESENCE_KERNEL_SIZE = 5


def state_presence_mass(n_hours_per_block: np.ndarray, n_days: int, t_block: int) -> float:
    """
    Compute the mass per state (mean-hourly).

    Args:
        n_hours_per_block: (24,) array of hours per time block (e.g., all 1s for hourly).
        n_days: number of days in the dataset.
        t_block: time block index into n_hours_per_block.

    Returns:
        float: 1.0 / (12.0 * n_hours_per_block[t_block] * n_days)
    """
    hours = float(n_hours_per_block[t_block])
    return 1.0 / (12.0 * hours * n_days)


def soft_delta_supply(
    probs_batch: torch.Tensor,
    cells: List[Tuple[int, int]],
    t_blocks: List[int],
    masses: List[float],
    signs: List[int],
    grid_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute differentiable delta-supply tensor by embedding soft probabilities
    into a grid and applying 5x5 box blur.

    Args:
        probs_batch: (B, ns, ns) soft assignment probabilities from SoftCellAssignment.
        cells: list of (cx, cy) cell tuples for each batch entry.
        t_blocks: list of time block indices for each batch entry.
        masses: list of masses to scale each entry.
        signs: list of signs (+1 or -1) for each entry.
        grid_shape: (gx, gy, T) grid dimensions.

    Returns:
        (gx, gy, T) torch.Tensor with accumulated soft delta-supply.
        Differentiable end-to-end through probs_batch.
    """
    batch_size = probs_batch.shape[0]
    gx, gy, T = grid_shape
    device = probs_batch.device
    dtype = probs_batch.dtype

    # Infer k from probs shape; assume (B, ns, ns) where ns = 2k+1
    ns = probs_batch.shape[1]
    assert ns == PRESENCE_KERNEL_SIZE, (
        f"probs window size {ns} != PRESENCE_KERNEL_SIZE {PRESENCE_KERNEL_SIZE}: "
        "soft_delta_supply's 5x5 blur kernel assumes SoftCellAssignment's neighborhood "
        "matches the presence-kernel box."
    )
    k = (ns - 1) // 2

    # Accumulator
    delta = torch.zeros((gx, gy, T), device=device, dtype=dtype)

    # Kernel for 5x5 box blur
    kernel = torch.ones(1, 1, PRESENCE_KERNEL_SIZE, PRESENCE_KERNEL_SIZE, device=device, dtype=dtype)

    for b in range(batch_size):
        probs_2d = probs_batch[b]  # (ns, ns)
        cx, cy = cells[b]
        t_block = t_blocks[b]
        mass = masses[b]
        sign = signs[b]

        # Clipped window: extract in-bounds rectangle from probs_2d
        x_lo = max(0, cx - k)
        x_hi = min(gx, cx + k + 1)
        y_lo = max(0, cy - k)
        y_hi = min(gy, cy + k + 1)

        # Corresponding rectangle inside probs_2d
        px_lo = x_lo - (cx - k)
        px_hi = px_lo + (x_hi - x_lo)
        py_lo = y_lo - (cy - k)
        py_hi = py_lo + (y_hi - y_lo)

        # Skip if entirely out of bounds
        if x_hi <= x_lo or y_hi <= y_lo:
            continue

        # Extract slice from probs, multiply by mass, pad to (gx, gy)
        contrib_2d = probs_2d[px_lo:px_hi, py_lo:py_hi] * mass  # (h, w)
        padded_2d = F.pad(contrib_2d, (y_lo, gy - y_hi, x_lo, gx - x_hi))  # (gx, gy)

        # Apply 5x5 box blur (plain conv2d; no post-hoc normalization — the blurred
        # output must remain a genuine function of probs_2d so gradients w.r.t.
        # sub-cell position survive downstream).
        plane = padded_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, gx, gy)
        blurred = F.conv2d(plane, kernel, padding=(PRESENCE_KERNEL_SIZE // 2, PRESENCE_KERNEL_SIZE // 2))[0, 0]  # (gx, gy)

        # Add to accumulator with sign
        delta[:, :, t_block] = delta[:, :, t_block] + sign * blurred

    return delta


def hard_delta_supply(
    positions_old: List[Tuple[int, int]],
    positions_new: List[Tuple[int, int]],
    t_blocks: List[int],
    masses: List[float],
    grid_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Compute discrete delta-supply array by adding/subtracting mass over 5x5 boxes.

    Args:
        positions_old: list of (cx, cy) positions to remove from (with sign=-1).
        positions_new: list of (cx, cy) positions to add to (with sign=+1).
        t_blocks: list of time block indices; t_blocks[i] for i-th old/new entry.
        masses: list of masses; masses[i] for i-th old/new entry.
        grid_shape: (gx, gy, T) grid dimensions.

    Returns:
        (gx, gy, T) np.ndarray with accumulated delta-supply.
    """
    gx, gy, T = grid_shape
    delta = np.zeros((gx, gy, T), dtype=np.float64)
    h = PRESENCE_KERNEL_SIZE // 2

    # Remove from old positions
    for i, (cx, cy) in enumerate(positions_old):
        mass = masses[i]
        t_block = t_blocks[i]
        # 5x5 box centered at (cx, cy)
        for dx in range(-h, h + 1):
            for dy in range(-h, h + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < gx and 0 <= y < gy:
                    delta[x, y, t_block] -= mass

    # Add to new positions
    for i, (cx, cy) in enumerate(positions_new):
        mass = masses[i]
        t_block = t_blocks[i]
        # 5x5 box centered at (cx, cy)
        for dx in range(-h, h + 1):
            for dy in range(-h, h + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < gx and 0 <= y < gy:
                    delta[x, y, t_block] += mass

    return delta
