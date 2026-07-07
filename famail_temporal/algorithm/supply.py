"""
Delta-supply math for supply-lift editing.

Differentiable (soft) and discrete (hard) computation of supply changes
when moving pickups between cells.
"""

from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from famail_temporal import config
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour

if TYPE_CHECKING:
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.data.loader import DataBundle


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


def supply_gradient_N(bundle: "DataBundle", objective: "FAMAILObjective") -> np.ndarray:
    """Compute dL/dS_i at the baseline supply, via the objective's optional
    ``delta_supply_N`` parameter (Task 3).

    Builds a zero leaf tensor the same shape as the active-unit supply
    vector, runs one forward + backward through ``objective``, and returns
    the leaf's gradient. Because ``delta_supply_N=0`` reproduces the
    baseline supply exactly (``clamp(active_taxis_N + 0, min=SUPPLY_FLOOR)``
    == ``active_taxis_N`` whenever the baseline already respects the floor),
    this gradient equals dL/dS_i at the unedited bundle.

    Note: at units where the baseline S_i == SUPPLY_FLOOR exactly, the
    ``clamp`` subgradient is 0 on the low side, so the returned gradient may
    be zeroed there even if the "true" unclamped derivative is nonzero.
    This is considered acceptable: those units are already floored and can
    only ever gain supply (never lose it) under any downstream edit, so a
    zero attribution simply means "no differentiable signal to rank by at
    this unit" rather than a modeling error.
    """
    n_units = bundle.unit_map.n_units
    delta_leaf = torch.zeros(n_units, dtype=torch.float32, requires_grad=True)
    soft_pickup_3d = torch.from_numpy(bundle.pickup_3d).float()
    total, _ = objective.forward(soft_pickup_3d, delta_supply_N=delta_leaf)
    total.backward()
    return delta_leaf.grad.detach().numpy()


def _box_sum_grid(grad_grid: np.ndarray, half_width: int) -> np.ndarray:
    """Precompute, for every (x, y, t), the sum of ``grad_grid`` over the
    ``(2*half_width+1)``-square box centered at (x, y) (same time slice),
    clipped at the grid edges (out-of-bounds cells simply do not
    contribute — the same clipping convention as ``hard_delta_supply``).

    Implemented via a 2D summed-area table (per time slice) so every
    (x, y, t) box sum is an O(1) lookup afterward — this is what lets
    ``lift_candidates`` screen 95k trajectories x 24 deltas without
    recomputing a 5x5 sum from scratch each time.
    """
    gx, gy, T = grad_grid.shape
    # Summed-area table with a leading zero row/column per time slice:
    # sat[i, j, t] = sum of grad_grid[:i, :j, t].
    sat = np.zeros((gx + 1, gy + 1, T), dtype=np.float64)
    sat[1:, 1:, :] = np.cumsum(np.cumsum(grad_grid, axis=0), axis=1)

    xs = np.arange(gx)
    ys = np.arange(gy)
    x0 = np.clip(xs - half_width, 0, gx)
    x1 = np.clip(xs + half_width + 1, 0, gx)
    y0 = np.clip(ys - half_width, 0, gy)
    y1 = np.clip(ys + half_width + 1, 0, gy)

    # Standard SAT box-sum inclusion-exclusion, broadcast over the whole
    # (gx, gy) grid at once: box(x, y) = sat[x1,y1] - sat[x0,y1] - sat[x1,y0] + sat[x0,y0].
    box_sum = (
        sat[x1][:, y1] - sat[x0][:, y1] - sat[x1][:, y0] + sat[x0][:, y0]
    )
    return box_sum


def lift_candidates(
    bundle: "DataBundle",
    grad_N: np.ndarray,
    tail_len: int = config.TAIL_LEN,
    epsilon: float = config.EPSILON_BALL,
) -> List[Tuple[int, float]]:
    """Linearized lift-candidate screen: score each trajectory by the best
    achievable gain from rigidly shifting its "seeking tail" by an integer
    offset delta in [-epsilon, epsilon]^2.

    Tail = the last ``min(tail_len, len(states) - 2)`` states before the
    pickup, plus the pickup state itself. The ``len(states) - 2`` cap
    guarantees at least one leading state (the anchor, index 0) is never
    part of the tail and therefore never moves — this holds regardless of
    ``tail_len``. Trajectories with fewer than 3 states are skipped (no
    room for an anchor + a tail).

    For each candidate delta, the score is the linearized gain of
    translating every tail state by delta (clipped to the grid):
        score(delta) = sum_states mass * (G_box[new_pos] - G_box[old_pos])
    where G_box is the 5x5-box sum of ``grad_N`` embedded on the grid, and
    "mass" is ``state_presence_mass`` at that state's time block. A
    trajectory's score is the max over delta of this linearized gain
    (fast screen; the optimizer refines the actual delta). delta=(0,0) is
    excluded from the candidate set — it is always exactly 0 by
    construction, so including or excluding it can never change any
    trajectory's best score or the resulting ranking.

    Returns ``[(trajectory_idx, score), ...]`` sorted descending by score.
    """
    gx, gy, T = bundle.mask_3d.shape
    grad_grid = np.zeros((gx, gy, T), dtype=np.float64)
    grad_grid[bundle.mask_3d] = grad_N

    half_width = PRESENCE_KERNEL_SIZE // 2
    G_box = _box_sum_grid(grad_grid, half_width)

    eps_int = int(round(epsilon))
    deltas = np.array(
        [
            (dx, dy)
            for dx in range(-eps_int, eps_int + 1)
            for dy in range(-eps_int, eps_int + 1)
            if not (dx == 0 and dy == 0)
        ],
        dtype=np.int64,
    )  # (n_deltas, 2); delta=(0,0) excluded (see docstring)

    results: List[Tuple[int, float]] = []
    for idx, traj in enumerate(bundle.trajectories):
        n = len(traj.states)
        if n < 3:
            continue
        k_tail = min(tail_len, n - 2)
        tail_states = traj.states[-(k_tail + 1):]  # k_tail preceding states + pickup

        positions = np.array(
            [(int(s.x_grid), int(s.y_grid)) for s in tail_states], dtype=np.int64
        )  # (n_tail, 2)
        positions[:, 0] = np.clip(positions[:, 0], 0, gx - 1)
        positions[:, 1] = np.clip(positions[:, 1], 0, gy - 1)
        t_blocks = np.array(
            [hour_to_block_index(time_bucket_to_hour(s.time_bucket)) for s in tail_states],
            dtype=np.int64,
        )  # (n_tail,)
        masses = np.array(
            [
                state_presence_mass(bundle.n_hours_per_block, bundle.n_days, int(tb))
                for tb in t_blocks
            ],
            dtype=np.float64,
        )  # (n_tail,)

        old_vals = G_box[positions[:, 0], positions[:, 1], t_blocks]  # (n_tail,)

        new_positions = positions[None, :, :] + deltas[:, None, :]  # (n_deltas, n_tail, 2)
        new_x = np.clip(new_positions[:, :, 0], 0, gx - 1)
        new_y = np.clip(new_positions[:, :, 1], 0, gy - 1)
        t_broadcast = np.broadcast_to(t_blocks[None, :], new_x.shape)
        new_vals = G_box[new_x, new_y, t_broadcast]  # (n_deltas, n_tail)

        gain = (new_vals - old_vals[None, :]) * masses[None, :]  # (n_deltas, n_tail)
        score_per_delta = gain.sum(axis=1)  # (n_deltas,)
        best_score = float(score_per_delta.max())
        results.append((idx, best_score))

    results.sort(key=lambda pair: pair[1], reverse=True)
    return results
