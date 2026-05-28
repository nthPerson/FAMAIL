"""
Attribution pipeline: per-cell fairness contribution → per-trajectory ranking.

Per-cell attribution comes from the canonical
``per_cell_fairness_attribution_causal`` function in ``fairness.causal``,
which returns the 1/N-shifted decomposition summing to F_causal:

    αᵢ > 0 → cell contributes more than uniform baseline to fairness
    αᵢ < 0 → cell drags fairness below baseline (priority for modification)

Each trajectory inherits the attribution of its pickup-cell's
``(x, y, time-block)`` unit. Trajectories in inactive units get a sentinel
score of ``+inf`` so they are placed at the end of the ascending ranking
(they are NOT priority targets) and are excluded from ``select_top_k`` —
which selects trajectories with strictly negative attribution (cells
actively dragging fairness down).

See ``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md`` for
the mathematical formulation and sign convention this module assumes.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.causal import per_cell_fairness_attribution_causal
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.utils.trajectory import Trajectory


# Sentinel score for trajectories whose pickup cell is inactive. Larger
# than any positive αᵢ in practice (αᵢ ∈ [−1, 1/N] for active cells), so
# inactive trajectories always sort to the end of an ascending ranking
# and never satisfy the "αᵢ < 0" filter in select_top_k.
_INACTIVE_SCORE = float("inf")


def compute_per_unit_attribution(
    bundle: DataBundle,
    pickup_3d: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the per-cell fairness attribution over active units.

    Wraps the canonical ``per_cell_fairness_attribution_causal`` and
    returns a length-N numpy array. Sum of the array equals F_causal.
    Sign convention: positive = above-baseline fairness contribution;
    negative = drags fairness below baseline.

    Args:
        bundle: A loaded ``DataBundle``. Uses bundle.active_taxis_3d,
            mask_3d, g0_func, and hat_matrices.
        pickup_3d: Optional pickup tensor to use in place of
            ``bundle.pickup_3d``. Use ``modifier.current_pickup_3d()`` to
            re-attribute against the post-modification state — required
            for iterative top-k with re-attribution between modifications.
            Same shape as ``bundle.pickup_3d``.

    Returns:
        np.ndarray of shape (N,) where N is the count of active cells.
    """
    if pickup_3d is None:
        pickup_3d = bundle.pickup_3d
    if pickup_3d.shape != bundle.pickup_3d.shape:
        raise ValueError(
            f"pickup_3d shape {pickup_3d.shape} != bundle.pickup_3d shape "
            f"{bundle.pickup_3d.shape}"
        )
    D = torch.from_numpy(pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D_np = bundle.g0_func(D.numpy())
    g0_D = torch.from_numpy(np.asarray(g0_D_np, dtype=np.float32))
    R = Y - g0_D

    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    X_demo = tensors['X_demo']
    XtX_inv = tensors['XtX_inv']
    return per_cell_fairness_attribution_causal(R, X_demo, XtX_inv).numpy()


def rank_trajectories(
    trajectories: List[Trajectory],
    unit_attribution: np.ndarray,
    unit_map: UnitIndexMap,
) -> List[Tuple[int, float]]:
    """Map each trajectory's pickup (cell, t) → fairness attribution score.

    Returns ``[(trajectory_idx, score), ...]`` sorted ASCENDING by score —
    the most-negative αᵢ (cells dragging fairness down most) come first.
    Trajectories in inactive units get a sentinel ``+inf`` score and sort
    to the end (never priority targets).

    The ascending direction is the canonical "rank-by-priority" semantics
    under the F-decomposition: trajectories whose pickup cell most
    detracts from fairness are highest priority for modification.
    """
    # Read gy from the unit_map so this works under both production
    # (config.GRID_DIMS[1] == 90) and test bundles (smaller grids).
    gy = unit_map.grid_shape[1]
    scored = []
    for i, traj in enumerate(trajectories):
        cx, cy = traj.pickup_cell
        time_bucket = traj.pickup_state.time_bucket
        hour = time_bucket_to_hour(time_bucket)
        t_block = hour_to_block_index(hour)
        flat_cell = cx * gy + cy
        unit_idx = unit_map.from_cell_time(flat_cell, t_block)
        score = (
            float(unit_attribution[unit_idx])
            if unit_idx >= 0
            else _INACTIVE_SCORE
        )
        scored.append((i, score))
    scored.sort(key=lambda x: x[1])  # ascending: most-negative first
    return scored


def select_top_k(
    scored: List[Tuple[int, float]],
    k: int,
    trajectories: List[Trajectory] | None = None,
    max_per_unit: int | None = None,
    max_per_cell: int | None = None,
) -> List[int]:
    """Return indices of the top-k trajectories with strictly negative scores.

    Strictly-negative filter: a trajectory only enters the top-k if its
    pickup cell has αᵢ < 0 (cell actively drags fairness below the 1/N
    baseline). Cells at the baseline (αᵢ ≈ 0) or above (αᵢ > 0) are
    helping fairness and have no priority for modification.

    Optional diversity constraints (see ``docs/TRAJECTORY_EDITING_METHODOLOGY.md``
    §8.3 for the empirical motivation):

    - ``max_per_unit``: maximum trajectories selected from any single
      (pickup_cell, t_block) unit. Default ``None`` (no cap). Setting
      ``max_per_unit = 1`` enforces that every selected trajectory comes
      from a distinct (cell, t_block), which prevents the source-side
      concentration that drives destination pile-up. Recommended for any
      run at production k where one unit might otherwise dominate.
    - ``max_per_cell``: maximum trajectories per pickup cell (across all
      time blocks). Default ``None``. Setting e.g. 1 enforces a fully
      cell-distinct selection.

    When either constraint is set, ``trajectories`` must be passed so
    each candidate's (cell, t_block) can be derived. ``scored`` indices
    must reference positions in ``trajectories``.
    """
    if (max_per_unit is not None or max_per_cell is not None) and trajectories is None:
        raise ValueError(
            "trajectories must be provided when max_per_unit or "
            "max_per_cell is set"
        )

    unit_counts: dict[Tuple[int, int, int], int] = {}
    cell_counts: dict[Tuple[int, int], int] = {}
    selected: List[int] = []

    for idx, score in scored:
        if score >= 0:
            break  # ascending order; no more strictly-negative candidates
        if len(selected) >= k:
            break

        if max_per_unit is not None or max_per_cell is not None:
            traj = trajectories[idx]
            cx, cy = traj.pickup_cell
            t_block = hour_to_block_index(
                time_bucket_to_hour(traj.pickup_state.time_bucket)
            )
            unit_key = (cx, cy, t_block)
            cell_key = (cx, cy)
            if (max_per_unit is not None
                    and unit_counts.get(unit_key, 0) >= max_per_unit):
                continue
            if (max_per_cell is not None
                    and cell_counts.get(cell_key, 0) >= max_per_cell):
                continue
            unit_counts[unit_key] = unit_counts.get(unit_key, 0) + 1
            cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1

        selected.append(idx)

    return selected
