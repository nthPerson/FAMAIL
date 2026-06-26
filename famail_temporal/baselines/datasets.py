"""Dataset-variant builders for the FAMAIL GAN baselines (Phase 1, data-level).

Defines the raw and *filtered* demand-grid variants used by the data-level
fairness x retention Pareto. The filtered variant removes the top-K
most-unfair seeking trajectories and subtracts each removed trajectory's
pickup contribution from the demand grid, using the SAME per-trajectory
pickup mass the editing modifier uses (1/(n_hours_per_block[t_block]*n_days),
see algorithm/modifier.py), so filtering and editing are accounted for
consistently. Supply (active_taxis) is left unchanged — filtering removes a
demand event, not taxi presence, mirroring the editing convention.
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories,
)
from famail_temporal.data.aggregation import (
    hour_to_block_index, time_bucket_to_hour,
)
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.trajectory import Trajectory


def pickup_unit_of(traj: Trajectory) -> Tuple[int, int, int]:
    """Return (cx, cy, t_block) for a trajectory's terminal pickup."""
    cx, cy = traj.pickup_cell
    t_block = hour_to_block_index(
        time_bucket_to_hour(traj.pickup_state.time_bucket)
    )
    return cx, cy, t_block


def pickup_mass(bundle: DataBundle, t_block: int) -> float:
    """Mean-hourly demand mass of one pickup event in t_block.

    Matches TrajectoryModifier: 1 / (n_hours_per_block[t_block] * n_days).
    """
    n_hours = int(bundle.n_hours_per_block[t_block])
    return 1.0 / (n_hours * bundle.n_days)


def rank_unfair_trajectory_indices(bundle: DataBundle) -> List[int]:
    """Indices into bundle.trajectories ordered most-unfair first.

    Only strictly-negative-attribution trajectories (pickup cells dragging
    fairness below the 1/N baseline) are returned; at/above-baseline
    (score >= 0) and inactive (+inf) trajectories are excluded — they are
    not filtering candidates.
    """
    attribution = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(
        bundle.trajectories, attribution, bundle.unit_map,
    )
    return [idx for idx, score in scored if score < 0]


def _most_fair_from_scored(scored, n=None):
    """From rank_trajectories output [(idx, score), ...], return indices ordered
    MOST-FAIR first (highest FINITE αᵢ), excluding inactive (+inf) cells.
    Top-n if n given."""
    import math
    finite = [(idx, s) for idx, s in scored if math.isfinite(s)]
    finite.sort(key=lambda x: x[1], reverse=True)   # highest (most-fair) first
    idxs = [idx for idx, _ in finite]
    return idxs[:n] if n is not None else idxs


def rank_fair_trajectory_indices(bundle: DataBundle, n=None) -> List[int]:
    """Indices into bundle.trajectories ordered MOST-FAIR first (highest finite
    per-cell attribution αᵢ; inactive +inf cells excluded). The mirror of
    rank_unfair_trajectory_indices — the 'select already-fair data' baseline."""
    attribution = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
    return _most_fair_from_scored(scored, n)


def build_filtered_pickup_3d(
    bundle: DataBundle, removed_trajs: List[Trajectory],
) -> np.ndarray:
    """Demand grid after removing the given trajectories' pickup events.

    Returns a fresh array (bundle.pickup_3d is not mutated). Only the
    modified cells are touched; each is floored at ``config.DEMAND_FLOOR``
    after subtraction. Flooring is required because the spatial-fairness path
    rejects negative demand, and demand cannot physically go negative. Over-
    subtraction is possible because ``pickup_3d`` is an independent mean-
    hourly counts artifact, not an exact sum of seeking-trajectory pickup
    masses; flooring handles that gracefully. Untouched cells (including
    legitimate zeros) are left exactly as in ``bundle.pickup_3d``.
    """
    pickup_3d = bundle.pickup_3d.copy()
    floor = config.DEMAND_FLOOR
    for traj in removed_trajs:
        cx, cy, t_block = pickup_unit_of(traj)
        reduced = float(pickup_3d[cx, cy, t_block]) - pickup_mass(bundle, t_block)
        pickup_3d[cx, cy, t_block] = max(reduced, floor)
    return pickup_3d
