"""Test helpers: build synthetic trajectories on a bundle's active units."""
from __future__ import annotations
from typing import List, Tuple

from famail_temporal import config
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def active_units(bundle, n: int) -> List[Tuple[int, int, int]]:
    """Return up to n active (cx, cy, t_block) triples from the bundle mask."""
    gx, gy, T = bundle.mask_3d.shape
    out: List[Tuple[int, int, int]] = []
    for t in range(T):
        for x in range(gx):
            for y in range(gy):
                if bundle.mask_3d[x, y, t]:
                    out.append((x, y, t))
                    if len(out) >= n:
                        return out
    return out


def time_bucket_for_block(t_block: int) -> int:
    """A 1-indexed 5-min time_bucket whose hour maps back to t_block.

    Uses the block's start hour from config.TIME_BLOCKS; time_bucket =
    start_hour*12 + 1 so time_bucket_to_hour(...) == start_hour and
    hour_to_block_index(start_hour) == t_block.
    """
    start_hour = config.TIME_BLOCKS[t_block][1]
    return start_hour * 12 + 1


def negative_attribution_units(bundle, n: int) -> List[Tuple[int, int, int]]:
    """Return up to n active (cx, cy, t_block) units with attribution < 0.

    These are the cells dragging fairness below the 1/N baseline — the only
    valid filtering candidates. The generic synthetic bundle is nearly
    uniformly fair, so tests must target these units explicitly to exercise
    ranking/filtering rather than placing trajectories on arbitrary active
    units (which may all be at/above baseline).
    """
    from famail_temporal.algorithm.attribution import compute_per_unit_attribution
    attr = compute_per_unit_attribution(bundle)
    gy = bundle.unit_map.grid_shape[1]
    out: List[Tuple[int, int, int]] = []
    for i in range(bundle.unit_map.n_units):
        if attr[i] < 0:
            fc = bundle.unit_map.to_flat_cell(i)
            tb = bundle.unit_map.to_time_block(i)
            cx, cy = fc // gy, fc % gy
            out.append((int(cx), int(cy), int(tb)))
            if len(out) >= n:
                break
    return out


def make_traj_at(cx: int, cy: int, t_block: int, traj_id: int) -> Trajectory:
    """A 2-state trajectory whose terminal (pickup) state is at (cx, cy, t_block)."""
    tb = time_bucket_for_block(t_block)
    states = [
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
    ]
    return Trajectory(trajectory_id=traj_id, driver_id=0, states=states)
