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


def make_traj_at(cx: int, cy: int, t_block: int, traj_id: int) -> Trajectory:
    """A 2-state trajectory whose terminal (pickup) state is at (cx, cy, t_block)."""
    tb = time_bucket_for_block(t_block)
    states = [
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
        TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tb, day_index=1),
    ]
    return Trajectory(trajectory_id=traj_id, driver_id=0, states=states)
