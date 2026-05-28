"""Trajectory <-> cell-token sequence encoding and conditioning context."""
from __future__ import annotations
from typing import List, Tuple

from famail_temporal.baselines.gan import config as gc
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.utils.trajectory import Trajectory


def flat_cell(x: int, y: int) -> int:
    return int(x) * gc.GY + int(y)


def unflat_cell(idx: int) -> Tuple[int, int]:
    return divmod(int(idx), gc.GY)


def trajectory_to_tokens(traj: Trajectory) -> List[int]:
    """[BOS, cell_0, ..., cell_{L-1}, EOS] of flat cell ids."""
    cells = [flat_cell(s.x_grid, s.y_grid) for s in traj.states]
    return [gc.BOS] + cells + [gc.EOS]


def trajectory_context(traj: Trajectory) -> Tuple[int, int]:
    """(start flat-cell, start time-block) for conditioning."""
    s0 = traj.states[0]
    t_block = hour_to_block_index(time_bucket_to_hour(s0.time_bucket))
    return flat_cell(s0.x_grid, s0.y_grid), t_block
