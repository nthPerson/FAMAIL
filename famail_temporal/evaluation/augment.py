"""Trajectory augmentation - widens 4-element states to 8-element states.

Output format is a drop-in replacement for passenger_seeking_trajs_45-800.pkl:
a dict keyed by driver_id, values are lists of trajectories, each trajectory
is a list of 8-element state lists. Indices 0-3 are 1-indexed on disk.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List

import numpy as np

from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.utils.trajectory import Trajectory


def augment_trajectories(
    trajectories: List[Trajectory],
    grid: np.ndarray,
) -> Dict[int, List[List[list]]]:
    """Produce the driver-keyed augmented dataset.

    Each state is widened from 4 to 8 elements:
        [x_grid, y_grid, time_bucket, day_index,
         spatial_attr, causal_attr, gini_decomp_dsr, gini_decomp_asr]

    Indices 0-3 written 1-indexed (drop-in compatibility with
    passenger_seeking_trajs_45-800.pkl). Indices 4-7 come from
    grid[x, y, t_block, :] (NaN for inactive cells).

    Every input trajectory is included in the output.
    """
    if grid.ndim != 4 or grid.shape[3] != 4:
        raise ValueError(
            f"grid must have shape (gx, gy, T, 4); got {grid.shape}"
        )

    out: Dict[int, List[List[list]]] = defaultdict(list)
    for traj in trajectories:
        augmented_states: List[list] = []
        for st in traj.states:
            x = int(st.x_grid)
            y = int(st.y_grid)
            t_block = hour_to_block_index(time_bucket_to_hour(st.time_bucket))
            fairness = grid[x, y, t_block, :]
            augmented_states.append([
                x + 1,
                y + 1,
                int(st.time_bucket),
                int(st.day_index),
                float(fairness[0]),
                float(fairness[1]),
                float(fairness[2]),
                float(fairness[3]),
            ])
        out[int(traj.driver_id)].append(augmented_states)

    return dict(out)
