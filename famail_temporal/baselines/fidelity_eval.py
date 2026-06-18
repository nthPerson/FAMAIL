"""Level-1 fidelity evaluation: discriminator (HuMID) + discriminator-free.

All functions are inference/analysis only — no training, no global state. The
HuMID discriminator (famail_temporal/fidelity) is consumed read-only and
forward-only (under torch.no_grad). See
docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.transmission import jensen_shannon_divergence


# ---------------------------------------------------------------- builders ----

def _xy_from_cells(cells: Sequence[int]) -> List[Tuple[int, int]]:
    """Flat cell ids -> [(x, y), ...] via x = c // GY, y = c % GY."""
    return [(int(c) // gc.GY, int(c) % gc.GY) for c in cells]


def _xy_from_traj(traj) -> List[Tuple[int, int]]:
    return [(int(s.x_grid), int(s.y_grid)) for s in traj.states]


def real_to_disc_tensor(traj) -> torch.Tensor:
    """Trajectory -> discriminator input [L, 4]: (x+1, y+1, time_bucket, day)."""
    rows = [
        [float(s.x_grid) + 1.0, float(s.y_grid) + 1.0,
         float(s.time_bucket), float(s.day_index)]
        for s in traj.states
    ]
    return torch.tensor(rows, dtype=torch.float32)


def generated_to_disc_tensor(
    cells: Sequence[int], time_bucket: int, day_index: int,
) -> torch.Tensor:
    """Generated flat cells -> discriminator input [L, 4].

    Un-flattens each cell to (x, y), adds +1 (1-indexed), and synthesizes a
    constant per-step (time_bucket, day_index) supplied by the caller (the
    paired real seed's temporal context; see plan Global Constraints note).
    """
    rows = [
        [float(x) + 1.0, float(y) + 1.0, float(time_bucket), float(day_index)]
        for (x, y) in _xy_from_cells(cells)
    ]
    return torch.tensor(rows, dtype=torch.float32)


# ------------------------------------------------------------- statistics ----

def trajectory_statistics(
    traj_or_cells: Union[object, Sequence[int]],
) -> Dict[str, float]:
    """{'length', 'mean_displacement', 'coverage'} for a Trajectory or cell list.

    - length: number of steps.
    - mean_displacement: mean Euclidean distance between consecutive (x, y)
      cells (0.0 if length < 2).
    - coverage: count of unique (x, y) cells visited.
    """
    if hasattr(traj_or_cells, "states"):
        xy = _xy_from_traj(traj_or_cells)
    else:
        xy = _xy_from_cells(traj_or_cells)
    length = len(xy)
    coverage = len(set(xy))
    if length < 2:
        mean_disp = 0.0
    else:
        dists = [
            float(np.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]))
            for i in range(length - 1)
        ]
        mean_disp = float(np.mean(dists))
    return {"length": length, "mean_displacement": mean_disp, "coverage": coverage}
