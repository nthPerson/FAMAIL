"""Level-1 fidelity evaluation: discriminator (HuMID) + discriminator-free.

All functions are inference/analysis only — no training, no global state. The
HuMID discriminator (famail_temporal/fidelity) is consumed read-only and
forward-only (under torch.no_grad). See
docs/superpowers/specs/2026-06-17-level1-data-quality-table-design.md.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.transmission import jensen_shannon_divergence

if TYPE_CHECKING:  # annotation-only; runtime dispatch is duck-typed (see below)
    from famail_temporal.utils.trajectory import Trajectory


# ---------------------------------------------------------------- builders ----

def _xy_from_cells(cells: Sequence[int]) -> List[Tuple[int, int]]:
    """Flat cell ids -> [(x, y), ...] via x = c // GY, y = c % GY."""
    return [(int(c) // gc.GY, int(c) % gc.GY) for c in cells]


def _xy_from_traj(traj) -> List[Tuple[int, int]]:
    return [(int(s.x_grid), int(s.y_grid)) for s in traj.states]


def real_to_disc_tensor(traj) -> torch.Tensor:
    """Trajectory -> discriminator input [L, 4]: (x+1, y+1, time_bucket, day).

    NOT equivalent to ``Trajectory.to_tensor()`` / ``to_discriminator_format()``
    (utils/trajectory.py): those return RAW 0-indexed coords. The HuMID
    discriminator expects 1-indexed coords (spec §3.7, mirrors
    fidelity/context.py), so this adds +1 to x and y. Always build discriminator
    inputs through this function, never the raw Trajectory methods.
    """
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
    traj_or_cells: Union["Trajectory", Sequence[int]],
) -> Dict[str, float]:
    """{'length', 'mean_displacement', 'coverage'} for a Trajectory or cell list.

    - length: number of steps (0 for an empty cell list).
    - mean_displacement: mean Euclidean distance between consecutive (x, y)
      cells (0.0 if length < 2).
    - coverage: count of unique (x, y) cells visited.
    """
    # Deliberate duck-typed dispatch (not isinstance): a real ``Trajectory`` has
    # ``.states``; anything else is treated as a flat cell-id sequence. This lets
    # callers and tests pass lightweight stand-ins without importing Trajectory.
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


# -------------------------------------------------- distributional fidelity ----

_STAT_KEYS = ("length", "mean_displacement", "coverage")
BINS = 50   # shared histogram bin count (spec §7: "Bin spec is a module constant")


def _hist(values: List[float], lo: float, hi: float, bins: int) -> np.ndarray:
    """Normalized histogram over [lo, hi]; uniform if the range is degenerate."""
    arr = np.asarray(values, dtype=np.float64)
    if hi <= lo:
        h = np.ones(bins, dtype=np.float64)
        return h / h.sum()
    counts, _ = np.histogram(arr, bins=bins, range=(lo, hi))
    total = counts.sum()
    if total == 0:
        return np.zeros(bins, dtype=np.float64)
    return counts.astype(np.float64) / total


def stat_ranges(stat_lists: List[List[Dict[str, float]]]) -> Dict[str, tuple]:
    """Pooled (lo, hi) per statistic across ALL given sources (spec §7).

    Pass ``[raw_stats, edited_stats, bc_stats, gan_stats]`` so every source is
    histogrammed on ONE shared grid and the per-source JS values are mutually
    comparable.
    """
    ranges: Dict[str, tuple] = {}
    for key in _STAT_KEYS:
        vals = [float(s[key]) for stats in stat_lists for s in stats]
        ranges[key] = (min(vals), max(vals)) if vals else (0.0, 0.0)
    return ranges


def distributional_fidelity(
    source_stats: List[Dict[str, float]],
    raw_stats: List[Dict[str, float]],
    *,
    bins: int = BINS,
    ranges: Dict[str, tuple] | None = None,
) -> Dict[str, object]:
    """Per-statistic JS divergence (bits, lower=better) of source vs raw.

    For each of {length, mean_displacement, coverage}, histogram the source and
    raw values on a shared bin grid, then take the Jensen-Shannon divergence.
    aggregate = mean of the three. ``ranges`` supplies the shared (lo, hi) per
    statistic — the orchestrator computes it once via ``stat_ranges`` over ALL
    sources (spec §7) so per-source numbers are comparable. If None, falls back
    to the per-call pooled src+raw range (used by the unit tests).
    """
    per_stat: Dict[str, float] = {}
    for key in _STAT_KEYS:
        src = [float(s[key]) for s in source_stats]
        raw = [float(s[key]) for s in raw_stats]
        if ranges is not None:
            lo, hi = ranges[key]
        else:
            pooled = src + raw
            lo, hi = (min(pooled), max(pooled)) if pooled else (0.0, 0.0)
        p = _hist(src, lo, hi, bins)
        q = _hist(raw, lo, hi, bins)
        per_stat[key] = float(jensen_shannon_divergence(p, q))
    aggregate = float(np.mean([per_stat[k] for k in _STAT_KEYS]))
    return {"per_stat": per_stat, "aggregate": aggregate}
