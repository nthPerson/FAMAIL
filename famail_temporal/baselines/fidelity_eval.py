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
        # No values pooled -> degenerate (0,0) grid -> _hist returns a uniform
        # histogram -> JS divergence 0 (the only sensible answer with no data).
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

    Both sides must be non-empty: an empty ``source_stats`` against a populated
    grid would yield an all-zero histogram and a positive-but-meaningless JS
    (divergence from a distribution that has no samples). The orchestrator
    excludes empty rollouts upstream (``n_empty``), so this is a contract guard.
    """
    if not source_stats or not raw_stats:
        raise ValueError(
            "distributional_fidelity requires non-empty source_stats and "
            f"raw_stats (got len(source)={len(source_stats)}, "
            f"len(raw)={len(raw_stats)})"
        )
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


# ------------------------------------------------- HuMID paired fidelity ----

GATE_MARGIN = 0.2   # min (high_real_real - max low) for the gate to pass


def _pad_pairs_to_batch(
    pairs: List[Tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (left[L,4], right[L',4]) pairs into padded [B, Lmax, 4] tensors
    PLUS boolean masks [B, Lmax] (True = real step).

    The mask is REQUIRED. The discriminator's seeking encoder only ignores
    padding when a mask is supplied (it uses pack_padded_sequence); with
    mask=None it runs the LSTM over zero-padded rows as if they were real
    (0,0) steps, so a pair's score would depend on the longest trajectory in
    its batch (non-deterministic w.r.t. batching). Mirrors the mask convention
    in famail_temporal/fidelity/context.py (True = valid step). Verify shape/
    dtype against that module when implementing.
    """
    lefts = [p[0] for p in pairs]
    rights = [p[1] for p in pairs]
    lmax = max(t.shape[0] for t in lefts)
    rmax = max(t.shape[0] for t in rights)

    def _stack(tensors: List[torch.Tensor], m: int) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(len(tensors), m, 4, dtype=torch.float32)
        mask = torch.zeros(len(tensors), m, dtype=torch.bool)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            out[i, :n] = t
            mask[i, :n] = True
        return out, mask

    x1, m1 = _stack(lefts, lmax)
    x2, m2 = _stack(rights, rmax)
    return x1.to(device), x2.to(device), m1.to(device), m2.to(device)


def _score_pairs(
    discriminator: torch.nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Forward the discriminator over all pairs; return per-pair probs (N,)."""
    probs: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            x1, x2, m1, m2 = _pad_pairs_to_batch(chunk, device)
            out = discriminator(x1, x2, mask1=m1, mask2=m2)
            probs.extend(out.reshape(-1).detach().cpu().tolist())
    return np.asarray(probs, dtype=np.float64)


def humid_paired_fidelity(
    discriminator: torch.nn.Module,
    pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Mean same-agent probability over (left, right) trajectory-tensor pairs."""
    device = device or torch.device("cpu")
    if not pairs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    probs = _score_pairs(discriminator, pairs, batch_size=batch_size, device=device)
    return {
        "mean": float(probs.mean()),
        "std": float(probs.std(ddof=1)) if probs.size > 1 else 0.0,
        "n": int(probs.size),
    }


def validation_gate(
    discriminator: torch.nn.Module,
    *,
    real_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    collapsed_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    shuffled_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    batch_size: int = 64,
    device: torch.device | None = None,
    margin: float = GATE_MARGIN,
) -> Dict[str, object]:
    """Does the discriminator rank real-vs-real above real-vs-garbage?

    Passes iff high_real_real - max(low_collapsed, low_shuffled) >= margin AND
    high_real_real exceeds both lows. All three means are returned regardless.
    """
    device = device or torch.device("cpu")
    high = humid_paired_fidelity(discriminator, real_pairs, batch_size=batch_size, device=device)["mean"]
    low_c = humid_paired_fidelity(discriminator, collapsed_pairs, batch_size=batch_size, device=device)["mean"]
    low_s = humid_paired_fidelity(discriminator, shuffled_pairs, batch_size=batch_size, device=device)["mean"]
    worst_low = max(low_c, low_s)
    passed = bool((high - worst_low) >= margin and high > low_c and high > low_s)
    return {
        "high_real_real": float(high),
        "low_collapsed": float(low_c),
        "low_shuffled": float(low_s),
        "margin": float(margin),
        "passed": passed,
    }
