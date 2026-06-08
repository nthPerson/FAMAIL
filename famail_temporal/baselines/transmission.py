"""Terminal-cell transmission check.

The fairness metric (F_causal, F_spatial) depends only on each rollout's
terminal pickup cell, so the model-level B0-vs-FAMAIL test reduces to: does
the LSTM reproduce a ~1% shift in the marginal distribution of one token?
This module measures that *before* the headline number is trusted.

Reported metrics (all JS in bits, bounded in [0, 1]):
- js_target           = JS(p_raw, p_edited)         - the signal we WANT to transmit
- js_generated        = JS(p_gen_B0, p_gen_FAMAIL)   - the signal that DID transmit
- transmission_ratio  = js_generated / js_target     - ~1 = faithful, <<1 = washed out
- js_b0_vs_raw, js_famail_vs_edited                  - per-variant fidelity to its own target
"""
from __future__ import annotations
from typing import Iterable, Tuple

import numpy as np

from famail_temporal.baselines.gan import config as gc


def terminal_cell_histogram(
    pickups: Iterable[Tuple[int, int, int]],
    n_cells: int = gc.N_CELLS,
) -> np.ndarray:
    """Build a normalized histogram over flat cell ids from pickup tuples.

    Each pickup is (x, y, t_block); only (x, y) is used (the metric is
    length/time-block-blind by design). Returns a length-`n_cells` array that
    sums to 1 (or all zeros if the input is empty). Out-of-vocab cells are
    dropped (no-op in production; matters for the small synthetic bundle).
    """
    h = np.zeros(n_cells, dtype=np.float64)
    for (x, y, _) in pickups:
        flat = int(x) * gc.GY + int(y)
        if 0 <= flat < n_cells:
            h[flat] += 1.0
    total = h.sum()
    return h / total if total > 0 else h


def trajectories_terminal_histogram(
    trajectories: Iterable, n_cells: int = gc.N_CELLS,
) -> np.ndarray:
    """Same as terminal_cell_histogram but reads from Trajectory.states[-1]."""
    h = np.zeros(n_cells, dtype=np.float64)
    for traj in trajectories:
        s = traj.states[-1]
        flat = int(s.x_grid) * gc.GY + int(s.y_grid)
        if 0 <= flat < n_cells:
            h[flat] += 1.0
    total = h.sum()
    return h / total if total > 0 else h


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """JS divergence in bits (log base 2). Symmetric, in [0, 1].

    Implemented as 0.5 KL(p || m) + 0.5 KL(q || m), m = 0.5 (p + q),
    with an eps-clip to avoid log(0). 0 if p == q; 1 if disjoint support.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)

    def _kl_bits(a: np.ndarray, b: np.ndarray) -> float:
        # Only nonzero rows of a contribute to KL(a||b).
        mask = a > 0
        return float(np.sum(
            a[mask] * (np.log2(a[mask] + eps) - np.log2(b[mask] + eps))
        ))

    return 0.5 * _kl_bits(p, m) + 0.5 * _kl_bits(q, m)


def transmission_metrics(
    p_raw: np.ndarray,
    p_edited: np.ndarray,
    p_gen_b0: np.ndarray,
    p_gen_famail: np.ndarray,
) -> dict:
    """Compute the full transmission bundle from four terminal-cell histograms."""
    js_target = jensen_shannon_divergence(p_raw, p_edited)
    js_generated = jensen_shannon_divergence(p_gen_b0, p_gen_famail)
    transmission_ratio = (
        js_generated / js_target if js_target > 0 else float("nan")
    )
    return {
        "js_target": float(js_target),
        "js_generated": float(js_generated),
        "transmission_ratio": float(transmission_ratio),
        "js_b0_vs_raw": float(jensen_shannon_divergence(p_gen_b0, p_raw)),
        "js_famail_vs_edited": float(jensen_shannon_divergence(p_gen_famail, p_edited)),
    }
