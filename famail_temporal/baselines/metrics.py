"""Data-level fairness metrics for a demand grid (Phase 1).

Reuses the canonical evaluation grid + scalar reduction so values match the
editing pipeline exactly (fairness convention: 1 = fairest).
"""
from __future__ import annotations

import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.runner import _scalar_metrics_from_grid


def data_level_fairness(
    bundle: DataBundle, pickup_3d: np.ndarray | None = None,
) -> dict:
    """Return {f_spatial, f_causal, gini_dsr, gini_asr} for a demand grid.

    pickup_3d=None evaluates the bundle's own demand grid (the raw variant).
    Pass a filtered/edited demand grid to evaluate a variant.
    """
    grid = build_fairness_grid(bundle, pickup_3d=pickup_3d)
    return _scalar_metrics_from_grid(grid)
