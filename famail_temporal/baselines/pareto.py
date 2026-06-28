"""Assemble the data-level fairness x retention Pareto (Phase 1)."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np

from famail_temporal.data.loader import DataBundle
from famail_temporal.baselines.datasets import (
    rank_unfair_trajectory_indices, build_filtered_pickup_3d,
)
from famail_temporal.baselines.metrics import data_level_fairness


@dataclass(frozen=True)
class ParetoPoint:
    label: str
    retention: float
    f_spatial: float
    f_causal: float
    gini_dsr: float
    gini_asr: float
    n_removed: int


def _point(
    label: str, bundle: DataBundle, pickup_3d: Optional[np.ndarray],
    retention: float, n_removed: int,
) -> ParetoPoint:
    m = data_level_fairness(bundle, pickup_3d=pickup_3d)
    return ParetoPoint(
        label=label, retention=retention,
        f_spatial=m["f_spatial"], f_causal=m["f_causal"],
        gini_dsr=m["gini_dsr"], gini_asr=m["gini_asr"],
        n_removed=n_removed,
    )


def raw_point(bundle: DataBundle) -> ParetoPoint:
    """No intervention: full retention, bundle's own demand grid."""
    return _point("raw", bundle, None, 1.0, 0)


def filtered_points(
    bundle: DataBundle, k_levels: List[int],
) -> List[ParetoPoint]:
    """Generate-then-filter sweep: remove the top-K most-unfair trajectories.

    Ranking is computed once on the raw grid (static generate-then-filter).
    Each k is capped at the number of strictly-unfair candidates.
    """
    n = len(bundle.trajectories)
    if n == 0:
        raise ValueError("bundle has no trajectories to filter")
    ranked = rank_unfair_trajectory_indices(bundle)
    pts: List[ParetoPoint] = []
    for k in k_levels:
        k_eff = min(k, len(ranked))
        removed = [bundle.trajectories[i] for i in ranked[:k_eff]]
        pickup_3d = build_filtered_pickup_3d(bundle, removed)
        retention = (n - k_eff) / n
        pts.append(_point(f"filter@{k}", bundle, pickup_3d, retention, k_eff))
    return pts


def edited_point(
    f_spatial: float, f_causal: float, gini_dsr: float, gini_asr: float,
) -> ParetoPoint:
    """The FAMAIL editing point (full retention) from run_experiment's
    post-edit metrics. Caller passes ExperimentResult.*_after fields."""
    return ParetoPoint(
        label="edit", retention=1.0,
        f_spatial=f_spatial, f_causal=f_causal,
        gini_dsr=gini_dsr, gini_asr=gini_asr, n_removed=0,
    )


def points_to_json(points: List[ParetoPoint]) -> str:
    return json.dumps([asdict(pt) for pt in points], indent=2)


def points_to_csv_rows(points: List[ParetoPoint]) -> List[dict]:
    """Return a list of flat dicts (one per point) suitable for csv.DictWriter (E17)."""
    return [asdict(p) for p in points]


def filtered_points_with_removed_ids(
    bundle: "DataBundle", k_levels: List[int],
) -> tuple:
    """Like filtered_points but also returns a parallel dict mapping label -> removed traj ids.

    Returns (List[ParetoPoint], dict[str, list[int]]) where the dict keys are
    "filter@{k}" and values are the trajectory ids of the ranked[:k_eff] removed
    trajectories.  ParetoPoint is kept frozen/unchanged (ids carried out-of-band).
    """
    n = len(bundle.trajectories)
    if n == 0:
        raise ValueError("bundle has no trajectories to filter")
    ranked = rank_unfair_trajectory_indices(bundle)
    pts: List[ParetoPoint] = []
    removed_ids: dict = {}
    for k in k_levels:
        k_eff = min(k, len(ranked))
        removed = [bundle.trajectories[i] for i in ranked[:k_eff]]
        pickup_3d = build_filtered_pickup_3d(bundle, removed)
        retention = (n - k_eff) / n
        label = f"filter@{k}"
        pts.append(_point(label, bundle, pickup_3d, retention, k_eff))
        # Collect trajectory ids (int-safe) for the driver to persist
        removed_ids[label] = [int(bundle.trajectories[i].trajectory_id)
                               for i in ranked[:k_eff]]
    return pts, removed_ids
