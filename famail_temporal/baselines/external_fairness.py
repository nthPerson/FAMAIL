"""Pure external fairness metrics over per-active-unit service ratios Y.

N-vector in, scalar/dict out; grid- and bundle-unaware (mirrors fairness/).
Group labels: 0 = advantaged (A), 1 = disadvantaged (D), -1 = excluded.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


def _group_means(Y: np.ndarray, groups: np.ndarray) -> Tuple[float, float]:
    d = Y[groups == 1]
    a = Y[groups == 0]
    mean_d = float(d.mean()) if d.size else float("nan")
    mean_a = float(a.mean()) if a.size else float("nan")
    return mean_d, mean_a


def supply_demand_ratio(Y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    mean_d, mean_a = _group_means(Y, groups)
    return {
        "mean_disadvantaged": mean_d,
        "mean_advantaged": mean_a,
        "gap": mean_a - mean_d,
    }


def demographic_parity(Y: np.ndarray, groups: np.ndarray) -> float:
    mean_d, mean_a = _group_means(Y, groups)
    return mean_a - mean_d               # signed gap; 0 = parity


def disparate_impact(Y: np.ndarray, groups: np.ndarray) -> float:
    mean_d, mean_a = _group_means(Y, groups)
    if not np.isfinite(mean_a) or mean_a == 0.0:
        return float("nan")
    return mean_d / mean_a               # 1 = parity; < 0.8 = adverse


def sdr_gap(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["gap"]


def sdr_mean_disadvantaged(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["mean_disadvantaged"]


def sdr_mean_advantaged(Y: np.ndarray, groups: np.ndarray) -> float:
    return supply_demand_ratio(Y, groups)["mean_advantaged"]
