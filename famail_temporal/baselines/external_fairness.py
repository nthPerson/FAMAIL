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


def theil_index(Y: np.ndarray, regions: np.ndarray) -> float:
    """Between-region Theil-T index of Y. regions: (N,) int, -1 excluded.

    T_between = sum_g (N_g/N) * (ybar_g/ybar) * ln(ybar_g/ybar).
    Zero-service units contribute 0 (limit y*ln y -> 0). Scale-invariant.
    """
    valid = regions >= 0
    y = Y[valid].astype(np.float64)
    r = regions[valid]
    n = y.size
    if n == 0:
        return float("nan")
    ybar = y.mean()
    if not np.isfinite(ybar) or ybar <= 0.0:
        return float("nan")
    total = 0.0
    for g in np.unique(r):
        yg = y[r == g]
        ybar_g = yg.mean()
        if ybar_g > 0.0:
            total += (yg.size / n) * (ybar_g / ybar) * np.log(ybar_g / ybar)
    return float(total)


def median_split(values: np.ndarray, disadvantaged_high: bool) -> np.ndarray:
    groups = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    if not finite.any():
        return groups
    med = np.median(values[finite])
    high = finite & (values > med)
    low = finite & (values <= med)
    if disadvantaged_high:
        groups[high] = 1
        groups[low] = 0
    else:
        groups[low] = 1
        groups[high] = 0
    return groups


def region_extremes(
    values: np.ndarray, disadvantaged_high: bool, frac: float = 1.0 / 3.0,
) -> np.ndarray:
    """Rank distinct region values; bottom/top `frac` of regions -> D/A."""
    groups = np.full(values.shape, -1, dtype=np.int64)
    finite = np.isfinite(values)
    uniq = np.unique(values[finite])
    n_reg = uniq.size
    if n_reg < 2:
        return groups
    k = max(1, int(round(frac * n_reg)))
    if 2 * k > n_reg:
        k = n_reg // 2
    low_vals = uniq[:k]
    high_vals = uniq[-k:]
    is_low = finite & np.isin(values, low_vals)
    is_high = finite & np.isin(values, high_vals)
    if disadvantaged_high:
        groups[is_high] = 1
        groups[is_low] = 0
    else:
        groups[is_low] = 1
        groups[is_high] = 0
    return groups


def regions_from_values(value_columns: Sequence[np.ndarray]) -> np.ndarray:
    """Map each unit's demographic profile (row across axes) to an int region
    id. Any NaN across the columns -> -1 (excluded)."""
    stacked = np.stack([np.asarray(c, dtype=np.float64) for c in value_columns],
                       axis=1)
    regions = np.full(stacked.shape[0], -1, dtype=np.int64)
    finite = np.all(np.isfinite(stacked), axis=1)
    if not finite.any():
        return regions
    _, inv = np.unique(stacked[finite], axis=0, return_inverse=True)
    regions[finite] = inv
    return regions
