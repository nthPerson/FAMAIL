"""IO layer: bundle -> service ratio Y, per-unit equity-axis demographics, and
the edited (after) demand grid. City-agnostic (reads cell_demographics.pkl)."""
from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np

from famail_temporal import config
from famail_temporal.data.demographics import enrich_demographics
from famail_temporal.data.loader import DataBundle

EQUITY_AXES: List[str] = ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]
DISADVANTAGED_HIGH: Dict[str, bool] = {
    "AvgHousingPricePerSqM": False,
    "CompPerCapita": False,
    "MigrantRatio": True,
}


def _enriched_selected_grid() -> np.ndarray:
    """(GX, GY, 3) raw values for EQUITY_AXES from cell_demographics.pkl."""
    # Pickle source is trusted: same in-package data file already loaded by
    # preprocess.py, district_metrics.py, and the existing test suite under
    # config.SOURCE_DATA_DIR (not untrusted/external input).
    with open(config.SOURCE_DATA_DIR / "cell_demographics.pkl", "rb") as f:
        raw = pickle.load(f)
    enriched, names = enrich_demographics(
        raw["demographics_grid"], list(raw["feature_names"]),
    )
    idx = [names.index(a) for a in EQUITY_AXES]
    return enriched[..., idx].astype(np.float64)


def per_unit_demographics(
    bundle: DataBundle, selected_grid: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """{axis: (N,) raw value per active unit}. NaN where cell unmapped."""
    sel = _enriched_selected_grid() if selected_grid is None else selected_grid
    mask = bundle.mask_3d
    out: Dict[str, np.ndarray] = {}
    for j, axis in enumerate(EQUITY_AXES):
        per_cell_t = np.broadcast_to(sel[:, :, j][:, :, None], mask.shape)
        out[axis] = per_cell_t[mask].astype(np.float64)
    return out


def service_ratio_Y(pickup_3d: np.ndarray, bundle: DataBundle) -> np.ndarray:
    """Y = supply/demand over active units (F_causal convention)."""
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    return supply_N / np.maximum(demand_N, config.DEMAND_FLOOR)
