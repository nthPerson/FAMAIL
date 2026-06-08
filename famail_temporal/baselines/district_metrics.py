"""District-level Disparate Impact (DI) ratio under both Y conventions.

For each district d, restrict to its active units (mask_3d=True), compute:
- Y_primary(d)       = mean(supply_N / max(demand_N, DEMAND_FLOOR))  (aligned w/ F_causal)
- Y_supplementary(d) = mean(demand_N / max(supply_N, supply_floor))  (demand pressure)

Then DI = mean(district_means in top-n_top hukou) / mean(district_means in bottom-n_bottom hukou).

Two-level averaging normalizes for both within-district size differences and
between-district population differences, and treats each district as one unit
of analysis (matches the spec's "top-3 vs bottom-3" framing). Districts that
have zero active units are dropped before grouping.

Data source notes (set during Task 4 implementation, see Step 1 inspection):
- `grid_to_district_mapping.pkl` is a dict with a pre-built
  `district_id_grid` (48, 90) int8 array (values 0..9, -1 for cells outside
  Shenzhen) and a `district_to_id` name->id map. We use `district_id_grid`
  directly.
- The plan referenced `all_demographics_by_district.csv`, which lives at the
  PROJECT-ROOT `source_data/`, NOT under `famail_temporal/source_data/`, and
  does not contain a literal `NonRegisteredRatio` column. The CSV would need
  the ratio computed from `NonRegisteredPermanentPop10k` /
  `YearEndPermanentPop10k`. We instead use `cell_demographics.pkl` (in-
  package, established by `famail_temporal/data/demographics.py:65-70`),
  whose `district_demographics` dict carries those same raw counts keyed by
  district name. That gives the same MigrantRatio used elsewhere in the
  codebase and keeps the loader self-contained inside
  `config.SOURCE_DATA_DIR`.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from famail_temporal import config
from famail_temporal.data.loader import DataBundle


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else float("nan")


def compute_di(
    *,
    demand_N: np.ndarray,           # (N,) demand per active unit
    supply_N: np.ndarray,           # (N,) supply per active unit
    district_of_unit: np.ndarray,   # (N,) int — district id per active unit
    hukou_ratios: np.ndarray,       # (n_districts,) NonRegisteredRatio per district
    n_top: int = 3,
    n_bottom: int = 3,
    demand_floor: float = config.DEMAND_FLOOR,
    supply_floor: float | None = None,
) -> dict:
    """Two-level DI ratio under both Y conventions.

    Returns:
        di_primary             — mean(Y_primary in top-hukou) / mean(... bottom)
        di_supplementary       — same for Y_supplementary
        per_district_y_primary — (n_districts,) per-district mean Y_primary
        per_district_y_supplementary — (n_districts,) per-district mean Y_supplementary
        top_district_ids, bottom_district_ids — which districts entered each group
        n_active_per_district  — sanity check (which districts have zero coverage)
    """
    if supply_floor is None:
        supply_floor = demand_floor

    demand_N = np.asarray(demand_N, dtype=np.float64)
    supply_N = np.asarray(supply_N, dtype=np.float64)
    district_of_unit = np.asarray(district_of_unit, dtype=np.int64)
    n_districts = len(hukou_ratios)

    y_primary_per_unit = supply_N / np.maximum(demand_N, demand_floor)
    y_supplementary_per_unit = demand_N / np.maximum(supply_N, supply_floor)

    per_district_y_primary = np.full(n_districts, np.nan, dtype=np.float64)
    per_district_y_supplementary = np.full(n_districts, np.nan, dtype=np.float64)
    n_active_per_district = np.zeros(n_districts, dtype=np.int64)
    for d in range(n_districts):
        mask = district_of_unit == d
        n_active_per_district[d] = int(mask.sum())
        if mask.any():
            per_district_y_primary[d] = _safe_mean(y_primary_per_unit[mask])
            per_district_y_supplementary[d] = _safe_mean(y_supplementary_per_unit[mask])

    has_coverage = ~np.isnan(per_district_y_primary)
    covered_ids = np.where(has_coverage)[0]
    if len(covered_ids) < n_top + n_bottom:
        raise ValueError(
            f"Need at least n_top + n_bottom = {n_top + n_bottom} covered "
            f"districts; have {len(covered_ids)}."
        )
    order = covered_ids[np.argsort(hukou_ratios[covered_ids])]
    bottom_ids = order[:n_bottom]
    top_ids = order[-n_top:]

    def _group_mean(district_y: np.ndarray, group: np.ndarray) -> float:
        return _safe_mean(district_y[group])

    di_primary = (
        _group_mean(per_district_y_primary, top_ids)
        / _group_mean(per_district_y_primary, bottom_ids)
    )
    di_supplementary = (
        _group_mean(per_district_y_supplementary, top_ids)
        / _group_mean(per_district_y_supplementary, bottom_ids)
    )

    return {
        "di_primary": float(di_primary),
        "di_supplementary": float(di_supplementary),
        "per_district_y_primary": per_district_y_primary,
        "per_district_y_supplementary": per_district_y_supplementary,
        "top_district_ids": top_ids.tolist(),
        "bottom_district_ids": bottom_ids.tolist(),
        "n_active_per_district": n_active_per_district.tolist(),
    }


def _load_grid_to_district() -> Tuple[np.ndarray, list[str], dict[str, int]]:
    """Return (district_id_grid (GX, GY) int64, district_names, name_to_id).

    The pkl is a dict with a pre-built `district_id_grid` (int8, -1 for cells
    outside Shenzhen) plus name<->id metadata; we just cast to int64 and pass
    the name list/map through so the hukou loader can align row order.
    """
    path = Path(config.SOURCE_DATA_DIR) / "grid_to_district_mapping.pkl"
    # Pickle source is trusted: in-package data file under
    # config.SOURCE_DATA_DIR (famail_temporal/source_data/), committed
    # alongside code; same convention used by preprocess.py and data/loader.py.
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    if not isinstance(mapping, dict) or "district_id_grid" not in mapping:
        raise ValueError(
            "Unexpected grid_to_district_mapping.pkl format: expected a dict "
            "with key 'district_id_grid'; got "
            f"{type(mapping).__name__} keys={list(mapping.keys()) if isinstance(mapping, dict) else 'n/a'}"
        )
    grid = np.asarray(mapping["district_id_grid"], dtype=np.int64)
    GX, GY = config.GRID_DIMS
    if grid.shape != (GX, GY):
        raise ValueError(
            f"district_id_grid shape {grid.shape} != config.GRID_DIMS {(GX, GY)}"
        )
    district_names = list(mapping["district_names"])
    name_to_id = dict(mapping["district_to_id"])
    return grid, district_names, name_to_id


def _load_hukou() -> Tuple[np.ndarray, list[str]]:
    """Return (hukou_ratios (n_districts,), district_names).

    Source: `cell_demographics.pkl` (in-package, lives in config.SOURCE_DATA_DIR).
    The plan referenced `all_demographics_by_district.csv`, which is at the
    PROJECT-ROOT `source_data/` and does not contain a `NonRegisteredRatio`
    column; the in-package pkl carries the same per-district raw counts and
    matches the MigrantRatio convention in `famail_temporal/data/demographics.py`.

    Ratio = NonRegisteredPermanentPop10k / YearEndPermanentPop10k, aligned to
    the same district name->id ordering as the grid mapping (so id i in the
    returned vector matches district id i on the grid).
    """
    path = Path(config.SOURCE_DATA_DIR) / "cell_demographics.pkl"
    # Pickle source is trusted: same in-package data file already loaded by
    # preprocess.py and the existing test suite under config.SOURCE_DATA_DIR.
    with open(path, "rb") as f:
        demo = pickle.load(f)
    if "district_demographics" not in demo or "district_to_id" not in demo:
        raise KeyError(
            "Expected 'district_demographics' and 'district_to_id' in "
            f"cell_demographics.pkl; got keys {list(demo.keys())}."
        )
    district_demographics = demo["district_demographics"]
    district_to_id = demo["district_to_id"]
    # Guard against drift between the two pkls: the grid mapping's name->id
    # must match the hukou source's, else district_of_active_units would label
    # units with ids that point to the wrong row in `hukou_ratios`.
    _, _, grid_name_to_id = _load_grid_to_district()
    if dict(grid_name_to_id) != dict(district_to_id):
        raise ValueError(
            "district_to_id mismatch between grid_to_district_mapping.pkl and "
            "cell_demographics.pkl; downstream DI would index the wrong "
            "hukou row per district."
        )
    n_districts = len(district_to_id)
    hukou = np.full(n_districts, np.nan, dtype=np.float64)
    names_by_id: list[str] = [""] * n_districts
    for name, did in district_to_id.items():
        if name not in district_demographics:
            raise KeyError(
                f"District {name!r} listed in district_to_id but missing from "
                "district_demographics"
            )
        row = district_demographics[name]
        non_reg = float(row["NonRegisteredPermanentPop10k"])
        total = float(row["YearEndPermanentPop10k"])
        if total <= 0:
            raise ValueError(
                f"District {name!r} has non-positive YearEndPermanentPop10k={total}"
            )
        hukou[did] = non_reg / total
        names_by_id[did] = name
    return hukou, names_by_id


def district_of_active_units(bundle: DataBundle) -> np.ndarray:
    """For each active unit (i in the flat active-unit vector), return its district id.

    The flat vector follows np.where(mask)'s ordering: lexicographic in
    (x, y, t), matching how F_causal / DI / JS pull active values via
    `array[mask]`. Cells outside Shenzhen (district id = -1 on the grid)
    should never appear in the active mask; if any do, the caller can detect
    them via the returned -1 ids and filter explicitly.
    """
    grid_to_dist, _, _ = _load_grid_to_district()
    mask = bundle.mask_3d
    xx, yy, _tt = np.where(mask)
    return grid_to_dist[xx, yy]


def di_from_bundle_and_pickup_grid(
    bundle: DataBundle,
    pickup_3d: np.ndarray,
    *,
    n_top: int = 3,
    n_bottom: int = 3,
) -> dict:
    """End-to-end DI from a bundle + a pickup demand grid (raw or generated)."""
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    district_of_unit = district_of_active_units(bundle)
    hukou_ratios, _ = _load_hukou()
    return compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou_ratios,
        n_top=n_top, n_bottom=n_bottom,
    )
