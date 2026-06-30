"""SF per-cell demographics via population-weighted areal interpolation (Task 3.3).

Interpolates ACS 2006-2010 tract estimates onto the faithful 0.01deg SF grid
using the TIGER 2010 tract polygons. For each grid cell, every overlapping tract
contributes weighted by the **estimated population captured** in the overlap
(`intersection_area x tract_pop / tract_area`) — a dasymetric-lite,
population-weighted areal interpolation (more accurate for per-capita / rate
features than pure area-weighting). Cells with ~zero residential population
(bay / SFO / commercial) are left NaN so the active-mask excludes them from
F_causal — mirroring how Shenzhen drops non-residential cells.

Output feature NAMES reuse the Shenzhen primary set so `config.DEMOGRAPHIC_FEATURES`
needs no change: AvgHousingPricePerSqM (median home value), CompPerCapita
(per-capita income), MigrantRatio (foreign-born share).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from famail_temporal.data.source_generation.sf_config import GridSpec

DEMO_FEATURE_NAMES: List[str] = [
    "AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio",
]
# Map of ACS source columns (from build_sf_demographics.py) -> FAMAIL feature names.
_ACS_TO_FEATURE = {
    "housing_median_value": "AvgHousingPricePerSqM",
    "income_percapita": "CompPerCapita",
    "migrant_share": "MigrantRatio",
}
AREA_CRS = "EPSG:3310"   # California Albers — equal-area, meters


def build_grid_cells(grid: GridSpec) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of the grid's square cells (1-indexed cell_x/cell_y)."""
    xs, ys, polys = [], [], []
    for x in range(1, grid.x_grid_max + 1):       # x = latitude axis
        for y in range(1, grid.y_grid_max + 1):   # y = longitude axis
            lat0 = grid.lat_min + (x - 1) * grid.cell_deg
            lon0 = grid.lon_min + (y - 1) * grid.cell_deg
            # shapely geometry uses (lon=x, lat=y)
            polys.append(box(lon0, lat0, lon0 + grid.cell_deg, lat0 + grid.cell_deg))
            xs.append(x)
            ys.append(y)
    return gpd.GeoDataFrame(
        {"cell_x": xs, "cell_y": ys, "geometry": polys}, crs="EPSG:4326",
    )


def areal_interpolate(
    cells: gpd.GeoDataFrame,
    tracts: gpd.GeoDataFrame,
    value_cols: List[str],
    pop_col: str,
    area_crs: str = AREA_CRS,
) -> pd.DataFrame:
    """Population-weighted areal interpolation of tract `value_cols` onto cells.

    Returns one row per input cell with the interpolated `value_cols` (NaN where a
    feature has no finite-valued overlapping tract) plus `pop_est` (estimated
    residential population captured by the cell; 0 if no overlap).
    """
    cells_p = cells.to_crs(area_crs)
    tracts_p = tracts.to_crs(area_crs).copy()
    tracts_p["_tract_area"] = tracts_p.geometry.area

    inter = gpd.overlay(
        cells_p[["cell_x", "cell_y", "geometry"]], tracts_p,
        how="intersection", keep_geom_type=True,
    )
    inter["_iarea"] = inter.geometry.area
    inter["_w"] = inter["_iarea"] * inter[pop_col] / inter["_tract_area"].replace(0, np.nan)

    rows = []
    for (cx, cy), g in inter.groupby(["cell_x", "cell_y"]):
        rec = {"cell_x": int(cx), "cell_y": int(cy), "pop_est": float(np.nansum(g["_w"]))}
        for c in value_cols:
            m = np.isfinite(g[c].to_numpy()) & np.isfinite(g["_w"].to_numpy())
            wsum = float(g["_w"].to_numpy()[m].sum())
            rec[c] = (float((g[c].to_numpy()[m] * g["_w"].to_numpy()[m]).sum()) / wsum
                      if wsum > 0 else np.nan)
        rows.append(rec)

    have = pd.DataFrame(rows, columns=["cell_x", "cell_y", "pop_est", *value_cols])
    allc = cells[["cell_x", "cell_y"]].astype(int).copy()
    out = allc.merge(have, on=["cell_x", "cell_y"], how="left")
    out["pop_est"] = out["pop_est"].fillna(0.0)
    return out


def load_tracts(acs_csv: str, tiger_zip: str) -> gpd.GeoDataFrame:
    """Join ACS tract estimates (CSV) to TIGER 2010 tract polygons by GEOID."""
    acs = pd.read_csv(acs_csv, dtype={"GEOID": str})
    poly = gpd.read_file(tiger_zip)
    geoid_col = "GEOID10" if "GEOID10" in poly.columns else "GEOID"
    poly = poly.rename(columns={geoid_col: "GEOID"})
    poly["GEOID"] = poly["GEOID"].astype(str)
    merged = poly.merge(acs, on="GEOID", how="inner")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=poly.crs)


def build_cell_demographics(
    grid: GridSpec, acs_csv: str, tiger_zip: str,
) -> Tuple[np.ndarray, List[str]]:
    """Build the `(x_grid_max, y_grid_max, 3)` demographics grid + feature names.

    NaN where a cell has no residential population (excluded from F_causal).
    """
    tracts = load_tracts(acs_csv, tiger_zip).rename(columns=_ACS_TO_FEATURE)
    cells = build_grid_cells(grid)
    out = areal_interpolate(
        cells, tracts, value_cols=DEMO_FEATURE_NAMES, pop_col="pop",
    )
    arr = np.full(
        (grid.x_grid_max, grid.y_grid_max, len(DEMO_FEATURE_NAMES)),
        np.nan, dtype=np.float32,
    )
    for _, r in out.iterrows():
        if r["pop_est"] <= 0:          # non-residential cell -> leave NaN
            continue
        x, y = int(r["cell_x"]) - 1, int(r["cell_y"]) - 1
        for j, c in enumerate(DEMO_FEATURE_NAMES):
            arr[x, y, j] = r[c]
    return arr, list(DEMO_FEATURE_NAMES)
