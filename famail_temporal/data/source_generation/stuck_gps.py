"""Detect & remove per-driver stuck-GPS pickup 'sinks' (frozen meter-on
coordinates emitting thousands of phantom pickups in a single cell).

Runs on the enriched event-stream df AFTER quantization+sort but BEFORE
transitions are computed, so pickups are detected from the raw
passenger_indicator 0->1 transition per driver (is_pickup does not exist yet).
"""
from __future__ import annotations
import pandas as pd


def pickup_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of pickup events (passenger_indicator 0->1 within driver)."""
    diff = df.groupby("plate_id")["passenger_indicator"].diff()
    return diff == 1


def detect_stuck_gps_sinks(
    df: pd.DataFrame, *, min_pickups: int, coord_dominance: float, coord_precision: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag (driver, exact-coord) pickup groups that are large AND dominate
    their grid cell's pickups. Returns (flagged, distribution)."""
    pk = df[pickup_mask(df)].copy()
    pk["lat_r"] = pk["latitude"].round(coord_precision)
    pk["lon_r"] = pk["longitude"].round(coord_precision)

    grp = pk.groupby(["plate_id", "lat_r", "lon_r"], sort=False)
    sizes = grp.size().rename("n_pickups").reset_index()
    cells = grp[["x_grid", "y_grid"]].first().reset_index()
    sizes = sizes.merge(cells, on=["plate_id", "lat_r", "lon_r"])

    cell_total = pk.groupby(["x_grid", "y_grid"]).size().rename("cell_total").reset_index()
    driver_total = pk.groupby("plate_id").size().rename("driver_total").reset_index()
    sizes = sizes.merge(cell_total, on=["x_grid", "y_grid"]).merge(driver_total, on="plate_id")
    sizes["cell_share"] = sizes["n_pickups"] / sizes["cell_total"]

    distribution = sizes.sort_values("n_pickups", ascending=False).reset_index(drop=True)
    flagged = distribution[
        (distribution["n_pickups"] >= min_pickups)
        & (distribution["cell_share"] >= coord_dominance)
    ].reset_index(drop=True)
    return flagged, distribution
