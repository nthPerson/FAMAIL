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

    out_cols = [
        "plate_id", "lat_r", "lon_r", "x_grid", "y_grid",
        "n_pickups", "cell_total", "cell_share", "driver_total",
    ]
    if pk.empty:
        # No pickups -> nothing to group. Return empty, well-typed frames
        # rather than relying on empty-groupby behaviour (which can drop the
        # expected columns or error on the downstream merges).
        empty = pd.DataFrame(columns=out_cols)
        return empty.copy(), empty.copy()

    pk["lat_r"] = pk["latitude"].round(coord_precision)
    pk["lon_r"] = pk["longitude"].round(coord_precision)

    grp = pk.groupby(["plate_id", "lat_r", "lon_r"], sort=False)
    sizes = grp.size().rename("n_pickups").reset_index()
    # A frozen coordinate that rounds to lat_r/lon_r can still straddle a grid-
    # cell boundary and map to >1 (x_grid, y_grid). Assign each group its MODAL
    # cell (the most frequent), tie-broken deterministically by cell coords, so
    # the flagged cell is reproducible run-to-run instead of an arbitrary pick.
    cell_counts = (
        pk.groupby(["plate_id", "lat_r", "lon_r", "x_grid", "y_grid"])
        .size()
        .rename("cell_count")
        .reset_index()
        .sort_values(
            ["plate_id", "lat_r", "lon_r", "cell_count", "x_grid", "y_grid"],
            ascending=[True, True, True, False, True, True],
        )
    )
    cells = cell_counts.drop_duplicates(
        subset=["plate_id", "lat_r", "lon_r"], keep="first",
    )[["plate_id", "lat_r", "lon_r", "x_grid", "y_grid"]]
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
