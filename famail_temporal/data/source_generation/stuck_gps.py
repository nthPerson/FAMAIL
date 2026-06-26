"""Detect & remove per-driver stuck-GPS pickup 'sinks' (frozen meter-on
coordinates emitting thousands of phantom pickups with ~zero dropoffs at that
exact coordinate).

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


def dropoff_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of dropoff events (passenger_indicator 1->0 within driver)."""
    diff = df.groupby("plate_id")["passenger_indicator"].diff()
    return diff == -1


def detect_stuck_gps_sinks(
    df: pd.DataFrame, *, min_pickups: int, max_dropoff_ratio: float, coord_precision: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag (driver, exact-coord) pickup groups that are large AND have ~zero
    dropoffs at that exact coordinate (the frozen meter-on signature).

    Returns (flagged, distribution) where:
    - flagged: groups meeting (n_pickups >= min_pickups) AND
               (n_dropoffs / n_pickups < max_dropoff_ratio).
    - distribution: all pickup groups sorted by n_pickups desc (for calibration).

    Both frames include cell_share/cell_total/driver_total (informative for the
    audit) as well as n_dropoffs and dropoff_ratio.
    """
    pk = df[pickup_mask(df)].copy()

    out_cols = [
        "plate_id", "lat_r", "lon_r", "x_grid", "y_grid",
        "n_pickups", "cell_total", "cell_share", "driver_total",
        "n_dropoffs", "dropoff_ratio",
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

    # Count dropoffs per (plate_id, rounded coord) and left-merge onto pickup groups.
    # A frozen-meter-on artifact has ~zero dropoffs at the frozen coordinate:
    # when the meter turns off, GPS geocodes normally elsewhere.
    do = df[dropoff_mask(df)].copy()
    if not do.empty:
        do["lat_r"] = do["latitude"].round(coord_precision)
        do["lon_r"] = do["longitude"].round(coord_precision)
        dropoff_counts = (
            do.groupby(["plate_id", "lat_r", "lon_r"])
            .size()
            .rename("n_dropoffs")
            .reset_index()
        )
        sizes = sizes.merge(dropoff_counts, on=["plate_id", "lat_r", "lon_r"], how="left")
    else:
        sizes["n_dropoffs"] = 0
    sizes["n_dropoffs"] = sizes["n_dropoffs"].fillna(0).astype(int)
    sizes["dropoff_ratio"] = sizes["n_dropoffs"] / sizes["n_pickups"]

    distribution = sizes.sort_values("n_pickups", ascending=False).reset_index(drop=True)
    flagged = distribution[
        (distribution["n_pickups"] >= min_pickups)
        & (distribution["dropoff_ratio"] < max_dropoff_ratio)
    ].reset_index(drop=True)
    return flagged, distribution


def filter_stuck_gps_sinks(
    df: pd.DataFrame, *, min_pickups: int, max_dropoff_ratio: float,
    coord_precision: int, expected_cells: set | None, drop: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Filter out flagged stuck-GPS pickup sinks. Returns (cleaned_df, audit).

    Args:
        df: Input DataFrame with pickup events.
        min_pickups: Minimum pickups to flag a sink.
        max_dropoff_ratio: Maximum dropoff_ratio (n_dropoffs/n_pickups) to flag
            a sink. Real frozen-meter-on artifacts have ~zero dropoffs at the
            frozen coordinate; use a small value like 0.02.
        coord_precision: Decimal places for coordinate rounding.
        expected_cells: If provided, assert flagged cells match this set.
        drop: If True, remove flagged pickups; if False, return df unchanged.

    Returns:
        (cleaned_df, audit) where audit keys: sinks, flagged_cells, n_pickups_removed, n_rows_removed.
        Each sink dict includes n_dropoffs and dropoff_ratio documenting the frozen signature.
    """
    flagged, _dist = detect_stuck_gps_sinks(
        df, min_pickups=min_pickups, max_dropoff_ratio=max_dropoff_ratio,
        coord_precision=coord_precision,
    )
    flagged_cells = sorted({(int(r.x_grid), int(r.y_grid)) for r in flagged.itertuples()})
    if expected_cells is not None:
        assert set(flagged_cells) == set(expected_cells), (
            f"stuck-GPS filter flagged {set(flagged_cells)} != expected {set(expected_cells)}"
        )

    audit = {
        "sinks": [
            {"plate_id": r.plate_id, "lat": float(r.lat_r), "lon": float(r.lon_r),
             "x_grid": int(r.x_grid), "y_grid": int(r.y_grid),
             "n_pickups": int(r.n_pickups), "cell_share": float(r.cell_share),
             "driver_total": int(r.driver_total),
             "n_dropoffs": int(r.n_dropoffs), "dropoff_ratio": float(r.dropoff_ratio)}
            for r in flagged.itertuples()
        ],
        "flagged_cells": flagged_cells,
        "n_pickups_removed": int(flagged["n_pickups"].sum()) if len(flagged) else 0,
    }

    if not drop or len(flagged) == 0:
        audit["n_rows_removed"] = 0
        return df.reset_index(drop=True), audit

    pk = pickup_mask(df)
    lat_r = df["latitude"].round(coord_precision)
    lon_r = df["longitude"].round(coord_precision)
    key = list(zip(df["plate_id"], lat_r, lon_r))
    flagged_keys = {(r.plate_id, float(r.lat_r), float(r.lon_r)) for r in flagged.itertuples()}
    is_flagged_pickup = pk.values & pd.Series(
        [k in flagged_keys for k in key], index=df.index
    ).values
    cleaned = df[~is_flagged_pickup].reset_index(drop=True)
    audit["n_rows_removed"] = int(is_flagged_pickup.sum())
    return cleaned, audit


def threshold_sensitivity(
    df: pd.DataFrame, thresholds: list[int], *, max_dropoff_ratio: float, coord_precision: int,
) -> list[dict]:
    """Compute threshold-sensitivity curve for stuck-GPS sink detection.

    For each threshold, returns the number of unique cells flagged as sinks.
    Backs the "the sink set is stable across a wide threshold band" figure.

    Args:
        df: Input DataFrame with pickup events.
        thresholds: List of min_pickups thresholds to probe.
        max_dropoff_ratio: Maximum dropoff_ratio to flag a sink (applied at all
            thresholds). Must match the value passed to filter_stuck_gps_sinks so
            the swept distribution reflects what the filter would actually flag.
        coord_precision: Decimal places for coordinate rounding. Must match the
            value passed to filter_stuck_gps_sinks so the swept distribution
            reflects what the filter would actually flag.

    Returns:
        List of dicts [{"min_pickups": t, "n_flagged_cells": k}, ...].
    """
    _, dist = detect_stuck_gps_sinks(
        df, min_pickups=1, max_dropoff_ratio=max_dropoff_ratio, coord_precision=coord_precision,
    )
    out = []
    for t in thresholds:
        hit = dist[(dist["n_pickups"] >= t) & (dist["dropoff_ratio"] < max_dropoff_ratio)]
        n_cells = hit.drop_duplicates(["x_grid", "y_grid"]).shape[0]
        out.append({"min_pickups": int(t), "n_flagged_cells": int(n_cells)})
    return out
