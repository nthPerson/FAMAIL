"""Tests for SF demographic areal interpolation (Task 3.3)."""
import numpy as np
import geopandas as gpd
from shapely.geometry import box

from famail_temporal.data.source_generation.sf_demographics import (
    areal_interpolate, build_grid_cells,
)
from famail_temporal.data.source_generation.sf_config import GridSpec


def test_areal_interpolate_is_population_weighted():
    # Two equal-area tracts; A is 3x denser than B. A cell covering both fully
    # must return the POPULATION-weighted mean (12.5), not the area mean (15.0).
    tracts = gpd.GeoDataFrame(
        {"val": [10.0, 20.0], "pop": [300.0, 100.0],
         "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        crs="EPSG:3310",
    )
    cells = gpd.GeoDataFrame(
        {"cell_x": [1], "cell_y": [1], "geometry": [box(0, 0, 2, 1)]},
        crs="EPSG:3310",
    )
    out = areal_interpolate(cells, tracts, value_cols=["val"], pop_col="pop",
                            area_crs="EPSG:3310")
    row = out.set_index(["cell_x", "cell_y"]).loc[(1, 1)]
    assert abs(row["val"] - 12.5) < 1e-6        # pop-weighted, not 15.0
    assert abs(row["pop_est"] - 400.0) < 1e-6   # estimated population in cell


def test_areal_interpolate_no_overlap_is_nan():
    tracts = gpd.GeoDataFrame(
        {"val": [10.0], "pop": [100.0], "geometry": [box(0, 0, 1, 1)]},
        crs="EPSG:3310",
    )
    cells = gpd.GeoDataFrame(
        {"cell_x": [5], "cell_y": [5], "geometry": [box(10, 10, 11, 11)]},
        crs="EPSG:3310",
    )
    out = areal_interpolate(cells, tracts, value_cols=["val"], pop_col="pop",
                            area_crs="EPSG:3310")
    row = out.set_index(["cell_x", "cell_y"]).loc[(5, 5)]
    assert np.isnan(row["val"])
    assert row["pop_est"] == 0.0


def test_build_grid_cells_covers_full_grid_1indexed():
    grid = GridSpec(lat_min=37.7, lon_min=-122.5, x_grid_max=3, y_grid_max=4, cell_deg=0.01)
    cells = build_grid_cells(grid)
    assert len(cells) == 3 * 4
    assert cells["cell_x"].min() == 1 and cells["cell_x"].max() == 3
    assert cells["cell_y"].min() == 1 and cells["cell_y"].max() == 4
    assert cells.crs is not None


def test_majority_overlap_assigns_dominant_tract():
    from famail_temporal.data.source_generation.sf_demographics import majority_overlap
    tracts = gpd.GeoDataFrame(
        {"val": [10.0, 20.0], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
        crs="EPSG:3310",
    )
    # cell (1,1): 1.0 overlap with A vs 0.4 with B -> A(10); cell (2,1): 0.3 vs 1.0 -> B(20)
    cells = gpd.GeoDataFrame(
        {"cell_x": [1, 2], "cell_y": [1, 1],
         "geometry": [box(0, 0, 1.4, 1), box(0.7, 0, 2, 1)]},
        crs="EPSG:3310",
    )
    out = majority_overlap(cells, tracts, ["val"]).set_index(["cell_x", "cell_y"])
    assert out.loc[(1, 1), "val"] == 10.0
    assert out.loc[(2, 1), "val"] == 20.0


def test_majority_overlap_no_overlap_is_nan():
    from famail_temporal.data.source_generation.sf_demographics import majority_overlap
    tracts = gpd.GeoDataFrame(
        {"val": [10.0], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:3310")
    cells = gpd.GeoDataFrame(
        {"cell_x": [9], "cell_y": [9], "geometry": [box(10, 10, 11, 11)]}, crs="EPSG:3310")
    out = majority_overlap(cells, tracts, ["val"]).set_index(["cell_x", "cell_y"])
    assert np.isnan(out.loc[(9, 9), "val"])
