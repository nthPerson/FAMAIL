import numpy as np
import pytest
from famail_temporal.visualization.gradient_heatmap import geometry as geom


def test_load_geometry_shapes_and_names():
    g = geom.load_district_geometry()
    assert g.district_id_grid.shape == (48, 90)
    assert g.valid_mask.shape == (48, 90)
    assert len(g.district_names) == 10
    assert "Nanshan" in g.district_names and "Dapeng" in g.district_names


def test_centroids_match_real_shenzhen_geography():
    g = geom.load_district_geometry()
    c = geom.district_centroids(g.district_id_grid, g.district_names)
    # (row=x_grid 0=S..47=N, col=y_grid 0=W..89=E)
    assert c["Nanshan"][0] < 20 and c["Nanshan"][1] < 25     # south-west
    assert c["Bao'an"][1] < 18                                # far west
    assert c["Dapeng"][1] > 65                                # far east
    assert c["Guangming"][0] > 30                             # north


def test_assert_canonical_orientation_passes_on_real_data():
    g = geom.load_district_geometry()
    geom.assert_canonical_orientation(g.district_id_grid, g.district_names)  # must not raise


def test_assert_canonical_orientation_fails_when_flipped():
    g = geom.load_district_geometry()
    flipped = np.flipud(g.district_id_grid)  # invert latitude -> Nanshan moves north
    with pytest.raises(AssertionError):
        geom.assert_canonical_orientation(flipped, g.district_names)


def test_boundary_segments_separate_differing_regions():
    grid = np.array([[0, 0, 1],
                     [0, 1, 1]], dtype=np.int8)
    xs, ys = geom.compute_boundary_segments(grid)
    # finite points come in (start, end) pairs separated by NaN
    assert np.isnan(xs[2]) and np.isnan(ys[2])
    assert np.isfinite(xs[:2]).all()
    # at least one vertical edge at x=1.5 (between col1 and col2 on row0) exists
    assert np.any(np.isclose(xs[np.isfinite(xs)], 1.5))
    # at least one horizontal edge at y=0.5 (between row0 and row1 at col1) exists
    assert np.any(np.isclose(ys[np.isfinite(ys)], 0.5))


def test_boundary_segments_empty_when_uniform():
    grid = np.zeros((3, 3), dtype=np.int8)
    xs, ys = geom.compute_boundary_segments(grid)
    assert xs.size == 0 and ys.size == 0
