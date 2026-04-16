"""Tests for data.active_mask."""
import numpy as np
import pytest

from famail_temporal.data.active_mask import UnitIndexMap


def _make_small_mask():
    mask = np.zeros((3, 2, 2), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[1, 0, 0] = True
    mask[2, 1, 1] = True
    return mask


def test_canonical_ordering():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.n_units == 4
    np.testing.assert_array_equal(umap.cell_indices, [0, 0, 2, 5])
    np.testing.assert_array_equal(umap.time_block_indices, [0, 1, 0, 1])


def test_from_cell_time_roundtrip():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.from_cell_time(0, 0) == 0
    assert umap.from_cell_time(0, 1) == 1
    assert umap.from_cell_time(2, 0) == 2
    assert umap.from_cell_time(5, 1) == 3
    assert umap.from_cell_time(1, 0) == -1  # inactive


def test_to_cell_time():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.to_cell_time(0) == (0, 0)
    assert umap.to_cell_time(3) == (5, 1)


def test_units_per_block():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    np.testing.assert_array_equal(umap.units_per_block, [2, 2])
