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


def test_empty_mask_no_active_units():
    mask = np.zeros((3, 2, 2), dtype=bool)
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    assert umap.n_units == 0
    assert umap.n_active_cells == 0
    assert umap.cell_indices.shape == (0,)
    assert umap.time_block_indices.shape == (0,)
    assert umap.units_per_block.tolist() == [0, 0]
    # from_cell_time on inactive cells returns -1
    assert umap.from_cell_time(0, 0) == -1


def test_arrays_are_read_only():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    with pytest.raises(ValueError):
        umap.cell_indices[0] = 99
    with pytest.raises(ValueError):
        umap.time_block_indices[0] = 99
    with pytest.raises(ValueError):
        umap.flat_lookup[0] = 99
    with pytest.raises(ValueError):
        umap.units_per_block[0] = 99


def test_from_cell_time_negative_inputs_return_minus_one():
    mask = _make_small_mask()
    umap = UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
    # Negative t should not alias to a valid slot
    assert umap.from_cell_time(1, -2) == -1
    assert umap.from_cell_time(-1, 0) == -1
    # t out of upper bound also returns -1
    assert umap.from_cell_time(0, 99) == -1


def test_non_3d_mask_raises():
    with pytest.raises(ValueError, match="must be 3D"):
        UnitIndexMap.from_mask(np.zeros((3, 2), dtype=bool), grid_shape=(3, 2))


def test_wrong_grid_shape_raises():
    with pytest.raises(ValueError, match="grid dims"):
        mask = np.zeros((4, 5, 2), dtype=bool)
        UnitIndexMap.from_mask(mask, grid_shape=(3, 2))
