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


from famail_temporal.data.active_mask import compute_active_mask


def test_active_mask_supply_threshold():
    active_3d = np.zeros((48, 90, 4), dtype=np.float32)
    active_3d[5, 10, 0] = 1.0
    active_3d[6, 11, 0] = 0.3
    valid_mask = np.ones((48, 90), dtype=bool)
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    mask = compute_active_mask(active_3d, valid_mask, demographics)
    assert mask.shape == (48, 90, 4)
    assert mask[5, 10, 0]
    assert not mask[6, 11, 0]
    assert mask.dtype == np.bool_


def test_active_mask_rejects_nan_demographics():
    active_3d = np.ones((48, 90, 4), dtype=np.float32) * 10.0
    valid_mask = np.ones((48, 90), dtype=bool)
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    demographics[5, 10, 0] = np.nan
    mask = compute_active_mask(active_3d, valid_mask, demographics)
    assert not mask[5, 10, 0]
    assert not mask[5, 10, 3]
    # Positive control: adjacent cell with finite demographics stays active
    assert mask[5, 11, 0]


def test_active_mask_rejects_invalid_cell():
    """A cell with active_taxis > threshold but valid_mask=False must be inactive."""
    active_3d = np.ones((48, 90, 4), dtype=np.float32) * 10.0
    valid_mask = np.ones((48, 90), dtype=bool)
    valid_mask[5, 10] = False  # mark cell (5, 10) as outside Shenzhen
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    mask = compute_active_mask(active_3d, valid_mask, demographics)
    # Cell is invalid → inactive across all blocks
    assert not mask[5, 10, 0]
    assert not mask[5, 10, 3]
    # Adjacent cell still active (positive control)
    assert mask[5, 11, 0]


def test_active_mask_rejects_mismatched_shapes():
    """Shape mismatches between inputs should raise ValueError."""
    valid_mask = np.ones((48, 90), dtype=bool)
    demographics = np.zeros((48, 90, 3), dtype=np.float32)
    # active_taxis_3d grid_x mismatch (47 vs 48)
    with pytest.raises(ValueError):
        compute_active_mask(
            np.zeros((47, 90, 4), dtype=np.float32),
            valid_mask,
            demographics,
        )
    # demographics grid_y mismatch (89 vs 90)
    with pytest.raises(ValueError):
        compute_active_mask(
            np.zeros((48, 90, 4), dtype=np.float32),
            valid_mask,
            np.zeros((48, 89, 3), dtype=np.float32),
        )
