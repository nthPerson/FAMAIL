"""Active-unit mask and canonical ordering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class UnitIndexMap:
    """Canonical ordering of active (cell, t) units.

    Ordering rule: cell-major, then time-block within cell.
    """
    cell_indices: np.ndarray
    time_block_indices: np.ndarray
    flat_lookup: np.ndarray
    n_units: int
    n_active_cells: int
    units_per_block: np.ndarray

    @classmethod
    def from_mask(cls, mask_3d: np.ndarray, grid_shape: Tuple[int, int]) -> "UnitIndexMap":
        gx, gy = grid_shape
        t = mask_3d.shape[2]
        assert mask_3d.shape == (gx, gy, t)

        cell_list, block_list = [], []
        for x in range(gx):
            for y in range(gy):
                flat_cell = x * gy + y
                for t_idx in range(t):
                    if mask_3d[x, y, t_idx]:
                        cell_list.append(flat_cell)
                        block_list.append(t_idx)

        cell_indices = np.asarray(cell_list, dtype=np.int32)
        time_block_indices = np.asarray(block_list, dtype=np.int8)
        n_units = len(cell_list)

        flat_lookup = np.full(gx * gy * t, -1, dtype=np.int32)
        for unit_idx, (c, b) in enumerate(zip(cell_list, block_list)):
            flat_lookup[c * t + b] = unit_idx

        units_per_block = np.zeros(t, dtype=np.int64)
        for b in block_list:
            units_per_block[b] += 1

        n_active_cells = len(set(cell_list))

        return cls(
            cell_indices=cell_indices,
            time_block_indices=time_block_indices,
            flat_lookup=flat_lookup,
            n_units=n_units,
            n_active_cells=n_active_cells,
            units_per_block=units_per_block,
        )

    def from_cell_time(self, cell: int, t: int) -> int:
        n_blocks = len(self.units_per_block)
        idx = cell * n_blocks + t
        if idx < 0 or idx >= len(self.flat_lookup):
            return -1
        return int(self.flat_lookup[idx])

    def to_cell_time(self, unit_idx: int) -> Tuple[int, int]:
        return int(self.cell_indices[unit_idx]), int(self.time_block_indices[unit_idx])

    def to_flat_cell(self, unit_idx: int) -> int:
        return int(self.cell_indices[unit_idx])

    def to_time_block(self, unit_idx: int) -> int:
        return int(self.time_block_indices[unit_idx])
