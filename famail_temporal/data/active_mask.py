"""Active-unit mask and canonical ordering."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from famail_temporal import config


@dataclass(frozen=True)
class UnitIndexMap:
    """Canonical ordering of active (cell, t) units.

    Ordering rule: cell-major, then time-block within cell.

    The ``grid_shape`` field (gx, gy) records the (grid_x, grid_y) dimensions
    used to encode flat cells as ``x * gy + y``. Consumers that need to
    encode/decode flat cells (e.g. ``rank_trajectories``) must read gy from
    here rather than from ``config.GRID_DIMS`` so they remain correct under
    smaller test bundles.
    """
    cell_indices: np.ndarray
    time_block_indices: np.ndarray
    flat_lookup: np.ndarray
    n_units: int
    n_active_cells: int
    units_per_block: np.ndarray
    grid_shape: Tuple[int, int]

    @classmethod
    def from_mask(cls, mask_3d: np.ndarray, grid_shape: Tuple[int, int]) -> "UnitIndexMap":
        if mask_3d.ndim != 3:
            raise ValueError(
                f"mask_3d must be 3D (grid_x, grid_y, T); got shape {mask_3d.shape}"
            )
        gx, gy = grid_shape
        if mask_3d.shape[:2] != (gx, gy):
            raise ValueError(
                f"Expected mask grid dims {(gx, gy)}, got {mask_3d.shape[:2]}"
            )
        t = mask_3d.shape[2]

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

        # Make arrays read-only so the canonical ordering can't be silently corrupted
        for arr in (cell_indices, time_block_indices, flat_lookup, units_per_block):
            arr.setflags(write=False)

        return cls(
            cell_indices=cell_indices,
            time_block_indices=time_block_indices,
            flat_lookup=flat_lookup,
            n_units=n_units,
            n_active_cells=n_active_cells,
            units_per_block=units_per_block,
            grid_shape=(int(gx), int(gy)),
        )

    def from_cell_time(self, cell: int, t: int) -> int:
        n_blocks = len(self.units_per_block)
        # Each coordinate must be in range (prevents negative-t aliasing)
        if cell < 0 or t < 0 or t >= n_blocks:
            return -1
        idx = cell * n_blocks + t
        if idx >= len(self.flat_lookup):
            return -1
        return int(self.flat_lookup[idx])

    def to_cell_time(self, unit_idx: int) -> Tuple[int, int]:
        return int(self.cell_indices[unit_idx]), int(self.time_block_indices[unit_idx])

    def to_flat_cell(self, unit_idx: int) -> int:
        return int(self.cell_indices[unit_idx])

    def to_time_block(self, unit_idx: int) -> int:
        return int(self.time_block_indices[unit_idx])


def compute_active_mask(
    active_taxis_3d: np.ndarray,
    valid_mask: np.ndarray,
    demographics: np.ndarray,
) -> np.ndarray:
    """A unit (c, t) is active iff:
      1. active_taxis_3d[c, t] > ACTIVE_SUPPLY_THRESHOLD
      2. valid_mask[c] is True
      3. No NaN in any demographic feature for cell c
    """
    if active_taxis_3d.ndim != 3:
        raise ValueError(
            f"active_taxis_3d must be 3D (grid_x, grid_y, T); got shape {active_taxis_3d.shape}"
        )
    if valid_mask.ndim != 2:
        raise ValueError(
            f"valid_mask must be 2D (grid_x, grid_y); got shape {valid_mask.shape}"
        )
    if demographics.ndim != 3:
        raise ValueError(
            f"demographics must be 3D (grid_x, grid_y, n_features); got shape {demographics.shape}"
        )
    gx, gy = valid_mask.shape
    t = active_taxis_3d.shape[2]
    if active_taxis_3d.shape != (gx, gy, t):
        raise ValueError(
            f"active_taxis_3d shape {active_taxis_3d.shape} does not match "
            f"valid_mask grid shape ({gx}, {gy}, T)"
        )
    if demographics.shape[:2] != (gx, gy):
        raise ValueError(
            f"demographics spatial dims {demographics.shape[:2]} do not match "
            f"valid_mask grid shape ({gx}, {gy})"
        )

    cell_finite = np.isfinite(demographics).all(axis=-1)
    cell_valid = valid_mask & cell_finite
    supply_ok = active_taxis_3d > config.ACTIVE_SUPPLY_THRESHOLD
    return supply_ok & cell_valid[:, :, None]
