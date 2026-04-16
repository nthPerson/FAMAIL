"""
DataBundle dataclass — immutable container for all data and artifacts needed
to instantiate FAMAILObjective and TrajectoryModifier.

The .load() classmethod is attached in a later task once preprocess.py and
cache_io.py exist.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch.nn as nn

from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.fairness.g0_power_basis import G0Function
from famail_temporal.fidelity.context import MultiStreamData
from famail_temporal.utils.trajectory import Trajectory


@dataclass(frozen=True, kw_only=True)
class DataBundle:
    """Immutable container for all data and artifacts needed by FAMAILObjective
    and TrajectoryModifier.

    Invariants:
    - pickup_3d, dropoff_3d, active_taxis_3d, and mask_3d all share the same
      (grid_x, grid_y, T) shape.
    - unit_map.n_units == hat_matrices['I_minus_H_demo'].shape[0] (Task 11
      enforces this at precompute time).
    - len(n_hours_per_block) == T (the block axis of the 3D tensors).
    - Fields are rebinding-frozen; contained numpy arrays and dicts may still
      be mutated by reference. Task 11's setflags ensures hat matrices are
      read-only; other tensors follow the same pattern downstream.

    The .load() classmethod is attached in a later task after preprocess.py
    and cache I/O are in place.
    """

    # Data tensors — all shape (48, 90, T), same spatial/block axes
    pickup_3d: np.ndarray        # (48, 90, T) float32 — mean hourly pickups per (cell, block)
    dropoff_3d: np.ndarray       # (48, 90, T) float32 — mean hourly dropoffs per (cell, block)
    active_taxis_3d: np.ndarray  # (48, 90, T) float32 — mean hourly active taxis per (cell, block)
    mask_3d: np.ndarray          # (48, 90, T) bool — active-unit mask

    # Aggregation metadata
    n_hours_per_block: np.ndarray  # (T,) int32 — hours spanned by each time block
    n_days: int                    # Number of days aggregated over

    # Derived artifacts
    unit_map: UnitIndexMap         # canonical ordering of active (cell, t) units
    g0_func: G0Function            # fitted g_0(D) power-basis function
    hat_matrices: Dict[str, np.ndarray]  # keys: 'I_minus_H_demo', 'M' (and optionally 'scaler_*')

    # Trajectories + multi-stream context
    trajectories: List[Trajectory]
    multi_stream: MultiStreamData

    # Model
    discriminator: nn.Module
