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


@dataclass(frozen=True)
class DataBundle:
    """All data and artifacts needed by FAMAILObjective and TrajectoryModifier."""
    pickup_3d: np.ndarray
    dropoff_3d: np.ndarray
    active_taxis_3d: np.ndarray
    mask_3d: np.ndarray
    unit_map: UnitIndexMap
    n_hours_per_block: np.ndarray
    n_days: int
    g0_func: G0Function
    hat_matrices: Dict[str, np.ndarray]
    trajectories: List[Trajectory]
    multi_stream: MultiStreamData
    discriminator: nn.Module
