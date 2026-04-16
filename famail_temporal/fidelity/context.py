"""
Multi-stream context builder for the V3 discriminator.

Full MultiStreamContextBuilder is added in Task 24. This file currently
contains only MultiStreamData.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class MultiStreamData:
    """Bundle of the five multi-stream inputs.

    All dicts keyed by driver_idx (int, 0..49) matching Trajectory.driver_id.
    Coordinates in driving_trajs / seeking_trajs are 1-indexed [1-48, 1-90].
    """
    driving_trajs: Dict[int, List]
    seeking_trajs: Dict[int, List]
    profile_features: Dict[int, np.ndarray]
    seeking_days: Dict[int, List[int]]
    driving_days: Dict[int, List[int]]
