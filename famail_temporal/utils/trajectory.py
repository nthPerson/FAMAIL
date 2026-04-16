"""Trajectory representation for famail_temporal."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Tuple

import numpy as np
import torch


@dataclass
class TrajectoryState:
    x_grid: float
    y_grid: float
    time_bucket: int
    day_index: int

    def to_array(self) -> np.ndarray:
        return np.array([self.x_grid, self.y_grid, self.time_bucket, self.day_index])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TrajectoryState":
        return cls(
            x_grid=float(arr[0]), y_grid=float(arr[1]),
            time_bucket=int(arr[2]), day_index=int(arr[3]),
        )


@dataclass
class Trajectory:
    trajectory_id: Any
    driver_id: Any
    states: List[TrajectoryState]
    metadata: dict = field(default_factory=dict)

    @property
    def pickup_state(self) -> TrajectoryState:
        return self.states[-1]

    @property
    def pickup_cell(self) -> Tuple[int, int]:
        s = self.pickup_state
        return (int(s.x_grid), int(s.y_grid))

    @property
    def n_states(self) -> int:
        return len(self.states)

    def to_discriminator_format(self) -> np.ndarray:
        return np.array([s.to_array() for s in self.states])

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.to_discriminator_format(), dtype=torch.float32)

    def clone(self) -> "Trajectory":
        return Trajectory(
            trajectory_id=self.trajectory_id,
            driver_id=self.driver_id,
            states=[TrajectoryState(s.x_grid, s.y_grid, s.time_bucket, s.day_index)
                    for s in self.states],
            metadata=self.metadata.copy(),
        )

    def apply_perturbation(self, delta: np.ndarray,
                           grid_dims: Tuple[int, int] = (48, 90)) -> "Trajectory":
        modified = self.clone()
        pickup = modified.states[-1]
        new_x = float(np.clip(pickup.x_grid + delta[0], 0, grid_dims[0] - 1))
        new_y = float(np.clip(pickup.y_grid + delta[1], 0, grid_dims[1] - 1))
        modified.states[-1] = TrajectoryState(
            x_grid=new_x, y_grid=new_y,
            time_bucket=pickup.time_bucket, day_index=pickup.day_index,
        )
        return modified
