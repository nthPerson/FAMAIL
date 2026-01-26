"""
Trajectory representation for FAMAIL.

A trajectory is a sequence of states representing a taxi's path from
passenger pickup to dropoff. Each state contains spatial (x_grid, y_grid)
and temporal (time_bucket, day_index) information.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrajectoryState:
    """Single state in a trajectory."""
    x_grid: float
    y_grid: float
    time_bucket: int
    day_index: int
    
    def to_array(self) -> np.ndarray:
        """Convert to [x, y, time, day] array."""
        return np.array([self.x_grid, self.y_grid, self.time_bucket, self.day_index])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TrajectoryState':
        """Create from [x, y, time, day] array."""
        return cls(
            x_grid=float(arr[0]),
            y_grid=float(arr[1]),
            time_bucket=int(arr[2]),
            day_index=int(arr[3]),
        )


@dataclass
class Trajectory:
    """
    A taxi trajectory from passenger-seeking to pickup.
    
    The trajectory consists of:
    - states[0:-1]: Passenger-seeking path (intermediate states)
    - states[-1]: Pickup location (the state we modify)
    
    Attributes:
        trajectory_id: Unique identifier
        driver_id: Driver who generated this trajectory
        states: List of TrajectoryState from start to pickup
    """
    trajectory_id: Any
    driver_id: Any
    states: List[TrajectoryState]
    metadata: dict = field(default_factory=dict)
    
    @property
    def pickup_state(self) -> TrajectoryState:
        """The final state (pickup location)."""
        return self.states[-1]
    
    @property
    def pickup_cell(self) -> Tuple[int, int]:
        """Pickup cell as (x, y) integer coordinates."""
        s = self.pickup_state
        return (int(s.x_grid), int(s.y_grid))
    
    @property
    def n_states(self) -> int:
        """Number of states in trajectory."""
        return len(self.states)
    
    def to_discriminator_format(self) -> np.ndarray:
        """
        Convert to discriminator input format.
        
        Returns:
            Array of shape [seq_len, 4] with [x, y, time, day] per state
        """
        return np.array([s.to_array() for s in self.states])
    
    def to_tensor(self) -> 'torch.Tensor':
        """Convert to PyTorch tensor [seq_len, 4]."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor conversion")
        return torch.tensor(self.to_discriminator_format(), dtype=torch.float32)
    
    def clone(self) -> 'Trajectory':
        """Create a deep copy of this trajectory."""
        return Trajectory(
            trajectory_id=self.trajectory_id,
            driver_id=self.driver_id,
            states=[TrajectoryState(s.x_grid, s.y_grid, s.time_bucket, s.day_index) 
                    for s in self.states],
            metadata=self.metadata.copy(),
        )
    
    def apply_perturbation(
        self,
        delta: np.ndarray,
        grid_dims: Tuple[int, int] = (48, 90),
    ) -> 'Trajectory':
        """
        Apply perturbation to pickup location.
        
        Args:
            delta: Perturbation [dx, dy] to apply to pickup
            grid_dims: Grid bounds for clamping
            
        Returns:
            New trajectory with modified pickup location
        """
        modified = self.clone()
        pickup = modified.states[-1]
        
        # Apply perturbation with grid clamping
        new_x = np.clip(pickup.x_grid + delta[0], 0, grid_dims[0] - 1)
        new_y = np.clip(pickup.y_grid + delta[1], 0, grid_dims[1] - 1)
        
        modified.states[-1] = TrajectoryState(
            x_grid=new_x,
            y_grid=new_y,
            time_bucket=pickup.time_bucket,
            day_index=pickup.day_index,
        )
        
        return modified
    
    def interpolate_to_pickup(
        self,
        n_interpolation_points: int = 3,
    ) -> 'Trajectory':
        """
        Interpolate points between second-to-last state and modified pickup.
        
        Creates smooth transition from unmodified path to new pickup location.
        
        Args:
            n_interpolation_points: Number of points to insert
            
        Returns:
            Trajectory with interpolated states
        """
        if len(self.states) < 2:
            return self.clone()
        
        modified = self.clone()
        prev_state = modified.states[-2]
        pickup = modified.states[-1]
        
        # Linear interpolation for spatial coordinates
        # Time bucket advances linearly
        new_states = []
        for i in range(1, n_interpolation_points + 1):
            t = i / (n_interpolation_points + 1)
            interp_state = TrajectoryState(
                x_grid=prev_state.x_grid + t * (pickup.x_grid - prev_state.x_grid),
                y_grid=prev_state.y_grid + t * (pickup.y_grid - prev_state.y_grid),
                time_bucket=prev_state.time_bucket + int(t * (pickup.time_bucket - prev_state.time_bucket)),
                day_index=pickup.day_index,  # Keep same day
            )
            new_states.append(interp_state)
        
        # Insert interpolated states before pickup
        modified.states = modified.states[:-1] + new_states + [pickup]
        return modified

    @classmethod
    def from_state_array(
        cls,
        states_array: np.ndarray,
        trajectory_id: Any,
        driver_id: Any,
    ) -> 'Trajectory':
        """
        Create trajectory from array of states.
        
        Args:
            states_array: Shape [seq_len, 4+] with [x, y, time, day, ...]
            trajectory_id: Unique ID
            driver_id: Driver ID
        """
        states = []
        for row in states_array:
            states.append(TrajectoryState(
                x_grid=float(row[0]),
                y_grid=float(row[1]),
                time_bucket=int(row[2]) if len(row) > 2 else 0,
                day_index=int(row[3]) if len(row) > 3 else 1,
            ))
        return cls(trajectory_id=trajectory_id, driver_id=driver_id, states=states)
