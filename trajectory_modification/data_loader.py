"""
Data loading utilities for FAMAIL trajectory modification.

Loads trajectories, pickup/dropoff counts, active taxis, and g(d) function.

Required Datasets:
- passenger_seeking_trajs_45-800.pkl: Trajectory data (Dict[driver_id, List[trajectory]])
- pickup_dropoff_counts.pkl: Pickup/dropoff counts per (x, y, time, day) state
- active_taxis_5x5_*.pkl: Active taxi counts per neighborhood
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable, Any
import pickle
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .trajectory import Trajectory, TrajectoryState


# Default paths relative to FAMAIL workspace
DEFAULT_PATHS = {
    'passenger_seeking': 'source_data/passenger_seeking_trajs_45-800.pkl',
    'pickup_dropoff_counts': 'source_data/pickup_dropoff_counts.pkl',
    'active_taxis_hourly': 'source_data/active_taxis_5x5_hourly.pkl',
    'active_taxis_time_bucket': 'source_data/active_taxis_5x5_time_bucket.pkl',
    'g_function_params': 'objective_function/causal_fairness/g_function_params.json',
}

# Grid dimensions for Shenzhen
GRID_DIMS = (48, 90)


def find_workspace_root() -> Path:
    """Find FAMAIL workspace root."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'objective_function').exists():
            return parent
    return Path.cwd()


def load_pickle(path: Path) -> Any:
    """Load pickle file with fallback encoding."""
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')


class TrajectoryLoader:
    """Loads trajectory data from pickle files."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or find_workspace_root()
    
    def load_passenger_seeking(
        self, 
        max_trajectories: Optional[int] = None,
        max_drivers: Optional[int] = None,
    ) -> List[Trajectory]:
        """
        Load passenger-seeking trajectories.
        
        The file contains a dict keyed by driver_id, where each value is
        a list of trajectories, and each trajectory is a list of states.
        Each state is [x_grid, y_grid, time_bucket, day_index].
        
        Args:
            max_trajectories: Maximum total trajectories to load
            max_drivers: Maximum number of drivers to load from
            
        Returns:
            List of Trajectory objects
        """
        path = self.workspace_root / DEFAULT_PATHS['passenger_seeking']
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        
        data = load_pickle(path)
        trajectories = []
        total_count = 0
        traj_id = 0
        
        # Data is dict[driver_id, list[trajectory]]
        driver_keys = list(data.keys())
        if max_drivers:
            driver_keys = driver_keys[:max_drivers]
        
        for driver_id in driver_keys:
            driver_trajs = data[driver_id]
            
            for traj_data in driver_trajs:
                if max_trajectories and total_count >= max_trajectories:
                    return trajectories
                
                try:
                    traj = self._parse_trajectory(
                        traj_data, 
                        driver_id=driver_id,
                        trajectory_id=traj_id,
                    )
                    if traj:
                        trajectories.append(traj)
                        total_count += 1
                        traj_id += 1
                except Exception:
                    continue
        
        return trajectories
    
    def _parse_trajectory(
        self,
        traj_data: List[List[int]], 
        driver_id: int = 0,
        trajectory_id: Optional[int] = None,
    ) -> Optional[Trajectory]:
        """
        Parse a single trajectory from list format.
        
        Expected format: List of states where each state is
        [x_grid, y_grid, time_bucket, day_index]
        """
        if not isinstance(traj_data, list) or len(traj_data) < 2:
            return None
        
        states = []
        for state_data in traj_data:
            if len(state_data) >= 4:
                state = TrajectoryState(
                    x_grid=int(state_data[0]),
                    y_grid=int(state_data[1]),
                    time_bucket=int(state_data[2]),
                    day_index=int(state_data[3]),
                )
                states.append(state)
        
        if len(states) < 2:
            return None
        
        return Trajectory(
            trajectory_id=trajectory_id,
            driver_id=driver_id,
            states=states,
        )


class PickupDropoffLoader:
    """Loads pickup/dropoff count data."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or find_workspace_root()
    
    def load(self) -> Dict[Tuple[int, int, int, int], Tuple[int, int]]:
        """
        Load pickup_dropoff_counts.pkl.
        
        Returns:
            Dictionary mapping (x, y, time_bucket, day) -> (pickup_count, dropoff_count)
        """
        path = self.workspace_root / DEFAULT_PATHS['pickup_dropoff_counts']
        if not path.exists():
            raise FileNotFoundError(f"Pickup/dropoff counts not found: {path}")
        
        return load_pickle(path)
    
    def extract_pickup_counts(
        self,
        data: Optional[Dict] = None,
    ) -> Dict[Tuple[int, int, int, int], int]:
        """Extract just pickup counts from the data."""
        if data is None:
            data = self.load()
        
        pickups = {}
        for key, counts in data.items():
            if isinstance(counts, (list, tuple)) and len(counts) >= 1:
                pickups[key] = int(counts[0])
            elif isinstance(counts, (int, float)):
                pickups[key] = int(counts)
        return pickups
    
    def extract_dropoff_counts(
        self,
        data: Optional[Dict] = None,
    ) -> Dict[Tuple[int, int, int, int], int]:
        """Extract just dropoff counts from the data."""
        if data is None:
            data = self.load()
        
        dropoffs = {}
        for key, counts in data.items():
            if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                dropoffs[key] = int(counts[1])
        return dropoffs
    
    def aggregate_to_grid(
        self,
        data: Optional[Dict] = None,
        aggregation: str = 'sum',
    ) -> np.ndarray:
        """
        Aggregate pickup counts to a 2D grid (sum across all time periods).
        
        Args:
            data: Pickup/dropoff data (loads if None)
            aggregation: 'sum' or 'mean'
            
        Returns:
            2D numpy array of shape (48, 90)
        """
        if data is None:
            data = self.load()
        
        grid = np.zeros(GRID_DIMS, dtype=np.float32)
        counts_per_cell = np.zeros(GRID_DIMS, dtype=np.float32)
        
        for key, counts in data.items():
            x, y = key[0], key[1]
            if 0 <= x < GRID_DIMS[0] and 0 <= y < GRID_DIMS[1]:
                pickup = counts[0] if isinstance(counts, (list, tuple)) else counts
                grid[x, y] += pickup
                counts_per_cell[x, y] += 1
        
        if aggregation == 'mean':
            mask = counts_per_cell > 0
            grid[mask] = grid[mask] / counts_per_cell[mask]
        
        return grid


class ActiveTaxisLoader:
    """Loads active taxi count data."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or find_workspace_root()
    
    def load(self, period_type: str = 'hourly') -> Dict[Tuple, int]:
        """
        Load active taxis data for specified period type.
        
        Args:
            period_type: 'hourly', 'time_bucket', 'daily', or 'all'
            
        Returns:
            Dictionary mapping state keys to active taxi counts
        """
        if period_type == 'hourly':
            path = self.workspace_root / DEFAULT_PATHS['active_taxis_hourly']
        elif period_type == 'time_bucket':
            path = self.workspace_root / DEFAULT_PATHS['active_taxis_time_bucket']
        else:
            # Try hourly as default
            path = self.workspace_root / DEFAULT_PATHS['active_taxis_hourly']
        
        if not path.exists():
            raise FileNotFoundError(f"Active taxis data not found: {path}")
        
        loaded = load_pickle(path)
        
        # Handle bundle format (with 'data', 'stats', 'config' keys)
        if isinstance(loaded, dict) and 'data' in loaded:
            return loaded['data']
        
        return loaded
    
    def aggregate_to_grid(
        self,
        data: Optional[Dict] = None,
        aggregation: str = 'mean',
    ) -> np.ndarray:
        """
        Aggregate active taxi counts to a 2D grid.
        
        Args:
            data: Active taxis data (loads if None)
            aggregation: 'sum', 'mean', or 'max'
            
        Returns:
            2D numpy array of shape (48, 90)
        """
        if data is None:
            data = self.load()
        
        grid = np.zeros(GRID_DIMS, dtype=np.float32)
        counts_per_cell = np.zeros(GRID_DIMS, dtype=np.float32)
        
        for key, count in data.items():
            # Handle different key formats
            x, y = key[0], key[1]
            if 0 < x <= GRID_DIMS[0] and 0 < y <= GRID_DIMS[1]:
                # active_taxis uses 1-indexed coords
                grid[x-1, y-1] += count
                counts_per_cell[x-1, y-1] += 1
        
        if aggregation == 'mean':
            mask = counts_per_cell > 0
            grid[mask] = grid[mask] / counts_per_cell[mask]
        elif aggregation == 'max':
            # For max, we need different logic
            grid = np.zeros(GRID_DIMS, dtype=np.float32)
            for key, count in data.items():
                x, y = key[0], key[1]
                if 0 < x <= GRID_DIMS[0] and 0 < y <= GRID_DIMS[1]:
                    grid[x-1, y-1] = max(grid[x-1, y-1], count)
        
        return grid


class GFunctionLoader:
    """Loads g(d) function parameters."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or find_workspace_root()
    
    def load_params(self) -> Optional[Dict[str, Any]]:
        """Load g(d) function parameters from JSON."""
        path = self.workspace_root / DEFAULT_PATHS['g_function_params']
        if not path.exists():
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def build_g_function(self) -> Callable:
        """Build g(d) function from saved parameters."""
        params = self.load_params()
        if params is None:
            # Return default log-linear function
            return lambda d: 0.5 * np.log(np.asarray(d) + 1)
        
        # Build function based on type
        func_type = params.get('type', 'log_linear')
        
        if func_type == 'log_linear':
            a = params.get('a', 0.5)
            b = params.get('b', 0.0)
            return lambda d: a * np.log(np.asarray(d) + 1) + b
        elif func_type == 'polynomial':
            coeffs = params.get('coefficients', [0.5])
            return lambda d: np.polyval(coeffs, np.asarray(d))
        elif func_type == 'power':
            k = params.get('k', 0.5)
            alpha = params.get('alpha', 0.5)
            return lambda d: k * np.power(np.asarray(d) + 1, alpha)
        else:
            return lambda d: 0.5 * np.log(np.asarray(d) + 1)


@dataclass
class DataBundle:
    """Bundle of all data needed for trajectory modification."""
    
    trajectories: List[Trajectory]
    pickup_dropoff_data: Dict[Tuple, Tuple[int, int]]  # Raw pickup/dropoff counts
    pickup_grid: np.ndarray  # Aggregated pickups (48, 90)
    dropoff_grid: np.ndarray  # Aggregated dropoffs (48, 90)
    active_taxis_data: Dict[Tuple, int]  # Raw active taxis counts
    active_taxis_grid: np.ndarray  # Aggregated active taxis (48, 90)
    g_function: Callable
    grid_dims: Tuple[int, int] = GRID_DIMS
    
    @classmethod
    def load_default(
        cls,
        max_trajectories: int = 100,
        max_drivers: Optional[int] = None,
        workspace_root: Optional[Path] = None,
        active_taxis_period: str = 'hourly',
    ) -> 'DataBundle':
        """
        Load default data bundle from workspace.
        
        Args:
            max_trajectories: Maximum trajectories to load
            max_drivers: Maximum drivers to load from
            workspace_root: Override workspace root path
            active_taxis_period: Period type for active taxis ('hourly' or 'time_bucket')
            
        Returns:
            DataBundle with all loaded data
        """
        root = workspace_root or find_workspace_root()
        
        # Load trajectories
        traj_loader = TrajectoryLoader(root)
        trajectories = traj_loader.load_passenger_seeking(
            max_trajectories=max_trajectories,
            max_drivers=max_drivers,
        )
        
        # Load pickup/dropoff counts
        pd_loader = PickupDropoffLoader(root)
        pickup_dropoff_data = pd_loader.load()
        pickup_grid = pd_loader.aggregate_to_grid(pickup_dropoff_data, aggregation='sum')
        
        # Extract and aggregate dropoffs
        dropoff_grid = np.zeros(GRID_DIMS, dtype=np.float32)
        for key, counts in pickup_dropoff_data.items():
            x, y = key[0], key[1]
            if 0 <= x < GRID_DIMS[0] and 0 <= y < GRID_DIMS[1]:
                if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                    dropoff_grid[x, y] += counts[1]
        
        # Load active taxis
        at_loader = ActiveTaxisLoader(root)
        active_taxis_data = at_loader.load(period_type=active_taxis_period)
        active_taxis_grid = at_loader.aggregate_to_grid(active_taxis_data, aggregation='mean')
        
        # Ensure no zero values in active_taxis_grid (for DSR calculation)
        active_taxis_grid = np.maximum(active_taxis_grid, 0.1)
        
        # Load g function
        g_loader = GFunctionLoader(root)
        g_function = g_loader.build_g_function()
        
        return cls(
            trajectories=trajectories,
            pickup_dropoff_data=pickup_dropoff_data,
            pickup_grid=pickup_grid,
            dropoff_grid=dropoff_grid,
            active_taxis_data=active_taxis_data,
            active_taxis_grid=active_taxis_grid,
            g_function=g_function,
        )
    
    def to_tensors(self, device: str = 'cpu') -> Dict[str, 'torch.Tensor']:
        """Convert numpy arrays to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        return {
            'pickup_grid': torch.tensor(self.pickup_grid, device=device, dtype=torch.float32),
            'dropoff_grid': torch.tensor(self.dropoff_grid, device=device, dtype=torch.float32),
            'active_taxis_grid': torch.tensor(self.active_taxis_grid, device=device, dtype=torch.float32),
        }
    
    def get_state_counts(
        self,
        x: int,
        y: int,
        time_bucket: int,
        day: int,
    ) -> Tuple[int, int, int]:
        """
        Get pickup, dropoff, and active taxi counts for a specific state.
        
        Args:
            x, y: Grid coordinates
            time_bucket: Time bucket (1-288)
            day: Day of week (1-6)
            
        Returns:
            Tuple of (pickup_count, dropoff_count, active_taxis)
        """
        key = (x, y, time_bucket, day)
        
        # Pickup/dropoff
        counts = self.pickup_dropoff_data.get(key, (0, 0))
        pickup = counts[0] if isinstance(counts, (list, tuple)) else counts
        dropoff = counts[1] if isinstance(counts, (list, tuple)) and len(counts) > 1 else 0
        
        # Active taxis - try hourly first (convert time_bucket to hour)
        hour = (time_bucket - 1) // 12  # Convert 1-288 to 0-23
        hourly_key = (x, y, hour, day)
        active = self.active_taxis_data.get(hourly_key, 0)
        
        # If not found, try time_bucket key
        if active == 0:
            active = self.active_taxis_data.get(key, 0)
        
        return (pickup, dropoff, active)
