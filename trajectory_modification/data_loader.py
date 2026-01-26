"""
Data loading utilities for FAMAIL trajectory modification.

Loads trajectories, supply/demand data, and g(d) function from various sources.
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
    'all_trajs': 'source_data/all_trajs.pkl',
    'passenger_seeking': 'source_data/passenger_seeking_trajs_45-800.pkl',
    'latest_traffic': 'source_data/latest_traffic.pkl',
    'latest_volume_pickups': 'source_data/latest_volume_pickups.pkl',
    'g_function_params': 'objective_function/causal_fairness/g_function_params.json',
}


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
    
    def load_all_trajs(self, max_trajectories: Optional[int] = None) -> List[Trajectory]:
        """
        Load trajectories from all_trajs.pkl.
        
        Expected format: List of trajectory dicts with 'traj' key containing state arrays.
        """
        path = self.workspace_root / DEFAULT_PATHS['all_trajs']
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        
        data = load_pickle(path)
        trajectories = []
        
        for i, item in enumerate(data):
            if max_trajectories and i >= max_trajectories:
                break
            
            try:
                traj = self._parse_trajectory(item, driver_id=i)
                if traj:
                    trajectories.append(traj)
            except Exception as e:
                continue  # Skip malformed entries
        
        return trajectories
    
    def load_passenger_seeking(
        self, 
        max_trajectories: Optional[int] = None,
    ) -> List[Trajectory]:
        """Load passenger-seeking trajectories."""
        path = self.workspace_root / DEFAULT_PATHS['passenger_seeking']
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        
        data = load_pickle(path)
        trajectories = []
        
        for i, item in enumerate(data):
            if max_trajectories and i >= max_trajectories:
                break
            
            try:
                traj = self._parse_trajectory(item, driver_id=i)
                if traj:
                    trajectories.append(traj)
            except Exception:
                continue
        
        return trajectories
    
    def _parse_trajectory(
        self, 
        item: Any, 
        driver_id: int = 0,
    ) -> Optional[Trajectory]:
        """Parse a single trajectory from various formats."""
        # Handle dict format
        if isinstance(item, dict):
            if 'traj' in item:
                states_data = item['traj']
                driver_id = item.get('driver_id', driver_id)
            else:
                return None
        # Handle array format
        elif isinstance(item, (list, np.ndarray)):
            states_data = item
        else:
            return None
        
        # Parse states
        states = []
        for state_data in states_data:
            if len(state_data) >= 4:
                state = TrajectoryState(
                    x_grid=float(state_data[0]),
                    y_grid=float(state_data[1]),
                    time_bucket=int(state_data[2]),
                    day_index=int(state_data[3]),
                )
                states.append(state)
        
        if len(states) < 2:
            return None
        
        return Trajectory(states=states, driver_id=driver_id)


class SupplyDemandLoader:
    """Loads supply/demand grid data."""
    
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or find_workspace_root()
    
    def load_latest_traffic(self) -> Dict[str, np.ndarray]:
        """
        Load latest_traffic.pkl containing supply/demand per cell.
        
        Returns dict with 'pickups', 'supply', 'active_taxis' arrays.
        """
        path = self.workspace_root / DEFAULT_PATHS['latest_traffic']
        if not path.exists():
            raise FileNotFoundError(f"Traffic data not found: {path}")
        
        data = load_pickle(path)
        return self._extract_counts(data)
    
    def load_latest_volume_pickups(self) -> Dict[str, np.ndarray]:
        """Load pickup volume data."""
        path = self.workspace_root / DEFAULT_PATHS['latest_volume_pickups']
        if not path.exists():
            raise FileNotFoundError(f"Pickup data not found: {path}")
        
        data = load_pickle(path)
        return self._extract_counts(data)
    
    def _extract_counts(self, data: Any) -> Dict[str, np.ndarray]:
        """Extract count arrays from various data formats."""
        result = {}
        
        if isinstance(data, dict):
            # Handle DataFrame-like dict
            if 'pickup' in data or 'pickups' in data:
                result['pickups'] = np.array(data.get('pickup', data.get('pickups', [])))
            if 'supply' in data:
                result['supply'] = np.array(data['supply'])
            if 'active_taxis' in data:
                result['active_taxis'] = np.array(data['active_taxis'])
        elif hasattr(data, 'values'):
            # Handle pandas DataFrame
            result['pickups'] = data.values
        elif isinstance(data, np.ndarray):
            result['pickups'] = data
        
        return result
    
    def aggregate_to_grid(
        self, 
        data: np.ndarray, 
        grid_dims: Tuple[int, int] = (48, 90),
        aggregation: str = 'mean',
    ) -> np.ndarray:
        """Aggregate multi-period data to single grid."""
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            # Assume shape is [time, grid_x, grid_y]
            if aggregation == 'mean':
                return data.mean(axis=0)
            elif aggregation == 'sum':
                return data.sum(axis=0)
            else:
                return data.mean(axis=0)
        else:
            return data.reshape(grid_dims)


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
    
    def build_g_function(self) -> Optional[Callable]:
        """Build g(d) function from saved parameters."""
        params = self.load_params()
        if params is None:
            # Return default log-linear function
            return lambda d: 0.5 * np.log(d + 1)
        
        # Build function based on type
        func_type = params.get('type', 'log_linear')
        
        if func_type == 'log_linear':
            a = params.get('a', 0.5)
            b = params.get('b', 0.0)
            return lambda d: a * np.log(d + 1) + b
        elif func_type == 'polynomial':
            coeffs = params.get('coefficients', [0.5])
            return lambda d: np.polyval(coeffs, d)
        elif func_type == 'power':
            k = params.get('k', 0.5)
            alpha = params.get('alpha', 0.5)
            return lambda d: k * np.power(d + 1, alpha)
        else:
            return lambda d: 0.5 * np.log(d + 1)


class DataBundle:
    """Bundle of all data needed for trajectory modification."""
    
    def __init__(
        self,
        trajectories: List[Trajectory],
        pickup_counts: np.ndarray,
        supply_counts: np.ndarray,
        active_taxis: np.ndarray,
        g_function: Optional[Callable] = None,
        grid_dims: Tuple[int, int] = (48, 90),
    ):
        self.trajectories = trajectories
        self.pickup_counts = pickup_counts
        self.supply_counts = supply_counts
        self.active_taxis = active_taxis
        self.g_function = g_function
        self.grid_dims = grid_dims
    
    @classmethod
    def load_default(
        cls,
        max_trajectories: int = 100,
        workspace_root: Optional[Path] = None,
    ) -> 'DataBundle':
        """Load default data bundle from workspace."""
        root = workspace_root or find_workspace_root()
        
        # Load trajectories
        traj_loader = TrajectoryLoader(root)
        try:
            trajectories = traj_loader.load_passenger_seeking(max_trajectories)
        except FileNotFoundError:
            trajectories = traj_loader.load_all_trajs(max_trajectories)
        
        # Load supply/demand
        sd_loader = SupplyDemandLoader(root)
        try:
            counts = sd_loader.load_latest_traffic()
        except FileNotFoundError:
            counts = {'pickups': np.zeros((48, 90)), 'supply': np.zeros((48, 90))}
        
        pickup_counts = counts.get('pickups', np.zeros((48, 90)))
        supply_counts = counts.get('supply', pickup_counts.copy())
        active_taxis = counts.get('active_taxis', np.ones_like(pickup_counts))
        
        # Load g function
        g_loader = GFunctionLoader(root)
        g_function = g_loader.build_g_function()
        
        return cls(
            trajectories=trajectories,
            pickup_counts=pickup_counts,
            supply_counts=supply_counts,
            active_taxis=active_taxis,
            g_function=g_function,
        )
    
    def to_tensors(self, device: str = 'cpu') -> Dict[str, 'torch.Tensor']:
        """Convert numpy arrays to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        return {
            'pickup_counts': torch.tensor(self.pickup_counts, device=device, dtype=torch.float32),
            'supply_counts': torch.tensor(self.supply_counts, device=device, dtype=torch.float32),
            'active_taxis': torch.tensor(self.active_taxis, device=device, dtype=torch.float32),
        }
