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
    'g_function_params': 'objective_function/causal_fairness/g_function_params.json', # TODO: this file does not exist and is not part of the implementation. Remove all references.
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
        
        # First pass: collect all valid trajectories with their driver IDs
        all_trajs = []
        for driver_id in driver_keys:
            driver_trajs = data[driver_id]
            for traj_data in driver_trajs:
                all_trajs.append((driver_id, traj_data))
        
        # If max_trajectories is set and less than total, sample randomly
        # This ensures we get trajectories from different drivers
        if max_trajectories and len(all_trajs) > max_trajectories:
            import random
            random.seed(42)  # Reproducible sampling
            all_trajs = random.sample(all_trajs, max_trajectories)
        
        # Parse the selected trajectories
        for driver_id, traj_data in all_trajs:
            try:
                traj = self._parse_trajectory(
                    traj_data, 
                    driver_id=driver_id,
                    trajectory_id=traj_id,
                )
                if traj:
                    trajectories.append(traj)
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
            # Data uses 1-indexed coordinates [1, 48] x [1, 90]
            # Convert to 0-indexed for grid array
            x, y = key[0] - 1, key[1] - 1
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
    """Loads or estimates g(d) function."""
    
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
        """Build g(d) function from saved parameters (DEPRECATED - use estimate_from_data)."""
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
    
    @staticmethod
    def estimate_from_data(
        pickup_dropoff_data: Dict[Tuple, Tuple[int, int]],
        active_taxis_data: Dict[Tuple, int],
        aggregation: str = 'mean',
        method: str = 'isotonic',
        min_demand: int = 1,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Estimate g(d) function from actual supply/demand data using isotonic regression.
        
        This is the RECOMMENDED approach: fit g(d) on the same temporal granularity
        as the data will be used during optimization.
        
        Args:
            pickup_dropoff_data: Raw pickup/dropoff counts {(x,y,time,day): [pickup, dropoff]}
            active_taxis_data: Active taxis counts {(x,y,time,day): count}
            aggregation: How to aggregate across time: 'mean', 'sum', or 'max'
            method: Estimation method: 'isotonic' (recommended), 'linear', 'polynomial'
            min_demand: Minimum demand to include cell in fitting (avoids division issues)
            
        Returns:
            Tuple of (g_function, diagnostics_dict)
        """
        # Import the well-tested utility functions from causal_fairness module
        try:
            from objective_function.causal_fairness.utils import (
                extract_demand_from_counts,
                aggregate_to_period,
                compute_service_ratios,
                extract_demand_ratio_arrays,
            )
        except ImportError:
            # Fall back to simple matching if utils not available
            return GFunctionLoader._estimate_from_data_simple( # TODO: do not fall back to simple matching, but throw error instead
                pickup_dropoff_data, active_taxis_data, aggregation, method, min_demand
            )
        
        # Extract demand from pickup counts
        demand = extract_demand_from_counts(pickup_dropoff_data)
        
        # Aggregate demand to hourly (same granularity as active_taxis)
        # This converts time_bucket (1-288) to hour (0-23)
        demand = aggregate_to_period(demand, 'hourly')
        
        # Compute service ratios Y = S/D
        ratios = compute_service_ratios(
            demand,
            active_taxis_data,
            min_demand=min_demand,
            include_zero_supply=False,
        )
        
        # Extract aligned arrays
        demands_arr, ratios_arr, common_keys = extract_demand_ratio_arrays(demand, ratios)
        
        if len(demands_arr) < 10:
            # Not enough data - return default function
            def default_g(d):
                return np.ones_like(np.atleast_1d(d), dtype=float)
            return default_g, {
                'method': 'default',
                'error': f'Insufficient matched data points: {len(demands_arr)}',
                'n_demand_entries': len(demand),
                'n_ratio_entries': len(ratios),
            }
        
        # Aggregate to per-cell level using specified method
        cell_demands = {}
        cell_ratios = {}
        
        for key in common_keys:
            cell = (key[0], key[1])
            d = demand[key]
            r = ratios[key]
            
            if cell not in cell_demands:
                cell_demands[cell] = []
                cell_ratios[cell] = []
            
            cell_demands[cell].append(d)
            cell_ratios[cell].append(r)
        
        # Apply aggregation
        final_demands = []
        final_ratios = []
        
        for cell in cell_demands:
            d_values = cell_demands[cell]
            r_values = cell_ratios[cell]
            
            if aggregation == 'mean':
                agg_demand = np.mean(d_values)
                agg_ratio = np.mean(r_values)
            elif aggregation == 'sum':
                agg_demand = np.sum(d_values)
                agg_ratio = np.sum(r_values)
            elif aggregation == 'max':
                agg_demand = np.max(d_values)
                agg_ratio = np.max(r_values)
            else:
                agg_demand = np.mean(d_values)
                agg_ratio = np.mean(r_values)
            
            final_demands.append(agg_demand)
            final_ratios.append(agg_ratio)
        
        final_demands = np.array(final_demands, dtype=float)
        final_ratios = np.array(final_ratios, dtype=float)
        
        # Estimate g(d) using isotonic regression
        if method == 'isotonic':
            try:
                from sklearn.isotonic import IsotonicRegression
                model = IsotonicRegression(increasing=False, out_of_bounds='clip')
                model.fit(final_demands, final_ratios)
                
                def g_function(d):
                    return model.predict(np.atleast_1d(d))
                
                # Compute R² for diagnostics
                predictions = g_function(final_demands)
                residuals = final_ratios - predictions
                var_residuals = np.var(residuals)
                var_total = np.var(final_ratios)
                r_squared = 1 - var_residuals / (var_total + 1e-10)
                
                diagnostics = {
                    'method': 'isotonic',
                    'increasing': False,
                    'n_points': len(final_demands),
                    'demand_range': (float(final_demands.min()), float(final_demands.max())),
                    'ratio_range': (float(final_ratios.min()), float(final_ratios.max())),
                    'aggregation': aggregation,
                    'r_squared': float(r_squared),
                }
                
                return g_function, diagnostics
                
            except ImportError:
                pass
        
        # Fallback: simple mean ratio
        mean_ratio = np.mean(final_ratios)
        def fallback_g(d):
            return np.full_like(np.atleast_1d(d), mean_ratio, dtype=float)
        
        return fallback_g, {
            'method': 'constant',
            'mean_ratio': float(mean_ratio),
            'n_points': len(final_demands),
        }
    
    @staticmethod
    def _estimate_from_data_simple(
        pickup_dropoff_data: Dict[Tuple, Tuple[int, int]],
        active_taxis_data: Dict[Tuple, int],
        aggregation: str = 'mean',
        method: str = 'isotonic',
        min_demand: int = 1,
    ) -> Tuple[Callable, Dict[str, Any]]:
        """
        Fallback estimation when causal_fairness utils not available.
        
        Uses simple key matching with time_bucket to hour conversion.
        """
        # Extract demand and compute service ratios
        demand_per_key = {}
        supply_per_key = {}
        
        for key, counts in pickup_dropoff_data.items():
            if len(key) >= 4:
                pickup = counts[0] if isinstance(counts, (list, tuple)) else counts
                if pickup >= min_demand:
                    demand_per_key[key] = pickup
        
        # Match with active taxis (convert time_bucket to hour)
        for key, demand in demand_per_key.items():
            x, y, time_val, day_val = key[0], key[1], key[2], key[3]
            
            # Convert time_bucket (1-288) to hour (0-23)
            hour = (time_val - 1) // 12
            hourly_key = (x, y, hour, day_val)
            supply = active_taxis_data.get(hourly_key, 0)
            
            if supply > 0:
                supply_per_key[key] = supply
        
        # Compute Y = S/D (service ratio)
        common_keys = set(demand_per_key.keys()) & set(supply_per_key.keys())
        
        if len(common_keys) < 10:
            # Not enough data - return default function
            def default_g(d):
                return np.ones_like(np.atleast_1d(d), dtype=float)
            return default_g, {
                'method': 'default',
                'error': f'Insufficient matched data points: {len(common_keys)}',
                'n_demand_entries': len(demand_per_key),
                'n_supply_entries': len(supply_per_key),
            }
        
        # Aggregate to per-cell level using specified method
        cell_demands = {}
        cell_ratios = {}
        cell_counts = {}
        
        for key in common_keys:
            cell = (key[0], key[1])
            demand = demand_per_key[key]
            supply = supply_per_key[key]
            ratio = supply / demand
            
            if cell not in cell_demands:
                cell_demands[cell] = []
                cell_ratios[cell] = []
            
            cell_demands[cell].append(demand)
            cell_ratios[cell].append(ratio)
        
        # Apply aggregation
        demands_arr = []
        ratios_arr = []
        
        for cell in cell_demands:
            d_values = cell_demands[cell]
            r_values = cell_ratios[cell]
            
            if aggregation == 'mean':
                agg_demand = np.mean(d_values)
                agg_ratio = np.mean(r_values)
            elif aggregation == 'sum':
                agg_demand = np.sum(d_values)
                agg_ratio = np.sum(r_values)  # May not be meaningful
            elif aggregation == 'max':
                agg_demand = np.max(d_values)
                agg_ratio = np.max(r_values)
            else:
                agg_demand = np.mean(d_values)
                agg_ratio = np.mean(r_values)
            
            demands_arr.append(agg_demand)
            ratios_arr.append(agg_ratio)
        
        demands_arr = np.array(demands_arr, dtype=float)
        ratios_arr = np.array(ratios_arr, dtype=float)
        
        # Estimate g(d) using specified method
        if method == 'isotonic':
            try:
                from sklearn.isotonic import IsotonicRegression
                model = IsotonicRegression(increasing=False, out_of_bounds='clip')
                model.fit(demands_arr, ratios_arr)
                
                def g_function(d):
                    return model.predict(np.atleast_1d(d))
                
                diagnostics = {
                    'method': 'isotonic',
                    'increasing': False,
                    'n_points': len(demands_arr),
                    'demand_range': (float(demands_arr.min()), float(demands_arr.max())),
                    'ratio_range': (float(ratios_arr.min()), float(ratios_arr.max())),
                    'aggregation': aggregation,
                }
                
                # Compute R² for diagnostics
                predictions = g_function(demands_arr)
                residuals = ratios_arr - predictions
                var_residuals = np.var(residuals)
                var_total = np.var(ratios_arr)
                r_squared = 1 - var_residuals / (var_total + 1e-10)
                diagnostics['r_squared'] = float(r_squared)
                
                return g_function, diagnostics
                
            except ImportError:
                # Fall back to simple mean if sklearn not available
                pass
        
        # Fallback: simple mean ratio
        mean_ratio = np.mean(ratios_arr)
        def fallback_g(d):
            return np.full_like(np.atleast_1d(d), mean_ratio, dtype=float)
        
        return fallback_g, {
            'method': 'constant',
            'mean_ratio': float(mean_ratio),
            'n_points': len(demands_arr),
        }


@dataclass
class DataBundle:
    """Bundle of all data needed for trajectory modification."""
    
    trajectories: List[Trajectory]
    pickup_dropoff_data: Dict[Tuple, Tuple[int, int]]  # Raw pickup/dropoff counts
    pickup_grid: np.ndarray  # Aggregated pickups (48, 90) - for spatial fairness
    dropoff_grid: np.ndarray  # Aggregated dropoffs (48, 90) - for spatial fairness
    active_taxis_data: Dict[Tuple, int]  # Raw active taxis counts
    active_taxis_grid: np.ndarray  # Aggregated active taxis (48, 90) - for DSR/ASR
    g_function: Callable
    grid_dims: Tuple[int, int] = GRID_DIMS
    
    # NEW: Separate tensors for causal fairness with proper temporal aggregation
    causal_demand_grid: Optional[np.ndarray] = None  # Mean demand per cell (for F_causal)
    causal_supply_grid: Optional[np.ndarray] = None  # Mean supply per cell (for F_causal)
    g_function_diagnostics: Optional[Dict[str, Any]] = None  # g(d) fitting info
    
    @classmethod
    def load_default(
        cls,
        max_trajectories: int = 100,
        max_drivers: Optional[int] = None,
        workspace_root: Optional[Path] = None,
        active_taxis_period: str = 'hourly',
        estimate_g_from_data: bool = True,
        aggregation: str = 'mean',
    ) -> 'DataBundle':
        """
        Load default data bundle from workspace.
        
        IMPORTANT: When estimate_g_from_data=True (default), g(d) is fitted using
        isotonic regression on the actual supply/demand data. This ensures the
        g(d) output scale matches the Y = S/D values in the data.
        
        Args:
            max_trajectories: Maximum trajectories to load
            max_drivers: Maximum drivers to load from
            workspace_root: Override workspace root path
            active_taxis_period: Period type for active taxis ('hourly' or 'time_bucket')
            estimate_g_from_data: If True, fit g(d) using isotonic regression on actual data.
                                  If False, use the old hardcoded g(d) function (NOT recommended).
            aggregation: Temporal aggregation method for causal fairness: 'mean' (recommended),
                        'sum', or 'max'. Must match the scale expected by g(d).
            
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
        
        # Spatial fairness uses SUM aggregation (total pickups/dropoffs per cell)
        pickup_grid = pd_loader.aggregate_to_grid(pickup_dropoff_data, aggregation='sum')
        
        # Extract and aggregate dropoffs (also sum for spatial fairness)
        dropoff_grid = np.zeros(GRID_DIMS, dtype=np.float32)
        for key, counts in pickup_dropoff_data.items():
            # Data uses 1-indexed coordinates, convert to 0-indexed
            x, y = key[0] - 1, key[1] - 1
            if 0 <= x < GRID_DIMS[0] and 0 <= y < GRID_DIMS[1]:
                if isinstance(counts, (list, tuple)) and len(counts) >= 2:
                    dropoff_grid[x, y] += counts[1]
        
        # Load active taxis
        at_loader = ActiveTaxisLoader(root)
        active_taxis_data = at_loader.load(period_type=active_taxis_period)
        active_taxis_grid = at_loader.aggregate_to_grid(active_taxis_data, aggregation='mean')
        
        # Ensure no zero values in active_taxis_grid (for DSR calculation)
        active_taxis_grid = np.maximum(active_taxis_grid, 0.1)
        
        # =====================================================================
        # G(D) FUNCTION + CAUSAL FAIRNESS TENSORS
        # =====================================================================
        # CRITICAL: g(d) is fitted on hourly data. For F_causal to be meaningful,
        # the causal demand/supply grids must use the SAME hourly-aggregated data
        # so that Y = S/D is in the same scale range as g(d) was trained on.
        
        g_diagnostics = None
        causal_demand_grid = None
        causal_supply_grid = None
        
        if estimate_g_from_data:
            # Import the utilities for proper data processing
            try:
                from objective_function.causal_fairness.utils import (
                    extract_demand_from_counts,
                    aggregate_to_period,
                    compute_service_ratios,
                )
                
                # Step 1: Extract demand from pickup counts
                hourly_demand = extract_demand_from_counts(pickup_dropoff_data)
                
                # Step 2: Aggregate to hourly (same granularity as active_taxis)
                hourly_demand = aggregate_to_period(hourly_demand, 'hourly')
                
                # Step 3: Fit g(d) using isotonic regression on hourly data
                g_function, g_diagnostics = GFunctionLoader.estimate_from_data(
                    pickup_dropoff_data=pickup_dropoff_data,
                    active_taxis_data=active_taxis_data,
                    aggregation=aggregation,
                    method='isotonic',
                )
                
                # Step 4: Create causal grids from the SAME hourly data
                # Aggregate hourly demand to per-cell mean
                causal_demand_grid = np.zeros(GRID_DIMS, dtype=np.float32)
                demand_counts_grid = np.zeros(GRID_DIMS, dtype=np.float32)
                
                for key, d in hourly_demand.items():
                    # hourly keys: (x, y, hour, day) - 1-indexed coords
                    x, y = key[0] - 1, key[1] - 1
                    if 0 <= x < GRID_DIMS[0] and 0 <= y < GRID_DIMS[1]:
                        causal_demand_grid[x, y] += d
                        demand_counts_grid[x, y] += 1
                
                # Apply mean aggregation
                mask = demand_counts_grid > 0
                causal_demand_grid[mask] = causal_demand_grid[mask] / demand_counts_grid[mask]
                
                # Create causal supply grid from hourly active_taxis data (same approach)
                causal_supply_grid = np.zeros(GRID_DIMS, dtype=np.float32)
                supply_counts_grid = np.zeros(GRID_DIMS, dtype=np.float32)
                
                for key, s in active_taxis_data.items():
                    # active_taxis keys: (x, y, hour, day) - 1-indexed coords
                    x, y = key[0] - 1, key[1] - 1
                    if 0 <= x < GRID_DIMS[0] and 0 <= y < GRID_DIMS[1]:
                        causal_supply_grid[x, y] += s
                        supply_counts_grid[x, y] += 1
                
                mask = supply_counts_grid > 0
                causal_supply_grid[mask] = causal_supply_grid[mask] / supply_counts_grid[mask]
                causal_supply_grid = np.maximum(causal_supply_grid, 0.1)  # Avoid division by zero
                
            except ImportError:
                # Fallback: use simple approach
                g_function, g_diagnostics = GFunctionLoader.estimate_from_data(
                    pickup_dropoff_data=pickup_dropoff_data,
                    active_taxis_data=active_taxis_data,
                    aggregation=aggregation,
                    method='isotonic',
                )
                causal_demand_grid = pd_loader.aggregate_to_grid(pickup_dropoff_data, aggregation='mean')
                causal_supply_grid = at_loader.aggregate_to_grid(active_taxis_data, aggregation='mean')
                causal_supply_grid = np.maximum(causal_supply_grid, 0.1)
        else:
            # DEPRECATED: Use hardcoded g(d) function (may cause scale mismatch)
            g_loader = GFunctionLoader(root)
            g_function = g_loader.build_g_function()
            g_diagnostics = {'method': 'hardcoded', 'warning': 'May cause scale mismatch with F_causal'}
            causal_demand_grid = pd_loader.aggregate_to_grid(pickup_dropoff_data, aggregation='mean')
            causal_supply_grid = at_loader.aggregate_to_grid(active_taxis_data, aggregation='mean')
            causal_supply_grid = np.maximum(causal_supply_grid, 0.1)
        
        return cls(
            trajectories=trajectories,
            pickup_dropoff_data=pickup_dropoff_data,
            pickup_grid=pickup_grid,
            dropoff_grid=dropoff_grid,
            active_taxis_data=active_taxis_data,
            active_taxis_grid=active_taxis_grid,
            g_function=g_function,
            causal_demand_grid=causal_demand_grid,
            causal_supply_grid=causal_supply_grid,
            g_function_diagnostics=g_diagnostics,
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
