"""
Attribution Integration Module.

This module combines the Spatial Fairness (LIS) and Causal Fairness (DCD)
attribution methods to identify which trajectories need modification to
improve overall fairness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Callable, Union
from dataclasses import dataclass, field
import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).resolve().parent
DASHBOARDS_DIR = SCRIPT_DIR.parent
OBJECTIVE_FUNCTION_DIR = DASHBOARDS_DIR.parent
PROJECT_ROOT = OBJECTIVE_FUNCTION_DIR.parent
sys.path.insert(0, str(OBJECTIVE_FUNCTION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class AttributionScores:
    """Attribution scores for a single trajectory."""
    trajectory_id: Any
    lis_score: float  # Local Inequality Score (spatial)
    dcd_score: float  # Demand-Conditional Deviation (causal)
    combined_score: float
    pickup_cell: Optional[Tuple[int, int]] = None
    dropoff_cell: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionResult:
    """Result from attribution computation."""
    trajectory_scores: List[AttributionScores]
    lis_weight: float
    dcd_weight: float
    method: str  # 'weighted_sum' or 'gradient'
    cell_stats: Optional[Dict[str, Any]] = None
    
    def get_top_n(self, n: int = 10) -> List[AttributionScores]:
        """Get top N trajectories by combined score."""
        return sorted(self.trajectory_scores, key=lambda x: x.combined_score, reverse=True)[:n]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([
            {
                'trajectory_id': s.trajectory_id,
                'lis_score': s.lis_score,
                'dcd_score': s.dcd_score,
                'combined_score': s.combined_score,
                'pickup_cell': s.pickup_cell,
                'dropoff_cell': s.dropoff_cell,
            }
            for s in self.trajectory_scores
        ])


# =============================================================================
# LOCAL INEQUALITY SCORE (LIS) - SPATIAL ATTRIBUTION
# =============================================================================

def compute_local_inequality_score(
    cell_counts: np.ndarray,
    cell_index: Tuple[int, int],
    global_mean: Optional[float] = None,
) -> float:
    """
    Compute Local Inequality Score for a cell.
    
    LIS_i = |c_i - μ| / μ
    
    Higher LIS indicates the cell contributes more to inequality.
    
    Args:
        cell_counts: Count array [grid_x, grid_y]
        cell_index: Cell (i, j) to compute LIS for
        global_mean: Pre-computed global mean (for efficiency)
        
    Returns:
        LIS value (0 = equal to mean, higher = more deviation)
    """
    i, j = cell_index
    c_i = cell_counts[i, j]
    
    if global_mean is None:
        global_mean = cell_counts.mean()
    
    if global_mean < 1e-8:
        return 0.0
    
    lis = abs(c_i - global_mean) / global_mean
    return float(lis)


def compute_all_lis_scores(
    cell_counts: np.ndarray,
) -> np.ndarray:
    """
    Compute LIS for all cells.
    
    Args:
        cell_counts: Count array [grid_x, grid_y]
        
    Returns:
        LIS array [grid_x, grid_y]
    """
    mean_val = cell_counts.mean()
    
    if mean_val < 1e-8:
        return np.zeros_like(cell_counts)
    
    lis = np.abs(cell_counts - mean_val) / mean_val
    return lis


def compute_trajectory_lis(
    pickup_cell: Tuple[int, int],
    dropoff_cell: Tuple[int, int],
    pickup_lis: np.ndarray,
    dropoff_lis: np.ndarray,
    aggregation: str = 'max',
) -> float:
    """
    Compute combined LIS for a trajectory.
    
    Args:
        pickup_cell: Pickup cell (i, j)
        dropoff_cell: Dropoff cell (i, j)
        pickup_lis: LIS array for pickups
        dropoff_lis: LIS array for dropoffs
        aggregation: 'max', 'mean', or 'sum'
        
    Returns:
        Combined LIS value
    """
    pi, pj = pickup_cell
    di, dj = dropoff_cell
    
    lis_pickup = pickup_lis[pi, pj] if 0 <= pi < pickup_lis.shape[0] and 0 <= pj < pickup_lis.shape[1] else 0
    lis_dropoff = dropoff_lis[di, dj] if 0 <= di < dropoff_lis.shape[0] and 0 <= dj < dropoff_lis.shape[1] else 0
    
    if aggregation == 'max':
        return max(lis_pickup, lis_dropoff)
    elif aggregation == 'mean':
        return (lis_pickup + lis_dropoff) / 2
    elif aggregation == 'sum':
        return lis_pickup + lis_dropoff
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# =============================================================================
# DEMAND-CONDITIONAL DEVIATION (DCD) - CAUSAL ATTRIBUTION
# =============================================================================

def compute_demand_conditional_deviation(
    demand_cell: float,
    supply_cell: float,
    g_function: Callable,
    eps: float = 1e-8,
) -> float:
    """
    Compute Demand-Conditional Deviation for a cell.
    
    DCD_i = |Y_i - g(D_i)| where Y_i = S_i / D_i
    
    Higher DCD indicates the cell deviates more from expected service ratio.
    
    Args:
        demand_cell: Demand at cell
        supply_cell: Supply at cell
        g_function: Pre-fitted g(d) function
        eps: Numerical stability
        
    Returns:
        DCD value
    """
    if demand_cell < eps:
        return 0.0
    
    Y = supply_cell / (demand_cell + eps)
    g_d = float(g_function(np.array([demand_cell]))[0])
    
    dcd = abs(Y - g_d)
    return float(dcd)


def compute_all_dcd_scores(
    demand_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function: Callable,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute DCD for all cells.
    
    Args:
        demand_counts: Demand array [grid_x, grid_y]
        supply_counts: Supply array [grid_x, grid_y]
        g_function: Pre-fitted g(d) function
        eps: Numerical stability
        
    Returns:
        DCD array [grid_x, grid_y]
    """
    dcd = np.zeros_like(demand_counts, dtype=float)
    
    # Mask for cells with sufficient demand
    mask = demand_counts > eps
    
    if not mask.any():
        return dcd
    
    # Compute Y = S / D for active cells
    Y = np.zeros_like(demand_counts, dtype=float)
    Y[mask] = supply_counts[mask] / (demand_counts[mask] + eps)
    
    # Compute g(D) for active cells
    D_flat = demand_counts.flatten()
    g_d_flat = g_function(D_flat)
    g_d = g_d_flat.reshape(demand_counts.shape)
    
    # DCD = |Y - g(D)|
    dcd[mask] = np.abs(Y[mask] - g_d[mask])
    
    return dcd


def compute_trajectory_dcd(
    pickup_cell: Tuple[int, int],
    dcd_scores: np.ndarray,
) -> float:
    """
    Compute DCD for a trajectory (based on pickup/demand).
    
    Args:
        pickup_cell: Pickup cell (i, j)
        dcd_scores: DCD array [grid_x, grid_y]
        
    Returns:
        DCD value for trajectory
    """
    pi, pj = pickup_cell
    
    if 0 <= pi < dcd_scores.shape[0] and 0 <= pj < dcd_scores.shape[1]:
        return float(dcd_scores[pi, pj])
    return 0.0


# =============================================================================
# COMBINED ATTRIBUTION
# =============================================================================

def compute_combined_attribution(
    trajectories: List[Dict[str, Any]],
    pickup_counts: np.ndarray,
    dropoff_counts: np.ndarray,
    supply_counts: np.ndarray,
    g_function: Callable,
    lis_weight: float = 0.5,
    dcd_weight: float = 0.5,
    lis_aggregation: str = 'max',
    normalize: bool = True,
) -> AttributionResult:
    """
    Compute combined attribution scores for all trajectories.
    
    Combined Score = w₁ · LIS_τ + w₂ · DCD_τ
    
    Args:
        trajectories: List of trajectory dicts with 'pickup_cell' and 'dropoff_cell'
        pickup_counts: Pickup count array [grid_x, grid_y]
        dropoff_counts: Dropoff count array [grid_x, grid_y]
        supply_counts: Supply array [grid_x, grid_y]
        g_function: Pre-fitted g(d) function
        lis_weight: Weight for LIS (spatial)
        dcd_weight: Weight for DCD (causal)
        lis_aggregation: How to aggregate pickup/dropoff LIS
        normalize: Normalize scores to [0, 1]
        
    Returns:
        AttributionResult with all trajectory scores
    """
    # Compute cell-level scores
    pickup_lis = compute_all_lis_scores(pickup_counts)
    dropoff_lis = compute_all_lis_scores(dropoff_counts)
    dcd_scores = compute_all_dcd_scores(pickup_counts, supply_counts, g_function)
    
    # Compute per-trajectory scores
    trajectory_scores = []
    
    for traj in trajectories:
        traj_id = traj.get('trajectory_id', traj.get('id', len(trajectory_scores)))
        pickup_cell = traj.get('pickup_cell', traj.get('pickup_grid_cell'))
        dropoff_cell = traj.get('dropoff_cell', traj.get('dropoff_grid_cell'))
        
        if pickup_cell is None or dropoff_cell is None:
            continue
        
        # Convert to tuple if needed
        if hasattr(pickup_cell, '__iter__') and not isinstance(pickup_cell, tuple):
            pickup_cell = tuple(pickup_cell)
        if hasattr(dropoff_cell, '__iter__') and not isinstance(dropoff_cell, tuple):
            dropoff_cell = tuple(dropoff_cell)
        
        # LIS for trajectory
        lis = compute_trajectory_lis(pickup_cell, dropoff_cell, pickup_lis, dropoff_lis, lis_aggregation)
        
        # DCD for trajectory
        dcd = compute_trajectory_dcd(pickup_cell, dcd_scores)
        
        trajectory_scores.append(AttributionScores(
            trajectory_id=traj_id,
            lis_score=lis,
            dcd_score=dcd,
            combined_score=0.0,  # Computed after normalization
            pickup_cell=pickup_cell,
            dropoff_cell=dropoff_cell,
            metadata=traj.get('metadata', {}),
        ))
    
    if not trajectory_scores:
        return AttributionResult(
            trajectory_scores=[],
            lis_weight=lis_weight,
            dcd_weight=dcd_weight,
            method='weighted_sum',
        )
    
    # Normalize if requested
    if normalize:
        lis_values = np.array([s.lis_score for s in trajectory_scores])
        dcd_values = np.array([s.dcd_score for s in trajectory_scores])
        
        lis_max = lis_values.max() if lis_values.max() > 0 else 1.0
        dcd_max = dcd_values.max() if dcd_values.max() > 0 else 1.0
        
        for s in trajectory_scores:
            s.lis_score = s.lis_score / lis_max
            s.dcd_score = s.dcd_score / dcd_max
    
    # Compute combined scores
    for s in trajectory_scores:
        s.combined_score = lis_weight * s.lis_score + dcd_weight * s.dcd_score
    
    # Cell statistics
    cell_stats = {
        'pickup_lis_mean': float(pickup_lis.mean()),
        'pickup_lis_max': float(pickup_lis.max()),
        'dropoff_lis_mean': float(dropoff_lis.mean()),
        'dropoff_lis_max': float(dropoff_lis.max()),
        'dcd_mean': float(dcd_scores.mean()),
        'dcd_max': float(dcd_scores.max()),
    }
    
    return AttributionResult(
        trajectory_scores=trajectory_scores,
        lis_weight=lis_weight,
        dcd_weight=dcd_weight,
        method='weighted_sum',
        cell_stats=cell_stats,
    )


# =============================================================================
# TRAJECTORY SELECTION
# =============================================================================

def select_trajectories_for_modification(
    attribution_result: AttributionResult,
    n_trajectories: int = 10,
    selection_method: str = 'top_n',
    selection_params: Optional[Dict[str, Any]] = None,
) -> List[AttributionScores]:
    """
    Select trajectories that need modification to improve fairness.
    
    Args:
        attribution_result: Result from compute_combined_attribution
        n_trajectories: Number of trajectories to select
        selection_method: Method for selection:
            - 'top_n': Select top N by combined score
            - 'threshold': Select all above threshold
            - 'diverse': Select diverse set (avoid same cells)
        selection_params: Additional parameters for selection method
        
    Returns:
        List of selected AttributionScores
    """
    scores = attribution_result.trajectory_scores
    params = selection_params or {}
    
    if selection_method == 'top_n':
        return attribution_result.get_top_n(n_trajectories)
    
    elif selection_method == 'threshold':
        threshold = params.get('threshold', 0.5)
        selected = [s for s in scores if s.combined_score >= threshold]
        return sorted(selected, key=lambda x: x.combined_score, reverse=True)[:n_trajectories]
    
    elif selection_method == 'diverse':
        return _select_diverse(scores, n_trajectories, params)
    
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")


def _select_diverse(
    scores: List[AttributionScores],
    n: int,
    params: Dict[str, Any],
) -> List[AttributionScores]:
    """
    Select diverse trajectories (avoid clustering in same cells).
    
    Uses greedy selection with penalty for cells already selected.
    """
    penalty_factor = params.get('penalty_factor', 0.5)
    
    # Sort by combined score
    sorted_scores = sorted(scores, key=lambda x: x.combined_score, reverse=True)
    
    selected = []
    cell_counts = {}  # Track how many times each cell is selected
    
    for s in sorted_scores:
        if len(selected) >= n:
            break
        
        # Apply penalty based on already-selected cells
        penalty = 0
        if s.pickup_cell:
            penalty += cell_counts.get(s.pickup_cell, 0) * penalty_factor
        if s.dropoff_cell:
            penalty += cell_counts.get(s.dropoff_cell, 0) * penalty_factor
        
        effective_score = s.combined_score - penalty
        
        # Simple greedy: accept if effective score still positive
        if effective_score > 0 or len(selected) == 0:
            selected.append(s)
            
            # Update cell counts
            if s.pickup_cell:
                cell_counts[s.pickup_cell] = cell_counts.get(s.pickup_cell, 0) + 1
            if s.dropoff_cell:
                cell_counts[s.dropoff_cell] = cell_counts.get(s.dropoff_cell, 0) + 1
    
    return selected


# =============================================================================
# GRADIENT-BASED ATTRIBUTION (Alternative Method)
# =============================================================================

if TORCH_AVAILABLE:
    def compute_gradient_based_attribution(
        pickup_coords: 'torch.Tensor',
        dropoff_coords: 'torch.Tensor',
        objective_module: 'torch.nn.Module',
        trajectory_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute attribution using gradient magnitude ||∇_τ L||.
        
        This directly measures each trajectory's influence on the objective.
        
        Args:
            pickup_coords: Pickup coordinates [n_pickups, 2] (requires_grad=True)
            dropoff_coords: Dropoff coordinates [n_dropoffs, 2] (requires_grad=True)
            objective_module: DifferentiableFAMAILObjective module
            trajectory_indices: Optional list mapping coords to trajectory IDs
            
        Returns:
            Dict with gradient-based attribution scores
        """
        import torch
        
        # Ensure gradients enabled
        pickup_coords = pickup_coords.detach().requires_grad_(True)
        dropoff_coords = dropoff_coords.detach().requires_grad_(True)
        
        # Forward pass
        total, _ = objective_module(pickup_coords, dropoff_coords)
        
        # Backward pass
        total.backward()
        
        # Compute gradient magnitudes per coordinate
        pickup_grad_norms = pickup_coords.grad.norm(dim=-1).detach().numpy()
        dropoff_grad_norms = dropoff_coords.grad.norm(dim=-1).detach().numpy()
        
        # Combine (assuming pickup and dropoff correspond)
        n_trajs = min(len(pickup_grad_norms), len(dropoff_grad_norms))
        combined_norms = (pickup_grad_norms[:n_trajs] + dropoff_grad_norms[:n_trajs]) / 2
        
        if trajectory_indices is None:
            trajectory_indices = list(range(n_trajs))
        
        return {
            'trajectory_ids': trajectory_indices[:n_trajs],
            'gradient_magnitudes': combined_norms,
            'pickup_magnitudes': pickup_grad_norms,
            'dropoff_magnitudes': dropoff_grad_norms,
            'rankings': np.argsort(-combined_norms).tolist(),
        }


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def load_trajectories_from_all_trajs(
    filepath: Union[str, Path],
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load trajectories from all_trajs.pkl file.
    
    Args:
        filepath: Path to all_trajs.pkl
        n_samples: Number of trajectories to sample (None = all)
        random_state: Random seed for sampling
        
    Returns:
        List of trajectory dictionaries
    """
    import pickle
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different data formats
    if isinstance(data, list):
        trajectories = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['trajectories', 'trajs', 'data']:
            if key in data:
                trajectories = data[key]
                break
        else:
            # Assume dict values are trajectories
            trajectories = list(data.values())
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    # Sample if requested
    if n_samples is not None and n_samples < len(trajectories):
        np.random.seed(random_state)
        indices = np.random.choice(len(trajectories), size=n_samples, replace=False)
        trajectories = [trajectories[i] for i in indices]
    
    # Standardize format
    result = []
    for i, traj in enumerate(trajectories):
        if isinstance(traj, dict):
            t = traj.copy()
            t['trajectory_id'] = traj.get('trajectory_id', traj.get('id', i))
        else:
            # Assume array-like with coords
            t = {
                'trajectory_id': i,
                'data': traj,
            }
        result.append(t)
    
    return result


def extract_cells_from_trajectories(
    trajectories: List[Dict[str, Any]],
    grid_dims: Tuple[int, int] = (48, 90),
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract pickup and dropoff cells from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        grid_dims: Grid dimensions for clamping
        
    Returns:
        Tuple of (pickup_counts, dropoff_counts, pickup_cells, dropoff_cells)
    """
    pickup_counts = np.zeros(grid_dims, dtype=float)
    dropoff_counts = np.zeros(grid_dims, dtype=float)
    
    pickup_cells = []
    dropoff_cells = []
    
    for traj in trajectories:
        # Try different field names
        pickup = traj.get('pickup_cell') or traj.get('pickup_grid_cell') or traj.get('start_cell')
        dropoff = traj.get('dropoff_cell') or traj.get('dropoff_grid_cell') or traj.get('end_cell')
        
        # Fall back to coords if cells not available
        if pickup is None and 'pickup_coords' in traj:
            coords = traj['pickup_coords']
            pickup = (int(coords[0]) % grid_dims[0], int(coords[1]) % grid_dims[1])
        if dropoff is None and 'dropoff_coords' in traj:
            coords = traj['dropoff_coords']
            dropoff = (int(coords[0]) % grid_dims[0], int(coords[1]) % grid_dims[1])
        
        if pickup is not None:
            pi, pj = int(pickup[0]) % grid_dims[0], int(pickup[1]) % grid_dims[1]
            pickup_counts[pi, pj] += 1
            pickup_cells.append((pi, pj))
            traj['pickup_cell'] = (pi, pj)
        else:
            pickup_cells.append(None)
        
        if dropoff is not None:
            di, dj = int(dropoff[0]) % grid_dims[0], int(dropoff[1]) % grid_dims[1]
            dropoff_counts[di, dj] += 1
            dropoff_cells.append((di, dj))
            traj['dropoff_cell'] = (di, dj)
        else:
            dropoff_cells.append(None)
    
    return pickup_counts, dropoff_counts, pickup_cells, dropoff_cells


def create_mock_supply_data(
    pickup_counts: np.ndarray,
    supply_factor: float = 0.8,
    noise_factor: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Create mock supply data based on pickup demand.
    
    Supply = supply_factor * demand + noise
    
    Args:
        pickup_counts: Demand (pickup) counts
        supply_factor: Base supply/demand ratio
        noise_factor: Noise level
        random_state: Random seed
        
    Returns:
        Supply array [grid_x, grid_y]
    """
    np.random.seed(random_state)
    
    supply = supply_factor * pickup_counts
    noise = noise_factor * pickup_counts.mean() * np.random.randn(*pickup_counts.shape)
    supply = np.maximum(supply + noise, 0)
    
    return supply
