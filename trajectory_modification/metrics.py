"""
Global metrics computation for FAMAIL trajectory modification.

Provides tools to track system-wide fairness metrics as trajectories are modified.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class FairnessSnapshot:
    """Snapshot of fairness metrics at a point in time."""
    gini_coefficient: float
    r_squared: float
    mean_fidelity: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    combined_objective: float
    num_trajectories_modified: int
    
    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            'gini': self.gini_coefficient,
            'r_squared': self.r_squared,
            'mean_fidelity': self.mean_fidelity,
            'f_spatial': self.f_spatial,
            'f_causal': self.f_causal,
            'f_fidelity': self.f_fidelity,
            'combined': self.combined_objective,
            'num_modified': self.num_trajectories_modified,
        }


class GlobalMetrics:
    """
    Tracks global fairness metrics across the entire taxi system.
    
    Maintains running counts of pickups, supply, and active taxis per cell,
    and computes fairness metrics on demand.
    """
    
    def __init__(
        self,
        grid_dims: Tuple[int, int] = (48, 90),
        g_function=None,
        alpha_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34),
    ):
        self.grid_dims = grid_dims
        self.g_function = g_function
        self.alpha_spatial, self.alpha_causal, self.alpha_fidelity = alpha_weights
        self.eps = 1e-8
        
        # Initialize counts as numpy arrays
        self.pickup_counts = np.zeros(grid_dims, dtype=np.float32)
        self.supply_counts = np.zeros(grid_dims, dtype=np.float32)
        self.active_taxis = np.zeros(grid_dims, dtype=np.float32)
        
        # Fidelity tracking
        self.fidelity_scores: List[float] = []
        self.num_modified = 0
        
        # History
        self.snapshots: List[FairnessSnapshot] = []
    
    def initialize_from_data(
        self,
        pickup_counts: np.ndarray,
        supply_counts: np.ndarray,
        active_taxis: Optional[np.ndarray] = None,
    ):
        """Initialize counts from existing data."""
        self.pickup_counts = pickup_counts.astype(np.float32).copy()
        self.supply_counts = supply_counts.astype(np.float32).copy()
        if active_taxis is not None:
            self.active_taxis = active_taxis.astype(np.float32).copy()
        else:
            self.active_taxis = np.ones_like(self.pickup_counts)
    
    def update_pickup(
        self,
        old_cell: Tuple[int, int],
        new_cell: Tuple[int, int],
        fidelity_score: Optional[float] = None,
    ):
        """Update counts when a pickup location changes."""
        ox, oy = old_cell
        nx, ny = new_cell
        
        # Move pickup count
        if 0 <= ox < self.grid_dims[0] and 0 <= oy < self.grid_dims[1]:
            self.pickup_counts[ox, oy] = max(0, self.pickup_counts[ox, oy] - 1)
        if 0 <= nx < self.grid_dims[0] and 0 <= ny < self.grid_dims[1]:
            self.pickup_counts[nx, ny] += 1
        
        self.num_modified += 1
        
        if fidelity_score is not None:
            self.fidelity_scores.append(fidelity_score)
    
    def compute_gini(self) -> float:
        """Compute Gini coefficient of DSR (demand-supply ratio)."""
        # DSR = pickups / active_taxis
        mask = self.active_taxis > self.eps
        dsr = np.zeros_like(self.pickup_counts)
        dsr[mask] = self.pickup_counts[mask] / self.active_taxis[mask]
        
        values = dsr.flatten()
        n = len(values)
        if n <= 1:
            return 0.0
        
        mean_val = values.mean() + self.eps
        diff_matrix = np.abs(values[:, None] - values[None, :])
        gini = diff_matrix.sum() / (2 * n * n * mean_val)
        
        return float(np.clip(gini, 0.0, 1.0))
    
    def compute_r_squared(self) -> float:
        """Compute R² for causal fairness."""
        if self.g_function is None:
            return 0.5
        
        # Filter active cells
        mask = self.pickup_counts.flatten() > 0.1
        if mask.sum() < 2:
            return 0.5
        
        D = self.pickup_counts.flatten()[mask]  # Demand proxy
        S = self.supply_counts.flatten()[mask]  # Supply
        Y = S / (D + self.eps)  # Outcome: supply per demand
        
        g_d = self.g_function(D)
        R = Y - g_d
        
        var_Y = Y.var() + self.eps
        var_R = R.var()
        r_squared = 1.0 - var_R / var_Y
        
        return float(np.clip(r_squared, 0.0, 1.0))
    
    def compute_mean_fidelity(self) -> float:
        """Compute mean fidelity across all modified trajectories."""
        if not self.fidelity_scores:
            return 0.5
        return float(np.mean(self.fidelity_scores))
    
    def compute_snapshot(self) -> FairnessSnapshot:
        """Compute current fairness snapshot."""
        gini = self.compute_gini()
        r_squared = self.compute_r_squared()
        mean_fidelity = self.compute_mean_fidelity()
        
        f_spatial = 1.0 - gini
        f_causal = max(0.0, r_squared)
        f_fidelity = mean_fidelity
        
        combined = (
            self.alpha_spatial * f_spatial +
            self.alpha_causal * f_causal +
            self.alpha_fidelity * f_fidelity
        )
        
        snapshot = FairnessSnapshot(
            gini_coefficient=gini,
            r_squared=r_squared,
            mean_fidelity=mean_fidelity,
            f_spatial=f_spatial,
            f_causal=f_causal,
            f_fidelity=f_fidelity,
            combined_objective=combined,
            num_trajectories_modified=self.num_modified,
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_improvement(self) -> Dict[str, float]:
        """Get improvement from first to last snapshot."""
        if len(self.snapshots) < 2:
            return {'delta_gini': 0.0, 'delta_combined': 0.0}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        return {
            'delta_gini': last.gini_coefficient - first.gini_coefficient,
            'delta_combined': last.combined_objective - first.combined_objective,
            'delta_spatial': last.f_spatial - first.f_spatial,
            'delta_causal': last.f_causal - first.f_causal,
            'delta_fidelity': last.f_fidelity - first.f_fidelity,
        }
    
    def to_tensors(self, device: str = 'cpu') -> Dict[str, 'torch.Tensor']:
        """Convert counts to PyTorch tensors."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        return {
            'pickup_counts': torch.tensor(self.pickup_counts, device=device, dtype=torch.float32),
            'supply_counts': torch.tensor(self.supply_counts, device=device, dtype=torch.float32),
            'active_taxis': torch.tensor(self.active_taxis, device=device, dtype=torch.float32),
        }
