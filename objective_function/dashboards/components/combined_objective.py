"""
Combined Objective Function Module.

This module implements the differentiable combined objective function
for FAMAIL trajectory optimization:

    L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity

All components are end-to-end differentiable to enable gradient-based
trajectory modification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Any, List, Union
from dataclasses import dataclass
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
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ObjectiveResult:
    """Result from objective function computation."""
    total: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    has_gradients: bool
    gradient_stats: Optional[Dict[str, Any]] = None


# =============================================================================
# DIFFERENTIABLE COMBINED OBJECTIVE FUNCTION
# =============================================================================

if TORCH_AVAILABLE:
    class DifferentiableFAMAILObjective(nn.Module):
        """
        Combined differentiable objective function for FAMAIL.
        
        L = α₁·F_causal + α₂·F_spatial + α₃·F_fidelity
        
        All terms output values in [0, 1] where higher is better.
        The optimization direction is maximization.
        
        Attributes:
            alpha_spatial: Weight for spatial fairness term
            alpha_causal: Weight for causal fairness term
            alpha_fidelity: Weight for fidelity term
            grid_dims: Grid dimensions (x, y)
            soft_assign: SoftCellAssignment module (shared)
        """
        
        def __init__(
            self,
            alpha_spatial: float = 0.33,
            alpha_causal: float = 0.33,
            alpha_fidelity: float = 0.34,
            grid_dims: Tuple[int, int] = (48, 90),
            neighborhood_size: int = 5,
            temperature: float = 1.0,
            g_function: Optional[Callable] = None,
            discriminator: Optional[nn.Module] = None,
            eps: float = 1e-8,
        ):
            """
            Initialize the combined objective function.
            
            Args:
                alpha_spatial: Weight for F_spatial
                alpha_causal: Weight for F_causal
                alpha_fidelity: Weight for F_fidelity
                grid_dims: Spatial grid dimensions
                neighborhood_size: Soft assignment neighborhood size
                temperature: Soft assignment temperature
                g_function: Pre-fitted g(d) function for causal term (frozen)
                discriminator: Pre-trained discriminator for fidelity (frozen)
                eps: Numerical stability constant
            """
            super().__init__()
            
            self.alpha_spatial = alpha_spatial
            self.alpha_causal = alpha_causal
            self.alpha_fidelity = alpha_fidelity
            self.grid_dims = grid_dims
            self.eps = eps
            
            # Import soft cell assignment
            from soft_cell_assignment import SoftCellAssignment
            
            # Shared soft cell assignment module
            self.soft_assign = SoftCellAssignment(
                grid_dims=grid_dims,
                neighborhood_size=neighborhood_size,
                initial_temperature=temperature,
            )
            
            # Store g(d) function (frozen - no parameters)
            self.g_function = g_function
            
            # Store discriminator (frozen - set to eval mode)
            self.discriminator = discriminator
            if self.discriminator is not None:
                self.discriminator.eval()
                for param in self.discriminator.parameters():
                    param.requires_grad = False
        
        def set_temperature(self, temperature: float) -> None:
            """Update soft assignment temperature."""
            self.soft_assign.set_temperature(temperature)
        
        def compute_soft_counts(
            self,
            coords: torch.Tensor,
            original_cells: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute soft counts from coordinates.
            
            Args:
                coords: Coordinates [n_points, 2] with (x, y)
                original_cells: Original cell indices [n_points, 2] (auto-computed if None)
                
            Returns:
                Soft counts per cell [grid_x, grid_y]
            """
            n_points = coords.shape[0]
            
            if original_cells is None:
                # Compute original cells from coordinates
                original_cells = coords.floor().long()
                original_cells = original_cells.clamp(
                    min=torch.tensor([0, 0], device=coords.device),
                    max=torch.tensor([self.grid_dims[0]-1, self.grid_dims[1]-1], device=coords.device)
                )
            
            # Get soft assignment for each point
            soft_counts = torch.zeros(self.grid_dims, device=coords.device)
            k = self.soft_assign.k
            
            for i in range(n_points):
                loc = coords[i:i+1]
                cell = original_cells[i:i+1]
                
                # Get probability distribution over neighborhood
                probs = self.soft_assign(loc.float(), cell.float())  # [1, ns, ns]
                
                # Add to soft counts at neighborhood cells
                cx, cy = cell[0, 0].item(), cell[0, 1].item()
                
                for di in range(-k, k+1):
                    for dj in range(-k, k+1):
                        ni, nj = int(cx + di), int(cy + dj)
                        if 0 <= ni < self.grid_dims[0] and 0 <= nj < self.grid_dims[1]:
                            soft_counts[ni, nj] += probs[0, di + k, dj + k]
            
            return soft_counts
        
        def compute_spatial_fairness(
            self,
            pickup_counts: torch.Tensor,
            dropoff_counts: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute differentiable spatial fairness (pairwise Gini).
            
            F_spatial = 1 - (G_pickup + G_dropoff) / 2
            
            Args:
                pickup_counts: Soft pickup counts [grid_x, grid_y]
                dropoff_counts: Soft dropoff counts [grid_x, grid_y]
                
            Returns:
                Spatial fairness score [0, 1]
            """
            # Flatten counts
            pickup_flat = pickup_counts.flatten()
            dropoff_flat = dropoff_counts.flatten()
            
            # Compute pairwise Gini for each
            gini_pickup = self._pairwise_gini(pickup_flat)
            gini_dropoff = self._pairwise_gini(dropoff_flat)
            
            # Spatial fairness = 1 - average Gini
            f_spatial = 1.0 - 0.5 * (gini_pickup + gini_dropoff)
            
            return f_spatial
        
        def _pairwise_gini(self, values: torch.Tensor) -> torch.Tensor:
            """Compute Gini coefficient using pairwise formula (differentiable)."""
            n = values.numel()
            if n <= 1:
                return torch.tensor(0.0, device=values.device)
            
            mean_val = values.mean() + self.eps
            
            # Pairwise absolute differences
            diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
            gini = diff_matrix.sum() / (2 * n * n * mean_val)
            
            return torch.clamp(gini, 0.0, 1.0)
        
        def compute_causal_fairness(
            self,
            pickup_counts: torch.Tensor,
            supply_tensor: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute differentiable causal fairness (R²).
            
            F_causal = 1 - Var(R) / Var(Y) where R = Y - g(D)
            
            Args:
                pickup_counts: Soft demand counts [grid_x, grid_y]
                supply_tensor: Supply per cell [grid_x, grid_y] (fixed)
                
            Returns:
                Causal fairness score [0, 1]
            """
            if self.g_function is None:
                # Return default value if g(d) not provided
                return torch.tensor(0.5, device=pickup_counts.device)
            
            # Flatten tensors
            demand = pickup_counts.flatten()
            supply = supply_tensor.flatten()
            
            # Filter cells with sufficient demand
            mask = demand > 0.1
            
            if mask.sum() < 2:
                return torch.tensor(0.5, device=pickup_counts.device)
            
            D = demand[mask]
            S = supply[mask]
            
            # Compute Y = S / D (differentiable)
            Y = S / (D + self.eps)
            
            # Get expected Y from frozen g(d)
            with torch.no_grad():
                D_np = D.detach().cpu().numpy()
                g_d_np = self.g_function(D_np)
                g_d = torch.tensor(g_d_np, device=Y.device, dtype=Y.dtype)
            
            # Compute residual (differentiable through Y)
            R = Y - g_d
            
            # R² computation
            # Note: R² = 1 - SS_res/SS_tot = 1 - Var(R)/Var(Y)
            # If g(d) explains variance well, Var(R) << Var(Y), so R² → 1
            # If g(d) is poor, Var(R) ≈ Var(Y), so R² → 0
            var_Y = Y.var() + self.eps
            var_R = R.var()
            
            # Handle edge case: if var_R > var_Y, R² would be negative
            # This can happen if g(d) is a poor fit
            r_squared = 1.0 - var_R / var_Y
            
            # Store debug info as attributes for inspection
            self._last_causal_debug = {
                'n_active_cells': int(mask.sum().item()),
                'demand_range': (D.min().item(), D.max().item()),
                'supply_range': (S.min().item(), S.max().item()),
                'Y_range': (Y.min().item(), Y.max().item()),
                'g_d_range': (g_d.min().item(), g_d.max().item()),
                'var_Y': var_Y.item(),
                'var_R': var_R.item(),
                'r_squared_raw': r_squared.item(),
            }
            
            return torch.clamp(r_squared, 0.0, 1.0)
        
        def compute_fidelity(
            self,
            trajectory_features: torch.Tensor,
            reference_features: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Compute differentiable fidelity using discriminator.
            
            Args:
                trajectory_features: Modified trajectory features [batch, seq_len, 4]
                reference_features: Original trajectory features [batch, seq_len, 4]
                
            Returns:
                Fidelity score [0, 1]
            """
            if self.discriminator is None:
                # Return default value if discriminator not provided
                return torch.tensor(0.5, device=trajectory_features.device)
            
            if reference_features is None:
                reference_features = trajectory_features.detach()
            
            # Get discriminator similarity score
            # Note: discriminator computes same-driver probability
            with torch.enable_grad():
                similarity = self.discriminator(trajectory_features, reference_features)
            
            # Average across batch
            f_fidelity = similarity.mean()
            
            return torch.clamp(f_fidelity, 0.0, 1.0)
        
        def forward(
            self,
            pickup_coords: torch.Tensor,
            dropoff_coords: torch.Tensor,
            supply_tensor: Optional[torch.Tensor] = None,
            demand_tensor: Optional[torch.Tensor] = None,
            trajectory_features: Optional[torch.Tensor] = None,
            reference_features: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            """
            Compute combined objective function.
            
            Args:
                pickup_coords: Pickup coordinates [n_pickups, 2]
                dropoff_coords: Dropoff coordinates [n_dropoffs, 2]
                supply_tensor: Supply per cell [grid_x, grid_y] (for causal term)
                demand_tensor: Optional demand tensor for causal fairness.
                    If provided, uses this instead of pickup_soft_counts for causal term.
                    This allows using real demand data at the correct temporal scale
                    while still computing spatial fairness from trajectory coordinates.
                trajectory_features: Trajectory features [batch, seq_len, 4] (for fidelity)
                reference_features: Reference features for fidelity comparison
                
            Returns:
                total_objective: Combined objective value (higher = better)
                term_values: Dict with individual term values
            """
            device = pickup_coords.device
            
            # Compute soft counts from trajectory coordinates
            # These are used for SPATIAL fairness (what we're optimizing via trajectory modification)
            pickup_soft_counts = self.compute_soft_counts(pickup_coords)
            dropoff_soft_counts = self.compute_soft_counts(dropoff_coords)
            
            # Compute spatial fairness using trajectory soft counts
            f_spatial = self.compute_spatial_fairness(pickup_soft_counts, dropoff_soft_counts)
            
            # Compute causal fairness
            # Use demand_tensor if provided (real demand data at correct temporal scale)
            # Otherwise fall back to pickup_soft_counts (trajectory-derived demand)
            if supply_tensor is not None:
                causal_demand = demand_tensor if demand_tensor is not None else pickup_soft_counts
                f_causal = self.compute_causal_fairness(causal_demand, supply_tensor)
            else:
                f_causal = torch.tensor(0.5, device=device)
            
            # Compute fidelity
            if trajectory_features is not None:
                f_fidelity = self.compute_fidelity(trajectory_features, reference_features)
            else:
                f_fidelity = torch.tensor(0.5, device=device)
            
            # Combined objective
            total = (
                self.alpha_causal * f_causal +
                self.alpha_spatial * f_spatial +
                self.alpha_fidelity * f_fidelity
            )
            
            term_values = {
                'f_spatial': f_spatial,
                'f_causal': f_causal,
                'f_fidelity': f_fidelity,
                'alpha_spatial': self.alpha_spatial,
                'alpha_causal': self.alpha_causal,
                'alpha_fidelity': self.alpha_fidelity,
            }
            
            return total, term_values
        
        @staticmethod
        def verify_gradients(
            grid_dims: Tuple[int, int] = (10, 10),
            n_pickups: int = 20,
            n_dropoffs: int = 20,
            temperature: float = 1.0,
        ) -> Dict[str, Any]:
            """
            Verify that gradients flow through all components.
            
            Returns:
                Dict with verification results
            """
            import torch
            
            # Create module (no g_function or discriminator for basic test)
            module = DifferentiableFAMAILObjective(
                grid_dims=grid_dims,
                temperature=temperature,
                alpha_spatial=0.5,
                alpha_causal=0.0,  # Disable causal for simple test
                alpha_fidelity=0.0,  # Disable fidelity for simple test
            )
            
            # Create test coordinates with gradients
            torch.manual_seed(42)
            pickup_coords = torch.rand(n_pickups, 2) * torch.tensor(grid_dims, dtype=torch.float32)
            dropoff_coords = torch.rand(n_dropoffs, 2) * torch.tensor(grid_dims, dtype=torch.float32)
            
            pickup_coords.requires_grad_(True)
            dropoff_coords.requires_grad_(True)
            
            # Forward pass
            total, term_values = module(pickup_coords, dropoff_coords)
            
            # Backward pass
            total.backward()
            
            # Check gradients
            results = {
                'passed': True,
                'total_objective': total.item(),
                'f_spatial': term_values['f_spatial'].item(),
            }
            
            # Check pickup gradients
            if pickup_coords.grad is None:
                results['passed'] = False
                results['pickup_grad_error'] = 'No gradient computed'
            elif torch.isnan(pickup_coords.grad).any():
                results['passed'] = False
                results['pickup_grad_error'] = 'NaN in gradients'
            elif torch.isinf(pickup_coords.grad).any():
                results['passed'] = False
                results['pickup_grad_error'] = 'Inf in gradients'
            else:
                results['pickup_grad_stats'] = {
                    'mean': pickup_coords.grad.mean().item(),
                    'std': pickup_coords.grad.std().item(),
                    'min': pickup_coords.grad.min().item(),
                    'max': pickup_coords.grad.max().item(),
                    'nonzero_count': (pickup_coords.grad.abs() > 1e-10).sum().item(),
                }
            
            # Check dropoff gradients
            if dropoff_coords.grad is None:
                results['passed'] = False
                results['dropoff_grad_error'] = 'No gradient computed'
            elif torch.isnan(dropoff_coords.grad).any():
                results['passed'] = False
                results['dropoff_grad_error'] = 'NaN in gradients'
            else:
                results['dropoff_grad_stats'] = {
                    'mean': dropoff_coords.grad.mean().item(),
                    'std': dropoff_coords.grad.std().item(),
                    'min': dropoff_coords.grad.min().item(),
                    'max': dropoff_coords.grad.max().item(),
                    'nonzero_count': (dropoff_coords.grad.abs() > 1e-10).sum().item(),
                }
            
            return results


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def compute_combined_objective(
    pickup_coords: np.ndarray,
    dropoff_coords: np.ndarray,
    alpha_spatial: float = 0.33,
    alpha_causal: float = 0.33,
    alpha_fidelity: float = 0.34,
    grid_dims: Tuple[int, int] = (48, 90),
    supply_data: Optional[np.ndarray] = None,
    g_function: Optional[Callable] = None,
) -> ObjectiveResult:
    """
    Compute combined objective function (NumPy interface).
    
    Args:
        pickup_coords: Pickup coordinates [n_pickups, 2]
        dropoff_coords: Dropoff coordinates [n_dropoffs, 2]
        alpha_*: Weights for each term
        grid_dims: Grid dimensions
        supply_data: Supply per cell (for causal term)
        g_function: Pre-fitted g(d) function (for causal term)
        
    Returns:
        ObjectiveResult with total and per-term values
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for objective computation")
    
    import torch
    
    # Convert to tensors
    pickup_tensor = torch.tensor(pickup_coords, dtype=torch.float32)
    dropoff_tensor = torch.tensor(dropoff_coords, dtype=torch.float32)
    
    supply_tensor = None
    if supply_data is not None:
        supply_tensor = torch.tensor(supply_data, dtype=torch.float32)
    
    # Create module
    module = DifferentiableFAMAILObjective(
        alpha_spatial=alpha_spatial,
        alpha_causal=alpha_causal,
        alpha_fidelity=alpha_fidelity,
        grid_dims=grid_dims,
        g_function=g_function,
    )
    
    # Compute
    with torch.no_grad():
        total, term_values = module(
            pickup_tensor,
            dropoff_tensor,
            supply_tensor=supply_tensor,
        )
    
    return ObjectiveResult(
        total=total.item(),
        f_spatial=term_values['f_spatial'].item(),
        f_causal=term_values['f_causal'].item(),
        f_fidelity=term_values['f_fidelity'].item(),
        has_gradients=False,
    )


def create_default_g_function(
    demands: np.ndarray,
    ratios: np.ndarray,
    method: str = 'isotonic',
) -> Callable:
    """
    Create g(d) function from demand-ratio data.
    
    Args:
        demands: Demand values
        ratios: Service ratio values (Y = S/D)
        method: Estimation method ('isotonic', 'binning', 'linear')
        
    Returns:
        g(d) function
    """
    from sklearn.isotonic import IsotonicRegression
    from scipy.stats import binned_statistic
    from sklearn.linear_model import LinearRegression
    
    if method == 'isotonic':
        model = IsotonicRegression(out_of_bounds='clip')
        model.fit(demands, ratios)
        return lambda d: model.predict(np.atleast_1d(d))
    
    elif method == 'binning':
        # Create 10 bins
        bin_means, bin_edges, _ = binned_statistic(demands, ratios, statistic='mean', bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        def g(d):
            d_arr = np.atleast_1d(d)
            result = np.zeros_like(d_arr, dtype=float)
            for i, d_val in enumerate(d_arr):
                idx = np.searchsorted(bin_centers, d_val)
                idx = min(max(0, idx), len(bin_means) - 1)
                result[i] = bin_means[idx] if not np.isnan(bin_means[idx]) else np.nanmean(bin_means)
            return result
        return g
    
    elif method == 'linear':
        model = LinearRegression()
        model.fit(demands.reshape(-1, 1), ratios)
        return lambda d: model.predict(np.atleast_1d(d).reshape(-1, 1))
    
    else:
        raise ValueError(f"Unknown method: {method}")
