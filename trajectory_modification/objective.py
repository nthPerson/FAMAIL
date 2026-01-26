"""
FAMAIL Objective Function for trajectory modification.

Implements L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity

All terms are differentiable to enable gradient-based trajectory modification.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FAMAILObjective(nn.Module):
    """
    Combined objective function for FAMAIL trajectory modification.
    
    L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
    
    The objective is maximized: higher L means fairer outcomes.
    """
    
    def __init__(
        self,
        alpha_spatial: float = 0.33,
        alpha_causal: float = 0.33,
        alpha_fidelity: float = 0.34,
        grid_dims: Tuple[int, int] = (48, 90),
        g_function: Optional[Callable] = None,
        discriminator: Optional[nn.Module] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_causal = alpha_causal
        self.alpha_fidelity = alpha_fidelity
        self.grid_dims = grid_dims
        self.g_function = g_function
        self.discriminator = discriminator
        self.eps = eps
        
        # Debug storage
        self.last_debug = {}
    
    def compute_spatial_fairness(
        self,
        pickup_counts: torch.Tensor,
        active_taxis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute spatial fairness as F_spatial = 1 - Gini(DSR).
        
        DSR_i = pickup_i / active_taxis_i (normalized service rate)
        """
        if active_taxis is not None:
            # Normalize by active taxis
            mask = active_taxis > self.eps
            dsr = torch.zeros_like(pickup_counts)
            dsr[mask] = pickup_counts[mask] / active_taxis[mask]
            values = dsr.flatten()
        else:
            values = pickup_counts.flatten()
        
        # Gini coefficient via pairwise differences
        n = values.numel()
        if n <= 1:
            return torch.tensor(1.0, device=pickup_counts.device)
        
        mean_val = values.mean() + self.eps
        diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
        gini = diff_matrix.sum() / (2 * n * n * mean_val)
        gini = torch.clamp(gini, 0.0, 1.0)
        
        f_spatial = 1.0 - gini
        
        self.last_debug['gini'] = gini.item()
        self.last_debug['f_spatial'] = f_spatial.item()
        
        return f_spatial
    
    def compute_causal_fairness(
        self,
        demand: torch.Tensor,
        supply: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal fairness as F_causal = max(0, R²).
        
        R² = 1 - Var(Y - g(D)) / Var(Y) where Y = S/D
        """
        if self.g_function is None:
            return torch.tensor(0.5, device=demand.device)
        
        # Filter active cells
        mask = demand > 0.1
        if mask.sum() < 2:
            return torch.tensor(0.5, device=demand.device)
        
        D = demand[mask]
        S = supply[mask]
        Y = S / (D + self.eps)
        
        # Get g(D) predictions (frozen)
        with torch.no_grad():
            D_np = D.detach().cpu().numpy()
            g_d = torch.tensor(self.g_function(D_np), device=Y.device, dtype=Y.dtype)
        
        # R² computation
        R = Y - g_d
        var_Y = Y.var() + self.eps
        var_R = R.var()
        r_squared = 1.0 - var_R / var_Y
        
        f_causal = torch.clamp(r_squared, 0.0, 1.0)
        
        self.last_debug['r_squared'] = r_squared.item()
        self.last_debug['f_causal'] = f_causal.item()
        
        return f_causal
    
    def compute_fidelity(
        self,
        tau_features: torch.Tensor,
        tau_prime_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fidelity as F_fidelity = f(τ, τ').
        
        The discriminator similarity indicates how well the modified
        trajectory preserves the original agent's behavior.
        """
        if self.discriminator is None:
            return torch.tensor(0.5, device=tau_features.device)
        
        # Get discriminator similarity (with gradients)
        similarity = self.discriminator(tau_features, tau_prime_features)
        f_fidelity = similarity.mean()
        
        self.last_debug['f_fidelity'] = f_fidelity.item()
        
        return torch.clamp(f_fidelity, 0.0, 1.0)
    
    def forward(
        self,
        pickup_counts: torch.Tensor,
        supply: torch.Tensor,
        tau_features: Optional[torch.Tensor] = None,
        tau_prime_features: Optional[torch.Tensor] = None,
        active_taxis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined objective L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity.
        
        Args:
            pickup_counts: Pickup counts per cell [grid_x, grid_y]
            supply: Supply (active taxis) per cell [grid_x, grid_y]
            tau_features: Original trajectory features [1, seq_len, 4]
            tau_prime_features: Modified trajectory features [1, seq_len, 4]
            active_taxis: Active taxis for DSR normalization [grid_x, grid_y]
            
        Returns:
            total: Combined objective value
            terms: Dict with individual term values
        """
        device = pickup_counts.device
        
        # Spatial fairness
        f_spatial = self.compute_spatial_fairness(pickup_counts, active_taxis)
        
        # Causal fairness
        f_causal = self.compute_causal_fairness(pickup_counts, supply)
        
        # Fidelity
        if tau_features is not None and tau_prime_features is not None:
            f_fidelity = self.compute_fidelity(tau_features, tau_prime_features)
        else:
            f_fidelity = torch.tensor(0.5, device=device)
            self.last_debug['f_fidelity'] = 0.5
        
        # Combined objective
        total = (
            self.alpha_spatial * f_spatial +
            self.alpha_causal * f_causal +
            self.alpha_fidelity * f_fidelity
        )
        
        self.last_debug['total'] = total.item()
        
        terms = {
            'f_spatial': f_spatial,
            'f_causal': f_causal,
            'f_fidelity': f_fidelity,
        }
        
        return total, terms
