"""
FAMAIL Objective Function for trajectory modification.

Implements L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity

All terms are differentiable to enable gradient-based trajectory modification.

The objective function uses soft cell assignment to ensure differentiability
when optimizing trajectory locations via gradient descent.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, Callable, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MissingDataError(Exception):
    """Raised when required data is missing for objective computation."""
    pass


class MissingComponentError(Exception):
    """Raised when a required component (g_function, discriminator) is not provided."""
    pass


class InsufficientDataError(Exception):
    """Raised when there is insufficient data for meaningful computation."""
    pass


class FAMAILObjective(nn.Module):
    """
    Combined objective function for FAMAIL trajectory modification.
    
    L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
    
    The objective is maximized: higher L means fairer outcomes.
    
    This implementation uses soft cell assignment for differentiability,
    allowing gradient-based trajectory optimization.
    
    Attributes:
        alpha_spatial: Weight for spatial fairness term
        alpha_causal: Weight for causal fairness term  
        alpha_fidelity: Weight for fidelity term
        grid_dims: Grid dimensions (x, y)
        soft_assign: SoftCellAssignment module for differentiable grid assignment
        g_function: Pre-fitted g(d) function for causal term (frozen)
        discriminator: Pre-trained discriminator for fidelity (frozen)
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
        Initialize the FAMAIL objective function.
        
        Args:
            alpha_spatial: Weight for F_spatial (default: 0.33)
            alpha_causal: Weight for F_causal (default: 0.33)
            alpha_fidelity: Weight for F_fidelity (default: 0.34)
            grid_dims: Spatial grid dimensions (x_cells, y_cells)
            neighborhood_size: Size of soft assignment neighborhood (must be odd)
            temperature: Initial temperature for soft cell assignment
            g_function: Pre-fitted g(d) function for causal fairness.
                       Must be provided for causal fairness computation.
            discriminator: Pre-trained discriminator for fidelity term.
                          Must be provided for fidelity computation.
            eps: Numerical stability constant
        """
        super().__init__()
        self.alpha_spatial = alpha_spatial
        self.alpha_causal = alpha_causal
        self.alpha_fidelity = alpha_fidelity
        self.grid_dims = grid_dims
        self.eps = eps
        
        # Import and initialize soft cell assignment
        try:
            from objective_function.soft_cell_assignment import SoftCellAssignment
            self.soft_assign = SoftCellAssignment(
                grid_dims=grid_dims,
                neighborhood_size=neighborhood_size,
                initial_temperature=temperature,
            )
        except ImportError:
            raise ImportError(
                "SoftCellAssignment module not found. "
                "Please ensure objective_function.soft_cell_assignment is available."
            )
        
        # Store g(d) function (frozen - no parameters)
        self.g_function = g_function
        
        # Store discriminator (frozen - set to eval mode)
        self.discriminator = discriminator
        if self.discriminator is not None:
            self.discriminator.eval()
            for param in self.discriminator.parameters():
                param.requires_grad = False
        
        # Debug storage for inspection
        self.last_debug = {}
        self._last_spatial_debug = {}
        self._last_causal_debug = {}
    
    def set_temperature(self, temperature: float) -> None:
        """
        Update the soft assignment temperature.
        
        Lower temperature = sharper assignment (closer to hard assignment)
        Higher temperature = softer assignment (more gradient flow)
        
        Args:
            temperature: New temperature value (must be > 0)
        """
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.soft_assign.set_temperature(temperature)
    
    def get_annealed_temperature(
        self,
        iteration: int,
        total_iterations: int,
        tau_max: float = 1.0,
        tau_min: float = 0.1,
    ) -> float:
        """
        Compute annealed temperature for a given iteration.
        
        Uses exponential annealing: τ_t = τ_max * (τ_min/τ_max)^(t/T)
        
        This allows starting with soft assignments (high temperature) and
        gradually transitioning to harder assignments (low temperature).
        
        Args:
            iteration: Current iteration (0-indexed)
            total_iterations: Total number of iterations
            tau_max: Starting temperature
            tau_min: Final temperature
            
        Returns:
            Annealed temperature value
        """
        return self.soft_assign.get_annealed_temperature(
            iteration, total_iterations, tau_max, tau_min
        )
    
    def compute_soft_counts(
        self,
        coords: torch.Tensor,
        original_cells: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute soft counts from continuous coordinates.
        
        This is the key differentiable operation that allows gradient-based
        optimization of trajectory locations.
        
        Args:
            coords: Continuous coordinates [n_points, 2] with (x, y)
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
    
    def _pairwise_gini(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute Gini coefficient using pairwise formula (differentiable).
        
        The pairwise Gini formula is:
            G = Σᵢ Σⱼ |xᵢ - xⱼ| / (2n²μ)
        
        where n = number of values and μ = mean of values.
        
        This formulation is fully differentiable since it only uses
        element-wise operations and reductions.
        
        Args:
            values: Flattened tensor of values to compute Gini over
            
        Returns:
            Gini coefficient in [0, 1] where 0 = perfect equality
        """
        # torch.numel() returns the total number of elements in the tensor
        n = values.numel()
        if n <= 1:
            return torch.tensor(0.0, device=values.device)
        
        mean_val = values.mean() + self.eps
        
        # Pairwise absolute differences: creates n×n matrix of |xᵢ - xⱼ|
        diff_matrix = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
        gini = diff_matrix.sum() / (2 * n * n * mean_val)
        
        return torch.clamp(gini, 0.0, 1.0)

    def compute_spatial_fairness(
        self,
        pickup_counts: torch.Tensor,
        dropoff_counts: torch.Tensor,
        active_taxis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial fairness: F_spatial = 1 - 0.5 * (Gini(DSR) + Gini(ASR))
        
        This computes the Gini coefficients on normalized service rates:
        - DSR (Departure Service Rate) = pickups / active_taxis
        - ASR (Arrival Service Rate) = dropoffs / active_taxis
        
        Normalizing by active taxis measures true service inequality, accounting
        for taxi availability in each cell.
        
        Args:
            pickup_counts: Pickup counts per cell [grid_x, grid_y]
            dropoff_counts: Dropoff counts per cell [grid_x, grid_y]
            active_taxis: Active taxis per cell [grid_x, grid_y].
                         REQUIRED - spatial fairness must be normalized by taxi availability.
            
        Returns:
            Spatial fairness score in [0, 1] where 1 = perfect fairness
            
        Raises:
            MissingDataError: If active_taxis is not provided
        """
        # Active taxis mask to avoid division by zero
        active_mask = active_taxis > self.eps
        n_active = active_mask.sum().item()
        
        if n_active == 0:
            raise InsufficientDataError(
                "No cells with active taxis found. Cannot compute spatial fairness. "
                "Ensure active_taxis tensor contains positive values."
            )
        
        # Compute DSR (Departure Service Rate) = pickups / active_taxis
        dsr = torch.zeros_like(pickup_counts)
        dsr[active_mask] = pickup_counts[active_mask] / active_taxis[active_mask]
        
        # Compute ASR (Arrival Service Rate) = dropoffs / active_taxis  
        asr = torch.zeros_like(dropoff_counts)
        asr[active_mask] = dropoff_counts[active_mask] / active_taxis[active_mask]
        
        # Flatten for Gini computation
        dsr_flat = dsr.flatten()
        asr_flat = asr.flatten()
        
        # Compute pairwise Gini for each rate
        gini_dsr = self._pairwise_gini(dsr_flat)
        gini_asr = self._pairwise_gini(asr_flat)
        
        # Spatial fairness = 1 - average Gini
        f_spatial = 1.0 - 0.5 * (gini_dsr + gini_asr)
        
        # Store debug info
        self._last_spatial_debug = {
            'normalization': 'active_taxis',
            'n_active_cells': int(n_active),
            'dsr_range': (dsr.min().item(), dsr.max().item()),
            'asr_range': (asr.min().item(), asr.max().item()),
            'active_taxis_range': (active_taxis.min().item(), active_taxis.max().item()),
            'gini_dsr': gini_dsr.item(),
            'gini_asr': gini_asr.item(),
        }
        self.last_debug['gini_dsr'] = gini_dsr.item()
        self.last_debug['gini_asr'] = gini_asr.item()
        self.last_debug['f_spatial'] = f_spatial.item()
        
        return f_spatial
    
    def compute_causal_fairness(
        self,
        demand: torch.Tensor,
        supply: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute causal fairness: F_causal = max(0, R²)
        
        R² = 1 - Var(Y - g(D)) / Var(Y) where Y = S/D
        
        The g(D) function represents the expected supply-to-demand ratio
        given demand level D, fitted on historical data. R² measures how
        well the current supply-demand relationship follows the expected
        pattern.
        
        Args:
            demand: Demand (pickup counts) per cell [grid_x, grid_y]
            supply: Supply (dropoff counts or taxi availability) per cell [grid_x, grid_y]
            
        Returns:
            Causal fairness score in [0, 1]
            
        Raises:
            MissingComponentError: If g_function is not provided
            InsufficientDataError: If there are too few active cells
        """
        if self.g_function is None:
            raise MissingComponentError(
                "g_function not provided. Causal fairness requires a pre-fitted g(d) function. "
                "Load g_function parameters from the data source."
            )
        
        # Filter cells with sufficient demand
        # CRITICAL: This threshold MUST match the min_demand used when fitting g(d)
        # (default min_demand=1.0 in GFunctionLoader.estimate_from_data)
        #
        # If threshold is too low (e.g., 0.1), we include cells with tiny demand
        # where Y = S/D becomes artificially high, inflating Var(Y) and 
        # making R² artificially low. Using 1.0 matches g(d) training data.
        DEMAND_THRESHOLD = 1.0
        mask = demand >= DEMAND_THRESHOLD
        n_active = mask.sum().item()
        
        # Require at least 2 cells to compute meaningful variance
        MIN_CELLS_FOR_VARIANCE = 2
        if n_active < MIN_CELLS_FOR_VARIANCE:
            raise InsufficientDataError(
                f"Only {n_active} cells have demand >= {DEMAND_THRESHOLD}. "
                f"Causal fairness requires at least {MIN_CELLS_FOR_VARIANCE} active cells. "
                "Check that demand tensor contains realistic values."
            )
        
        D = demand[mask]
        S = supply[mask]
        
        # Compute Y = S/D (supply-to-demand ratio)
        Y = S / (D + self.eps)
        
        # Get g(D) predictions (frozen - no gradient tracking)
        with torch.no_grad():
            D_np = D.detach().cpu().numpy()
            g_d_np = self.g_function(D_np)
            g_d = torch.tensor(g_d_np, device=Y.device, dtype=Y.dtype)
        
        # Compute residual R = Y - g(D)
        # This is differentiable through Y
        R = Y - g_d
        
        # R² computation
        # R² = 1 - SS_res/SS_tot = 1 - Var(R)/Var(Y)
        # High R² means g(d) explains the variance well (fair supply allocation)
        var_Y = Y.var() + self.eps
        var_R = R.var()
        r_squared = 1.0 - var_R / var_Y
        
        f_causal = torch.clamp(r_squared, 0.0, 1.0)
        
        # Store debug info
        self._last_causal_debug = {
            'n_active_cells': int(n_active),
            'demand_range': (D.min().item(), D.max().item()),
            'supply_range': (S.min().item(), S.max().item()),
            'Y_range': (Y.min().item(), Y.max().item()),
            'g_d_range': (g_d.min().item(), g_d.max().item()),
            'var_Y': var_Y.item(),
            'var_R': var_R.item(),
            'r_squared_raw': r_squared.item(),
        }
        self.last_debug['r_squared'] = r_squared.item()
        self.last_debug['f_causal'] = f_causal.item()
        
        return f_causal
    
    def compute_fidelity(
        self,
        tau_features: torch.Tensor,
        tau_prime_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fidelity: F_fidelity = Discriminator(τ, τ')
        
        The discriminator outputs a similarity score in [0, 1]:
        - Score ≈ 1: Modified trajectory preserves original agent behavior
        - Score ≈ 0: Modified trajectory deviates significantly from agent behavior
        
        During optimization, fidelity acts as a regularizer to prevent
        modifications that would make the trajectory unrealistic for the driver.
        
        Args:
            tau_features: Original trajectory features [batch, seq_len, 4]
            tau_prime_features: Modified trajectory features [batch, seq_len, 4]
            
        Returns:
            Fidelity score in [0, 1]
            
        Raises:
            MissingComponentError: If discriminator is not provided
        """
        if self.discriminator is None:
            raise MissingComponentError(
                "Discriminator not provided. Fidelity computation requires a trained "
                "discriminator model. Load from discriminator/model/checkpoints/."
            )
        
        # Get discriminator similarity score
        # Gradients flow through tau_prime_features for optimization
        with torch.enable_grad():
            similarity = self.discriminator(tau_features, tau_prime_features)
        
        # Handle batch dimension - discriminator should return scalar per pair
        if similarity.dim() > 0:
            f_fidelity = similarity.mean()
        else:
            f_fidelity = similarity
        
        self.last_debug['f_fidelity'] = f_fidelity.item()
        
        return torch.clamp(f_fidelity, 0.0, 1.0)
    
    def forward(
        self,
        pickup_counts: torch.Tensor,
        dropoff_counts: torch.Tensor,
        supply: torch.Tensor,
        active_taxis: torch.Tensor,
        tau_features: Optional[torch.Tensor] = None,
        tau_prime_features: Optional[torch.Tensor] = None,
        causal_demand: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined objective L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity.
        
        All three terms are required for proper objective computation.
        
        IMPORTANT: For causal fairness to work correctly, supply and causal_demand should
        be aggregated using 'mean' (not 'sum'), so that Y = S/D is at the same scale
        as the g(d) function was fitted on.
        
        Args:
            pickup_counts: Pickup counts per cell [grid_x, grid_y] - for spatial fairness
            dropoff_counts: Dropoff counts per cell [grid_x, grid_y] - for spatial fairness
            supply: Supply (taxi availability) per cell [grid_x, grid_y] - for causal fairness
                   Should be mean-aggregated for proper F_causal computation.
            active_taxis: Active taxis for DSR/ASR normalization [grid_x, grid_y]
            tau_features: Original trajectory features [batch, seq_len, 4]
            tau_prime_features: Modified trajectory features [batch, seq_len, 4]
            causal_demand: Mean demand per cell for causal fairness [grid_x, grid_y].
                          If not provided, uses pickup_counts (may cause scale mismatch).
            
        Returns:
            total: Combined objective value (higher = better)
            terms: Dict with individual term values
            
        Raises:
            MissingDataError: If required data tensors are not provided
            MissingComponentError: If g_function or discriminator is missing
        """
        device = pickup_counts.device
        
        # Spatial fairness (requires pickup, dropoff, and active_taxis)
        f_spatial = self.compute_spatial_fairness(pickup_counts, dropoff_counts, active_taxis)
        
        # Causal fairness (requires demand and supply)
        # Use causal_demand if provided, otherwise fall back to pickup_counts
        demand_for_causal = causal_demand if causal_demand is not None else pickup_counts
        f_causal = self.compute_causal_fairness(demand_for_causal, supply)
        
        # Fidelity (requires trajectory features and discriminator)
        # Skip if alpha_fidelity is 0 to allow running without discriminator
        if self.alpha_fidelity > 0:
            if tau_features is None or tau_prime_features is None:
                raise MissingDataError(
                    "Trajectory features (tau_features and tau_prime_features) are required "
                    "for fidelity computation. Provide trajectory tensors [batch, seq_len, 4]."
                )
            f_fidelity = self.compute_fidelity(tau_features, tau_prime_features)
        else:
            f_fidelity = torch.tensor(0.0, device=device)
        
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
            'alpha_spatial': self.alpha_spatial,
            'alpha_causal': self.alpha_causal,
            'alpha_fidelity': self.alpha_fidelity,
        }
        
        return total, terms
    
    def forward_spatial_only(
        self,
        pickup_counts: torch.Tensor,
        dropoff_counts: torch.Tensor,
        active_taxis: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute objective with only spatial fairness (for testing/debugging).
        
        This is useful when g_function or discriminator are not available.
        
        Args:
            pickup_counts: Pickup counts per cell [grid_x, grid_y]
            dropoff_counts: Dropoff counts per cell [grid_x, grid_y]
            active_taxis: Active taxis per cell [grid_x, grid_y]
            
        Returns:
            f_spatial: Spatial fairness value
            terms: Dict with term values (causal and fidelity will be None)
        """
        f_spatial = self.compute_spatial_fairness(pickup_counts, dropoff_counts, active_taxis)
        
        terms = {
            'f_spatial': f_spatial,
            'f_causal': None,
            'f_fidelity': None,
        }
        
        return f_spatial, terms
    
    @staticmethod
    def verify_gradients(
        grid_dims: Tuple[int, int] = (10, 10),
        n_pickups: int = 20,
        n_dropoffs: int = 20,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Verify that gradients flow through all components.
        
        Creates test data and verifies backward pass produces valid gradients.
        
        Args:
            grid_dims: Test grid dimensions
            n_pickups: Number of test pickup points
            n_dropoffs: Number of test dropoff points
            temperature: Soft assignment temperature
            
        Returns:
            Dict with verification results including gradient statistics
        """
        import torch
        
        # Create module (spatial only for basic test)
        module = FAMAILObjective(
            grid_dims=grid_dims,
            temperature=temperature,
            alpha_spatial=1.0,
            alpha_causal=0.0,
            alpha_fidelity=0.0,
        )
        
        # Create test data
        torch.manual_seed(42)
        pickup_counts = torch.rand(grid_dims) * 10
        dropoff_counts = torch.rand(grid_dims) * 10
        active_taxis = torch.rand(grid_dims) * 5 + 1  # Ensure > 0
        
        pickup_counts.requires_grad_(True)
        
        # Forward pass (spatial only)
        f_spatial, _ = module.forward_spatial_only(pickup_counts, dropoff_counts, active_taxis)
        
        # Backward pass
        f_spatial.backward()
        
        # Check gradients
        results = {
            'passed': True,
            'f_spatial': f_spatial.item(),
        }
        
        if pickup_counts.grad is None:
            results['passed'] = False
            results['error'] = 'No gradient computed'
        elif torch.isnan(pickup_counts.grad).any():
            results['passed'] = False
            results['error'] = 'NaN in gradients'
        elif torch.isinf(pickup_counts.grad).any():
            results['passed'] = False
            results['error'] = 'Inf in gradients'
        else:
            results['grad_stats'] = {
                'mean': pickup_counts.grad.mean().item(),
                'std': pickup_counts.grad.std().item(),
                'min': pickup_counts.grad.min().item(),
                'max': pickup_counts.grad.max().item(),
                'nonzero_count': (pickup_counts.grad.abs() > 1e-10).sum().item(),
            }
        
        return results
        
        return total, terms
