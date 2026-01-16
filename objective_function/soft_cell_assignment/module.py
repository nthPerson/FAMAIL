"""
Soft Cell Assignment Module Implementation.

This module implements the differentiable soft cell assignment
for gradient-based trajectory optimization in FAMAIL.
"""

from typing import Tuple, Optional, Union
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# PYTORCH MODULE IMPLEMENTATION
# =============================================================================

if TORCH_AVAILABLE:
    class SoftCellAssignment(nn.Module):
        """
        Differentiable soft cell assignment for trajectory optimization.
        
        This module computes a probability distribution over grid cells
        based on continuous location coordinates, enabling gradient-based
        optimization of trajectory locations.
        
        The soft assignment uses a Gaussian kernel:
            σ_c(x, y) = exp(-d_c²/(2τ²)) / Z
        
        where d_c is the distance from location to cell center and τ is
        the temperature controlling distribution sharpness.
        
        Attributes:
            grid_dims: Tuple of (x_cells, y_cells) for the spatial grid
            neighborhood_size: Size of the neighborhood window (e.g., 5 for 5x5)
            temperature: Current temperature value
            
        Example:
            >>> module = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5)
            >>> location = torch.tensor([[24.3, 45.7]], requires_grad=True)
            >>> original_cell = torch.tensor([[24, 45]])
            >>> probs = module(location, original_cell)
            >>> probs.sum(dim=[1,2])  # Should be ~1.0
            >>> probs.backward(torch.ones_like(probs))
            >>> location.grad  # Gradients flow back to location
        """
        
        def __init__(
            self,
            grid_dims: Tuple[int, int] = (48, 90),
            neighborhood_size: int = 5,
            initial_temperature: float = 1.0,
            eps: float = 1e-8,
        ):
            """
            Initialize the soft cell assignment module.
            
            Args:
                grid_dims: Spatial grid dimensions (x_cells, y_cells)
                neighborhood_size: Size of neighborhood window (must be odd)
                initial_temperature: Starting temperature for soft assignment
                eps: Small constant for numerical stability
            """
            super().__init__()
            
            if neighborhood_size % 2 == 0:
                raise ValueError(f"neighborhood_size must be odd, got {neighborhood_size}")
            
            self.grid_dims = grid_dims
            self.neighborhood_size = neighborhood_size
            self.eps = eps
            
            # Temperature as a buffer (not a parameter - controlled externally)
            self.register_buffer('temperature', torch.tensor(initial_temperature))
            
            # Pre-compute neighborhood offsets
            k = (neighborhood_size - 1) // 2
            offsets = torch.arange(-k, k + 1, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(offsets, offsets, indexing='ij')
            offset_grid = torch.stack([grid_x, grid_y], dim=-1)  # [ns, ns, 2]
            self.register_buffer('offset_grid', offset_grid)
            self.k = k
        
        def forward(
            self,
            location: torch.Tensor,
            original_cell: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute soft assignment over neighborhood cells.
            
            Args:
                location: Continuous coordinates [batch, 2] with (x, y)
                original_cell: Original cell indices [batch, 2] with (x, y)
                
            Returns:
                Probability distribution over neighborhood [batch, ns, ns]
                where ns = neighborhood_size
            """
            batch_size = location.shape[0]
            ns = self.neighborhood_size
            
            # Ensure inputs are float
            location = location.float()
            original_cell = original_cell.float()
            
            # Compute cell centers in the neighborhood
            # Shape: [batch, ns, ns, 2]
            cell_centers = original_cell[:, None, None, :] + self.offset_grid
            
            # Compute squared distances from location to each cell center
            # Shape: [batch, ns, ns]
            diff = location[:, None, None, :] - cell_centers
            sq_distances = (diff ** 2).sum(dim=-1)
            
            # Soft assignment via softmax over negative distances
            logits = -sq_distances / (2 * self.temperature ** 2 + self.eps)
            
            # Flatten for softmax, then reshape back
            logits_flat = logits.view(batch_size, -1)
            probs_flat = F.softmax(logits_flat, dim=-1)
            probs = probs_flat.view(batch_size, ns, ns)
            
            return probs
        
        def forward_with_boundary_check(
            self,
            location: torch.Tensor,
            original_cell: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute soft assignment with boundary validity mask.
            
            Returns both probabilities and a mask indicating which cells
            are within the grid bounds.
            
            Args:
                location: Continuous coordinates [batch, 2]
                original_cell: Original cell indices [batch, 2]
                
            Returns:
                Tuple of (probs [batch, ns, ns], valid_mask [batch, ns, ns])
            """
            batch_size = location.shape[0]
            ns = self.neighborhood_size
            
            # Get base probabilities
            probs = self.forward(location, original_cell)
            
            # Compute actual cell indices
            original_cell_long = original_cell.long()
            cell_indices_x = original_cell_long[:, 0:1, None] + self.offset_grid[:, :, 0].long()
            cell_indices_y = original_cell_long[:, 1:2, None] + self.offset_grid[:, :, 1].long()
            
            # Create validity mask
            valid_x = (cell_indices_x >= 0) & (cell_indices_x < self.grid_dims[0])
            valid_y = (cell_indices_y >= 0) & (cell_indices_y < self.grid_dims[1])
            valid_mask = valid_x & valid_y
            
            return probs, valid_mask.squeeze(1)
        
        def set_temperature(self, temperature: float) -> None:
            """
            Update the temperature parameter.
            
            Args:
                temperature: New temperature value (should be > 0)
            """
            self.temperature.fill_(temperature)
        
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
            
            Args:
                iteration: Current iteration (0-indexed)
                total_iterations: Total number of iterations
                tau_max: Starting temperature
                tau_min: Final temperature
                
            Returns:
                Annealed temperature value
            """
            if total_iterations <= 1:
                return tau_min
            progress = iteration / (total_iterations - 1)
            return tau_max * (tau_min / tau_max) ** progress
        
        def discretize(
            self,
            location: torch.Tensor,
            original_cell: torch.Tensor,
            method: str = 'argmax',
        ) -> torch.Tensor:
            """
            Convert soft location to discrete cell assignment.
            
            Args:
                location: Continuous coordinates [batch, 2]
                original_cell: Original cell indices [batch, 2]
                method: 'argmax' for hard assignment, 'sample' for stochastic
                
            Returns:
                Discrete cell indices [batch, 2]
            """
            probs = self.forward(location, original_cell)
            batch_size = probs.shape[0]
            ns = self.neighborhood_size
            
            if method == 'argmax':
                # Find cell with highest probability
                flat_probs = probs.view(batch_size, -1)
                max_idx = flat_probs.argmax(dim=-1)
                
                # Convert flat index to 2D offset
                offset_x = max_idx // ns - self.k
                offset_y = max_idx % ns - self.k
                
            elif method == 'sample':
                # Sample from probability distribution
                flat_probs = probs.view(batch_size, -1)
                sampled_idx = torch.multinomial(flat_probs, 1).squeeze(-1)
                
                offset_x = sampled_idx // ns - self.k
                offset_y = sampled_idx % ns - self.k
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Compute final cell indices
            final_cell = original_cell.long() + torch.stack([offset_x, offset_y], dim=-1)
            
            # Clamp to grid bounds
            final_cell[:, 0] = final_cell[:, 0].clamp(0, self.grid_dims[0] - 1)
            final_cell[:, 1] = final_cell[:, 1].clamp(0, self.grid_dims[1] - 1)
            
            return final_cell


# =============================================================================
# FUNCTIONAL API
# =============================================================================

def soft_cell_assignment(
    location: 'torch.Tensor',
    original_cell: 'torch.Tensor',
    neighborhood_size: int = 5,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> 'torch.Tensor':
    """
    Compute soft cell assignment (functional API).
    
    This is a stateless version of SoftCellAssignment.forward() for
    cases where a module instance is not needed.
    
    Args:
        location: Continuous coordinates [batch, 2] with (x, y)
        original_cell: Original cell indices [batch, 2] with (x, y)
        neighborhood_size: Size of neighborhood window (must be odd)
        temperature: Temperature parameter for softmax
        eps: Numerical stability constant
        
    Returns:
        Probability distribution over neighborhood [batch, ns, ns]
    
    Example:
        >>> location = torch.tensor([[24.3, 45.7]], requires_grad=True)
        >>> original_cell = torch.tensor([[24, 45]])
        >>> probs = soft_cell_assignment(location, original_cell, temperature=0.5)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for soft_cell_assignment")
    
    if neighborhood_size % 2 == 0:
        raise ValueError(f"neighborhood_size must be odd, got {neighborhood_size}")
    
    batch_size = location.shape[0]
    k = (neighborhood_size - 1) // 2
    ns = neighborhood_size
    
    # Create neighborhood offsets
    offsets = torch.arange(-k, k + 1, dtype=torch.float32, device=location.device)
    grid_x, grid_y = torch.meshgrid(offsets, offsets, indexing='ij')
    offset_grid = torch.stack([grid_x, grid_y], dim=-1)  # [ns, ns, 2]
    
    # Compute cell centers
    location = location.float()
    original_cell = original_cell.float()
    cell_centers = original_cell[:, None, None, :] + offset_grid
    
    # Compute squared distances
    diff = location[:, None, None, :] - cell_centers
    sq_distances = (diff ** 2).sum(dim=-1)
    
    # Softmax over negative distances
    logits = -sq_distances / (2 * temperature ** 2 + eps)
    logits_flat = logits.view(batch_size, -1)
    probs_flat = torch.softmax(logits_flat, dim=-1)
    probs = probs_flat.view(batch_size, ns, ns)
    
    return probs


def compute_soft_counts(
    trajectory_probs: 'torch.Tensor',
    original_cells: 'torch.Tensor',
    grid_dims: Tuple[int, int],
    base_counts: Optional['torch.Tensor'] = None,
) -> 'torch.Tensor':
    """
    Aggregate soft assignments into grid-level counts.
    
    This function converts per-trajectory probability distributions
    over neighborhoods into a full grid of soft counts.
    
    Args:
        trajectory_probs: Soft assignments [num_traj, ns, ns]
        original_cells: Original cell indices [num_traj, 2]
        grid_dims: Grid dimensions (x_size, y_size)
        base_counts: Optional base counts to add to [x_size, y_size]
        
    Returns:
        Soft counts tensor [x_size, y_size]
    
    Example:
        >>> probs = torch.rand(100, 5, 5)  # 100 trajectories, 5x5 neighborhoods
        >>> probs = probs / probs.sum(dim=[1,2], keepdim=True)  # Normalize
        >>> cells = torch.randint(2, 46, (100, 2))  # Original cells
        >>> counts = compute_soft_counts(probs, cells, (48, 90))
        >>> counts.sum()  # Should be ~100 (total probability mass)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for compute_soft_counts")
    
    num_traj = trajectory_probs.shape[0]
    ns = trajectory_probs.shape[1]
    k = (ns - 1) // 2
    
    # Initialize counts
    device = trajectory_probs.device
    dtype = trajectory_probs.dtype
    
    if base_counts is not None:
        counts = base_counts.clone()
    else:
        counts = torch.zeros(grid_dims, device=device, dtype=dtype)
    
    # Get original cell indices as long tensors
    original_cells_long = original_cells.long()
    
    # Scatter-add soft assignments to grid
    # This is done in a loop for clarity; could be optimized with scatter_add
    for traj_idx in range(num_traj):
        ox, oy = original_cells_long[traj_idx]
        
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ci = ox + di
                cj = oy + dj
                
                # Check bounds
                if 0 <= ci < grid_dims[0] and 0 <= cj < grid_dims[1]:
                    prob_idx_i = di + k
                    prob_idx_j = dj + k
                    counts[ci, cj] = counts[ci, cj] + trajectory_probs[traj_idx, prob_idx_i, prob_idx_j]
    
    return counts


def compute_soft_counts_vectorized(
    trajectory_probs: 'torch.Tensor',
    original_cells: 'torch.Tensor',
    grid_dims: Tuple[int, int],
    base_counts: Optional['torch.Tensor'] = None,
) -> 'torch.Tensor':
    """
    Vectorized version of compute_soft_counts (faster for large batches).
    
    Uses index_put_ with accumulate=True for efficient aggregation.
    
    Args:
        trajectory_probs: Soft assignments [num_traj, ns, ns]
        original_cells: Original cell indices [num_traj, 2]
        grid_dims: Grid dimensions (x_size, y_size)
        base_counts: Optional base counts to add to
        
    Returns:
        Soft counts tensor [x_size, y_size]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    num_traj = trajectory_probs.shape[0]
    ns = trajectory_probs.shape[1]
    k = (ns - 1) // 2
    device = trajectory_probs.device
    dtype = trajectory_probs.dtype
    
    # Initialize counts
    if base_counts is not None:
        counts = base_counts.clone()
    else:
        counts = torch.zeros(grid_dims, device=device, dtype=dtype)
    
    # Create offset grids
    offsets = torch.arange(-k, k + 1, device=device)
    offset_x, offset_y = torch.meshgrid(offsets, offsets, indexing='ij')
    offset_x = offset_x.reshape(1, -1)  # [1, ns*ns]
    offset_y = offset_y.reshape(1, -1)  # [1, ns*ns]
    
    # Compute target cell indices for all trajectories
    # [num_traj, ns*ns]
    target_x = original_cells[:, 0:1] + offset_x
    target_y = original_cells[:, 1:2] + offset_y
    
    # Flatten probabilities
    probs_flat = trajectory_probs.view(num_traj, -1)  # [num_traj, ns*ns]
    
    # Create validity mask
    valid = (
        (target_x >= 0) & (target_x < grid_dims[0]) &
        (target_y >= 0) & (target_y < grid_dims[1])
    )
    
    # Flatten everything and filter by validity
    target_x_flat = target_x.view(-1)
    target_y_flat = target_y.view(-1)
    probs_all = probs_flat.view(-1)
    valid_flat = valid.view(-1)
    
    # Get valid indices and probabilities
    valid_x = target_x_flat[valid_flat].long()
    valid_y = target_y_flat[valid_flat].long()
    valid_probs = probs_all[valid_flat]
    
    # Accumulate using index_put_
    counts.index_put_(
        (valid_x, valid_y),
        valid_probs,
        accumulate=True,
    )
    
    return counts


def update_counts_with_soft_assignment(
    base_counts: 'torch.Tensor',
    soft_probs: 'torch.Tensor',
    original_cell: 'torch.Tensor',
    subtract_original: bool = True,
) -> 'torch.Tensor':
    """
    Update counts by removing hard assignment and adding soft assignment.
    
    This is used during optimization to replace a trajectory's
    original discrete contribution with its current soft assignment.
    
    Args:
        base_counts: Grid counts [x_size, y_size]
        soft_probs: Soft assignment for one trajectory [1, ns, ns] or [ns, ns]
        original_cell: Original cell index [2] or [1, 2]
        subtract_original: If True, subtract 1 from original cell
        
    Returns:
        Updated counts tensor [x_size, y_size]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    counts = base_counts.clone()
    
    # Handle dimensions
    if soft_probs.dim() == 2:
        soft_probs = soft_probs.unsqueeze(0)
    if original_cell.dim() == 1:
        original_cell = original_cell.unsqueeze(0)
    
    ns = soft_probs.shape[1]
    k = (ns - 1) // 2
    grid_dims = counts.shape
    
    ox, oy = original_cell[0].long()
    
    # Subtract original hard assignment
    if subtract_original:
        if 0 <= ox < grid_dims[0] and 0 <= oy < grid_dims[1]:
            counts[ox, oy] = counts[ox, oy] - 1.0
    
    # Add soft assignment
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            ci = ox + di
            cj = oy + dj
            if 0 <= ci < grid_dims[0] and 0 <= cj < grid_dims[1]:
                prob = soft_probs[0, di + k, dj + k]
                counts[ci, cj] = counts[ci, cj] + prob
    
    return counts


# =============================================================================
# NUMPY COMPATIBILITY FUNCTIONS
# =============================================================================

def soft_cell_assignment_numpy(
    location: np.ndarray,
    original_cell: np.ndarray,
    neighborhood_size: int = 5,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    NumPy version of soft cell assignment (for non-gradient use).
    
    Args:
        location: Continuous coordinates [batch, 2] or [2]
        original_cell: Original cell indices [batch, 2] or [2]
        neighborhood_size: Size of neighborhood window
        temperature: Temperature parameter
        eps: Numerical stability constant
        
    Returns:
        Probability distribution [batch, ns, ns] or [ns, ns]
    """
    # Handle single location
    single_input = location.ndim == 1
    if single_input:
        location = location[np.newaxis, :]
        original_cell = original_cell[np.newaxis, :]
    
    batch_size = location.shape[0]
    k = (neighborhood_size - 1) // 2
    ns = neighborhood_size
    
    # Create offset grid
    offsets = np.arange(-k, k + 1)
    grid_x, grid_y = np.meshgrid(offsets, offsets, indexing='ij')
    offset_grid = np.stack([grid_x, grid_y], axis=-1)  # [ns, ns, 2]
    
    # Compute cell centers
    cell_centers = original_cell[:, np.newaxis, np.newaxis, :] + offset_grid
    
    # Compute squared distances
    diff = location[:, np.newaxis, np.newaxis, :] - cell_centers
    sq_distances = (diff ** 2).sum(axis=-1)
    
    # Softmax
    logits = -sq_distances / (2 * temperature ** 2 + eps)
    logits_flat = logits.reshape(batch_size, -1)
    
    # Stable softmax
    logits_max = logits_flat.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_flat - logits_max)
    probs_flat = exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + eps)
    
    probs = probs_flat.reshape(batch_size, ns, ns)
    
    if single_input:
        return probs[0]
    return probs


def compute_soft_counts_numpy(
    trajectory_probs: np.ndarray,
    original_cells: np.ndarray,
    grid_dims: Tuple[int, int],
    base_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    NumPy version of compute_soft_counts.
    
    Args:
        trajectory_probs: Soft assignments [num_traj, ns, ns]
        original_cells: Original cell indices [num_traj, 2]
        grid_dims: Grid dimensions (x_size, y_size)
        base_counts: Optional base counts
        
    Returns:
        Soft counts array [x_size, y_size]
    """
    num_traj = trajectory_probs.shape[0]
    ns = trajectory_probs.shape[1]
    k = (ns - 1) // 2
    
    if base_counts is not None:
        counts = base_counts.copy()
    else:
        counts = np.zeros(grid_dims, dtype=np.float64)
    
    for traj_idx in range(num_traj):
        ox, oy = int(original_cells[traj_idx, 0]), int(original_cells[traj_idx, 1])
        
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ci, cj = ox + di, oy + dj
                if 0 <= ci < grid_dims[0] and 0 <= cj < grid_dims[1]:
                    counts[ci, cj] += trajectory_probs[traj_idx, di + k, dj + k]
    
    return counts
