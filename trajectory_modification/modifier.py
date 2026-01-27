"""
FAMAIL Trajectory Modifier using modified ST-iFGSM algorithm.

Implements the fairness-aware trajectory editing algorithm:
δ = clip(α · sign[∇L], -ε, ε)

The algorithm iteratively modifies trajectory pickup locations
to maximize the combined objective L = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .trajectory import Trajectory, TrajectoryState


@dataclass
class ModificationResult:
    """Result of a trajectory modification iteration."""
    trajectory: Trajectory
    objective_value: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    gradient_norm: float
    perturbation: np.ndarray
    

@dataclass
class ModificationHistory:
    """Full history of trajectory modification process."""
    original: Trajectory
    modified: Trajectory
    iterations: List[ModificationResult]
    converged: bool
    total_iterations: int
    final_objective: float
    

class TrajectoryModifier:
    """
    ST-iFGSM based trajectory modifier for fairness optimization.
    
    Algorithm (Modified ST-iFGSM):
    1. Initialize: τ' = τ (modified trajectory starts as original)
    2. For each iteration t:
       a. Compute objective L(τ') = α₁·F_spatial + α₂·F_causal + α₃·F_fidelity
       b. Compute gradient ∇L w.r.t. pickup location
       c. Apply perturbation: δ = clip(α · sign[∇L], -ε, ε)
       d. Update pickup: τ' = project(τ' + δ)
    3. Terminate when converged or max_iterations reached
    
    Gradient Modes:
    - 'soft_cell': Use soft cell assignment for differentiable gradient flow.
                  Gradients flow from objective through soft counts to pickup location.
    - 'heuristic': Use heuristic gradient pointing toward underserved areas.
                   Based on local DSR differences in neighboring cells.
    """
    
    def __init__(
        self,
        objective_fn: 'nn.Module',
        grid_dims: Tuple[int, int] = (48, 90),
        alpha: float = 0.1,       # Step size
        epsilon: float = 3.0,     # Max perturbation per dimension
        max_iterations: int = 50,
        convergence_threshold: float = 1e-4,
        gradient_mode: str = 'soft_cell',  # 'soft_cell' or 'heuristic'
        temperature: float = 1.0,
        temperature_annealing: bool = False,
        tau_max: float = 1.0,
        tau_min: float = 0.1,
        neighborhood_size: int = 5,
    ):
        self.objective_fn = objective_fn
        self.grid_dims = grid_dims
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Gradient mode configuration
        if gradient_mode not in ('soft_cell', 'heuristic'):
            raise ValueError(f"gradient_mode must be 'soft_cell' or 'heuristic', got '{gradient_mode}'")
        self.gradient_mode = gradient_mode
        
        # Soft cell assignment configuration
        self.temperature = temperature
        self.temperature_annealing = temperature_annealing
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.neighborhood_size = neighborhood_size
        
        # Initialize soft cell assignment module if using soft_cell mode
        self.soft_assign = None
        if gradient_mode == 'soft_cell':
            try:
                from objective_function.soft_cell_assignment import SoftCellAssignment
                self.soft_assign = SoftCellAssignment(
                    grid_dims=grid_dims,
                    neighborhood_size=neighborhood_size,
                    initial_temperature=temperature,
                )
            except ImportError:
                raise ImportError(
                    "SoftCellAssignment required for gradient_mode='soft_cell'. "
                    "Install objective_function.soft_cell_assignment or use gradient_mode='heuristic'."
                )
        
        # State accumulators (updated during modification)
        self.pickup_counts = None
        self.dropoff_counts = None
        self.supply_counts = None
        self.active_taxis = None
        
        # Base counts (original counts without current trajectory's contribution)
        self._base_pickup_counts = None
        self._base_dropoff_counts = None
    
    def set_global_state(
        self,
        pickup_counts: torch.Tensor,
        dropoff_counts: torch.Tensor,
        active_taxis: torch.Tensor,
        supply_counts: Optional[torch.Tensor] = None,
    ):
        """
        Set the global counts for fairness computation.
        
        Args:
            pickup_counts: Pickup counts per cell [grid_x, grid_y]
            dropoff_counts: Dropoff counts per cell [grid_x, grid_y]
            active_taxis: Active taxis per cell [grid_x, grid_y]
            supply_counts: Optional separate supply tensor for causal fairness.
                          If not provided, dropoff_counts is used as supply.
        """
        self.pickup_counts = pickup_counts.clone()
        self.dropoff_counts = dropoff_counts.clone()
        self.active_taxis = active_taxis.clone()
        # Supply defaults to dropoff counts if not provided separately 
        self.supply_counts = supply_counts.clone() if supply_counts is not None else self.dropoff_counts.clone()
        
        # Store base counts for soft cell computation (without current trajectory)
        # These are used to compute soft counts during optimization
        self._base_pickup_counts = pickup_counts.clone()
        self._base_dropoff_counts = dropoff_counts.clone() 
    
    def _get_pickup_index(self, trajectory: Trajectory) -> int:
        """
        Get the index of the pickup state in the trajectory.
        
        The pickup location is always the final state in the trajectory,
        as trajectories represent the path from passenger-seeking to pickup.
        
        Returns:
            Index of the last state (pickup location)
        """
        return len(trajectory.states) - 1
    
    def _update_counts(
        self,
        old_cell: Tuple[int, int],
        new_cell: Tuple[int, int],
    ):
        """Update pickup counts when a trajectory's pickup moves."""
        if self.pickup_counts is None:
            return
            
        ox, oy = old_cell
        nx, ny = new_cell
        
        # Decrement old cell, increment new cell
        if 0 <= ox < self.grid_dims[0] and 0 <= oy < self.grid_dims[1]:
            self.pickup_counts[ox, oy] -= 1
        if 0 <= nx < self.grid_dims[0] and 0 <= ny < self.grid_dims[1]:
            self.pickup_counts[nx, ny] += 1
    
    def _clip_to_grid(self, position: np.ndarray) -> np.ndarray:
        """Clip position to valid grid bounds."""
        clipped = position.copy()
        clipped[0] = np.clip(clipped[0], 0, self.grid_dims[0] - 1)
        clipped[1] = np.clip(clipped[1], 0, self.grid_dims[1] - 1)
        return clipped
    
    def _get_annealed_temperature(self, iteration: int) -> float:
        """Compute annealed temperature for current iteration."""
        if not self.temperature_annealing:
            return self.temperature
        if self.max_iterations <= 1:
            return self.tau_min
        progress = iteration / (self.max_iterations - 1)
        return self.tau_max * (self.tau_min / self.tau_max) ** progress
    
    def _compute_soft_pickup_counts(
        self,
        pickup_location: torch.Tensor,
        original_cell: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute differentiable pickup counts using soft cell assignment.
        
        The soft counts enable gradient flow from the objective back to
        the pickup location, which is essential for gradient-based optimization.
        
        Args:
            pickup_location: Current pickup location [2] (x, y) - requires_grad=True
            original_cell: Original cell index [2] (x, y)
            
        Returns:
            Soft pickup counts [grid_x, grid_y] with gradient connection to pickup_location
        """
        device = pickup_location.device
        k = self.soft_assign.k
        
        # Get soft assignment probabilities [1, ns, ns]
        loc = pickup_location.unsqueeze(0)  # [1, 2]
        cell = original_cell.unsqueeze(0).float()  # [1, 2]
        probs = self.soft_assign(loc, cell)  # [1, ns, ns]
        
        # Start with base counts (without this trajectory's original contribution)
        # We need to subtract the original hard assignment
        soft_counts = self._base_pickup_counts.clone()
        ox, oy = int(original_cell[0].item()), int(original_cell[1].item())
        if 0 <= ox < self.grid_dims[0] and 0 <= oy < self.grid_dims[1]:
            soft_counts[ox, oy] = soft_counts[ox, oy] - 1.0
        
        # Add soft assignment to counts
        # This is where gradients flow: probs depends on pickup_location
        cx, cy = int(original_cell[0].item()), int(original_cell[1].item())
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ni, nj = cx + di, cy + dj
                if 0 <= ni < self.grid_dims[0] and 0 <= nj < self.grid_dims[1]:
                    # Add probability mass to this cell
                    # The gradient will flow: soft_counts -> probs -> pickup_location
                    soft_counts[ni, nj] = soft_counts[ni, nj] + probs[0, di + k, dj + k]
        
        return soft_counts

    def modify_single(
        self,
        trajectory: Trajectory,
        discriminator_fn: Optional[Callable] = None,
    ) -> ModificationHistory:
        """
        Modify a single trajectory to optimize fairness.
        
        This is the core ST-iFGSM loop:
        δ = clip(α · sign[∇L], -ε, ε)
        
        When gradient_mode='soft_cell', gradients flow through soft cell assignment:
            pickup_location -> soft_probs -> soft_counts -> objective -> gradient
        
        When gradient_mode='heuristic', gradients are estimated from local DSR:
            pickup_location -> heuristic (neighbor DSR comparison) -> gradient
        
        Args:
            trajectory: Original trajectory to modify
            discriminator_fn: Optional function(τ, τ') → similarity score
            
        Returns:
            ModificationHistory with full optimization trace
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for trajectory modification")
        
        device = self.pickup_counts.device if self.pickup_counts is not None else 'cpu'
        
        # Move soft_assign to device if using soft cell mode
        if self.gradient_mode == 'soft_cell' and self.soft_assign is not None:
            self.soft_assign = self.soft_assign.to(device)
        
        # Get pickup location index (always the last state in trajectory)
        pickup_idx = self._get_pickup_index(trajectory)
        pickup_state = trajectory.states[pickup_idx]
        
        # Initialize modified trajectory: \tau' = \tau
        modified = trajectory.clone()
        
        # Track cumulative perturbation
        cumulative_delta = np.zeros(2)  # [δx, δy]
        
        # Original pickup location for fidelity computation
        original_pickup = np.array([pickup_state.x_grid, pickup_state.y_grid], dtype=np.float32)
        original_cell = torch.tensor([int(pickup_state.x_grid), int(pickup_state.y_grid)], device=device)
        current_pickup = original_pickup.copy()
        
        iterations: List[ModificationResult] = []
        prev_objective = float('-inf')
        
        for t in range(self.max_iterations):
            # Update temperature if annealing is enabled
            if self.gradient_mode == 'soft_cell' and self.temperature_annealing:
                current_temp = self._get_annealed_temperature(t)
                self.soft_assign.set_temperature(current_temp)
            
            # Current pickup as differentiable tensor
            pickup_tensor = torch.tensor(
                current_pickup, 
                dtype=torch.float32, 
                device=device,
                requires_grad=True
            )
            
            # Build fidelity tensors for discriminator
            # to_tensor() returns [seq_len, 4], we add batch dimension [1, seq_len, 4]
            tau_features = trajectory.to_tensor().unsqueeze(0).to(device)
            tau_prime_features = modified.to_tensor().unsqueeze(0).to(device)
            
            # Compute counts based on gradient mode
            if self.gradient_mode == 'soft_cell':
                # Use soft cell assignment for differentiable counts
                soft_pickup_counts = self._compute_soft_pickup_counts(pickup_tensor, original_cell)
                
                # Compute objective with soft counts (gradients flow through)
                total, terms = self.objective_fn(
                    pickup_counts=soft_pickup_counts,
                    dropoff_counts=self.dropoff_counts,
                    supply=self.supply_counts,
                    active_taxis=self.active_taxis,
                    tau_features=tau_features,
                    tau_prime_features=tau_prime_features,
                )
                
                # Backward pass - gradients should flow to pickup_tensor
                total.backward(retain_graph=True)
                
                if pickup_tensor.grad is not None:
                    grad = pickup_tensor.grad.detach().cpu().numpy()
                    grad_norm = np.linalg.norm(grad)
                else:
                    # Fallback to heuristic if soft cell didn't produce gradients
                    grad = self._compute_heuristic_gradient(current_pickup)
                    grad_norm = np.linalg.norm(grad)
                    
            else:  # gradient_mode == 'heuristic'
                # Use pre-aggregated counts (no gradient flow)
                total, terms = self.objective_fn(
                    pickup_counts=self.pickup_counts,
                    dropoff_counts=self.dropoff_counts,
                    supply=self.supply_counts,
                    active_taxis=self.active_taxis,
                    tau_features=tau_features,
                    tau_prime_features=tau_prime_features,
                )
                
                # Use heuristic gradient
                grad = self._compute_heuristic_gradient(current_pickup)
                grad_norm = np.linalg.norm(grad)
            
            # ST-iFGSM perturbation: δ = clip(α · sign[∇L], -ε, ε)
            if grad_norm > 1e-8:
                delta = self.alpha * np.sign(grad)
                # Clip cumulative perturbation to ε-ball
                cumulative_delta = np.clip(
                    cumulative_delta + delta, 
                    -self.epsilon, 
                    self.epsilon
                )
            else:
                delta = np.zeros(2)
            
            # Apply perturbation
            old_cell = (int(current_pickup[0]), int(current_pickup[1]))
            new_pickup = self._clip_to_grid(original_pickup + cumulative_delta)
            new_cell = (int(new_pickup[0]), int(new_pickup[1]))
            
            # Update global counts if cell changed (for heuristic mode tracking)
            if old_cell != new_cell:
                self._update_counts(old_cell, new_cell)
            
            current_pickup = new_pickup
            
            # Update modified trajectory (apply_perturbation always modifies the last state)
            modified = modified.apply_perturbation(cumulative_delta)
            
            # Record iteration
            result = ModificationResult(
                trajectory=modified.clone(),
                objective_value=total.item(),
                f_spatial=terms['f_spatial'].item(),
                f_causal=terms['f_causal'].item(),
                f_fidelity=terms['f_fidelity'].item(),
                gradient_norm=grad_norm,
                perturbation=cumulative_delta.copy(),
            )
            iterations.append(result)
            
            # Check convergence
            if abs(total.item() - prev_objective) < self.convergence_threshold:
                break
            prev_objective = total.item()
        
        # Build history
        history = ModificationHistory(
            original=trajectory,
            modified=modified,
            iterations=iterations,
            converged=(len(iterations) < self.max_iterations),
            total_iterations=len(iterations),
            final_objective=iterations[-1].objective_value if iterations else 0.0,
        )
        
        return history
    
    def _compute_heuristic_gradient(self, current_pickup: np.ndarray) -> np.ndarray:
        """
        Compute heuristic gradient toward underserved areas.
        
        When gradients don't flow (e.g., non-differentiable counts),
        we estimate gradient by looking at nearby cell service levels.
        """
        if self.pickup_counts is None or self.active_taxis is None:
            return np.zeros(2)
        
        x, y = int(current_pickup[0]), int(current_pickup[1])
        grad = np.zeros(2)
        
        # Check neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_dims[0] and 0 <= ny < self.grid_dims[1]:
                # DSR = pickups / active_taxis
                active = self.active_taxis[nx, ny].item()
                if active > 0:
                    dsr = self.pickup_counts[nx, ny].item() / active
                    # Gradient points toward LOWER DSR (underserved areas)
                    grad[0] -= dx * dsr
                    grad[1] -= dy * dsr
        
        # Normalize
        norm = np.linalg.norm(grad)
        if norm > 1e-8:
            grad = grad / norm
        
        return grad
    
    def modify_batch(
        self,
        trajectories: List[Trajectory],
        discriminator_fn: Optional[Callable] = None,
    ) -> List[ModificationHistory]:
        """Modify multiple trajectories sequentially."""
        histories = []
        for traj in trajectories:
            history = self.modify_single(traj, discriminator_fn)
            histories.append(history)
        return histories
