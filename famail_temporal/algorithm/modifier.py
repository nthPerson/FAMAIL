"""
TrajectoryModifier — ST-iFGSM with soft cell assignment and pooled (cell, t)
fairness.

Algorithm per trajectory:
  1. Determine the trajectory's time block t* from its pickup's time_bucket
  2. Compute pickup_mass = 1 / (n_hours_per_block[t*] * n_days)
  3. Subtract trajectory's contribution from _base_pickup_3d[orig_cell, t*]
  4. For each iteration t:
     a. Anneal temperature
     b. Build pickup_tensor with requires_grad=True
     c. Compute soft probs; inject * pickup_mass into t* slice
     d. Forward + backward through FAMAILObjective
     e. Apply delta = clip(alpha * sign(grad), -epsilon, epsilon)
     f. Update cumulative delta, clip pickup to grid bounds
  5. Persist changes to shared _base_pickup_3d

The modifier is the only module that MUTATES _base_pickup_3d. This creates
ordering dependencies across trajectories when modify_batch is called:
trajectory B's optimization sees the updated baseline after trajectory A's
modification. This is intentional.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import warnings

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.soft_cell_assignment import (
    SoftCellAssignment, inject_soft_counts_into_3d,
)
from famail_temporal.data.aggregation import (
    hour_to_block_index, time_bucket_to_hour,
)
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.trajectory import Trajectory


@dataclass
class ModificationResult:
    """Single iteration record from the ST-iFGSM loop."""
    iteration: int
    objective_value: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    gradient_norm: float
    cumulative_delta: np.ndarray


@dataclass
class ModificationHistory:
    """Full history of modifying one trajectory."""
    original: Trajectory
    modified: Trajectory
    iterations: List[ModificationResult] = field(default_factory=list)
    converged: bool = False
    total_iterations: int = 0
    final_objective: float = 0.0


class TrajectoryModifier:
    """ST-iFGSM trajectory modifier with soft cell assignment.

    Perturbation rule:  delta = clip(alpha * sign(grad), -epsilon, epsilon)
    Cumulative delta is clipped to the epsilon-ball: ||delta||_inf <= epsilon.
    Mass balance is maintained: pickup_3d.sum() is preserved after each
    modification (subtract at old cell, add at new cell).
    """

    def __init__(
        self,
        objective: FAMAILObjective,
        bundle: DataBundle,
        multi_stream_builder=None,
        alpha: float = config.STEP_SIZE_ALPHA,
        epsilon: float = config.EPSILON_BALL,
        max_iterations: int = config.MAX_ITERATIONS,
        convergence_tol: float = config.CONVERGENCE_TOL,
    ):
        self.objective = objective
        self.bundle = bundle
        self.multi_stream_builder = multi_stream_builder
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        self.soft_assign = SoftCellAssignment()
        # Clone so we don't mutate the original bundle array
        self._base_pickup_3d = torch.from_numpy(bundle.pickup_3d).float().clone()

    def _get_annealed_temperature(self, iteration: int) -> float:
        """Exponential temperature annealing: tau_max * (tau_min/tau_max)^(t/T)."""
        if not config.ANNEAL_TEMPERATURE or self.max_iterations <= 1:
            return config.TAU_MIN
        progress = iteration / (self.max_iterations - 1)
        return config.TAU_MAX * (config.TAU_MIN / config.TAU_MAX) ** progress

    def _neighborhood_has_active_units(
        self, cell_xy, t_block: int,
    ) -> bool:
        """Check if any cell in the soft-assignment neighborhood is active."""
        k = self.soft_assign.k
        cx, cy = cell_xy
        gx, gy = config.GRID_DIMS
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ni, nj = cx + di, cy + dj
                if 0 <= ni < gx and 0 <= nj < gy:
                    if self.bundle.mask_3d[ni, nj, t_block]:
                        return True
        return False

    def modify_single(self, trajectory: Trajectory) -> ModificationHistory:
        """Run the ST-iFGSM loop on a single trajectory.

        Steps:
        1. Determine time block and pickup mass
        2. Subtract this trajectory's contribution from the shared base
        3. Iteratively perturb the pickup location using signed gradients
        4. Persist the final change to the shared _base_pickup_3d
        """
        pickup_state = trajectory.states[-1]
        orig_cx = int(pickup_state.x_grid)
        orig_cy = int(pickup_state.y_grid)
        hour = time_bucket_to_hour(pickup_state.time_bucket)
        t_block = hour_to_block_index(hour)

        # Safeguard: skip if no active units in the neighborhood
        if not self._neighborhood_has_active_units((orig_cx, orig_cy), t_block):
            warnings.warn(
                f"Trajectory {trajectory.trajectory_id} pickup at "
                f"({orig_cx}, {orig_cy}, t={t_block}) has no active "
                f"neighbors — skipping.",
            )
            return ModificationHistory(
                original=trajectory, modified=trajectory.clone(),
                iterations=[], converged=False, total_iterations=0,
            )

        # pickup_mass = 1 / (n_hours_per_block[t*] * n_days)
        n_hours = int(self.bundle.n_hours_per_block[t_block])
        pickup_mass = 1.0 / (n_hours * self.bundle.n_days)

        # Subtract this trajectory's contribution BEFORE the loop
        base_3d = self._base_pickup_3d.clone()
        base_3d[orig_cx, orig_cy, t_block] -= pickup_mass

        original_pickup = np.array([float(orig_cx), float(orig_cy)], dtype=np.float32)
        cumulative_delta = np.zeros(2, dtype=np.float32)

        iterations: List[ModificationResult] = []
        prev_objective = float("-inf")
        converged = False

        for it in range(self.max_iterations):
            # (a) Anneal temperature
            if config.ANNEAL_TEMPERATURE:
                self.soft_assign.set_temperature(
                    self._get_annealed_temperature(it)
                )

            # (b) Build pickup_tensor with requires_grad=True
            current_pickup = original_pickup + cumulative_delta
            pickup_tensor = torch.tensor(
                current_pickup, dtype=torch.float32, requires_grad=True,
            )
            cell_tensor = torch.tensor(
                [orig_cx, orig_cy], dtype=torch.float32,
            ).unsqueeze(0)

            # (c) Compute soft probs -> inject into t_block slice
            probs = self.soft_assign(
                pickup_tensor.unsqueeze(0), cell_tensor,
            )[0]  # (ns, ns)

            soft_3d = inject_soft_counts_into_3d(
                base_3d, probs, (orig_cx, orig_cy), t_block,
                k=self.soft_assign.k, pickup_mass=pickup_mass,
            )

            # Build fidelity features if needed
            tau_features = None
            tau_prime_features = None
            ms_kwargs = None
            if self.objective.alpha_fidelity > 0:
                tau_features = trajectory.to_tensor().unsqueeze(0)
                tau_prime_features = tau_features.clone()
                tau_prime_features[0, -1, 0] = pickup_tensor[0]
                tau_prime_features[0, -1, 1] = pickup_tensor[1]
                if self.multi_stream_builder is not None:
                    modified = trajectory.apply_perturbation(cumulative_delta)
                    ms_kwargs = self.multi_stream_builder.build_fidelity_kwargs(
                        trajectory, modified,
                    )
                    # Inject gradient through slot 0 of x2 (+1 for 1-indexed coords)
                    x2 = ms_kwargs["x2"]
                    x2_new = x2.clone()
                    x2_new[0, 0, -1, 0] = pickup_tensor[0] + 1
                    x2_new[0, 0, -1, 1] = pickup_tensor[1] + 1
                    ms_kwargs["x2"] = x2_new

            # (d) Forward through FAMAILObjective
            total, terms = self.objective(
                soft_pickup_3d=soft_3d,
                tau_features=tau_features,
                tau_prime_features=tau_prime_features,
                multi_stream_kwargs=ms_kwargs,
            )

            # (e) Backward — zero_grad before backward to clear accumulated gradients
            self.objective.zero_grad()
            total.backward(retain_graph=True)

            if pickup_tensor.grad is None:
                grad = np.zeros(2)
            else:
                grad = pickup_tensor.grad.detach().cpu().numpy()
            grad_norm = float(np.linalg.norm(grad))

            # (f) ST-iFGSM: delta = clip(alpha * sign(grad), -epsilon, epsilon)
            if grad_norm > 1e-8:
                delta = self.alpha * np.sign(grad)
                cumulative_delta = np.clip(
                    cumulative_delta + delta, -self.epsilon, self.epsilon,
                ).astype(np.float32)

            # Clip pickup to grid bounds
            new_pickup = np.clip(
                original_pickup + cumulative_delta,
                [0.0, 0.0],
                [config.GRID_DIMS[0] - 1, config.GRID_DIMS[1] - 1],
            ).astype(np.float32)
            # Re-sync cumulative_delta after grid-clip
            cumulative_delta = new_pickup - original_pickup

            result = ModificationResult(
                iteration=it,
                objective_value=float(total.detach()),
                f_spatial=float(terms["f_spatial"].detach()),
                f_causal=float(terms["f_causal"].detach()),
                f_fidelity=float(terms["f_fidelity"].detach()),
                gradient_norm=grad_norm,
                cumulative_delta=cumulative_delta.copy(),
            )
            iterations.append(result)

            # (g) Convergence check
            if abs(float(total.detach()) - prev_objective) < self.convergence_tol:
                converged = True
                break
            prev_objective = float(total.detach())

        # Create the modified trajectory
        modified = trajectory.apply_perturbation(cumulative_delta)

        # Persist change to shared _base_pickup_3d (sub at old, add at new)
        new_cx = int(modified.pickup_state.x_grid)
        new_cy = int(modified.pickup_state.y_grid)
        if (new_cx, new_cy) != (orig_cx, orig_cy):
            self._base_pickup_3d[orig_cx, orig_cy, t_block] -= pickup_mass
            self._base_pickup_3d[new_cx, new_cy, t_block] += pickup_mass

        return ModificationHistory(
            original=trajectory,
            modified=modified,
            iterations=iterations,
            converged=converged,
            total_iterations=len(iterations),
            final_objective=iterations[-1].objective_value if iterations else 0.0,
        )

    def modify_batch(
        self, trajectories: List[Trajectory],
    ) -> List[ModificationHistory]:
        """Sequentially modify trajectories. Each sees the updated baseline
        from all prior modifications."""
        return [self.modify_single(t) for t in trajectories]
