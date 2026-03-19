"""
Per-term gradient decomposition for the FAMAIL objective function.

Computes individual gradient vectors ∂F_spatial/∂pos, ∂F_causal/∂pos,
∂F_fidelity/∂pos at each ST-iFGSM iteration. This requires 3 separate
backward passes (one per term) plus the combined backward pass,
so computational cost is roughly 4x per iteration vs standard mode.

The decomposition reveals which objective term is driving each modification
and whether terms cooperate (similar gradient direction) or conflict
(opposing gradient direction).

Key insight: F_fidelity gradient w.r.t. the soft-cell pickup position is
typically zero because the fidelity term receives trajectory features via
a numpy→tensor conversion that breaks the computational graph. The fidelity
gradient acts through a separate path (the multi-stream x2 tensor).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from experiment_framework.experiment_config import ExperimentConfig
from experiment_framework.experiment_result import (
    TrajectoryResult,
    IterationRecord,
    GradientDecomposition,
)

logger = logging.getLogger(__name__)


class GradientDecomposer:
    """Compute per-term gradients for the FAMAIL objective.

    Usage:
        decomp = GradientDecomposer.decompose(pickup_tensor, terms, total)
    """

    @staticmethod
    def decompose(
        pickup_tensor: torch.Tensor,
        terms: Dict[str, torch.Tensor],
        total: torch.Tensor,
    ) -> GradientDecomposition:
        """Compute per-term gradients via separate backward passes.

        Must be called BEFORE any standalone backward() on total,
        because all calls use retain_graph=True to preserve the graph.

        Args:
            pickup_tensor: [2] tensor with requires_grad=True
            terms: {'f_spatial': tensor, 'f_causal': tensor, 'f_fidelity': tensor}
            total: combined objective tensor

        Returns:
            GradientDecomposition with per-term gradient vectors and derived metrics
        """
        grads = {}

        for term_name in ('f_spatial', 'f_causal', 'f_fidelity'):
            term_val = terms.get(term_name)
            if term_val is None or not term_val.requires_grad:
                grads[term_name] = np.zeros(2)
                continue

            pickup_tensor.grad = None
            try:
                term_val.backward(retain_graph=True)
                if pickup_tensor.grad is not None:
                    grads[term_name] = pickup_tensor.grad.detach().cpu().numpy().copy()
                else:
                    grads[term_name] = np.zeros(2)
            except RuntimeError:
                # If the term doesn't have a path to pickup_tensor
                grads[term_name] = np.zeros(2)

        # Combined gradient (this is what ST-iFGSM actually uses)
        pickup_tensor.grad = None
        total.backward(retain_graph=True)
        grad_combined = (
            pickup_tensor.grad.detach().cpu().numpy().copy()
            if pickup_tensor.grad is not None
            else np.zeros(2)
        )

        # Compute derived metrics
        eps = 1e-10
        norms = {k: float(np.linalg.norm(v)) for k, v in grads.items()}
        total_norm = sum(norms.values()) + eps

        spatial_frac = norms['f_spatial'] / total_norm
        causal_frac = norms['f_causal'] / total_norm
        fidelity_frac = norms['f_fidelity'] / total_norm

        # Cosine similarity between spatial and causal gradients
        gs = grads['f_spatial']
        gc = grads['f_causal']
        ns = norms['f_spatial']
        nc = norms['f_causal']
        if ns > eps and nc > eps:
            alignment = float(np.dot(gs, gc) / (ns * nc))
        else:
            alignment = 0.0

        return GradientDecomposition(
            grad_spatial=grads['f_spatial'].tolist(),
            grad_causal=grads['f_causal'].tolist(),
            grad_fidelity=grads['f_fidelity'].tolist(),
            grad_combined=grad_combined.tolist(),
            spatial_fraction=spatial_frac,
            causal_fraction=causal_frac,
            fidelity_fraction=fidelity_frac,
            alignment_spatial_causal=alignment,
        )


def run_modification_with_decomposition(
    trajectory,
    traj_idx: int,
    modifier,
    device: str,
    config: ExperimentConfig,
) -> TrajectoryResult:
    """Re-implements the soft_cell ST-iFGSM loop with gradient decomposition.

    This mirrors TrajectoryModifier.modify_single() but inserts per-term
    backward passes at each iteration before the combined backward.

    Args:
        trajectory: Original Trajectory object
        traj_idx: Index in the trajectory list
        modifier: TrajectoryModifier instance (provides objective_fn, soft_assign, counts)
        device: torch device string
        config: ExperimentConfig for parameters

    Returns:
        TrajectoryResult with gradient decomposition in each IterationRecord
    """
    objective_fn = modifier.objective_fn
    soft_assign = modifier.soft_assign

    # Move soft_assign to device
    if soft_assign is not None:
        soft_assign = soft_assign.to(device)

    # Pickup state
    pickup_idx = modifier._get_pickup_index(trajectory)
    pickup_state = trajectory.states[pickup_idx]
    original_pickup = np.array([pickup_state.x_grid, pickup_state.y_grid], dtype=np.float32)
    original_cell = torch.tensor([int(pickup_state.x_grid), int(pickup_state.y_grid)], device=device)
    current_pickup = original_pickup.copy()

    modified = trajectory.clone()
    cumulative_delta = np.zeros(2)
    iterations: List[IterationRecord] = []
    prev_objective = float('-inf')

    for t in range(config.max_iterations):
        # Temperature annealing
        if config.temperature_annealing and soft_assign is not None:
            current_temp = modifier._get_annealed_temperature(t)
            soft_assign.set_temperature(current_temp)

        # Differentiable pickup tensor
        pickup_tensor = torch.tensor(
            current_pickup, dtype=torch.float32, device=device, requires_grad=True,
        )

        # Trajectory features for fidelity
        tau_features = trajectory.to_tensor().unsqueeze(0).to(device)
        tau_prime_features = modified.to_tensor().unsqueeze(0).to(device)

        # Multi-stream fidelity kwargs
        fidelity_kwargs = {}
        if modifier.multi_stream_context is not None:
            fidelity_kwargs = modifier.multi_stream_context.build_fidelity_kwargs(
                trajectory, modified,
            )

        # Soft cell assignment → differentiable counts
        soft_pickup_counts = modifier._compute_soft_pickup_counts(pickup_tensor, original_cell)

        # Forward pass
        total, terms = objective_fn(
            pickup_counts=soft_pickup_counts,
            dropoff_counts=modifier.dropoff_counts,
            supply=modifier.causal_supply,
            active_taxis=modifier.active_taxis,
            tau_features=tau_features,
            tau_prime_features=tau_prime_features,
            causal_demand=modifier.causal_demand,
            **fidelity_kwargs,
        )

        # --- Gradient decomposition (3 per-term + 1 combined backward) ---
        decomp = GradientDecomposer.decompose(pickup_tensor, terms, total)

        # The combined backward was done inside decompose; extract gradient
        grad = (
            pickup_tensor.grad.detach().cpu().numpy()
            if pickup_tensor.grad is not None
            else np.zeros(2)
        )
        grad_norm = float(np.linalg.norm(grad))

        # ST-iFGSM perturbation
        if grad_norm > 1e-8:
            delta = config.alpha * np.sign(grad)
            cumulative_delta = np.clip(
                cumulative_delta + delta, -config.epsilon, config.epsilon,
            )

        # Apply perturbation
        old_cell = (int(current_pickup[0]), int(current_pickup[1]))
        new_pickup = modifier._clip_to_grid(original_pickup + cumulative_delta)
        new_cell = (int(new_pickup[0]), int(new_pickup[1]))

        if old_cell != new_cell:
            modifier._update_counts(old_cell, new_cell)

        current_pickup = new_pickup
        modified = trajectory.apply_perturbation(cumulative_delta)

        # Record iteration
        iterations.append(IterationRecord(
            iteration=t,
            objective=total.item(),
            f_spatial=terms['f_spatial'].item(),
            f_causal=terms['f_causal'].item(),
            f_fidelity=terms['f_fidelity'].item(),
            gradient_norm=grad_norm,
            perturbation=cumulative_delta.copy().tolist(),
            gradient_decomposition=decomp,
        ))

        # Convergence check
        if abs(total.item() - prev_objective) < config.convergence_threshold:
            break
        prev_objective = total.item()

    # Sync base counts
    if modifier._base_pickup_counts is not None:
        modifier._base_pickup_counts = modifier.pickup_counts.clone()
    if modifier._base_dropoff_counts is not None:
        modifier._base_dropoff_counts = modifier.dropoff_counts.clone()

    # Build result
    last = iterations[-1] if iterations else None
    orig_cell = trajectory.pickup_cell
    mod_cell = modified.pickup_cell
    dx = mod_cell[0] - orig_cell[0]
    dy = mod_cell[1] - orig_cell[1]
    perturbation_mag = float(np.sqrt(dx**2 + dy**2))

    return TrajectoryResult(
        trajectory_index=traj_idx,
        trajectory_id=str(getattr(trajectory, 'trajectory_id', traj_idx)),
        driver_id=int(trajectory.driver_id),
        original_pickup=list(orig_cell),
        modified_pickup=list(mod_cell),
        converged=(len(iterations) < config.max_iterations),
        total_iterations=len(iterations),
        final_objective=total.item() if iterations else 0.0,
        final_f_spatial=float(last.f_spatial) if last else 0.0,
        final_f_causal=float(last.f_causal) if last else 0.0,
        final_f_fidelity=float(last.f_fidelity) if last else 0.0,
        perturbation_magnitude=perturbation_mag,
        iterations=iterations,
    )
