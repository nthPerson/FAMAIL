"""End-to-end gradient flow tests for the full FAMAIL objective.

These tests verify the load-bearing property: gradient from the total objective
flows back to a pickup_tensor (x, y) through the entire chain:
    pickup_tensor -> SoftCellAssignment -> inject_soft_counts_into_3d ->
    FAMAILObjective -> total -> backward -> pickup_tensor.grad
"""
import numpy as np
import torch

from famail_temporal import config
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.soft_cell_assignment import (
    SoftCellAssignment, inject_soft_counts_into_3d,
)
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_gradient_flows_through_pooled_objective():
    """Gradient from total objective flows to a pickup_tensor (x, y)."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    cell = bundle.unit_map.to_flat_cell(0)
    t_block = bundle.unit_map.to_time_block(0)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy

    pickup_tensor = torch.tensor([float(x), float(y)], requires_grad=True)
    soft = SoftCellAssignment()
    cell_t = torch.tensor([x, y]).float().unsqueeze(0)
    probs = soft(pickup_tensor.unsqueeze(0), cell_t)[0]

    base_3d = torch.from_numpy(bundle.pickup_3d).float()
    pickup_mass = 1.0 / (int(bundle.n_hours_per_block[t_block]) * bundle.n_days)
    soft_3d = inject_soft_counts_into_3d(
        base_3d, probs, (x, y), t_block, k=soft.k, pickup_mass=pickup_mass,
    )

    total, _ = obj(soft_pickup_3d=soft_3d)
    total.backward()

    assert pickup_tensor.grad is not None
    assert not torch.isnan(pickup_tensor.grad).any()
    assert not torch.isinf(pickup_tensor.grad).any()


def test_gradient_only_flows_through_correct_t_block():
    """The gradient should affect only the target time block's slice."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    cell = bundle.unit_map.to_flat_cell(0)
    t_block = bundle.unit_map.to_time_block(0)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy

    pickup_tensor = torch.tensor([float(x), float(y)], requires_grad=True)
    soft = SoftCellAssignment()
    cell_t = torch.tensor([x, y]).float().unsqueeze(0)
    probs = soft(pickup_tensor.unsqueeze(0), cell_t)[0]

    base_3d = torch.from_numpy(bundle.pickup_3d).float()
    soft_3d = inject_soft_counts_into_3d(
        base_3d, probs, (x, y), t_block, k=soft.k, pickup_mass=1.0,
    )

    # Only the t_block slice was modified; others are identical to base
    for t in range(config.T):
        if t == t_block:
            continue
        assert torch.equal(soft_3d[:, :, t], base_3d[:, :, t])


def test_gradient_magnitude_is_nonzero():
    """The gradient at pickup_tensor should have nonzero magnitude.

    If the gradient is exactly zero, the ST-iFGSM loop cannot make progress.
    This guards against silent gradient-killing bugs in the chain.
    """
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    cell = bundle.unit_map.to_flat_cell(0)
    t_block = bundle.unit_map.to_time_block(0)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy

    pickup_tensor = torch.tensor([float(x), float(y)], requires_grad=True)
    soft = SoftCellAssignment()
    cell_t = torch.tensor([x, y]).float().unsqueeze(0)
    probs = soft(pickup_tensor.unsqueeze(0), cell_t)[0]

    base_3d = torch.from_numpy(bundle.pickup_3d).float()
    pickup_mass = 1.0 / (int(bundle.n_hours_per_block[t_block]) * bundle.n_days)
    soft_3d = inject_soft_counts_into_3d(
        base_3d, probs, (x, y), t_block, k=soft.k, pickup_mass=pickup_mass,
    )

    total, _ = obj(soft_pickup_3d=soft_3d)
    total.backward()

    grad_norm = pickup_tensor.grad.norm().item()
    assert grad_norm > 1e-10, (
        f"Gradient norm {grad_norm} is effectively zero — "
        f"ST-iFGSM loop would make no progress"
    )
