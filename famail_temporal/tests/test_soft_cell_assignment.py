"""Tests for algorithm.soft_cell_assignment."""
import torch

from famail_temporal.algorithm.soft_cell_assignment import SoftCellAssignment


def test_soft_assignment_probs_shape():
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    loc = torch.tensor([[10.3, 20.7]])
    cell = torch.tensor([[10, 20]]).float()
    probs = s(loc, cell)
    assert probs.shape == (1, 5, 5)


def test_soft_assignment_probs_sum_to_one():
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    loc = torch.tensor([[10.3, 20.7]])
    cell = torch.tensor([[10, 20]]).float()
    probs = s(loc, cell)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_soft_assignment_set_temperature():
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    s.set_temperature(0.2)
    assert abs(float(s.temperature) - 0.2) < 1e-6


def test_gradient_flows_to_loc():
    """Gradient should flow from probs back to loc (requires_grad)."""
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    loc = torch.tensor([[10.3, 20.7]], requires_grad=True)
    cell = torch.tensor([[10, 20]]).float()
    probs = s(loc, cell)
    probs.sum().backward()
    assert loc.grad is not None
    assert not torch.isnan(loc.grad).any()


# === Hardening tests ===


def test_temperature_annealing_monotonic_decrease():
    """Annealed temperature should monotonically decrease from tau_max to tau_min."""
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    tau_max, tau_min = 1.0, 0.1
    total = 50
    temps = [s.get_annealed_temperature(i, total, tau_max, tau_min)
             for i in range(total)]
    # Monotonically decreasing
    for i in range(1, len(temps)):
        assert temps[i] <= temps[i - 1] + 1e-10, (
            f"Temperature increased at step {i}: {temps[i-1]:.6f} -> {temps[i]:.6f}"
        )
    # Endpoints
    assert abs(temps[0] - tau_max) < 1e-6
    assert abs(temps[-1] - tau_min) < 1e-6


def test_soft_to_hard_convergence():
    """At very low temperature, mass should concentrate on the nearest cell."""
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=0.001)
    # loc at (10.3, 20.7) -> nearest cell center is (10+0.5, 20+0.5)=(10.5,20.5)
    # which is cell (10, 20) in the neighborhood, i.e. offset (0,0) -> index (2,2)
    loc = torch.tensor([[10.3, 20.7]])
    cell = torch.tensor([[10, 20]]).float()
    probs = s(loc, cell)
    # The center cell (offset 0,0) should have nearly all the mass
    assert probs[0, 2, 2] > 0.99, (
        f"Center cell prob {probs[0, 2, 2]:.4f} should be > 0.99 at low temperature"
    )


def test_probability_non_negativity():
    """All probabilities should be >= 0 (guaranteed by softmax, but verify)."""
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    loc = torch.tensor([[10.3, 20.7], [5.9, 80.1], [0.0, 0.0]])
    cell = torch.tensor([[10, 20], [5, 80], [0, 0]]).float()
    probs = s(loc, cell)
    assert (probs >= 0).all(), "Found negative probabilities"


def test_batch_independent():
    """Two different locations in a batch should produce different distributions."""
    s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                           initial_temperature=1.0)
    loc = torch.tensor([[10.3, 20.7], [10.8, 20.2]])
    cell = torch.tensor([[10, 20], [10, 20]]).float()
    probs = s(loc, cell)
    # The two distributions should differ
    assert not torch.allclose(probs[0], probs[1]), (
        "Distributions for different locations should differ"
    )


# === inject_soft_counts_into_3d tests ===

from famail_temporal.algorithm.soft_cell_assignment import inject_soft_counts_into_3d


def test_inject_only_modifies_t_block_slice():
    base = torch.zeros(48, 90, 4)
    base[:, :, 1] = 5.0
    probs = torch.ones(5, 5) / 25.0
    out = inject_soft_counts_into_3d(
        base_counts_3d=base, probs_2d=probs,
        cell_xy=(10, 20), t_block=0, k=2, pickup_mass=1.0,
    )
    assert torch.equal(out[:, :, 1], base[:, :, 1])
    changed = (out[:, :, 0] != base[:, :, 0]).sum()
    assert changed == 25


def test_inject_mass_balance():
    base = torch.zeros(48, 90, 4)
    probs = torch.rand(5, 5)
    probs = probs / probs.sum()
    pickup_mass = 0.01
    out = inject_soft_counts_into_3d(
        base, probs, cell_xy=(10, 20), t_block=0, k=2, pickup_mass=pickup_mass,
    )
    total_injected = (out[:, :, 0] - base[:, :, 0]).sum()
    assert torch.isclose(total_injected, torch.tensor(pickup_mass), atol=1e-5)


def test_inject_preserves_gradient():
    base = torch.zeros(48, 90, 4)
    probs = torch.rand(5, 5, requires_grad=True)
    out = inject_soft_counts_into_3d(
        base, probs, cell_xy=(10, 20), t_block=0, k=2, pickup_mass=1.0,
    )
    out.sum().backward()
    assert probs.grad is not None
    assert not torch.isnan(probs.grad).any()


# === inject_soft_counts_into_3d hardening tests ===


def test_inject_edge_cell_boundary():
    """Cell at (0, 0) with k=2 should clip neighborhood to valid cells only."""
    base = torch.zeros(48, 90, 4)
    probs = torch.ones(5, 5) / 25.0
    out = inject_soft_counts_into_3d(
        base, probs, cell_xy=(0, 0), t_block=0, k=2, pickup_mass=1.0,
    )
    # Only 3x3 cells in-bounds (rows 0-2, cols 0-2)
    changed = (out[:, :, 0] != base[:, :, 0]).sum()
    assert changed == 9
    # Mass should be less than 1.0 since some cells are clipped
    total = (out[:, :, 0] - base[:, :, 0]).sum()
    assert total < 1.0


def test_inject_edge_cell_top_right():
    """Cell at (47, 89) with k=2 should clip at the upper grid boundary."""
    base = torch.zeros(48, 90, 4)
    probs = torch.ones(5, 5) / 25.0
    out = inject_soft_counts_into_3d(
        base, probs, cell_xy=(47, 89), t_block=0, k=2, pickup_mass=1.0,
    )
    # Only 3x3 cells in-bounds (rows 45-47, cols 87-89)
    changed = (out[:, :, 0] != base[:, :, 0]).sum()
    assert changed == 9


def test_inject_t_block_out_of_range():
    """t_block out of range should raise AssertionError."""
    base = torch.zeros(48, 90, 4)
    probs = torch.ones(5, 5) / 25.0
    import pytest
    with pytest.raises(AssertionError):
        inject_soft_counts_into_3d(
            base, probs, cell_xy=(10, 20), t_block=5, k=2, pickup_mass=1.0,
        )


def test_inject_zero_pickup_mass():
    """Zero pickup_mass should produce no modification."""
    base = torch.zeros(48, 90, 4)
    probs = torch.rand(5, 5)
    probs = probs / probs.sum()
    out = inject_soft_counts_into_3d(
        base, probs, cell_xy=(10, 20), t_block=0, k=2, pickup_mass=0.0,
    )
    assert torch.equal(out, base)
