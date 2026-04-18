"""Tests for fairness.spatial."""
import torch
import pytest

from famail_temporal.fairness.spatial import pairwise_gini, compute_fspatial


def test_gini_equal_values_zero():
    values = torch.full((20,), 3.0)
    assert float(pairwise_gini(values)) < 1e-6


def test_gini_one_hot_approaches_max():
    values = torch.zeros(10)
    values[0] = 100.0
    g = float(pairwise_gini(values))
    assert 0.85 < g <= 0.91


def test_fspatial_perfect_equality():
    N = 30
    pickup = torch.full((N,), 2.0)
    dropoff = torch.full((N,), 2.0)
    active = torch.full((N,), 4.0)
    f, _ = compute_fspatial(pickup, dropoff, active)
    assert float(f) > 0.999


def test_fspatial_bounded():
    N = 50
    torch.manual_seed(42)
    pickup = torch.rand(N) * 5.0
    dropoff = torch.rand(N) * 5.0
    active = torch.rand(N) * 3.0 + 1.0
    f, _ = compute_fspatial(pickup, dropoff, active)
    assert 0.0 <= float(f) <= 1.0


# --- Hardening tests ---

def test_fspatial_shape_mismatch_raises():
    pickup = torch.ones(10)
    dropoff = torch.ones(10)
    active = torch.ones(9)  # wrong length
    with pytest.raises(ValueError, match="same shape"):
        compute_fspatial(pickup, dropoff, active)


def test_fspatial_not_1d_raises():
    pickup = torch.ones(5, 2)
    dropoff = torch.ones(5, 2)
    active = torch.ones(5, 2)
    with pytest.raises(ValueError, match="1-D"):
        compute_fspatial(pickup, dropoff, active)


def test_fspatial_negative_values_raises():
    pickup = torch.tensor([-1.0, 1.0, 2.0])
    dropoff = torch.ones(3)
    active = torch.ones(3)
    with pytest.raises(ValueError, match="negative"):
        compute_fspatial(pickup, dropoff, active)


def test_gini_scale_invariance():
    """Gini(c·x) == Gini(x) for c > 0 — scale invariance."""
    torch.manual_seed(7)
    values = torch.rand(30) * 10.0
    g1 = pairwise_gini(values)
    g2 = pairwise_gini(values * 5.0)
    assert abs(float(g1) - float(g2)) < 1e-5


def test_fspatial_gradient_flows():
    """Gradient must flow from F_spatial back to pickup_N (needed for ST-iFGSM)."""
    N = 20
    pickup = torch.rand(N, requires_grad=True)
    dropoff = torch.rand(N)
    active = torch.rand(N) + 1.0
    f, _ = compute_fspatial(pickup, dropoff, active)
    f.backward()
    assert pickup.grad is not None
    assert not torch.all(pickup.grad == 0), "all-zero gradient — no gradient flow"


def test_gini_n_le_one_returns_zero():
    """Single-element and empty tensors return 0 Gini."""
    assert float(pairwise_gini(torch.tensor([3.0]))) == 0.0
    assert float(pairwise_gini(torch.tensor([]))) == 0.0


def test_gini_all_zeros_returns_zero():
    """All-zero tensor returns ~0 Gini (mean clamp doesn't affect uniform zero case)."""
    assert float(pairwise_gini(torch.zeros(10))) < 1e-6


def test_debug_dict_has_keys():
    """compute_fspatial debug dict must contain gini_dsr and gini_asr."""
    N = 10
    pickup = torch.ones(N)
    dropoff = torch.ones(N)
    active = torch.ones(N) * 2.0
    _, debug = compute_fspatial(pickup, dropoff, active)
    assert "gini_dsr" in debug
    assert "gini_asr" in debug
