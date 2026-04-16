"""Tests for fairness.causal."""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.fairness.causal import (
    compute_fcausal,
    per_unit_attribution,
    per_unit_attribution_signed,
)
from famail_temporal.fairness.hat_matrices import (
    precompute_hat_matrices,
    compute_fcausal_torch,
)


def _make_hat(N, seed):
    rng = np.random.RandomState(seed)
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo'].copy()).float()
    M = torch.from_numpy(hat['M'].copy()).float()
    return D, demo, IH, M


# ---------------------------------------------------------------------------
# Required tests (from the task description)
# ---------------------------------------------------------------------------

def test_fcausal_in_unit_interval():
    N = 40
    D, _, IH, M = _make_hat(N, seed=10)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(11).randn(N)) * 2.0 + 1.0
    ).float()
    d_t = torch.from_numpy(D).float()
    g0_D = torch.full((N,), 0.5)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert 0.0 <= float(f) <= 1.0


def test_attribution_sums_to_one_minus_fcausal():
    N = 80
    D, _, IH, M = _make_hat(N, seed=12)
    R = torch.from_numpy(np.random.RandomState(13).randn(N) * 2.0).float()
    f = compute_fcausal_torch(R, IH, M)
    attr = per_unit_attribution(R, IH, M)
    assert abs(float(attr.sum()) - (1.0 - float(f))) < 1e-5


def test_attribution_shape():
    N = 50
    D, _, IH, M = _make_hat(N, seed=14)
    R = torch.randn(N)
    attr = per_unit_attribution(R, IH, M)
    assert attr.shape == (N,)


# ---------------------------------------------------------------------------
# Hardening: gradient flow through compute_fcausal
# ---------------------------------------------------------------------------

def test_compute_fcausal_gradient_flows_through_demand():
    """Gradient should flow: demand_N -> Y -> R -> F_causal."""
    N = 40
    D_np, _, IH, M = _make_hat(N, seed=20)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(21).randn(N)) * 2.0 + 1.0
    ).float()
    # Use perturbed demand so Y won't exactly match g0_D (avoid degenerate branch)
    d_t = torch.from_numpy(D_np).float().clone().requires_grad_(True)
    g0_D = torch.zeros(N)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    f.backward()
    assert d_t.grad is not None
    assert not torch.isnan(d_t.grad).any()
    assert not torch.isinf(d_t.grad).any()
    assert (d_t.grad.abs() > 0).any(), "Expected nonzero gradient w.r.t. demand"


def test_compute_fcausal_debug_dict_keys():
    """Debug dict should contain the documented keys."""
    N = 30
    D_np, _, IH, M = _make_hat(N, seed=22)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(23).randn(N)) * 2.0 + 1.0
    ).float()
    d_t = torch.from_numpy(D_np).float()
    g0_D = torch.zeros(N)
    _, debug = compute_fcausal(d_t, supply, g0_D, IH, M)
    for key in ('Y_min', 'Y_max', 'R_min', 'R_max', 'f_causal'):
        assert key in debug


# ---------------------------------------------------------------------------
# Hardening: demand clamping (floor) doesn't break gradient
# ---------------------------------------------------------------------------

def test_compute_fcausal_demand_floor_clamp():
    """Zero demand should be clamped to DEMAND_FLOOR; Y must be finite."""
    N = 40
    _, _, IH, M = _make_hat(N, seed=24)
    # Some demand values at or below zero
    rng = np.random.RandomState(25)
    D_np = rng.uniform(0.5, 2.0, N)
    D_np[:5] = 0.0
    D_np[5:10] = -0.01  # force clamp
    supply = torch.from_numpy(rng.uniform(0.5, 3.0, N)).float()
    d_t = torch.from_numpy(D_np).float()
    g0_D = torch.zeros(N)
    f, debug = compute_fcausal(d_t, supply, g0_D, IH, M)
    # Y_max should be bounded: supply/floor is finite, no inf
    assert np.isfinite(debug['Y_min'])
    assert np.isfinite(debug['Y_max'])
    assert 0.0 <= float(f) <= 1.0
    # Upper bound: Y_max <= max(supply) / DEMAND_FLOOR
    max_possible = float(supply.max()) / config.DEMAND_FLOOR
    assert debug['Y_max'] <= max_possible + 1e-6


# ---------------------------------------------------------------------------
# Hardening: per-unit attribution under torch.no_grad
# ---------------------------------------------------------------------------

def test_per_unit_attribution_no_grad():
    """per_unit_attribution must not build a graph — it's a reporting fn."""
    N = 30
    _, _, IH, M = _make_hat(N, seed=30)
    R = torch.randn(N, requires_grad=True)
    attr = per_unit_attribution(R, IH, M)
    assert attr.grad_fn is None
    assert not attr.requires_grad


# ---------------------------------------------------------------------------
# Hardening: per-unit attribution decomposition invariant
# (bolted-on verification at the consumer-facing API)
# ---------------------------------------------------------------------------

def test_attribution_invariant_matches_manual_decomposition():
    """attribution_i == ((MR)_i^2 - ((I-H)R)_i^2) / R'MR — checks the formula."""
    N = 60
    _, _, IH, M = _make_hat(N, seed=40)
    rng = np.random.RandomState(41)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    attr = per_unit_attribution(R, IH, M)

    # Manual computation
    MR = M @ R
    IHR = IH @ R
    ss_tot = (MR ** 2).sum() + config.EPS
    manual = (MR ** 2 - IHR ** 2) / ss_tot

    torch.testing.assert_close(attr, manual, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Hardening: signed attribution magnitudes match unsigned
# ---------------------------------------------------------------------------

def test_signed_attribution_magnitudes_equal_unsigned_magnitudes():
    """|signed| == |unsigned| — sign() * x has same magnitude as x.

    Note: unsigned attribution can itself be negative for individual units
    (when ((I-H)R)_i^2 > (MR)_i^2), so we compare absolute values, not the
    raw unsigned against the signed magnitude.
    """
    N = 50
    _, _, IH, M = _make_hat(N, seed=50)
    rng = np.random.RandomState(51)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    unsigned = per_unit_attribution(R, IH, M)
    signed = per_unit_attribution_signed(R, IH, M)
    torch.testing.assert_close(signed.abs(), unsigned.abs(), rtol=1e-5, atol=1e-6)


def test_signed_attribution_sign_correctness():
    """If demographics predict R positive for a unit, signed attribution > 0.

    Construct R that is (up to noise) in the demographic span (from the
    intercept + the standardized demographic columns). Then HR ≈ R, which is
    positive-heavy, so signed attribution should be positive for those units.
    """
    from sklearn.preprocessing import StandardScaler

    N = 60
    rng = np.random.RandomState(60)
    D_np = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D_np, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo'].copy()).float()
    M = torch.from_numpy(hat['M'].copy()).float()

    # Construct R that lies (mostly) in the demographic span
    X_scaled = StandardScaler().fit_transform(demo)
    # R_demo = intercept + beta * demo (positive for some units, negative for others)
    R_np = 1.0 + 2.0 * X_scaled[:, 0] + 0.5 * X_scaled[:, 1]
    R = torch.from_numpy(R_np).float()

    signed = per_unit_attribution_signed(R, IH, M)
    # HR should track R closely for units where R is well-explained by demographics
    HR = R - IH @ R
    # For every unit, sign(signed_attribution) must match sign(HR)
    # (modulo zeros, which we skip)
    nonzero = HR.abs() > 1e-4
    assert torch.all(torch.sign(signed[nonzero]) == torch.sign(HR[nonzero]))


def test_signed_attribution_no_grad():
    """Signed attribution is also a reporting function — no gradient."""
    N = 30
    _, _, IH, M = _make_hat(N, seed=70)
    R = torch.randn(N, requires_grad=True)
    signed = per_unit_attribution_signed(R, IH, M)
    assert signed.grad_fn is None
    assert not signed.requires_grad


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_compute_fcausal_zero_residual_degenerate():
    """If R = 0 everywhere (Y == g_0(D) identically), fall into degenerate branch -> 1.0."""
    N = 30
    _, _, IH, M = _make_hat(N, seed=80)
    # supply / demand == g0 -> R = 0 -> ss_tot ~ 0 -> degenerate branch returns 1.0
    d_t = torch.full((N,), 2.0)
    supply = torch.full((N,), 1.0)  # Y = 0.5 everywhere
    g0_D = torch.full((N,), 0.5)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert float(f) == 1.0


# ---------------------------------------------------------------------------
# I3: NaN / Inf guard on g0_D_N
# ---------------------------------------------------------------------------

def test_compute_fcausal_rejects_nan_g0():
    """NaN in g0_D_N must raise ValueError at entry, not propagate."""
    N = 40
    rng = np.random.RandomState(17)
    hat = precompute_hat_matrices(
        rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"]
    )
    IH = torch.from_numpy(hat['I_minus_H_demo'].copy()).float()
    M = torch.from_numpy(hat['M'].copy()).float()
    D = torch.from_numpy(rng.uniform(0.5, 5.0, N)).float()
    S = torch.from_numpy(rng.uniform(1.0, 10.0, N)).float()
    g0 = torch.full((N,), 0.5)
    g0[0] = float('nan')
    with pytest.raises(ValueError, match="non-finite"):
        compute_fcausal(D, S, g0, IH, M)


def test_compute_fcausal_rejects_inf_g0():
    """Inf in g0_D_N must raise ValueError at entry."""
    N = 40
    rng = np.random.RandomState(18)
    hat = precompute_hat_matrices(
        rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"]
    )
    IH = torch.from_numpy(hat['I_minus_H_demo'].copy()).float()
    M = torch.from_numpy(hat['M'].copy()).float()
    D = torch.from_numpy(rng.uniform(0.5, 5.0, N)).float()
    S = torch.from_numpy(rng.uniform(1.0, 10.0, N)).float()
    g0 = torch.full((N,), 0.5)
    g0[5] = float('inf')
    with pytest.raises(ValueError, match="non-finite"):
        compute_fcausal(D, S, g0, IH, M)
