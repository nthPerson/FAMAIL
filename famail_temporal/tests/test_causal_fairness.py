"""Tests for fairness.causal — F_causal + canonical per-cell attribution.

The single canonical attribution is ``per_cell_fairness_attribution_causal``,
which uses the compact FWL form (X_demo, XtX_inv) and sums to F_causal
(not 1 - F_causal). See ``docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``.
"""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.fairness.causal import (
    compute_fcausal,
    per_cell_fairness_attribution_causal,
)
from famail_temporal.fairness.hat_matrices import (
    precompute_hat_matrices,
    compute_fcausal_torch,
    hat_matrices_to_torch,
    apply_i_minus_h,
)


def _make_hat(N, seed):
    rng = np.random.RandomState(seed)
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo'].copy()).float()
    M = torch.from_numpy(hat['M'].copy()).float()
    tensors = hat_matrices_to_torch(hat)
    return D, demo, IH, M, tensors


# ---------------------------------------------------------------------------
# F_causal scalar metric
# ---------------------------------------------------------------------------

def test_fcausal_in_unit_interval():
    N = 40
    D, _, IH, M, _ = _make_hat(N, seed=10)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(11).randn(N)) * 2.0 + 1.0
    ).float()
    d_t = torch.from_numpy(D).float()
    g0_D = torch.full((N,), 0.5)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert 0.0 <= float(f) <= 1.0


def test_compute_fcausal_gradient_flows_through_demand():
    """Gradient should flow: demand_N -> Y -> R -> F_causal."""
    N = 40
    D_np, _, IH, M, _ = _make_hat(N, seed=20)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(21).randn(N)) * 2.0 + 1.0
    ).float()
    d_t = torch.from_numpy(D_np).float().clone().requires_grad_(True)
    g0_D = torch.zeros(N)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    f.backward()
    assert d_t.grad is not None
    assert not torch.isnan(d_t.grad).any()
    assert not torch.isinf(d_t.grad).any()
    assert (d_t.grad.abs() > 0).any()


def test_compute_fcausal_debug_dict_keys():
    """Debug dict should contain the documented keys."""
    N = 30
    D_np, _, IH, M, _ = _make_hat(N, seed=22)
    supply = torch.from_numpy(
        np.abs(np.random.RandomState(23).randn(N)) * 2.0 + 1.0
    ).float()
    d_t = torch.from_numpy(D_np).float()
    g0_D = torch.zeros(N)
    _, debug = compute_fcausal(d_t, supply, g0_D, IH, M)
    for key in ('Y_min', 'Y_max', 'R_min', 'R_max', 'f_causal'):
        assert key in debug


def test_compute_fcausal_demand_floor_clamp():
    """Zero demand should be clamped to DEMAND_FLOOR; Y must be finite."""
    N = 40
    _, _, IH, M, _ = _make_hat(N, seed=24)
    rng = np.random.RandomState(25)
    D_np = rng.uniform(0.5, 2.0, N)
    D_np[:5] = 0.0
    D_np[5:10] = -0.01
    supply = torch.from_numpy(rng.uniform(0.5, 3.0, N)).float()
    d_t = torch.from_numpy(D_np).float()
    g0_D = torch.zeros(N)
    f, debug = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert np.isfinite(debug['Y_min'])
    assert np.isfinite(debug['Y_max'])
    assert 0.0 <= float(f) <= 1.0
    max_possible = float(supply.max()) / config.DEMAND_FLOOR
    assert debug['Y_max'] <= max_possible + 1e-6


def test_compute_fcausal_zero_residual_degenerate():
    """If R = 0 everywhere (Y == g_0(D) identically), fall into degenerate branch -> 1.0."""
    N = 30
    _, _, IH, M, _ = _make_hat(N, seed=80)
    d_t = torch.full((N,), 2.0)
    supply = torch.full((N,), 1.0)
    g0_D = torch.full((N,), 0.5)
    f, _ = compute_fcausal(d_t, supply, g0_D, IH, M)
    assert float(f) == 1.0


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


# ---------------------------------------------------------------------------
# per_cell_fairness_attribution_causal — canonical 1/N-shifted decomposition
# ---------------------------------------------------------------------------

def test_attribution_shape():
    """Returns a 1-D tensor of length N."""
    N = 50
    _, _, _, _, tensors = _make_hat(N, seed=14)
    R = torch.randn(N)
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    assert attr.dim() == 1
    assert attr.shape == (N,)


def test_attribution_sums_to_fcausal():
    """Σᵢ αᵢ == F_causal — the load-bearing decomposition identity."""
    N = 80
    _, _, IH, M, tensors = _make_hat(N, seed=12)
    R = torch.from_numpy(np.random.RandomState(13).randn(N) * 2.0).float()
    f = compute_fcausal_torch(R, IH, M)
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    assert abs(float(attr.sum()) - float(f)) < 1e-5


def test_attribution_no_grad():
    """per_cell_fairness_attribution_causal must not build a graph — reporting fn."""
    N = 30
    _, _, _, _, tensors = _make_hat(N, seed=30)
    R = torch.randn(N, requires_grad=True)
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    assert attr.grad_fn is None
    assert not attr.requires_grad


def test_attribution_invariant_matches_manual_decomposition():
    """αᵢ == 1/N - ((MR)ᵢ² - ((I-H)R)ᵢ²) / R'MR — checks the formula directly."""
    N = 60
    _, _, IH, M, tensors = _make_hat(N, seed=40)
    rng = np.random.RandomState(41)
    R = torch.from_numpy(rng.randn(N) * 2.0).float()
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])

    MR = M @ R
    IHR = IH @ R
    ss_tot = (MR ** 2).sum() + config.EPS
    manual = (1.0 / N) - (MR ** 2 - IHR ** 2) / ss_tot

    torch.testing.assert_close(attr, manual, rtol=1e-5, atol=1e-6)


def test_attribution_compact_matches_dense_via_apply_i_minus_h():
    """Compact-form (X_demo, XtX_inv) matches the dense-(I-H) computation."""
    N = 40
    _, _, IH, _, tensors = _make_hat(N, seed=42)
    rng = np.random.RandomState(43)
    R = torch.from_numpy(rng.randn(N) * 1.5).float()
    # Cross-check: apply_i_minus_h via X_demo/XtX_inv should match IH @ R.
    IHR_compact = apply_i_minus_h(R, tensors['X_demo'], tensors['XtX_inv'])
    IHR_dense = IH @ R
    torch.testing.assert_close(IHR_compact, IHR_dense, rtol=1e-5, atol=1e-5)


def test_attribution_uniform_baseline_when_R_constant():
    """When R is constant, MR = 0 so attribution = 1/N at every cell.

    Constant R has zero centered residual variance. The degenerate branch in
    apply_i_minus_h / the EPS guard in attribution leaves ss_explained ≈ 0,
    so each cell gets the 1/N baseline term unmodified.
    """
    N = 25
    _, _, _, _, tensors = _make_hat(N, seed=44)
    R = torch.full((N,), 2.5)
    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    expected = torch.full((N,), 1.0 / N)
    # ss_tot is tiny but finite (EPS), so attribution may have some drift
    # from the exact 1/N baseline. The sum invariant is what matters.
    assert torch.allclose(attr.sum(), torch.tensor(1.0), atol=1e-3)
    # Each cell should be near 1/N.
    assert torch.allclose(attr, expected, atol=1e-3)


def test_attribution_negative_for_strongly_demographic_cells():
    """When R is in the demographic span, demographics explain most of R's
    variance — so cells with high (MR)² (= ss_tot contribution) but near-zero
    ((I-H)R)² (= ss_res contribution) get αᵢ < 1/N (and often αᵢ < 0).
    """
    from sklearn.preprocessing import StandardScaler

    N = 60
    rng = np.random.RandomState(60)
    D_np = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D_np, demo, ["f1", "f2", "f3"])
    tensors = hat_matrices_to_torch(hat)

    X_scaled = StandardScaler().fit_transform(demo)
    R_np = 1.0 + 2.0 * X_scaled[:, 0] + 0.5 * X_scaled[:, 1]
    R = torch.from_numpy(R_np).float()

    attr = per_cell_fairness_attribution_causal(R, tensors['X_demo'], tensors['XtX_inv'])
    # F_causal ≈ 0 when R ∈ span([1, X_demo]), so Σ attr ≈ 0.
    assert abs(float(attr.sum())) < 1e-3
    # And there should be cells with αᵢ < 0 (demographics explain MORE than baseline).
    assert (attr < 0).any()
