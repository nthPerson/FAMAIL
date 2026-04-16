"""Tests for fairness.hat_matrices."""
import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal.fairness.hat_matrices import (
    compute_fcausal_torch,
    precompute_hat_matrices,
)


def test_shapes():
    rng = np.random.RandomState(0)
    N = 50
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    assert hat['I_minus_H_demo'].shape == (N, N)
    assert hat['M'].shape == (N, N)
    assert hat['n_units'] == N


def test_I_minus_H_idempotent():
    rng = np.random.RandomState(1)
    N = 40
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = hat['I_minus_H_demo']
    np.testing.assert_allclose(IH @ IH, IH, atol=1e-10)


def test_M_centering():
    rng = np.random.RandomState(2)
    N = 30
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    M = hat['M']
    np.testing.assert_allclose(M @ M, M, atol=1e-10)
    np.testing.assert_allclose(M @ np.ones(N), np.zeros(N), atol=1e-10)


def test_rank_deficient_raises():
    rng = np.random.RandomState(3)
    N = 30
    col1 = rng.randn(N)
    demo = np.column_stack([col1, col1, rng.randn(N)])
    with pytest.raises(AssertionError, match="rank"):
        precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["a", "b", "c"])


def test_demands_must_be_1d():
    with pytest.raises(ValueError, match="1-D"):
        precompute_hat_matrices(np.zeros((10, 2)), np.zeros((10, 2)), ["a", "b"])


def test_demo_must_be_2d():
    with pytest.raises(ValueError, match="2-D"):
        precompute_hat_matrices(np.zeros(10), np.zeros(10), ["a"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        # demands has length 10, demo has length 12
        precompute_hat_matrices(np.zeros(10), np.zeros((12, 2)), ["a", "b"])


def test_empty_feature_names_raises():
    with pytest.raises(ValueError):
        precompute_hat_matrices(np.zeros(10), np.zeros((10, 0)), [])


def test_small_N_raises():
    with pytest.raises(ValueError):
        precompute_hat_matrices(np.zeros(5), np.zeros((5, 2)), ["a", "b"])


def test_nan_demands_raises():
    rng = np.random.RandomState(0)
    D = np.ones(20)
    D[0] = np.nan
    with pytest.raises(ValueError):
        precompute_hat_matrices(D, rng.randn(20, 2), ["a", "b"])


def test_nan_demo_raises():
    rng = np.random.RandomState(1)
    demo = rng.randn(20, 2)
    demo[0, 0] = np.nan
    with pytest.raises(ValueError):
        precompute_hat_matrices(np.ones(20), demo, ["a", "b"])


def test_zero_variance_demo_raises():
    rng = np.random.RandomState(2)
    N = 20
    demo = np.column_stack([rng.randn(N), np.ones(N)])  # second column is constant
    with pytest.raises(ValueError, match="zero-variance"):
        precompute_hat_matrices(rng.uniform(0.5, 5.0, N), demo, ["real", "constant"])


def test_return_arrays_read_only():
    rng = np.random.RandomState(3)
    hat = precompute_hat_matrices(
        rng.uniform(0.5, 5.0, 30),
        rng.randn(30, 3),
        ["a", "b", "c"],
    )
    for key in ("I_minus_H_demo", "M", "scaler_mean", "scaler_std"):
        with pytest.raises(ValueError):
            hat[key].flat[0] = 99.0


def test_fcausal_zero_when_R_in_demographic_span():
    N = 50
    rng = np.random.RandomState(4)
    D = rng.uniform(0.5, 5.0, N)
    demo = rng.randn(N, 3)
    hat = precompute_hat_matrices(D, demo, ["f1", "f2", "f3"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    X_scaled = StandardScaler().fit_transform(demo)
    R = torch.from_numpy(2.0 + 1.5 * X_scaled[:, 0]).float()
    f = compute_fcausal_torch(R, IH, M)
    assert float(f) < 1e-4


def test_fcausal_bounded():
    N = 80
    rng = np.random.RandomState(5)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    R = torch.randn(N) * 3.0
    f = compute_fcausal_torch(R, IH, M)
    assert 0.0 <= float(f) <= 1.0


def test_fcausal_degenerate_returns_one():
    N = 30
    rng = np.random.RandomState(6)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    R = torch.full((N,), 0.5)
    f = compute_fcausal_torch(R, IH, M)
    assert float(f) == 1.0


def test_fcausal_gradient_flows():
    N = 30
    rng = np.random.RandomState(7)
    hat = precompute_hat_matrices(rng.uniform(0.5, 5.0, N), rng.randn(N, 3), ["a", "b", "c"])
    IH = torch.from_numpy(hat['I_minus_H_demo']).float()
    M = torch.from_numpy(hat['M']).float()
    R = torch.randn(N, requires_grad=True)
    f = compute_fcausal_torch(R, IH, M)
    f.backward()
    assert R.grad is not None
    assert not torch.isnan(R.grad).any()
    assert not torch.isinf(R.grad).any()
    # R is a random N-vector -> not in degenerate branch, gradient should be nonzero.
    assert (R.grad.abs() > 0).any()
