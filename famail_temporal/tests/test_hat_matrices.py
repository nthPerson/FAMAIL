"""Tests for fairness.hat_matrices."""
import numpy as np
import pytest

from famail_temporal.fairness.hat_matrices import precompute_hat_matrices


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
