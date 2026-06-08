"""Unit tests for localized F_causal."""
import numpy as np
import pytest

from famail_temporal.baselines import localized_metrics as lm


def test_localized_f_causal_one_when_residual_orthogonal_to_demo():
    # Residual R is orthogonal to the demographic columns -> r²=0, F_causal=1.
    rng = np.random.default_rng(0)
    n = 50
    X_demo = rng.standard_normal((n, 3))
    # Construct R orthogonal to X_demo by removing X_demo's projection.
    R = rng.standard_normal(n)
    H = X_demo @ np.linalg.pinv(X_demo.T @ X_demo) @ X_demo.T
    R_orth = R - H @ R
    f = lm.f_causal_orthogonality(R_orth, X_demo)
    assert f == pytest.approx(1.0, abs=1e-8)


def test_localized_f_causal_zero_when_residual_fully_in_demo_span():
    rng = np.random.default_rng(1)
    n = 50
    X_demo = rng.standard_normal((n, 3))
    # R is a linear combination of X_demo -> r²=1, F_causal=0.
    beta = np.array([1.5, -0.7, 2.3])
    R = X_demo @ beta
    f = lm.f_causal_orthogonality(R, X_demo)
    assert f == pytest.approx(0.0, abs=1e-8)


def test_localized_f_causal_zero_safe_on_zero_residual():
    # Degenerate: R = 0. Define F_causal = 1.0 (no residual to explain).
    n = 10
    X_demo = np.random.default_rng(0).standard_normal((n, 3))
    R = np.zeros(n)
    assert lm.f_causal_orthogonality(R, X_demo) == 1.0
