"""Tests for fairness.g0_power_basis."""
import numpy as np

from famail_temporal.fairness.g0_power_basis import (
    build_power_basis_features,
    G0Function,
)


def test_power_basis_shape_with_intercept():
    D = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    X = build_power_basis_features(D, include_intercept=True)
    assert X.shape == (5, 4)
    np.testing.assert_array_equal(X[:, 0], np.ones(5))


def test_power_basis_shape_without_intercept():
    D = np.array([1.0, 2.0, 3.0])
    X = build_power_basis_features(D, include_intercept=False)
    assert X.shape == (3, 3)


def test_g0function_shape():
    coefficients = np.array([0.1, 0.5, 0.2, 0.05])
    g0 = G0Function(coefficients=coefficients, d_min=0.01, d_max=10.0)
    D = np.array([1.0, 2.0, 3.0])
    assert g0(D).shape == (3,)


def test_g0function_clips():
    coefficients = np.array([0.0, 1.0, 0.0, 0.0])   # g0 = 1/(D+1)
    g0 = G0Function(coefficients=coefficients, d_min=1.0, d_max=5.0)
    np.testing.assert_allclose(g0(np.array([100.0])), 1.0 / 6.0)
    np.testing.assert_allclose(g0(np.array([0.0])), 1.0 / 2.0)


def test_g0function_coefficients_readonly():
    """Hardening: coefficients array must be read-only after construction."""
    coefficients = np.array([0.1, 0.5, 0.2, 0.05])
    g0 = G0Function(coefficients=coefficients, d_min=0.01, d_max=10.0)
    import pytest
    with pytest.raises((ValueError, TypeError)):
        g0.coefficients[0] = 999.0


def test_g0function_coefficients_wrong_length():
    """Hardening: must raise ValueError if coefficients length != 4."""
    import pytest
    with pytest.raises(ValueError, match="coefficients"):
        G0Function(coefficients=np.array([0.1, 0.5]), d_min=0.01, d_max=10.0)
