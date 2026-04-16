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


def test_g0function_coefficients_wrong_ndim():
    """2-D coefficient shapes (4, 1) and (1, 4) should also be rejected."""
    import pytest
    with pytest.raises(ValueError, match="1-D"):
        G0Function(
            coefficients=np.ones((4, 1)),
            d_min=0.01, d_max=10.0,
        )
    with pytest.raises(ValueError, match="1-D"):
        G0Function(
            coefficients=np.ones((1, 4)),
            d_min=0.01, d_max=10.0,
        )


def test_power_basis_column_semantics():
    """Pin down column ordering: [1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)]."""
    D = np.array([3.0, 8.0])
    X = build_power_basis_features(D, include_intercept=True)
    np.testing.assert_allclose(X[:, 0], np.ones(2))
    np.testing.assert_allclose(X[:, 1], 1.0 / (D + 1))
    np.testing.assert_allclose(X[:, 2], 1.0 / np.sqrt(D + 1))
    np.testing.assert_allclose(X[:, 3], np.sqrt(D + 1))


def test_g0function_rejects_negative_d_min():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        G0Function(
            coefficients=np.array([0.1, 0.2, 0.3, 0.4]),
            d_min=-0.5, d_max=10.0,
        )


def test_g0function_rejects_inverted_bounds():
    import pytest
    with pytest.raises(ValueError, match="d_max"):
        G0Function(
            coefficients=np.array([0.1, 0.2, 0.3, 0.4]),
            d_min=5.0, d_max=1.0,
        )


def test_g0function_accepts_scalar():
    g0 = G0Function(
        coefficients=np.array([0.0, 1.0, 0.0, 0.0]),  # g0(D) = 1/(D+1)
        d_min=0.01, d_max=10.0,
    )
    # Scalar input should return a 1-element array
    result = g0(3.0)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, np.array([0.25]))  # 1/(3+1)


from famail_temporal.fairness.g0_power_basis import fit as fit_g0


def test_fit_recovers_hyperbolic():
    rng = np.random.RandomState(42)
    D = np.linspace(0.5, 10.0, 500)
    Y = 2.0 / D + 0.05 * rng.randn(len(D))
    g0, diag = fit_g0(D, Y)
    assert diag['n_points'] == 500
    assert diag['power_r2'] > 0.8


def test_fit_diagnostics():
    D = np.linspace(0.5, 10.0, 100)
    Y = 1.0 / D + 0.01
    _, diag = fit_g0(D, Y)
    assert 'agreement_max_abs_diff' in diag
    assert 'isotonic_r2' in diag
    assert 'power_r2' in diag
