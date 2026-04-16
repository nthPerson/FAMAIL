"""
Fit g_0(D) using power basis [1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)].

Captures hyperbolic Y ~ a/D with four linear parameters.
Fitted once during preprocessing at active-unit block-mean scale.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from famail_temporal import config

_N_COEFFICIENTS = 4


def build_power_basis_features(demands: np.ndarray, include_intercept: bool = True) -> np.ndarray:
    """Feature matrix [1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)] per cell.

    Parameters
    ----------
    demands:
        Scalar or 1-D array of demand values (D ≥ 0).
    include_intercept:
        If True, prepend a column of ones (default). Produces shape (n, 4).
        If False, omit the intercept column. Produces shape (n, 3).

    Returns
    -------
    np.ndarray of shape (n, 4) or (n, 3).
    """
    d_arr = np.asarray(demands, dtype=np.float64)
    if d_arr.ndim == 0:
        d_arr = d_arr.reshape(1)  # accept scalar input
    if d_arr.ndim != 1:
        raise ValueError(
            f"demands must be 1-D or scalar; got shape {d_arr.shape}"
        )
    d_safe = d_arr + 1.0
    feats = np.column_stack([
        1.0 / d_safe,
        1.0 / np.sqrt(d_safe),
        np.sqrt(d_safe),
    ])
    if include_intercept:
        n = d_arr.shape[0]
        feats = np.column_stack([np.ones(n), feats])
    return feats


@dataclass(frozen=True)
class G0Function:
    """Fitted g_0(D) with power basis coefficients.

    Coefficient order: [intercept, c_{1/(D+1)}, c_{1/sqrt(D+1)}, c_{sqrt(D+1)}]

    Parameters
    ----------
    coefficients:
        1-D array of length 4.
    d_min:
        Lower clip bound applied to demand before evaluation.
    d_max:
        Upper clip bound applied to demand before evaluation.
    """
    coefficients: np.ndarray
    d_min: float
    d_max: float

    def __post_init__(self) -> None:
        coeffs = np.asarray(self.coefficients, dtype=np.float64)
        if coeffs.ndim != 1 or len(coeffs) != _N_COEFFICIENTS:
            raise ValueError(
                f"coefficients must be a 1-D array of length {_N_COEFFICIENTS}, "
                f"got shape {coeffs.shape}"
            )
        if self.d_min < 0:
            raise ValueError(
                f"d_min must be non-negative (demand is always >= 0); got {self.d_min}"
            )
        if self.d_max < self.d_min:
            raise ValueError(
                f"d_max ({self.d_max}) must be >= d_min ({self.d_min})"
            )
        # Make the array read-only and store via object.__setattr__ (frozen dataclass).
        coeffs.setflags(write=False)
        object.__setattr__(self, "coefficients", coeffs)

    def __call__(self, d: np.ndarray) -> np.ndarray:
        d_arr = np.asarray(d, dtype=np.float64)
        d_clipped = np.clip(d_arr, self.d_min, self.d_max)
        X = build_power_basis_features(d_clipped, include_intercept=True)
        return X @ self.coefficients


def fit(demands: np.ndarray, supplies_over_demands: np.ndarray) -> tuple[G0Function, dict]:
    """Fit g_0(D) on (D, Y=S/D) pairs at block-mean scale.

    Fits a power basis regression [1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)] via
    LinearRegression and cross-checks with IsotonicRegression (monotone-
    decreasing) for diagnostic comparison.

    Parameters
    ----------
    demands:
        1-D array of demand values (D). Values below ``config.DEMAND_FLOOR``
        are clamped before fitting to avoid numerical instability.
    supplies_over_demands:
        1-D array of Y = S/D values corresponding to each demand entry.
        Must have the same length as ``demands``.

    Returns
    -------
    g0_func : G0Function
        Fitted power-basis function with coefficients learned from data.
    diagnostics : dict
        Plain dict with keys:
          - ``'n_points'``: number of (D, Y) pairs used.
          - ``'power_r2'``: R² of power basis fit against Y.
          - ``'isotonic_r2'``: R² of isotonic regression fit against Y.
          - ``'agreement_max_abs_diff'``: max |g0(D) - iso(D)| over training points.

    Raises
    ------
    ValueError
        If ``demands`` or ``supplies_over_demands`` are not 1-D arrays of
        equal length, or if fewer than 10 points are provided.
    """
    D_raw = np.asarray(demands, dtype=np.float64)
    Y = np.asarray(supplies_over_demands, dtype=np.float64)

    if D_raw.ndim != 1:
        raise ValueError(
            f"demands must be a 1-D array; got shape {D_raw.shape}"
        )
    if Y.ndim != 1:
        raise ValueError(
            f"supplies_over_demands must be a 1-D array; got shape {Y.shape}"
        )
    if D_raw.shape != Y.shape:
        raise ValueError(
            f"demands and supplies_over_demands must have the same length; "
            f"got {D_raw.shape} vs {Y.shape}"
        )
    if len(D_raw) < 10:
        raise ValueError(
            f"At least 10 data points required for a meaningful fit; "
            f"got {len(D_raw)}"
        )

    D = np.maximum(D_raw, config.DEMAND_FLOOR)

    X = build_power_basis_features(D, include_intercept=True)
    lr = LinearRegression(fit_intercept=False).fit(X, Y)
    g0 = G0Function(
        coefficients=lr.coef_,
        d_min=float(D.min()),
        d_max=float(D.max()),
    )

    iso = IsotonicRegression(increasing=False, out_of_bounds='clip').fit(D, Y)
    y_power = g0(D)
    y_iso = iso.predict(D)
    max_abs_diff = float(np.max(np.abs(y_power - y_iso)))

    y_var = float(np.var(Y)) + 1e-10
    diagnostics = {
        'n_points': int(len(D)),
        'power_r2': float(1.0 - np.var(Y - y_power) / y_var),
        'isotonic_r2': float(1.0 - np.var(Y - y_iso) / y_var),
        'agreement_max_abs_diff': max_abs_diff,
    }
    return g0, diagnostics
