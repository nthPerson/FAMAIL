"""
Fit g_0(D) using power basis [1, 1/(D+1), 1/sqrt(D+1), sqrt(D+1)].

Captures hyperbolic Y ~ a/D with four linear parameters.
Fitted once during preprocessing at active-unit block-mean scale.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

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
