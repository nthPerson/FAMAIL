"""
Pre-compute hat matrices for pooled Option B F_causal.

Inputs are active-unit vectors (length N). Constants during optimization —
only the residual vector R changes across forward passes.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler


def precompute_hat_matrices(
    demands: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Build (I - H_demo), M, and diagnostics.

    H_demo projects onto [1, standardized(demographics)] (intercept included).
    M = I - 11'/N is the centering matrix.
    Asserts H_demo has full rank.
    """
    D = np.asarray(demands, dtype=np.float64)
    demo = np.asarray(demographic_features, dtype=np.float64)

    # Input validation (hardening: fail loud on mis-shaped or non-finite inputs).
    if D.ndim != 1:
        raise ValueError(f"demands must be 1-D; got shape {D.shape}")
    if demo.ndim != 2:
        raise ValueError(
            f"demographic_features must be 2-D; got shape {demo.shape}"
        )
    if not feature_names:
        raise ValueError("feature_names must be a non-empty list")

    N = len(D)
    if demo.shape != (N, len(feature_names)):
        raise ValueError(
            f"demographic_features shape {demo.shape} "
            f"inconsistent with N={N} and {len(feature_names)} features"
        )
    # X has (1 + n_features) columns; rank check requires N >= that many rows.
    min_N = 1 + len(feature_names)
    if N < max(10, min_N):
        raise ValueError(
            f"Need at least max(10, 1+n_features)={max(10, min_N)} units "
            f"for a well-defined rank check; got N={N}"
        )
    if not np.all(np.isfinite(D)):
        raise ValueError("demands contains non-finite values (NaN/Inf)")
    if not np.all(np.isfinite(demo)):
        raise ValueError(
            "demographic_features contains non-finite values (NaN/Inf) — "
            "would silently poison hat matrix downstream"
        )

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demo)
    X = np.column_stack([np.ones(N), X_demo_scaled])

    H = X @ np.linalg.pinv(X)
    rank_H = int(np.linalg.matrix_rank(H))
    expected_rank = X.shape[1]
    assert rank_H == expected_rank, (
        f"H_demo rank {rank_H}, expected {expected_rank}. "
        "Demographic collinearity — check feature set."
    )

    I_minus_H_demo = np.eye(N) - H
    M = np.eye(N) - np.ones((N, N)) / N

    # Sanity-check the constructed hat matrices before freezing.
    if not np.all(np.isfinite(I_minus_H_demo)):
        raise RuntimeError("I_minus_H_demo contains non-finite values")
    if not np.all(np.isfinite(M)):
        raise RuntimeError("M contains non-finite values")

    # Freeze load-bearing constants so downstream code can't mutate them.
    I_minus_H_demo.setflags(write=False)
    M.setflags(write=False)
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
    scaler_std = np.asarray(scaler.scale_, dtype=np.float64)
    scaler_mean.setflags(write=False)
    scaler_std.setflags(write=False)

    return {
        'I_minus_H_demo': I_minus_H_demo,
        'M': M,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'n_units': N,
        'n_demo_features': len(feature_names),
        'feature_names': feature_names,
        'rank_H_demo': rank_H,
    }
