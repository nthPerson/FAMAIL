"""
Pre-compute hat-matrix building blocks for pooled Option B F_causal.

Exports both a **compact** representation of the demographic projection
(`X_demo`, `XtX_inv`) and — at small N only — the classic **dense** hat
matrices `I_minus_H_demo` and `M`. The compact form is O(Np) in memory;
the dense form is O(N²) and becomes untenable as N grows (≈19 GB at
N=34,524). Production code paths use the compact form; the dense form
is retained for test compatibility and for debug-level introspection
on small problems.

Frisch–Waugh–Lovell identities that let us skip the N×N materialization:
    (I − H) R  =  R − X (XᵀX)⁻¹ (Xᵀ R)
    Rᵀ(I − H)R = RᵀR − (Xᵀ R)ᵀ (XᵀX)⁻¹ (Xᵀ R)
    M R        =  R − (1ᵀR / N) · 1
    Rᵀ M R     =  RᵀR − (1ᵀR)² / N
Both sides are algebraically identical; the right-hand sides use only
O(Np) and O(p²) intermediates.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from famail_temporal import config


# Emit the N×N dense matrices only when N is below this threshold.
# At N=5,834 (T=4) we use ~544 MB; at N=34,524 (T=24) we would need ~19 GB.
# 10_000 keeps the dense form available for small-N problems and tests
# while skipping it for production-scale caches.
_DENSE_MATERIALIZATION_MAX_N = 10_000


def precompute_hat_matrices(
    demands: np.ndarray,
    demographic_features: np.ndarray,
    feature_names: List[str],
) -> Dict[str, np.ndarray]:
    """Build the compact demographic-projection representation + diagnostics.

    Always-emitted compact fields:
      - ``X_demo``: (N, p+1) design matrix = [1 | standardized(demographics)]
      - ``XtX_inv``: (p+1, p+1) = (XᵀX)⁻¹
      - ``scaler_mean``, ``scaler_std``: per-demographic standardization params
      - ``n_units``, ``n_demo_features``, ``feature_names``, ``rank_H_demo``

    Conditionally emitted dense fields (only when N ≤ _DENSE_MATERIALIZATION_MAX_N):
      - ``I_minus_H_demo``: (N, N) residual-maker matrix
      - ``M``: (N, N) centering matrix

    Production callers MUST use the compact form (``compute_fcausal_compact``);
    the dense form is retained for small-N debug/testing and the legacy
    ``compute_fcausal_torch(R, I_minus_H_demo, M)`` API.
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

    # Zero-variance preflight: StandardScaler silently replaces std=0 with
    # std=1, producing an all-zero scaled column that triggers the rank check
    # with a misleading "collinearity" message when the real cause is different.
    col_std = demo.std(axis=0)
    zero_var_cols = np.where(col_std < 1e-12)[0]
    if len(zero_var_cols) > 0:
        bad = [feature_names[i] for i in zero_var_cols]
        raise ValueError(
            f"demographic_features has zero-variance columns: {bad}. "
            f"StandardScaler would silently replace std=0 with std=1, producing "
            f"an all-zero scaled column that fails the rank check."
        )

    scaler = StandardScaler()
    X_demo_scaled = scaler.fit_transform(demo)
    X = np.column_stack([np.ones(N), X_demo_scaled])  # (N, p+1)

    # rank(X) on (N, p+1) is O(N * p^2) — cheap even at large N.
    rank_X = int(np.linalg.matrix_rank(X))
    expected_rank = X.shape[1]
    assert rank_X == expected_rank, (
        f"X has rank {rank_X}, expected {expected_rank}. "
        "Demographic collinearity or zero-variance column — check feature set."
    )

    # Compact form: XtX_inv is (p+1) x (p+1) — always cheap.
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    rank_H_demo = rank_X  # rank(H) == rank(X)

    # Sanity-check compact form before freezing.
    if not np.all(np.isfinite(XtX_inv)):
        raise RuntimeError("XtX_inv contains non-finite values")

    # Freeze compact form.
    X.setflags(write=False)
    XtX_inv.setflags(write=False)
    scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
    scaler_std = np.asarray(scaler.scale_, dtype=np.float64)
    scaler_mean.setflags(write=False)
    scaler_std.setflags(write=False)

    out: Dict[str, np.ndarray] = {
        'X_demo': X,
        'XtX_inv': XtX_inv,
        'scaler_mean': scaler_mean,
        'scaler_std': scaler_std,
        'n_units': N,
        'n_demo_features': len(feature_names),
        'feature_names': list(feature_names),
        'rank_H_demo': rank_H_demo,
    }

    # Conditionally emit the dense (N, N) matrices at small N only.
    if N <= _DENSE_MATERIALIZATION_MAX_N:
        H = X @ np.linalg.pinv(X)
        I_minus_H_demo = np.eye(N) - H
        M = np.eye(N) - np.ones((N, N)) / N
        if not np.all(np.isfinite(I_minus_H_demo)):
            raise RuntimeError("I_minus_H_demo contains non-finite values")
        if not np.all(np.isfinite(M)):
            raise RuntimeError("M contains non-finite values")
        I_minus_H_demo.setflags(write=False)
        M.setflags(write=False)
        out['I_minus_H_demo'] = I_minus_H_demo
        out['M'] = M

    return out


def compute_fcausal_compact(
    R: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
    eps: float = config.EPS,
) -> torch.Tensor:
    """Pooled Option B F_causal via the compact (FWL) form.

    Equivalent to ``R'(I − H)R / R'MR`` but computed without any N×N
    materialization — memory is O(Np + p²). Required for production-scale
    caches where N can reach tens of thousands.

    Algebra:
        Rᵀ(I − H)R  =  RᵀR  −  (XᵀR)ᵀ (XᵀX)⁻¹ (XᵀR)
        RᵀMR        =  RᵀR  −  (1ᵀR)² / N
        F_causal    =  (Rᵀ(I − H)R) / (RᵀMR)  clamped to [0, 1]

    Parameters
    ----------
    R : torch.Tensor, shape (N,)
        Residual vector Y − g_0(D). Typically requires_grad=True.
    X_demo : torch.Tensor, shape (N, p+1)
        Design matrix [1 | standardized(demographics)], constant wrt R.
    XtX_inv : torch.Tensor, shape (p+1, p+1)
        Precomputed inverse of XᵀX, constant wrt R.
    eps : float, optional
        Numerical guard for the degenerate Rᵀ M R ~ 0 branch.

    Returns
    -------
    torch.Tensor, scalar
        F_causal in [0, 1], higher = fairer. See
        ``compute_fcausal_torch`` for the full orientation and limit-case
        documentation (they apply identically to this compact form).
    """
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D; got shape {tuple(R.shape)}")
    if X_demo.ndim != 2:
        raise ValueError(f"X_demo must be 2-D; got shape {tuple(X_demo.shape)}")
    if XtX_inv.ndim != 2:
        raise ValueError(
            f"XtX_inv must be 2-D; got shape {tuple(XtX_inv.shape)}"
        )
    N = R.shape[0]
    if X_demo.shape[0] != N:
        raise ValueError(
            f"X_demo.shape[0]={X_demo.shape[0]} but R has length {N}"
        )
    p1 = X_demo.shape[1]
    if XtX_inv.shape != (p1, p1):
        raise ValueError(
            f"XtX_inv shape {tuple(XtX_inv.shape)} inconsistent with "
            f"X_demo.shape[1]={p1}"
        )

    RtR = R @ R                           # scalar
    XtR = X_demo.T @ R                    # shape (p+1,)
    ss_res_demo = RtR - XtR @ XtX_inv @ XtR
    sum_R = R.sum()
    ss_tot = RtR - sum_R * sum_R / N
    f_causal = torch.where(
        ss_tot < eps,
        torch.ones_like(ss_tot),
        ss_res_demo / (ss_tot + eps),
    )
    return torch.clamp(f_causal, 0.0, 1.0)


def compute_fcausal_torch(
    R: torch.Tensor,
    I_minus_H_demo: torch.Tensor,
    M: torch.Tensor,
    eps: float = config.EPS,
) -> torch.Tensor:
    """Pooled Option B: F_causal = R'(I-H)R / R'MR, clamped to [0, 1].

    LEGACY DENSE-MATRIX FORM. Retained for small-N debug/testing and for
    backward-compatible call sites. Production code should prefer
    ``compute_fcausal_compact(R, X_demo, XtX_inv)``, which is O(Np) in
    memory and algebraically identical.

    Differentiable wrt R. The hat matrices are constants (frozen read-only
    numpy arrays from ``precompute_hat_matrices``, wrapped in torch tensors
    by the caller). When ``R'MR < eps`` (no total variance to explain, e.g.,
    constant R), returns 1.0 by convention — gradients flow through the
    non-degenerate branch in the typical case via ``torch.where``.

    NaN in R propagates to F_causal; this is intentional — callers must
    ensure R is finite.

    Parameters
    ----------
    R : torch.Tensor, shape (N,)
        Residual vector Y - g_0(D). Typically requires_grad=True during
        optimization.
    I_minus_H_demo : torch.Tensor, shape (N, N)
        Residual-maker matrix, constant wrt R.
    M : torch.Tensor, shape (N, N)
        Centering matrix, constant wrt R.
    eps : float, optional
        Numerical guard for the degenerate ss_tot ~ 0 branch.

    Returns
    -------
    torch.Tensor, scalar
        F_causal in [0, 1], higher = fairer. Algebraically
            F_causal = R'(I - H_demo)R / R'MR
                     = SSR_demo / SST
                     = 1 - r²_demo
        where r²_demo is the coefficient of determination from regressing R
        on [1, standardized_demographics]. Orientation:
        - High F_causal (near 1) ⇔ low r²_demo ⇔ demographics explain LITTLE
          of R ⇔ service deviations from the demand baseline are not driven
          by demographics ⇔ FAIR.
        - Low F_causal (near 0) ⇔ high r²_demo ⇔ demographics explain MOST
          of R ⇔ service deviations are predicted by demographics ⇔ UNFAIR.
        Per the design spec: R ∈ span(X_demo) → F_causal = 0 (fully unfair);
        R ⊥ X_demo → F_causal = 1 (fully fair).
    """
    # Shape / dim validation — fail loud rather than producing silent garbage
    # from a broadcasting surprise.
    if R.ndim != 1:
        raise ValueError(f"R must be 1-D; got shape {tuple(R.shape)}")
    if I_minus_H_demo.ndim != 2 or M.ndim != 2:
        raise ValueError(
            f"I_minus_H_demo and M must be 2-D; got shapes "
            f"{tuple(I_minus_H_demo.shape)}, {tuple(M.shape)}"
        )
    N = R.shape[0]
    if I_minus_H_demo.shape != (N, N) or M.shape != (N, N):
        raise ValueError(
            f"Matrix shapes inconsistent with R of length {N}: "
            f"I_minus_H_demo={tuple(I_minus_H_demo.shape)}, "
            f"M={tuple(M.shape)}"
        )

    ss_res_demo = R @ I_minus_H_demo @ R
    ss_tot = R @ M @ R
    f_causal = torch.where(
        ss_tot < eps,
        torch.ones_like(ss_tot),
        ss_res_demo / (ss_tot + eps),
    )
    return torch.clamp(f_causal, 0.0, 1.0)


def apply_i_minus_h(
    R: torch.Tensor,
    X_demo: torch.Tensor,
    XtX_inv: torch.Tensor,
) -> torch.Tensor:
    """Compute (I − H_demo) R using the compact representation.

    Returns a length-N tensor. O(Np) memory. Required by attribution
    routines that need the residual-after-projection vector.

    Identity: (I − H) R = R − X (XᵀX)⁻¹ (XᵀR).
    """
    XtR = X_demo.T @ R             # (p+1,)
    return R - X_demo @ (XtX_inv @ XtR)


def hat_matrices_to_torch(
    hat: Dict[str, np.ndarray],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Convert the numpy hat dict to torch tensors.

    Always converts the compact representation (``X_demo``, ``XtX_inv``).
    When present in the input dict, also converts the dense forms
    (``I_minus_H_demo``, ``M``) for backward compatibility.

    Internally calls `.copy()` on each array before conversion because the
    arrays returned by ``precompute_hat_matrices`` are read-only (frozen
    via ``setflags``), and ``torch.from_numpy()`` emits UserWarning on
    non-writable arrays.

    Args:
        hat: Dict returned from ``precompute_hat_matrices()``.
        dtype: Target torch dtype (default ``torch.float32``).
        device: Target device string (default ``"cpu"``).

    Returns:
        Dict with torch tensors on the specified device. Always contains
        ``X_demo`` and ``XtX_inv``. Contains ``I_minus_H_demo`` and ``M``
        only if they are present in the input ``hat`` dict.
    """
    out = {
        'X_demo': torch.from_numpy(hat['X_demo'].copy()).to(dtype=dtype, device=device),
        'XtX_inv': torch.from_numpy(hat['XtX_inv'].copy()).to(dtype=dtype, device=device),
    }
    if 'I_minus_H_demo' in hat:
        out['I_minus_H_demo'] = torch.from_numpy(
            hat['I_minus_H_demo'].copy(),
        ).to(dtype=dtype, device=device)
    if 'M' in hat:
        out['M'] = torch.from_numpy(hat['M'].copy()).to(dtype=dtype, device=device)
    return out
