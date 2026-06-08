"""F_causal restricted to the active units the edit touches.

The global F_causal dilutes the editing signal across ~34,524 active units;
locally (the ~1,186-3,773 units the edit relocates pickups *from*) the effect
is concentrated. Localized F_causal uses the same residual definition as the
global metric — R = Y − g_0(D), Y = supply/demand, g_0 from the bundle — but
restricts the orthogonality computation to the touched units.

With M = I (uniform weighting), F_causal_localized = R'(I−H_demo)R / R'R,
which is 1 − r²_demo on the touched subset (residual SS over total SS).
Higher = fairer (residual unexplained by demographics).
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.data.loader import DataBundle


def edited_units_from_histories(
    edit_dir: str | Path,
) -> List[Tuple[int, int, int]]:
    """Return (x, y, t_block) of each edited trajectory's ORIGINAL pickup unit.

    Reads <edit_dir>/histories.pkl. We use the *original* pickup unit (not the
    modified one) because the editing moves the pickup OUT OF that unit's
    demand — that's the unit where the change is concentrated.
    """
    # Trusted source: histories.pkl is produced by our own editing pipeline
    # (famail_temporal.editing) on the same machine — not external input. This
    # follows the existing project pattern for intermediate artifact I/O.
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    out = []
    for h in histories:
        s = h.original.states[-1]
        # Identify t_block from the original pickup state's time bucket.
        # Use the same convention as Phase-2 sequences: t_block = hour_to_block(time_bucket_to_hour(...))
        t_block = hour_to_block_index(time_bucket_to_hour(s.time_bucket))
        out.append((int(s.x_grid), int(s.y_grid), int(t_block)))
    return out


def active_unit_index_of(
    bundle: DataBundle, units: Iterable[Tuple[int, int, int]],
) -> np.ndarray:
    """Map (x, y, t_block) units to their flat active-unit index.

    Returns a 1-D int array of indices into the N-vector ordering used by
    pickup_N, supply_N, X_demo, etc. (i.e., C-order traversal of mask_3d).
    Drops units that fall outside mask_3d (inactive cells).
    """
    mask = bundle.mask_3d  # (GX, GY, T)
    # Build a (GX, GY, T) lookup table: -1 for inactive, else its flat index.
    flat_index = np.full(mask.shape, -1, dtype=np.int64)
    flat_index[mask] = np.arange(int(mask.sum()))
    seen = set()
    out: list[int] = []
    for (x, y, t) in units:
        idx = int(flat_index[int(x), int(y), int(t)])
        if idx >= 0 and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return np.array(out, dtype=np.int64)


def residual_and_demo(
    bundle: DataBundle, pickup_3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R = supply/demand − g_0(demand) and X_demo over ALL active units.

    Uses bundle.g0_func (frozen) on the clamped demand, exactly as
    FAMAILObjective does (so the residual is comparable to the global F_causal).
    Returns (R, X_demo) both as numpy arrays.
    """
    mask = bundle.mask_3d
    demand_N = pickup_3d[mask].astype(np.float64)
    supply_N = bundle.active_taxis_3d[mask].astype(np.float64)
    Y = supply_N / np.maximum(demand_N, config.DEMAND_FLOOR)
    D_clamped = np.maximum(demand_N, config.DEMAND_FLOOR)
    # g0_func.eval_torch lives on tensors; convert in/out.
    D_t = torch.from_numpy(D_clamped).float()
    g0 = bundle.g0_func.eval_torch(D_t).cpu().numpy().astype(np.float64)
    R = Y - g0
    # X_demo is a torch tensor stored as a buffer on FAMAILObjective in the
    # gradient path; the bundle's hat_matrices dict carries the raw numpy.
    X_demo = np.asarray(bundle.hat_matrices["X_demo"], dtype=np.float64)
    return R, X_demo


def f_causal_orthogonality(R: np.ndarray, X_demo: np.ndarray) -> float:
    """F_causal under M = I: 1 − R'(I−H_demo)R / R'R = SS_residual / SS_total.

    Higher = fairer (residual is less explained by demographics).

    Implemented via the FWL identity R'(I−H)R = R'R − (X'R)'(X'X)⁻¹(X'R) to
    avoid materializing the dense N×N hat matrix at production N≈34k (matches
    the O(N·p) convention in famail_temporal/fairness/hat_matrices.py).

    Degenerate cases:
    - R has zero norm -> return 1.0 (no residual variance to explain).
    - X_demo has zero columns / rank 0 -> H_demo = 0, return F_causal = 0
      (documented plan-spec fallback; the formula limit at H=0 is actually 1.0,
      but the plan keeps 0.0 as a defensive fallback for malformed input).
    """
    R = np.asarray(R, dtype=np.float64).ravel()
    n = R.shape[0]
    if n == 0:
        return float("nan")
    rr = float(R @ R)
    if rr <= 0.0:
        return 1.0
    if X_demo.size == 0 or X_demo.shape[1] == 0:
        return 0.0
    XtX = X_demo.T @ X_demo                # (p, p)
    XtX_inv = np.linalg.pinv(XtX)
    XtR = X_demo.T @ R                     # (p,)
    ss_explained = float(XtR @ XtX_inv @ XtR)   # R' H R
    ss_res = rr - ss_explained             # R' (I-H) R
    return ss_res / rr


def localized_f_causal(
    bundle: DataBundle, pickup_3d: np.ndarray, edited_units: Iterable[Tuple[int, int, int]],
) -> dict:
    """Return localized + global F_causal under the SAME M=I formula.

    Both fields use the M=I (uniform-weighting) form of F_causal, which is
    `1 - r²_demo`. The localized field restricts the regression to the active
    units the edit touched (typically ~1k-4k units); the global field uses ALL
    active units (~34k). Both are directly comparable since they use the SAME
    formula at different N.

    NOTE: `f_causal_global` here is NOT the same number as
    `data_level_fairness(bundle, pickup_3d=...)["f_causal"]` — production
    F_causal uses M = I − 11'/N (centering), not M = I. The orchestrator (Task 6)
    reports the production number SEPARATELY via `b0_fairness["f_causal"]` /
    `famail_fairness["f_causal"]`; this function provides the localized signal
    plus its directly-comparable global counterpart for the "did the local
    signal beat the global dilution?" analysis.

    Returns:
        f_causal_localized      — M=I F_causal on the edited-units subset.
        f_causal_global         — M=I F_causal on ALL active units (paired with localized).
        n_edited_active_units   — size of the subset (0 → localized = nan).
    """
    R, X_demo = residual_and_demo(bundle, pickup_3d)
    f_global = f_causal_orthogonality(R, X_demo)
    idx = active_unit_index_of(bundle, edited_units)
    if idx.size == 0:
        return {
            "f_causal_localized": float("nan"),
            "f_causal_global": float(f_global),
            "n_edited_active_units": 0,
        }
    f_local = f_causal_orthogonality(R[idx], X_demo[idx])
    return {
        "f_causal_localized": float(f_local),
        "f_causal_global": float(f_global),
        "n_edited_active_units": int(idx.size),
    }
