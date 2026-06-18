# famail_temporal/visualization/gradient_heatmap/render.py
"""Field selection, combination math, color scaling, and figure builders."""
from __future__ import annotations

import numpy as np

QUANTITIES = ["Gradient", "Attribution", "Concentration"]
TERMS = ["F_spatial", "F_causal", "F_fidelity", "Combined", "Spatial+Causal"]


def is_signed(quantity: str) -> bool:
    return quantity in ("Gradient", "Attribution")


def _fidelity_field(active_mask: np.ndarray) -> np.ndarray:
    f = np.full(active_mask.shape, np.nan, dtype=np.float32)
    f[active_mask] = 0.0
    return f


def select_field(bundle, quantity, term, alpha_spatial, alpha_causal, alpha_fidelity):
    """Return the (48,90,24) field for the chosen quantity x term.

    Combined = a_sp*sp + a_ca*ca + a_fi*fidelity(=0 on active); equals
    Spatial+Causal at the per-cell level because fidelity has no per-cell field.
    """
    if quantity == "Concentration":
        return bundle.pickup
    if quantity == "Gradient":
        sp, ca = bundle.grad_spatial, bundle.grad_causal
    elif quantity == "Attribution":
        sp, ca = bundle.attr_spatial, bundle.attr_causal
    else:
        raise ValueError(f"unknown quantity {quantity!r}")

    if term == "F_spatial":
        return sp
    if term == "F_causal":
        return ca
    if term == "F_fidelity":
        return _fidelity_field(bundle.active_mask)
    if term == "Spatial+Causal":
        return alpha_spatial * sp + alpha_causal * ca
    if term == "Combined":
        return alpha_spatial * sp + alpha_causal * ca + alpha_fidelity * _fidelity_field(bundle.active_mask)
    raise ValueError(f"unknown term {term!r}")


def color_range(values: np.ndarray, signed: bool, clip_pct: float = 99.0):
    """Return (zmin, zmax, zmid_or_None, colorscale) using a robust percentile clip."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0, None, "Viridis")
    if signed:
        v = float(np.percentile(np.abs(finite), clip_pct))
        if v <= 0:
            v = float(np.abs(finite).max()) or 1.0
        return (-v, v, 0.0, "RdBu_r")
    hi = float(np.percentile(finite, clip_pct))
    if hi <= 0:
        hi = float(finite.max()) or 1.0
    return (0.0, hi, None, "Viridis")
