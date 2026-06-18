# famail_temporal/visualization/gradient_heatmap/render.py
"""Field selection, combination math, color scaling, and figure builders."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

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
        return (-1.0, 1.0, 0.0, "RdBu_r") if signed else (0.0, 1.0, None, "Viridis")
    if signed:
        v = float(np.percentile(np.abs(finite), clip_pct))
        if v <= 0:
            v = float(np.abs(finite).max()) or 1.0
        return (-v, v, 0.0, "RdBu_r")
    hi = float(np.percentile(finite, clip_pct))
    if hi <= 0:
        hi = float(finite.max()) or 1.0
    return (0.0, hi, None, "Viridis")


def build_heatmap_figure(slice2d, geometry, *, title, zmin, zmax, zmid,
                         colorscale, show_boundaries=True):
    """Square-cell heatmap, South at the bottom, West at the left, with optional
    district boundary overlay. slice2d is (48,90) indexed [row=x_grid][col=y_grid]."""
    rows, cols = slice2d.shape
    fig = go.Figure(
        go.Heatmap(
            z=slice2d, x=np.arange(cols), y=np.arange(rows),
            zmin=zmin, zmax=zmax, zmid=zmid, colorscale=colorscale,
            colorbar=dict(title="value"),
            hovertemplate="y_grid(col)=%{x}<br>x_grid(row)=%{y}<br>value=%{z}<extra></extra>",
        )
    )
    if show_boundaries:
        fig.add_trace(go.Scatter(
            x=geometry.boundary_x, y=geometry.boundary_y, mode="lines",
            line=dict(color="black", width=1), hoverinfo="skip", showlegend=False,
        ))
    fig.update_xaxes(title="y_grid (West → East)", constrain="domain")
    fig.update_yaxes(title="x_grid (South → North)",
                     scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def build_contour_overlay(fig, pickup_slice):
    """Overlay pickup-concentration iso-lines on an existing figure."""
    rows, cols = pickup_slice.shape
    fig.add_trace(go.Contour(
        z=pickup_slice, x=np.arange(cols), y=np.arange(rows),
        showscale=False, contours_coloring="lines", line_width=1,
        colorscale="Greys", hoverinfo="skip", opacity=0.6,
    ))
    return fig


def export_png(slice2d, geometry, *, title, vmin, vmax, cmap, show_boundaries=True):
    """Render a publication-quality PNG (Matplotlib, origin='lower'); return bytes."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(slice2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    if show_boundaries:
        ax.plot(geometry.boundary_x, geometry.boundary_y, color="black", lw=0.8)
    ax.set_xlabel("y_grid (West → East)")
    ax.set_ylabel("x_grid (South → North)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
