"""Streamlit app: interactive gradient/attribution/concentration heatmaps.

Run:  streamlit run famail_temporal/visualization/gradient_heatmap/app.py
"""
from __future__ import annotations

import numpy as np

from . import render as rd
from .loader import DEFAULT_BUNDLE_PATH, load_bundle


def build_views(bundle, state) -> dict:
    """Pure view builder: state dict -> {'main': fig[, 'concentration': fig]}."""
    field = rd.select_field(bundle, state["quantity"], state["term"],
                            state["alpha_spatial"], state["alpha_causal"],
                            state["alpha_fidelity"])
    hour = state["hour"]
    signed = rd.is_signed(state["quantity"])
    use_mag = bool(state["magnitude"]) and signed

    scale_src = field if state["shared_scale"] else field[:, :, hour]
    if use_mag:
        scale_src = np.abs(scale_src)
    zmin, zmax, zmid, cs = rd.color_range(scale_src, signed and not use_mag, state["clip_pct"])

    slice2d = field[:, :, hour]
    if use_mag:
        slice2d = np.abs(slice2d)

    title = f"{state['quantity']} — {state['term']} — hour {hour:02d}"
    main = rd.build_heatmap_figure(slice2d, bundle.geometry, title=title,
                                   zmin=zmin, zmax=zmax, zmid=zmid, colorscale=cs,
                                   show_boundaries=state["show_boundaries"])
    if state["contour_overlay"]:
        rd.build_contour_overlay(main, bundle.pickup[:, :, hour])

    out = {"main": main}
    if state["show_concentration_panel"]:
        csrc = bundle.pickup if state["shared_scale"] else bundle.pickup[:, :, hour]
        czmin, czmax, _, ccs = rd.color_range(csrc, signed=False, clip_pct=state["clip_pct"])
        out["concentration"] = rd.build_heatmap_figure(
            bundle.pickup[:, :, hour], bundle.geometry,
            title=f"Concentration — hour {hour:02d}", zmin=czmin, zmax=czmax,
            zmid=None, colorscale=ccs, show_boundaries=state["show_boundaries"])
    return out


def main() -> None:  # pragma: no cover - Streamlit UI
    import streamlit as st

    st.set_page_config(page_title="FAMAIL Gradient Heatmap", layout="wide")
    st.title("FAMAIL Temporal — Objective Gradient Heatmap")

    try:
        bundle = load_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    m = bundle.meta
    with st.sidebar:
        st.caption(f"Source: {m.get('source', '?')}  |  created {m.get('created', '?')}")
        quantity = st.radio("Quantity", rd.QUANTITIES, index=0)
        term = st.selectbox("Term / filter", rd.TERMS, index=1)
        hour = st.slider("Hour (0–23)", 0, 23, 8)
        cols = st.columns(2)
        if cols[0].button("◀ prev"):
            hour = (hour - 1) % 24
        if cols[1].button("next ▶"):
            hour = (hour + 1) % 24
        st.markdown("**Display**")
        magnitude = st.checkbox("|magnitude|", value=False)
        shared_scale = st.checkbox("Shared scale across 24 hours", value=True)
        clip_pct = st.slider("Percentile clip", 80.0, 100.0, 99.0)
        show_boundaries = st.checkbox("District boundaries", value=True)
        contour_overlay = st.checkbox("Concentration contour overlay", value=False)
        show_conc = st.checkbox("Concentration panel", value=False)
        st.markdown("**α weights (Combined)**")
        a_sp = st.slider("α spatial", 0.0, 1.0, float(m.get("default_alpha_spatial", 0.33)))
        a_ca = st.slider("α causal", 0.0, 1.0, float(m.get("default_alpha_causal", 0.33)))
        a_fi = st.slider("α fidelity", 0.0, 1.0, float(m.get("default_alpha_fidelity", 0.34)))

    if term == "F_fidelity":
        st.warning("F_fidelity has no per-cell spatial gradient (≈0 by construction). "
                   "It is a per-trajectory realism constraint, not a spatial steering "
                   "force — shown flat for that reason.")
    if term == "Combined":
        st.info("At the per-cell level, Combined ≡ Spatial+Causal "
                "(fidelity contributes no per-cell field).")

    state = dict(quantity=quantity, term=term, hour=hour,
                 alpha_spatial=a_sp, alpha_causal=a_ca, alpha_fidelity=a_fi,
                 magnitude=magnitude, shared_scale=shared_scale, clip_pct=clip_pct,
                 show_boundaries=show_boundaries, contour_overlay=contour_overlay,
                 show_concentration_panel=show_conc)
    views = build_views(bundle, state)

    if show_conc:
        c1, c2 = st.columns(2)
        c1.plotly_chart(views["main"], use_container_width=True)
        c2.plotly_chart(views["concentration"], use_container_width=True)
    else:
        st.plotly_chart(views["main"], use_container_width=True)

    field = rd.select_field(bundle, quantity, term, a_sp, a_ca, a_fi)
    signed = rd.is_signed(quantity) and not (magnitude and rd.is_signed(quantity))
    scale_src = field if shared_scale else field[:, :, hour]
    if magnitude and rd.is_signed(quantity):
        scale_src = np.abs(scale_src)
    vmin, vmax, _, _ = rd.color_range(scale_src, signed, clip_pct)
    slice2d = np.abs(field[:, :, hour]) if (magnitude and rd.is_signed(quantity)) else field[:, :, hour]
    png = rd.export_png(slice2d, bundle.geometry,
                        title=f"{quantity} — {term} — hour {hour:02d}",
                        vmin=vmin, vmax=vmax,
                        cmap="RdBu_r" if signed else "viridis",
                        show_boundaries=show_boundaries)
    st.download_button("Download publication PNG", data=png,
                       file_name=f"gradient_{quantity}_{term}_h{hour:02d}.png",
                       mime="image/png")


if __name__ == "__main__":  # pragma: no cover
    main()
