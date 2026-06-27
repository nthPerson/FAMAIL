"""Before/after gradient-heatmap pair driver (E16).

Pure helper: write_heatmap_png
CLI driver: main (pair + difference panel) [DEFERRED EXECUTION — needs two
gradient_viz_bundle.npz files produced by precompute, one per bundle variant;
do NOT run while the experiment sequence is live]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from famail_temporal.visualization.gradient_heatmap.loader import load_bundle
from famail_temporal.visualization.gradient_heatmap.render import (
    select_field,
    color_range,
    export_png,
    is_signed,
)


def write_heatmap_png(
    bundle_npz_path,
    *,
    quantity: str,
    term: str,
    hour: int,
    out_png,
    alpha_spatial: float = 0.2,
    alpha_causal: float = 0.7,
    alpha_fidelity: float = 0.1,
    show_boundaries: bool = True,
) -> Path:
    """Render a single heatmap slice as a publication-quality PNG.

    Loads ``bundle_npz_path`` via the viz loader, picks the (48,90,24) field
    for ``quantity`` x ``term``, slices to ``hour`` (0-indexed, 0..23),
    computes a robust color range, and writes the PNG to ``out_png``.

    Parameters
    ----------
    bundle_npz_path : path-like
        Path to a ``gradient_viz_bundle.npz`` produced by precompute.
    quantity : str
        One of "Gradient", "Attribution", "Concentration".
    term : str
        One of "F_spatial", "F_causal", "F_fidelity", "Combined",
        "Spatial+Causal".  Ignored when quantity=="Concentration".
    hour : int
        Time-block index (0..23) to slice from the 24-hour axis.
    out_png : path-like
        Destination PNG file path.  Parent directory is created if needed.
    alpha_spatial, alpha_causal, alpha_fidelity : float
        Weights used when term is "Combined" or "Spatial+Causal".

    Returns
    -------
    Path
        The resolved path of the written PNG file.
    """
    bundle_npz_path = Path(bundle_npz_path)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(bundle_npz_path)

    # (48,90,24) field selection via the shared render helper
    field3d = select_field(
        bundle, quantity, term,
        alpha_spatial, alpha_causal, alpha_fidelity,
    )
    # Slice to the requested hour: shape (48,90)
    slice2d = field3d[:, :, hour]

    signed = is_signed(quantity)
    zmin, zmax, _zmid, colorscale = color_range(slice2d, signed)
    # color_range returns Plotly colorscale names (e.g. "Viridis", "RdBu_r");
    # export_png uses matplotlib, which requires lowercase ("viridis", "RdBu_r").
    # RdBu_r is already valid in matplotlib; only "Viridis" needs lowercasing.
    mpl_cmap = colorscale.lower() if colorscale in ("Viridis",) else colorscale

    # export_png uses matplotlib parameter names vmin/vmax/cmap
    png_bytes = export_png(
        slice2d, bundle.geometry,
        title=f"{quantity} / {term}  (hour={hour})",
        vmin=zmin, vmax=zmax,
        cmap=mpl_cmap,
        show_boundaries=show_boundaries,
    )
    out_png.write_bytes(png_bytes)
    return out_png


# ---------------------------------------------------------------------------
# CLI driver (DEFERRED — do NOT execute while baaigffdf sequence is live)
# ---------------------------------------------------------------------------

def _difference_panel(
    bundle_npz_with_sinks: Path,
    bundle_npz_cleaned: Path,
    *,
    quantity: str,
    term: str,
    hour: int,
    out_png: Path,
    alpha_spatial: float = 0.2,
    alpha_causal: float = 0.7,
    alpha_fidelity: float = 0.1,
) -> Path:
    """Render the (cleaned − with_sinks) difference heatmap."""
    b_dirty = load_bundle(bundle_npz_with_sinks)
    b_clean = load_bundle(bundle_npz_cleaned)

    f_dirty = select_field(b_dirty, quantity, term,
                           alpha_spatial, alpha_causal, alpha_fidelity)
    f_clean = select_field(b_clean, quantity, term,
                           alpha_spatial, alpha_causal, alpha_fidelity)

    diff2d = f_clean[:, :, hour] - f_dirty[:, :, hour]
    zmin, zmax, _zmid, _ = color_range(diff2d, signed=True)

    png_bytes = export_png(
        diff2d, b_clean.geometry,
        title=f"Δ{quantity}/{term} (cleaned−with_sinks, hour={hour})",
        vmin=zmin, vmax=zmax,
        cmap="RdBu_r",
        show_boundaries=True,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_png.write_bytes(png_bytes)
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.heatmap_pair",
        description=(
            "DEFERRED EXECUTION: requires two gradient_viz_bundle.npz files "
            "from precompute (one with sinks, one cleaned). "
            "Do NOT run while the experiment sequence is live."
        ),
    )
    ap.add_argument("--with-sinks-bundle", type=Path, required=True,
                    help="gradient_viz_bundle_with_sinks.npz path")
    ap.add_argument("--cleaned-bundle", type=Path, required=True,
                    help="gradient_viz_bundle_cleaned.npz path")
    ap.add_argument("--quantity", default="Attribution",
                    choices=["Gradient", "Attribution", "Concentration"])
    ap.add_argument("--term", default="F_spatial",
                    choices=["F_spatial", "F_causal", "F_fidelity",
                             "Combined", "Spatial+Causal"])
    ap.add_argument("--hour", type=int, default=10)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("famail_temporal/results/analysis/heatmap_pair"))
    args = ap.parse_args(argv)

    slug = f"{args.quantity}_{args.term}"
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    p_dirty = write_heatmap_png(
        args.with_sinks_bundle,
        quantity=args.quantity, term=args.term, hour=args.hour,
        out_png=out_dir / f"heatmap_with_sinks_{slug}.png",
    )
    p_clean = write_heatmap_png(
        args.cleaned_bundle,
        quantity=args.quantity, term=args.term, hour=args.hour,
        out_png=out_dir / f"heatmap_cleaned_{slug}.png",
    )
    p_diff = _difference_panel(
        args.with_sinks_bundle, args.cleaned_bundle,
        quantity=args.quantity, term=args.term, hour=args.hour,
        out_png=out_dir / f"heatmap_diff_{slug}.png",
    )

    for p in (p_dirty, p_clean, p_diff):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
