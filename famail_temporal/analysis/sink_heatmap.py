"""E16 — before/after sink heatmaps (dirty vs clean), the visual proof the
stuck-GPS sinks existed and were removed.

NO source_data swap: reads the two editor runs' grid_before.pkl directly and
renders the spatial-attribution channel (channel 0) dirty vs clean, with the
calibrated sink cells circled. Orientation follows the verified FAMAIL
convention: x_grid row 0 = South (origin='lower' => South at bottom),
y_grid col 0 = West.
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np

from famail_temporal.analysis.sink_decomposition import DEFAULT_SINK_CELLS


def _spatial_2d(run_dir: Path) -> np.ndarray:
    """grid_before.pkl channel-0 ('spatial_attr'), summed over t (NaN->0) -> (gx,gy)."""
    # pickle is safe here: grid_before.pkl is a project-internal numpy artifact
    # produced by famail's own evaluation.persistence layer, never external input.
    with open(Path(run_dir) / "grid_before.pkl", "rb") as f:
        data = pickle.load(f)
    assert data["channel_names"][0] == "spatial_attr", data["channel_names"]
    return np.nansum(data["grid"][..., 0], axis=2)


def render_pair(dirty_run: Path, clean_run: Path, out_png: Path,
                sink_cells=DEFAULT_SINK_CELLS) -> Path:
    """Write a 3-panel figure: dirty | clean | (clean-dirty) spatial attribution,
    sinks circled. Returns out_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _spatial_2d(dirty_run)
    c = _spatial_2d(clean_run)
    diff = c - d
    # Shared symmetric scale for dirty/clean (signed attribution); diverging cmap.
    vmax = float(np.nanpercentile(np.abs(np.concatenate([d.ravel(), c.ravel()])), 99.5))
    vmax = max(vmax, 1e-6)
    # 0-indexed sink positions for scatter (imshow plots (col=y, row=x)).
    sx = [x - 1 for (x, y) in sink_cells]   # row (x_grid)
    sy = [y - 1 for (x, y) in sink_cells]   # col (y_grid)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    panels = [("dirty (with sinks)", d, -vmax, vmax, "RdBu_r"),
              ("cleaned (sinks removed)", c, -vmax, vmax, "RdBu_r"),
              ("clean − dirty", diff, -float(np.nanmax(np.abs(diff))),
               float(np.nanmax(np.abs(diff))), "PuOr_r")]
    for ax, (title, arr, vmn, vmx, cmap) in zip(axes, panels):
        im = ax.imshow(arr, origin="lower", aspect="equal", cmap=cmap,
                       vmin=vmn, vmax=vmx, interpolation="nearest")
        ax.scatter(sy, sx, s=80, facecolors="none", edgecolors="lime", linewidths=1.4)
        ax.set_title(title)
        ax.set_xlabel("y_grid (West → East)")
        ax.set_ylabel("x_grid (South → North)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("E16 — per-cell spatial fairness attribution (αᵢ), dirty vs cleaned; "
                 "circled = calibrated stuck-GPS sinks", y=1.02)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    return out_png


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.analysis.sink_heatmap")
    ap.add_argument("--editor-dirty", type=Path, required=True)
    ap.add_argument("--editor-clean", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path("famail_temporal/results/analysis/sink_heatmap/"
                                 "sink_spatial_attr_before_after.png"))
    a = ap.parse_args(argv)
    render_pair(a.editor_dirty, a.editor_clean, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
