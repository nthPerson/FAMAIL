"""Precompute CLI: build the cached viz bundle from DataBundle.

Run once (CPU-only; needs no GPU and no trained discriminator — the gradient/
attribution layers depend only on spatial + causal terms):

    python -m famail_temporal.visualization.gradient_heatmap.precompute
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from . import geometry as geom

DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parent / "cache" / "gradient_viz_bundle.npz"


def compute_layers(bundle) -> dict:
    """Compute the six (48,90,24) layers from a loaded DataBundle."""
    from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
    from famail_temporal.evaluation.grid import build_fairness_grid

    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)   # (48,90,24,2)
    fair = build_fairness_grid(bundle, bundle.pickup_3d)            # (48,90,24,4)
    return {
        "grad_spatial": sens[..., 0].astype(np.float32),
        "grad_causal":  sens[..., 1].astype(np.float32),
        "attr_spatial": fair[..., 0].astype(np.float32),
        "attr_causal":  fair[..., 1].astype(np.float32),
        "pickup":       np.asarray(bundle.pickup_3d, dtype=np.float32),
        "active_mask":  np.asarray(bundle.mask_3d, dtype=bool),
    }


def assemble_bundle(layers: dict, geometry: geom.DistrictGeometry, meta: dict) -> dict:
    """Flat dict ready for np.savez_compressed (no object arrays / no pickle)."""
    d = dict(layers)
    d["district_id_grid"] = geometry.district_id_grid
    d["valid_mask"] = geometry.valid_mask
    d["district_names"] = np.asarray(list(geometry.district_names))  # unicode array
    d["boundary_x"] = geometry.boundary_x
    d["boundary_y"] = geometry.boundary_y
    d["meta_json"] = np.asarray(json.dumps(meta))                    # 0-d unicode
    return d


def save_bundle(path, bundle_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle_dict)


def main(argv=None) -> None:
    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle

    parser = argparse.ArgumentParser(description="Build gradient-heatmap viz cache.")
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_PATH)
    args = parser.parse_args(argv)

    geometry = geom.load_district_geometry(args.mapping)
    geom.assert_canonical_orientation(geometry.district_id_grid, geometry.district_names)

    print("Loading DataBundle (CPU)...")
    bundle = DataBundle.load()
    print("Computing gradient + attribution + concentration layers...")
    layers = compute_layers(bundle)

    meta = {
        "default_alpha_spatial": config.ALPHA_SPATIAL,
        "default_alpha_causal": config.ALPHA_CAUSAL,
        "default_alpha_fidelity": config.ALPHA_FIDELITY,
        "grid_dims": list(config.GRID_DIMS),
        "T": config.T,
        "source": "DataBundle.load() preprocess cache (before-edit dataset)",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    save_bundle(args.out, assemble_bundle(layers, geometry, meta))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
