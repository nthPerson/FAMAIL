"""Load + validate the cached viz bundle (pure numpy; no torch)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import DistrictGeometry
from .precompute import DEFAULT_BUNDLE_PATH

__all__ = ["VizBundle", "load_bundle", "DEFAULT_BUNDLE_PATH"]


@dataclass(frozen=True)
class VizBundle:
    grad_spatial: np.ndarray
    grad_causal: np.ndarray
    attr_spatial: np.ndarray
    attr_causal: np.ndarray
    pickup: np.ndarray
    active_mask: np.ndarray
    geometry: DistrictGeometry
    meta: dict


def load_bundle(path=DEFAULT_BUNDLE_PATH) -> VizBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"viz bundle not found at {path}; run "
            f"`python -m famail_temporal.visualization.gradient_heatmap.precompute` first"
        )
    z = np.load(path, allow_pickle=False)
    geometry = DistrictGeometry(
        district_id_grid=z["district_id_grid"],
        valid_mask=z["valid_mask"].astype(bool),
        district_names=[str(s) for s in z["district_names"].tolist()],
        boundary_x=z["boundary_x"],
        boundary_y=z["boundary_y"],
    )
    for key in ("grad_spatial", "grad_causal", "attr_spatial", "attr_causal", "pickup"):
        if z[key].shape != (48, 90, 24):
            raise ValueError(f"{key} has shape {z[key].shape}, expected (48,90,24)")
    return VizBundle(
        grad_spatial=z["grad_spatial"], grad_causal=z["grad_causal"],
        attr_spatial=z["attr_spatial"], attr_causal=z["attr_causal"],
        pickup=z["pickup"], active_mask=z["active_mask"].astype(bool),
        geometry=geometry, meta=json.loads(str(z["meta_json"])),
    )
