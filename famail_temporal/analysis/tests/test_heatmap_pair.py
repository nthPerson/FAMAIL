"""TDD test for before/after gradient-heatmap pair (E16).

Strategy: build a minimal synthetic gradient_viz_bundle.npz with the correct
(48,90,24) shapes required by loader.load_bundle, then call write_heatmap_png
end-to-end and assert a non-empty PNG is written to disk.

No DataBundle, no torch, no precompute — pure numpy + matplotlib.
"""
import json
import numpy as np
import pytest
from pathlib import Path

from famail_temporal.analysis.heatmap_pair import write_heatmap_png


def _make_synthetic_bundle_npz(tmp_path: Path) -> Path:
    """Write a minimal gradient_viz_bundle.npz with correct shapes."""
    rng = np.random.default_rng(42)
    shape = (48, 90, 24)

    # Six (48,90,24) float32 layers
    grad_spatial  = rng.random(shape).astype(np.float32) * 1e-4
    grad_causal   = rng.random(shape).astype(np.float32) * 1e-4
    attr_spatial  = rng.random(shape).astype(np.float32)
    attr_causal   = rng.random(shape).astype(np.float32)
    pickup        = rng.random(shape).astype(np.float32)
    active_mask   = (rng.random(shape) > 0.5)

    # Geometry: flat district grid (all district 0) with minimal boundary
    district_id_grid = np.zeros((48, 90), dtype=np.int8)
    valid_mask       = np.ones((48, 90), dtype=bool)
    district_names   = np.asarray(["TestDistrict"] * 10)
    # Single boundary segment (start, end, NaN)
    boundary_x = np.array([0.0, 89.0, np.nan])
    boundary_y = np.array([0.0, 47.0, np.nan])
    meta_json  = np.asarray(json.dumps({"synthetic": True}))

    out = tmp_path / "gradient_viz_bundle.npz"
    np.savez_compressed(
        out,
        grad_spatial=grad_spatial, grad_causal=grad_causal,
        attr_spatial=attr_spatial, attr_causal=attr_causal,
        pickup=pickup, active_mask=active_mask,
        district_id_grid=district_id_grid, valid_mask=valid_mask,
        district_names=district_names,
        boundary_x=boundary_x, boundary_y=boundary_y,
        meta_json=meta_json,
    )
    return out


def test_write_heatmap_png_produces_nonempty_file(tmp_path):
    bundle_npz = _make_synthetic_bundle_npz(tmp_path)
    out_png = tmp_path / "heatmap_test.png"

    result = write_heatmap_png(
        bundle_npz,
        quantity="Attribution",
        term="F_causal",
        hour=0,
        out_png=out_png,
    )

    assert result == out_png
    assert out_png.exists()
    assert out_png.stat().st_size > 1000, "PNG should be non-trivially large"


def test_write_heatmap_png_spatial_term(tmp_path):
    bundle_npz = _make_synthetic_bundle_npz(tmp_path)
    out_png = tmp_path / "heatmap_spatial.png"

    result = write_heatmap_png(
        bundle_npz,
        quantity="Attribution",
        term="F_spatial",
        hour=3,
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 1000


def test_write_heatmap_png_concentration(tmp_path):
    bundle_npz = _make_synthetic_bundle_npz(tmp_path)
    out_png = tmp_path / "heatmap_concentration.png"

    result = write_heatmap_png(
        bundle_npz,
        quantity="Concentration",
        term="F_spatial",  # ignored for Concentration
        hour=12,
        out_png=out_png,
    )

    assert out_png.exists()
    assert out_png.stat().st_size > 1000
