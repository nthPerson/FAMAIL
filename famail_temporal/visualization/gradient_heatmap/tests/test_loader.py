import numpy as np
import pytest
from famail_temporal.visualization.gradient_heatmap import precompute as pc
from famail_temporal.visualization.gradient_heatmap import geometry as geom
from famail_temporal.visualization.gradient_heatmap import loader as ld


def _make_bundle(tmp_path):
    rng = np.random.default_rng(1)
    mask = np.zeros((48, 90, 24), dtype=bool)
    mask[5:15, 5:15, :] = True
    def f():
        a = np.full((48, 90, 24), np.nan, dtype=np.float32)
        a[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
        return a
    layers = {"grad_spatial": f(), "grad_causal": f(), "attr_spatial": f(),
              "attr_causal": f(), "pickup": rng.random((48, 90, 24)).astype(np.float32),
              "active_mask": mask}
    g = geom.load_district_geometry()
    out = tmp_path / "b.npz"
    pc.save_bundle(out, pc.assemble_bundle(layers, g, {"source": "test", "T": 24}))
    return out, layers


def test_load_bundle_roundtrip(tmp_path):
    out, layers = _make_bundle(tmp_path)
    vb = ld.load_bundle(out)
    assert vb.grad_spatial.shape == (48, 90, 24)
    np.testing.assert_array_equal(vb.active_mask, layers["active_mask"])
    assert vb.geometry.district_id_grid.shape == (48, 90)
    assert vb.meta["source"] == "test"
    assert "Nanshan" in vb.geometry.district_names


def test_load_bundle_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ld.load_bundle(tmp_path / "nope.npz")


def test_load_bundle_rejects_wrong_shape(tmp_path):
    from famail_temporal.visualization.gradient_heatmap import precompute as pc
    from famail_temporal.visualization.gradient_heatmap import geometry as geom
    g = geom.load_district_geometry()
    bad = {"grad_spatial": np.zeros((10, 10, 10), np.float32),
           "grad_causal": np.zeros((48, 90, 24), np.float32),
           "attr_spatial": np.zeros((48, 90, 24), np.float32),
           "attr_causal": np.zeros((48, 90, 24), np.float32),
           "pickup": np.zeros((48, 90, 24), np.float32),
           "active_mask": np.zeros((48, 90, 24), bool)}
    out = tmp_path / "bad.npz"
    pc.save_bundle(out, pc.assemble_bundle(bad, g, {"source": "t"}))
    with pytest.raises(ValueError):
        ld.load_bundle(out)
