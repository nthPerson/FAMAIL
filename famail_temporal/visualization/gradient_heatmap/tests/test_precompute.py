import json
import numpy as np
from famail_temporal.visualization.gradient_heatmap import precompute as pc
from famail_temporal.visualization.gradient_heatmap import geometry as geom


def _synthetic_layers():
    rng = np.random.default_rng(0)
    mask = np.zeros((48, 90, 24), dtype=bool)
    mask[10:20, 10:20, :] = True
    def f():
        a = np.full((48, 90, 24), np.nan, dtype=np.float32)
        a[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
        return a
    return {
        "grad_spatial": f(), "grad_causal": f(),
        "attr_spatial": f(), "attr_causal": f(),
        "pickup": rng.random((48, 90, 24)).astype(np.float32),
        "active_mask": mask,
    }


def test_assemble_bundle_has_all_keys():
    g = geom.load_district_geometry()
    meta = {"source": "test", "default_alpha_spatial": 0.33}
    d = pc.assemble_bundle(_synthetic_layers(), g, meta)
    for k in ("grad_spatial", "grad_causal", "attr_spatial", "attr_causal",
              "pickup", "active_mask", "district_id_grid", "valid_mask",
              "district_names", "boundary_x", "boundary_y", "meta_json"):
        assert k in d, f"missing {k}"
    assert json.loads(str(d["meta_json"]))["source"] == "test"


def test_save_bundle_roundtrips_arrays(tmp_path):
    g = geom.load_district_geometry()
    layers = _synthetic_layers()
    d = pc.assemble_bundle(layers, g, {"source": "test"})
    out = tmp_path / "b.npz"
    pc.save_bundle(out, d)
    assert out.exists()
    z = np.load(out, allow_pickle=False)
    np.testing.assert_array_equal(z["active_mask"], layers["active_mask"])
    # NaN preserved on inactive cells
    assert np.isnan(z["grad_spatial"][0, 0, 0])
    assert [str(s) for s in z["district_names"].tolist()][7] == "Nanshan"
