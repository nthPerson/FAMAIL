# famail_temporal/visualization/gradient_heatmap/tests/test_render.py
import numpy as np
import pytest
from famail_temporal.visualization.gradient_heatmap import render as rd
from famail_temporal.visualization.gradient_heatmap.geometry import load_district_geometry
from famail_temporal.visualization.gradient_heatmap.loader import VizBundle


def _bundle():
    rng = np.random.default_rng(2)
    mask = np.zeros((48, 90, 24), dtype=bool)
    mask[0:10, 0:10, :] = True
    def f():
        a = np.full((48, 90, 24), np.nan, dtype=np.float32)
        a[mask] = rng.standard_normal(mask.sum()).astype(np.float32)
        return a
    return VizBundle(
        grad_spatial=f(), grad_causal=f(), attr_spatial=f(), attr_causal=f(),
        pickup=rng.random((48, 90, 24)).astype(np.float32), active_mask=mask,
        geometry=load_district_geometry(), meta={},
    )


def test_select_spatial_returns_grad_spatial():
    b = _bundle()
    out = rd.select_field(b, "Gradient", "F_spatial", 0.33, 0.33, 0.34)
    assert np.allclose(out, b.grad_spatial, equal_nan=True)


def test_combined_equals_spatial_plus_causal_at_cell_level():
    b = _bundle()
    comb = rd.select_field(b, "Gradient", "Combined", 0.33, 0.33, 0.34)
    spca = rd.select_field(b, "Gradient", "Spatial+Causal", 0.33, 0.33, 0.34)
    assert np.allclose(comb, spca, equal_nan=True)


def test_weighted_sum_is_exact():
    b = _bundle()
    spca = rd.select_field(b, "Gradient", "Spatial+Causal", 0.2, 0.7, 0.1)
    expected = 0.2 * b.grad_spatial + 0.7 * b.grad_causal
    assert np.allclose(spca, expected, equal_nan=True)


def test_fidelity_field_is_zero_on_active_nan_inactive():
    b = _bundle()
    out = rd.select_field(b, "Gradient", "F_fidelity", 0.33, 0.33, 0.34)
    assert np.all(out[b.active_mask] == 0.0)
    assert np.all(np.isnan(out[~b.active_mask]))


def test_concentration_ignores_term():
    b = _bundle()
    a = rd.select_field(b, "Concentration", "F_spatial", 0.33, 0.33, 0.34)
    c = rd.select_field(b, "Concentration", "F_causal", 0.33, 0.33, 0.34)
    np.testing.assert_array_equal(a, b.pickup)
    np.testing.assert_array_equal(a, c)


def test_is_signed():
    assert rd.is_signed("Gradient") and rd.is_signed("Attribution")
    assert not rd.is_signed("Concentration")


def test_color_range_signed_symmetric_about_zero():
    vals = np.array([[-3.0, 1.0], [np.nan, 2.0]])
    zmin, zmax, zmid, cs = rd.color_range(vals, signed=True, clip_pct=100.0)
    assert zmin == -zmax and zmid == 0.0 and zmax == pytest.approx(3.0)


def test_color_range_sequential_starts_at_zero():
    vals = np.array([[0.0, 5.0], [10.0, np.nan]])
    zmin, zmax, zmid, cs = rd.color_range(vals, signed=False, clip_pct=100.0)
    assert zmin == 0.0 and zmid is None and zmax == pytest.approx(10.0)


def test_unknown_quantity_raises():
    with pytest.raises(ValueError):
        rd.select_field(_bundle(), "Bogus", "F_spatial", 0.1, 0.1, 0.1)


def test_unknown_term_raises():
    with pytest.raises(ValueError):
        rd.select_field(_bundle(), "Gradient", "Bogus", 0.1, 0.1, 0.1)


def test_color_range_all_nan_signed_returns_diverging():
    zmin, zmax, zmid, cs = rd.color_range(np.full((5, 5), np.nan), signed=True)
    assert np.isfinite(zmin) and np.isfinite(zmax)
    assert zmid == 0.0 and cs == "RdBu_r"


def test_color_range_all_nan_unsigned_returns_sequential():
    zmin, zmax, zmid, cs = rd.color_range(np.full((5, 5), np.nan), signed=False)
    assert np.isfinite(zmin) and np.isfinite(zmax)
    assert zmid is None and cs == "Viridis"
