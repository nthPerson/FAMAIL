"""Unit tests for the variance suite's pure aggregation helpers."""
import json
import math

import numpy as np

from famail_temporal.baselines import run_variance_suite as vs


def test_mean_std_basic():
    out = vs.mean_std([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0
    assert math.isclose(out["std"], 1.0)  # sample std, ddof=1
    assert out["min"] == 1.0 and out["max"] == 3.0 and out["n"] == 3


def test_mean_std_single_value_has_zero_std():
    out = vs.mean_std([5.0])
    assert out["mean"] == 5.0 and out["std"] == 0.0 and out["n"] == 1


def test_mean_std_empty_is_nan_not_crash():
    # Single-seed runs have zero within-variant JS pairs; aggregation must
    # not throw away a completed (GPU-expensive) suite.
    out = vs.mean_std([])
    assert out["n"] == 0
    assert math.isnan(out["mean"]) and math.isnan(out["std"])
    assert math.isnan(out["min"]) and math.isnan(out["max"])


def test_paired_delta_stats_subtracts_b0_from_famail():
    out = vs.paired_delta_stats(b0=[1.0, 2.0], famail=[1.5, 2.1])
    assert math.isclose(out["mean"], 0.3)
    assert out["n"] == 2


def test_pairwise_js_stats_zero_for_identical_histograms():
    h = np.array([0.5, 0.5, 0.0])
    out = vs.pairwise_js_stats([h, h.copy(), h.copy()])
    assert out["n"] == 3  # C(3,2) pairs
    assert math.isclose(out["mean"], 0.0, abs_tol=1e-12)


def test_pairwise_js_stats_one_for_disjoint_histograms():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    out = vs.pairwise_js_stats([a, b])
    assert out["n"] == 1
    assert math.isclose(out["mean"], 1.0, rel_tol=1e-6)


def test_cross_js_values_paired_by_index():
    a = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    b = [np.array([1.0, 0.0]), np.array([0.5, 0.5])]
    vals = vs.cross_js_values(a, b)
    assert len(vals) == 2
    assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in vals)


def test_result_to_json_roundtrips_numpy_floats():
    blob = vs.result_to_json({"x": np.float64(1.5), "y": {"z": [1, 2]}})
    loaded = json.loads(blob)
    assert loaded["x"] == 1.5 and loaded["y"]["z"] == [1, 2]


def test_adv_curve_or_none():
    # adv_epochs=0 -> empty lists -> None
    assert vs.adv_curve_or_none({"adv_losses": {"g_losses": [], "d_losses": [],
                                                "g_batch_losses": [], "d_batch_losses": []}}) is None
    assert vs.adv_curve_or_none({}) is None
    # populated -> dict with float lists
    out = vs.adv_curve_or_none({"adv_losses": {
        "g_losses": [0.7], "d_losses": [1.3],
        "g_batch_losses": [0.71, 0.69], "d_batch_losses": [1.3, 1.31]}})
    assert out["g_epoch_losses"] == [0.7]
    assert out["g_batch_losses"] == [0.71, 0.69]
    assert out["d_batch_losses"] == [1.3, 1.31]
