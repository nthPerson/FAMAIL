"""Tests for the pure helpers in fcausal_feature_sensitivity.

Covers the rank-correlation, Jaccard, top-K, and VIF helpers — the small pure
functions the sweep depends on. The end-to-end metric path is covered by the
sanity gate (it must reproduce the editor's 0.8069) and is exercised when the
module is run, so it is not duplicated here.
"""
from __future__ import annotations

import numpy as np
import pytest

from famail_temporal.analysis import fcausal_feature_sensitivity as F


def test_spearman_perfect_monotone():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert F._spearman(a, b) == pytest.approx(1.0)
    assert F._spearman(a, -b) == pytest.approx(-1.0)


def test_spearman_matches_scipy_with_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=200).astype(float)  # heavy ties
    b = a * 2.0 + rng.normal(size=200)
    expected = scipy_stats.spearmanr(a, b).statistic
    assert F._spearman(a, b) == pytest.approx(expected, abs=1e-9)


def test_rankdata_average_ties():
    # scipy.stats.rankdata default = average ties.
    x = np.array([3.0, 1.0, 1.0, 2.0])
    # sorted: 1,1,2,3 → the two 1s share average rank (1+2)/2 = 1.5
    r = F._rankdata_average(x)
    np.testing.assert_allclose(r, [4.0, 1.5, 1.5, 3.0])


def test_topk_unfair_picks_smallest_alpha():
    # Most-unfair = smallest (most negative) alpha.
    alpha = np.array([0.5, -0.3, 0.1, -0.9, 0.2])
    idx = F._topk_unfair_indices(alpha, 2)
    assert set(idx.tolist()) == {3, 1}  # -0.9 and -0.3


def test_topk_clamps_to_length():
    alpha = np.array([0.1, 0.2, 0.3])
    idx = F._topk_unfair_indices(alpha, 10)
    assert len(idx) == 3


def test_jaccard_basic():
    a = np.array([1, 2, 3, 4])
    b = np.array([3, 4, 5, 6])
    # intersection {3,4}=2, union {1..6}=6
    assert F._jaccard(a, b) == pytest.approx(2 / 6)
    assert F._jaccard(a, a) == pytest.approx(1.0)


def test_vif_single_feature_is_one():
    x = np.random.default_rng(1).normal(size=(50, 1))
    vifs = F._vifs(x, ["only"])
    assert vifs == {"only": 1.0}


def test_vif_independent_features_near_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 3))  # independent columns
    vifs = F._vifs(x, ["a", "b", "c"])
    for v in vifs.values():
        assert v == pytest.approx(1.0, abs=0.15)


def test_vif_collinear_features_blow_up():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(500, 1))
    # Two near-duplicate columns → high VIF.
    x = np.column_stack([base[:, 0], base[:, 0] + 1e-3 * rng.normal(size=500),
                         rng.normal(size=500)])
    vifs = F._vifs(x, ["a", "b", "c"])
    assert vifs["a"] > 50.0
    assert vifs["b"] > 50.0
    assert vifs["c"] == pytest.approx(1.0, abs=0.3)


def test_max_abs_offdiag_corr():
    rng = np.random.default_rng(4)
    a = rng.normal(size=400)
    x = np.column_stack([a, 0.9 * a + 0.1 * rng.normal(size=400),
                         rng.normal(size=400)])
    m = F._max_abs_offdiag_corr(x)
    assert 0.9 < m <= 1.0  # cols 0,1 strongly correlated
    # single column → 0
    assert F._max_abs_offdiag_corr(x[:, :1]) == 0.0


def test_redundant_pairs_flags_near_duplicates():
    names = ["h", "logh", "indep"]
    corr = {
        "h": {"h": 1.0, "logh": 0.999, "indep": 0.1},
        "logh": {"h": 0.999, "logh": 1.0, "indep": 0.05},
        "indep": {"h": 0.1, "logh": 0.05, "indep": 1.0},
    }
    pairs = F._redundant_pairs(corr, names, thresh=0.95)
    assert pairs == [("h", "logh", 0.999)]


def test_pareto_domination_and_verdicts():
    base_f = 0.80
    rows = [
        {"set": "baseline_h-g-c", "n_features": 3, "max_vif": 2.5,
         "max_abs_corr": 0.7, "topk_jaccard": 1.0, "spearman_alpha": 1.0,
         "f_causal": 0.80, "axes": ["housing", "income"]},
        {"set": "better", "n_features": 4, "max_vif": 4.0,
         "max_abs_corr": 0.8, "topk_jaccard": 0.93, "spearman_alpha": 0.87,
         "f_causal": 0.72, "axes": ["housing", "income", "pop_structure"]},
        {"set": "highvif", "n_features": 4, "max_vif": 15.0,
         "max_abs_corr": 0.9, "topk_jaccard": 0.95, "spearman_alpha": 0.9,
         "f_causal": 0.71, "axes": ["housing", "income", "pop_structure"]},
    ]
    out, summary = F._pareto_and_verdicts(rows, base_f)
    by = {r["set"]: r for r in out}
    assert by["better"]["verdict"] == "ROBUST-AND-BETTER"
    assert by["highvif"]["verdict"] == "HIGH-VIF/UNSTABLE"
    assert by["baseline_h-g-c"]["verdict"] == "ROBUST-EQUIVALENT"
    assert "better" in summary["sets_dominating_base3"]
    assert "highvif" not in summary["sets_dominating_base3"]  # VIF>=10 excluded
    assert summary["any_low_vif_pop_axis_beats_base3"] == "better"
