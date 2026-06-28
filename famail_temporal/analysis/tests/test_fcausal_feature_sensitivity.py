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
