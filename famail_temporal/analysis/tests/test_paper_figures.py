import math

import numpy as np

from famail_temporal.analysis.paper_figures import t_ci_from_values


def test_t_ci_matches_scipy_reference():
    # Known sample; compare against an independent t-interval computation.
    vals = [0.0324, 0.0247, 0.0264, 0.0296, 0.0266, 0.0250]
    lo, hi = t_ci_from_values(vals, confidence=0.95)
    mean = float(np.mean(vals))
    # Interval is centered on the mean and strictly positive (this is the
    # edited_w30 paired-diff sample -> significant gain).
    assert lo < mean < hi
    assert lo > 0.0
    assert math.isclose((lo + hi) / 2.0, mean, rel_tol=0, abs_tol=1e-12)

    # Reproduce half-width from first principles.
    from scipy.stats import t

    n = len(vals)
    sem = float(np.std(vals, ddof=1)) / math.sqrt(n)
    expected_half = sem * float(t.ppf(0.975, n - 1))
    assert math.isclose((hi - lo) / 2.0, expected_half, rel_tol=1e-12)


def test_t_ci_too_few_values_is_nan():
    lo, hi = t_ci_from_values([0.5])
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = t_ci_from_values([])
    assert math.isnan(lo) and math.isnan(hi)


def test_t_ci_zero_variance_collapses_to_point():
    lo, hi = t_ci_from_values([0.73, 0.73, 0.73, 0.73])
    assert math.isclose(lo, 0.73, abs_tol=1e-12)
    assert math.isclose(hi, 0.73, abs_tol=1e-12)


def test_t_ci_drops_non_finite_and_none():
    finite = [0.1, 0.2, 0.3]
    a = t_ci_from_values(finite)
    b = t_ci_from_values(finite + [float("nan"), None])
    assert math.isclose(a[0], b[0], rel_tol=1e-12)
    assert math.isclose(a[1], b[1], rel_tol=1e-12)


def test_t_ci_wider_for_higher_confidence():
    vals = [0.01, 0.02, 0.015, 0.025, 0.018]
    lo95, hi95 = t_ci_from_values(vals, 0.95)
    lo99, hi99 = t_ci_from_values(vals, 0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


def test_t_ci_rejects_bad_confidence():
    import pytest

    with pytest.raises(ValueError):
        t_ci_from_values([0.1, 0.2, 0.3], confidence=1.0)
    with pytest.raises(ValueError):
        t_ci_from_values([0.1, 0.2, 0.3], confidence=0.0)
