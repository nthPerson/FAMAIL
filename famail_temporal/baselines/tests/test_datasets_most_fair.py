import math
from famail_temporal.baselines.datasets import _most_fair_from_scored


def test_most_fair_from_scored_takes_highest_finite_descending():
    scored = [(0, -2.0), (1, -0.5), (2, 0.1), (3, 0.9), (4, float("inf")), (5, 0.4)]
    # most-fair first, inactive (+inf at idx 4) excluded
    assert _most_fair_from_scored(scored) == [3, 5, 2, 1, 0]
    assert _most_fair_from_scored(scored, 2) == [3, 5]          # top-2 most fair
    assert 4 not in _most_fair_from_scored(scored)              # +inf never selected


def test_most_fair_from_scored_n_larger_than_available():
    scored = [(0, 0.2), (1, float("inf"))]
    assert _most_fair_from_scored(scored, 5) == [0]            # only 1 finite
