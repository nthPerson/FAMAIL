import numpy as np
import pytest

from famail_temporal.baselines import external_fairness as ef


def test_perfect_parity_dp_zero_di_one():
    Y = np.array([2.0, 2.0, 2.0, 2.0])
    groups = np.array([0, 0, 1, 1])          # A, A, D, D
    assert ef.demographic_parity(Y, groups) == pytest.approx(0.0)
    assert ef.disparate_impact(Y, groups) == pytest.approx(1.0)
    sdr = ef.supply_demand_ratio(Y, groups)
    assert sdr["gap"] == pytest.approx(0.0)


def test_skewed_case_hand_computed():
    # D units under-served: mean(D)=1, mean(A)=4
    Y = np.array([4.0, 4.0, 1.0, 1.0])
    groups = np.array([0, 0, 1, 1])
    assert ef.demographic_parity(Y, groups) == pytest.approx(3.0)   # A - D
    assert ef.disparate_impact(Y, groups) == pytest.approx(0.25)    # D / A
    sdr = ef.supply_demand_ratio(Y, groups)
    assert sdr["mean_disadvantaged"] == pytest.approx(1.0)
    assert sdr["mean_advantaged"] == pytest.approx(4.0)
    assert sdr["gap"] == pytest.approx(3.0)


def test_excluded_units_ignored():
    Y = np.array([4.0, 1.0, 99.0])
    groups = np.array([0, 1, -1])            # last excluded
    assert ef.demographic_parity(Y, groups) == pytest.approx(3.0)


def test_empty_group_returns_nan():
    Y = np.array([1.0, 2.0])
    groups = np.array([0, 0])                # no disadvantaged
    assert np.isnan(ef.demographic_parity(Y, groups))
    assert np.isnan(ef.disparate_impact(Y, groups))
