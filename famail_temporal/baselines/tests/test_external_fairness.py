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


def test_theil_zero_when_all_regions_equal():
    Y = np.array([3.0, 3.0, 3.0, 3.0])
    regions = np.array([0, 0, 1, 1])
    assert ef.theil_index(Y, regions) == pytest.approx(0.0, abs=1e-12)


def test_theil_scale_invariant():
    Y = np.array([1.0, 1.0, 5.0, 5.0])
    regions = np.array([0, 0, 1, 1])
    t1 = ef.theil_index(Y, regions)
    t2 = ef.theil_index(10.0 * Y, regions)
    assert t1 == pytest.approx(t2)
    assert t1 > 0.0


def test_theil_hand_computed_two_regions():
    # region means 1 and 3, equal sizes -> ybar=2
    # T = 0.5*(1/2)ln(1/2) + 0.5*(3/2)ln(3/2)
    Y = np.array([1.0, 1.0, 3.0, 3.0])
    regions = np.array([0, 0, 1, 1])
    expected = 0.5 * (0.5) * np.log(0.5) + 0.5 * (1.5) * np.log(1.5)
    assert ef.theil_index(Y, regions) == pytest.approx(expected)


def test_theil_excludes_negative_region_and_survives_zero_service():
    Y = np.array([0.0, 4.0, 4.0, 99.0])
    regions = np.array([0, 0, 1, -1])        # last excluded; zero-service in region 0
    # region0 mean=2, region1 mean=4, ybar over valid = (0+4+4)/3
    val = ef.theil_index(Y, regions)
    assert np.isfinite(val)
