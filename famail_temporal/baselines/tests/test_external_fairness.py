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


def test_median_split_disadvantaged_low():
    values = np.array([1.0, 2.0, 3.0, 4.0])       # median 2.5
    g = ef.median_split(values, disadvantaged_high=False)
    # low (<=2.5) is disadvantaged
    assert list(g) == [1, 1, 0, 0]


def test_median_split_disadvantaged_high_and_nan_excluded():
    values = np.array([1.0, 4.0, np.nan])
    g = ef.median_split(values, disadvantaged_high=True)
    assert g[2] == -1
    assert g[0] == 0 and g[1] == 1                # high=disadvantaged


def test_region_extremes_top_bottom_third():
    # 6 distinct region values -> frac 1/3 -> k=2 each end
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    g = ef.region_extremes(values, disadvantaged_high=False)  # low = disadv
    assert g[0] == 1 and g[1] == 1                # 10,20 bottom -> D
    assert g[4] == 0 and g[5] == 0               # 50,60 top -> A
    assert g[2] == -1 and g[3] == -1             # middle excluded


def test_region_extremes_groups_by_distinct_value():
    # region-constant values: two regions of size 3
    values = np.array([5.0, 5.0, 5.0, 9.0, 9.0, 9.0])
    g = ef.region_extremes(values, disadvantaged_high=True)   # high = disadv
    assert list(g[:3]) == [0, 0, 0]              # value 5 = advantaged
    assert list(g[3:]) == [1, 1, 1]              # value 9 = disadvantaged


def test_regions_from_values_maps_profiles():
    housing = np.array([1.0, 1.0, 2.0, np.nan])
    comp = np.array([9.0, 9.0, 8.0, 8.0])
    r = ef.regions_from_values([housing, comp])
    assert r[0] == r[1]                          # same profile
    assert r[0] != r[2]                          # different profile
    assert r[3] == -1                            # NaN -> excluded


def test_bootstrap_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    Yb = rng.uniform(0.5, 2.0, size=200)
    Ya = Yb + 0.3                                    # uniform improvement
    groups = np.where(np.arange(200) % 2 == 0, 0, 1)
    specs = [("dp", ef.demographic_parity, groups)]
    out1 = ef.paired_bootstrap(Yb, Ya, specs, B=200, seed=7)
    out2 = ef.paired_bootstrap(Yb, Ya, specs, B=200, seed=7)
    assert out1 == out2                              # determinism
    lo, hi = out1["dp"]["delta"]
    assert lo <= 0.0 <= hi or (lo <= (ef.demographic_parity(Ya, groups)
                                      - ef.demographic_parity(Yb, groups)) <= hi)


def test_bootstrap_counts_empty_group_drops():
    # a group that can vanish under resampling of a tiny sample
    Yb = np.array([1.0, 1.0, 2.0])
    Ya = np.array([1.5, 1.5, 2.0])
    groups = np.array([0, 0, 1])                     # single disadvantaged unit
    specs = [("di", ef.disparate_impact, groups)]
    out = ef.paired_bootstrap(Yb, Ya, specs, B=300, seed=1)
    assert out["di"]["n_dropped"] >= 1               # some resamples drop unit 2
