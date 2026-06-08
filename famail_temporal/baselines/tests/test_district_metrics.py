"""Unit tests for district-level DI on synthetic district + grid setup."""
import numpy as np
import pytest

from famail_temporal.baselines import district_metrics as dm


def _synthetic_inputs():
    """3 districts x 4 cells/t_blocks each = 12 active units. Hukou ratios
    chosen so that district 0 = top-3 hukou, district 2 = bottom-3 hukou
    (with only 3 districts each is both top-3 and bottom-3, so we use 6
    districts to exercise the grouping cleanly)."""
    n_districts = 6
    # Hukou ratios increasing: districts 3,4,5 are top-3 (high hukou);
    # districts 0,1,2 are bottom-3 (low hukou).
    hukou_ratios = np.array([0.10, 0.15, 0.20, 0.60, 0.70, 0.80])
    # 2 active units per district -> 12 units total.
    # district_of_unit: which district each active unit belongs to.
    district_of_unit = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    # demand_N, supply_N per active unit (12 vector each)
    demand_N = np.array([1.0]*6 + [2.0]*6)        # higher demand in top-3 hukou
    supply_N = np.array([5.0]*6 + [5.0]*6)        # equal supply
    return n_districts, hukou_ratios, district_of_unit, demand_N, supply_N


def test_di_primary_below_one_when_high_hukou_has_lower_supply_demand_ratio():
    # supply/demand: low-hukou districts get 5/1=5, top-hukou get 5/2=2.5.
    # DI_primary = mean(top-hukou Y) / mean(low-hukou Y) = 2.5 / 5.0 = 0.5
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert out["di_primary"] == pytest.approx(0.5, rel=1e-6)


def test_di_supplementary_is_inverse_of_primary_under_equal_supply():
    # demand/supply: top-hukou get 2/5=0.4, low-hukou get 1/5=0.2.
    # DI_supplementary = 0.4 / 0.2 = 2.0 (the inverse of 0.5)
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert out["di_supplementary"] == pytest.approx(2.0, rel=1e-6)


def test_di_returns_per_district_means_for_traceability():
    n_d, hukou, district_of_unit, demand_N, supply_N = _synthetic_inputs()
    out = dm.compute_di(
        demand_N=demand_N, supply_N=supply_N,
        district_of_unit=district_of_unit,
        hukou_ratios=hukou,
        n_top=3, n_bottom=3, demand_floor=1e-3, supply_floor=1e-3,
    )
    assert "per_district_y_primary" in out
    assert out["per_district_y_primary"].shape == (6,)
    # district 0 (low-hukou): supply/demand = 5/1 = 5
    assert out["per_district_y_primary"][0] == pytest.approx(5.0, rel=1e-6)
    # district 3 (high-hukou): supply/demand = 5/2 = 2.5
    assert out["per_district_y_primary"][3] == pytest.approx(2.5, rel=1e-6)
