import numpy as np
import pytest

from famail_temporal.baselines import run_external_fairness as rx
from famail_temporal.baselines import external_fairness_io as io


def _synthetic_arrays(n_per_region=20):
    # 4 regions with distinct housing/comp/migrant profiles
    housing, comp, migrant, Yb = [], [], [], []
    profiles = [(1.0, 1.0, 0.8), (2.0, 2.0, 0.6),
                (3.0, 3.0, 0.4), (4.0, 4.0, 0.2)]
    base_Y = [1.0, 2.0, 3.0, 4.0]                 # poor regions under-served
    for (h, c, m), y in zip(profiles, base_Y):
        housing += [h] * n_per_region
        comp += [c] * n_per_region
        migrant += [m] * n_per_region
        Yb += [y] * n_per_region
    demo = {"AvgHousingPricePerSqM": np.array(housing),
            "CompPerCapita": np.array(comp),
            "MigrantRatio": np.array(migrant)}
    Yb = np.array(Yb)
    Ya = Yb + np.where(Yb < 2.5, 1.0, 0.0)        # lift under-served regions
    return Yb, Ya, demo


def test_assemble_results_schema_and_improvement():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=100)
    assert set(res["metrics"].keys()) == set(io.EQUITY_AXES)
    for axis in io.EQUITY_AXES:
        for g in rx.GROUPINGS:
            entry = res["metrics"][axis][g]
            assert "demographic_parity" in entry
            # lifting under-served regions reduces the parity gap magnitude
            dp = entry["demographic_parity"]
            assert abs(dp["after"]) <= abs(dp["before"]) + 1e-9
            # disparate impact moves toward 1
            di = entry["disparate_impact"]
            assert di["after"] >= di["before"] - 1e-9
    assert "delta" in res["theil"]
    assert res["theil"]["after"] <= res["theil"]["before"] + 1e-9


def test_assemble_results_ci_present():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=50)
    entry = res["metrics"]["MigrantRatio"]["district_extremes"]
    lo, hi = entry["demographic_parity"]["delta_ci"]
    assert lo <= entry["demographic_parity"]["delta"] <= hi
