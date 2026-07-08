import numpy as np
import pytest

from famail_temporal.baselines import run_external_fairness as rx
from famail_temporal.baselines import external_fairness_io as io
from famail_temporal import config as fm_config


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


import json


def test_write_json_and_markdown(tmp_path):
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=30)
    meta = {"dataset": "shenzhen-primary", "edit_dir": "x", "seed": 0, "B": 30}
    path = rx.write_json(res, tmp_path, meta)
    loaded = json.loads(path.read_text())
    assert loaded["meta"]["dataset"] == "shenzhen-primary"
    assert "theil" in loaded

    md = rx.render_markdown(res, meta)
    assert "Demographic parity" in md
    assert "Disparate impact" in md
    assert "Theil" in md
    assert "| Before | After | Delta |" in md or "Before" in md
    assert "Supply/demand ratio (disadvantaged)" in md
    assert "(advantaged)" in md


def test_combined_table():
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=20)
    md = rx.render_combined_table([("shenzhen", res), ("sf", res)])
    assert "shenzhen" in md and "sf" in md


def test_delta_supply_flag_applies_to_after_side_only(tmp_path, monkeypatch):
    import pickle

    from famail_temporal.tests.test_objective import _make_synthetic_bundle

    bundle = _make_synthetic_bundle()
    monkeypatch.setattr(rx.DataBundle, "load", staticmethod(lambda *a, **kw: bundle))

    edit_dir = tmp_path / "edit"
    edit_dir.mkdir()
    with open(edit_dir / "histories.pkl", "wb") as f:
        pickle.dump([], f)  # no relocations: after_pickup == bundle.pickup_3d

    delta_supply_3d = np.zeros_like(bundle.active_taxis_3d)
    delta_supply_3d[bundle.mask_3d] = 3.0
    delta_path = edit_dir / "delta_supply_3d.npz"
    np.savez_compressed(delta_path, delta_supply_3d=delta_supply_3d)

    calls = []
    orig_service_ratio_Y = io.service_ratio_Y

    def spy(pickup_3d, b, supply_3d=None):
        calls.append(supply_3d)
        return orig_service_ratio_Y(pickup_3d, b, supply_3d=supply_3d)

    monkeypatch.setattr(io, "service_ratio_Y", spy)
    # per_unit_demographics reads the real (48, 90) cell_demographics.pkl via
    # _enriched_selected_grid(), which doesn't match the synthetic bundle's
    # small grid; stub it out since this test only cares about the
    # before/after supply_3d wiring, not the demographics values.
    n_active = int(bundle.mask_3d.sum())
    monkeypatch.setattr(
        io, "per_unit_demographics",
        lambda b, selected_grid=None: {a: np.ones(n_active) for a in io.EQUITY_AXES},
    )

    out_dir = tmp_path / "out"
    rx._run_one(edit_dir, "test-dataset", out_dir, seed=0, B=10,
                delta_supply_path=delta_path)

    assert len(calls) == 2
    before_supply, after_supply = calls
    assert before_supply is None  # BEFORE side untouched by the flag
    expected_after = np.clip(
        bundle.active_taxis_3d + delta_supply_3d, fm_config.SUPPLY_FLOOR, None,
    )
    np.testing.assert_allclose(after_supply, expected_after)


def test_write_figure_creates_png(tmp_path):
    Yb, Ya, demo = _synthetic_arrays()
    res = rx.assemble_results(Yb, Ya, demo, seed=0, B=30)
    meta = {"dataset": "shenzhen-primary"}
    path = rx.write_figure(res, tmp_path, meta)
    assert path.exists()
    assert path.suffix == ".png"
    assert path.stat().st_size > 0
