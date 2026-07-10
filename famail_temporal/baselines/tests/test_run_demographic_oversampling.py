"""CLI-level tests on a synthetic bundle via monkeypatched seams (pattern:
test_run_stifgsm_baseline.py)."""
import json
import pickle
from types import SimpleNamespace

import numpy as np

from famail_temporal.baselines import run_demographic_oversampling as rdo
from famail_temporal.baselines.assemble_baseline_table import _flatten_arm_metrics
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj(cells, traj_id, driver="d0"):
    states = [TrajectoryState(x_grid=float(x), y_grid=float(y),
                              time_bucket=13, day_index=0) for x, y in cells]
    return Trajectory(trajectory_id=traj_id, driver_id=driver, states=states)


def _selected_grid():
    vals = np.arange(6, dtype=np.float64)
    grid = np.zeros((6, 4, 3))
    for j in range(3):
        grid[:, :, j] = vals[:, None]
    return grid


def _stub_bundle():
    from famail_temporal import config as cfg
    from famail_temporal.data.aggregation import block_n_hours
    T = cfg.T
    # 12 trajectories: 4 originate in housing/comp-D rows (0-1), 4 in
    # migrant-D rows (4-5), 4 in neutral rows.
    trajs = (
        [_traj([(0, j % 4), (2, 2)], f"h{j}", driver=f"dh{j}") for j in range(4)]
        + [_traj([(5, j % 4), (3, 3)], f"m{j}", driver=f"dm{j}") for j in range(4)]
        + [_traj([(3, j % 4), (2, 1)], f"n{j}", driver=f"dn{j}") for j in range(4)]
    )
    return SimpleNamespace(
        trajectories=trajs,
        pickup_3d=np.ones((6, 4, T), dtype=np.float32),
        active_taxis_3d=np.ones((6, 4, T), dtype=np.float32),
        n_hours_per_block=np.array([block_n_hours(t) for t in range(T)],
                                   dtype=np.int32),
        n_days=1,
    )


def _patch_seams(monkeypatch, bundle):
    monkeypatch.setattr(rdo, "_load_bundle", lambda: bundle)
    monkeypatch.setattr(rdo, "_selected_grid", lambda: _selected_grid())
    monkeypatch.setattr(
        rdo, "_rescore_fairness",
        lambda bundle, D, S: {
            "f_spatial_before": 0.1, "f_spatial_after": 0.2,
            "f_causal_before": 0.8, "f_causal_after": 0.9,
            "deltas": {"f_spatial": 0.1, "f_causal": 0.1},
        },
    )
    monkeypatch.setattr(
        rdo, "_external",
        lambda bundle, D, S, arm_dir, meta, seed, B: {"stub": True},
    )


def test_run_targeted_writes_arm_contract(tmp_path, monkeypatch):
    bundle = _stub_bundle()
    _patch_seams(monkeypatch, bundle)
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "targeted", "--dose", "6", "--seed", "0",
         "--out-root", str(tmp_path)]))
    assert arm_dir.is_dir()
    assert "demo_oversample_targeted_d6_s0" in arm_dir.name

    # duplicates.pkl round-trips; deliberately NO histories.pkl
    assert not (arm_dir / "histories.pkl").exists()
    with open(arm_dir / "duplicates.pkl", "rb") as f:
        dup = pickle.load(f)
    assert len(dup["specs"]) == len(dup["phantoms"]) == 6
    real_ids = {t.driver_id for t in bundle.trajectories}
    assert all(p.driver_id not in real_ids for p in dup["phantoms"])

    meta = json.loads((arm_dir / "metrics.json").read_text())
    arm = meta["arm"]
    assert arm["mode"] == "oversample-targeted-d6"
    assert arm["variant"] == "targeted" and arm["dose"] == 6 and arm["seed"] == 0
    assert arm["n_edited"] == 6
    assert arm["n_corpus"] == 12
    assert arm["corpus_inflation"] == 6 / 12
    assert set(arm["per_stratum_draws"]) == {
        "AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"}
    assert sum(arm["per_stratum_draws"].values()) == 6
    assert isinstance(arm["adjacency_violation_rate"], float)
    assert "origin_escape_frac" in arm and "pickup_outside_frac" in arm
    assert "n_with_replacement" in arm and "n_clipped_states" in arm
    assert meta["fairness"]["f_causal_after"] == 0.9


def test_arm_metrics_ingest_into_baseline_table(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, _stub_bundle())
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "targeted", "--dose", "6", "--seed", "0",
         "--out-root", str(tmp_path)]))
    flat = _flatten_arm_metrics(json.loads((arm_dir / "metrics.json").read_text()))
    assert flat["label"] == "oversample-targeted-d6"
    assert flat["n"] == 6
    assert flat["f_causal_before"] == 0.8 and flat["f_causal_after"] == 0.9
    assert flat["fidelity_a"] is None          # not scored: by construction


def test_run_placebo_ignores_pools(tmp_path, monkeypatch):
    _patch_seams(monkeypatch, _stub_bundle())
    arm_dir = rdo.run(rdo.parse_args(
        ["--variant", "placebo", "--dose", "5", "--seed", "1",
         "--out-root", str(tmp_path)]))
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert meta["arm"]["mode"] == "oversample-placebo-d5"
    assert meta["arm"]["per_stratum_draws"] == {"placebo": 5}
    assert meta["arm"]["origin_escape_frac"] is None


def test_run_dose_zero_grids_identity(tmp_path, monkeypatch):
    """dose=0 end-to-end: the additive grids the runner hands to the scoring
    seams must equal the bundle's own grids exactly."""
    bundle = _stub_bundle()
    captured = {}

    def _capture_rescore(bundle_, D, S):
        captured["D"], captured["S"] = D, S
        return {"f_spatial_before": 0.0, "f_spatial_after": 0.0,
                "f_causal_before": 0.0, "f_causal_after": 0.0,
                "deltas": {"f_spatial": 0.0, "f_causal": 0.0}}

    monkeypatch.setattr(rdo, "_load_bundle", lambda: bundle)
    monkeypatch.setattr(rdo, "_selected_grid", lambda: _selected_grid())
    monkeypatch.setattr(rdo, "_rescore_fairness", _capture_rescore)
    monkeypatch.setattr(rdo, "_external",
                        lambda *a, **k: {"stub": True})
    rdo.run(rdo.parse_args(["--variant", "targeted", "--dose", "0",
                            "--seed", "0", "--out-root", str(tmp_path)]))
    assert np.array_equal(captured["D"], np.float64(bundle.pickup_3d))
    assert np.array_equal(captured["S"], np.float64(bundle.active_taxis_3d))
