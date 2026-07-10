"""CLI-level test: run_baseline on a synthetic bundle via monkeypatching."""
import json
import pickle

import numpy as np
import pytest

import famail_temporal.baselines.run_stifgsm_baseline as rb
from famail_temporal.algorithm.modifier import ModificationHistory
from famail_temporal.baselines.tests.test_stifgsm_baseline import (
    StubDisc, _profiles, _traj,
)


class _StubBundle:
    def __init__(self, trajs):
        self.trajectories = trajs
        self.pickup_3d = np.ones((48, 90, 24), dtype=np.float32)


def test_run_baseline_end_to_end(tmp_path, monkeypatch):
    trajs = [_traj(1), _traj(2, n_states=3)]
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    with open(seed_dir / "histories.pkl", "wb") as f:
        pickle.dump([ModificationHistory(original=t, modified=t) for t in trajs], f)

    monkeypatch.setattr(rb, "_load_bundle", lambda: _StubBundle(trajs))
    monkeypatch.setattr(rb, "_load_disc", lambda device: StubDisc())
    monkeypatch.setattr(rb, "_driver_profiles", lambda bundle: _profiles())
    monkeypatch.setattr(
        rb, "_rescore",
        lambda bundle, arm_dir: {"f_spatial_before": 0.1, "f_spatial_after": 0.1,
                                 "f_causal_before": 0.8, "f_causal_after": 0.8},
    )

    arm_dir = rb.run_baseline(rb.parse_args([
        "--edit-dir", str(seed_dir), "--mode", "random",
        "--out-root", str(tmp_path), "--seed", "0", "--device", "cpu",
    ]))
    meta = json.loads((arm_dir / "metrics.json").read_text())
    assert meta["arm"]["mode"] == "random" and meta["arm"]["n_edited"] == 2
    assert "fairness" in meta and "f_causal_before" in meta["fairness"]
    with open(arm_dir / "histories.pkl", "rb") as f:
        assert len(pickle.load(f)) == 2
