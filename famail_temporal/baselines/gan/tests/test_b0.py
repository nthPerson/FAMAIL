"""End-to-end B0 on a tiny synthetic bundle."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import b0


def test_run_b0_returns_generated_and_corpus_fairness():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = b0.run_b0(
        bundle, epochs=3, max_len=8, device=torch.device("cpu"), seed=0,
    )
    assert set(out) == {"generated", "corpus", "n_generated"}
    for key in ("generated", "corpus"):
        m = out[key]
        assert set(m) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
        assert 0.0 <= m["f_causal"] <= 1.0
    assert out["n_generated"] == len(bundle.trajectories)
