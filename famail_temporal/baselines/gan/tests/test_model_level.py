"""End-to-end MLE -> adversarial -> generate -> grid -> fairness on a tiny bundle."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_level


def test_fit_and_evaluate_returns_fairness_and_histories():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=2, adv_epochs=2, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert set(out) == {
        "generated", "corpus", "n_generated", "mle_losses", "adv_losses",
    }
    for key in ("generated", "corpus"):
        m = out[key]
        assert set(m) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}
        assert 0.0 <= m["f_causal"] <= 1.0
    assert out["n_generated"] == len(bundle.trajectories)
    assert len(out["mle_losses"]) == 2
    assert set(out["adv_losses"]) == {"g_losses", "d_losses"}
    assert len(out["adv_losses"]["g_losses"]) == 2
    assert len(out["adv_losses"]["d_losses"]) == 2
