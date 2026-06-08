"""fit_and_evaluate train_trajectories param + pickups exposure."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_level


def test_train_trajectories_controls_generation_count():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 20)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    subset = bundle.trajectories[:10]
    out = model_level.fit_and_evaluate(
        bundle, train_trajectories=subset,
        mle_epochs=2, adv_epochs=0, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == len(subset)
    assert set(out["corpus"]) == {"f_spatial", "f_causal", "gini_dsr", "gini_asr"}


def test_default_train_trajectories_is_full_corpus_and_exposes_pickups():
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 12)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=2, adv_epochs=0, max_len=8,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == len(bundle.trajectories)
    # pickups exposed for downstream metric work (transmission, DI, etc.)
    assert "pickups" in out
    assert len(out["pickups"]) == out["n_generated"]
    assert all(len(p) == 3 for p in out["pickups"])  # (x, y, t_block)
