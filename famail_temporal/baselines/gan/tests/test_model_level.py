"""End-to-end MLE -> adversarial -> generate -> grid -> fairness on a tiny bundle."""
import torch

from famail_temporal.tests.test_objective import _make_synthetic_bundle
from famail_temporal.baselines.tests._helpers import active_units, make_traj_at
from famail_temporal.baselines.gan import model_level
from famail_temporal.baselines.gan.sequences import trajectory_to_tokens
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


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


def test_fit_and_evaluate_excludes_overlong_trajectories():
    """Trajectories whose token sequence exceeds max_tokens are dropped from
    training and generation (the memory-bounding fix for the long length tail)."""
    bundle = _make_synthetic_bundle()
    units = active_units(bundle, 6)
    bundle.trajectories.extend(
        make_traj_at(cx, cy, tb, traj_id=i) for i, (cx, cy, tb) in enumerate(units)
    )
    # One overlong trajectory (30 states -> 32 tokens), excluded at max_tokens=20.
    long_states = [
        TrajectoryState(x_grid=2.0, y_grid=3.0, time_bucket=97, day_index=1)
        for _ in range(30)
    ]
    bundle.trajectories.append(
        Trajectory(trajectory_id=999, driver_id=0, states=long_states)
    )
    expected = sum(
        1 for t in bundle.trajectories if len(trajectory_to_tokens(t)) <= 20
    )
    out = model_level.fit_and_evaluate(
        bundle, mle_epochs=1, adv_epochs=1, max_len=8, max_tokens=20,
        device=torch.device("cpu"), seed=0,
    )
    assert out["n_generated"] == expected
    assert out["n_generated"] < len(bundle.trajectories)   # the long one dropped
