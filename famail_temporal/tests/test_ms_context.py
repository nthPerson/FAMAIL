"""Tests for fidelity.context MultiStreamContextBuilder."""
import numpy as np
import torch

from famail_temporal.fidelity.context import (
    MultiStreamContextBuilder,
    MultiStreamData,
)
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _synthetic_ms_data(n_drivers=3, n_trajs=10, seq_len=20):
    driving, seeking, profile = {}, {}, {}
    seeking_days, driving_days = {}, {}
    for d in range(n_drivers):
        driving[d] = [
            [[float(i + 1), float(i + 2), i, 1] for i in range(seq_len)]
            for _ in range(n_trajs)
        ]
        seeking[d] = [
            [[float(i + 1), float(i + 2), i, 1] for i in range(seq_len)]
            for _ in range(n_trajs)
        ]
        profile[d] = np.zeros(11, dtype=np.float32)
        seeking_days[d] = [1] * n_trajs
        driving_days[d] = [1] * n_trajs
    return MultiStreamData(
        driving_trajs=driving, seeking_trajs=seeking,
        profile_features=profile,
        seeking_days=seeking_days, driving_days=driving_days,
    )


def _make_trajectory(driver_id, seq_len=20):
    states = [
        TrajectoryState(
            x_grid=float(i), y_grid=float(i + 1),
            time_bucket=50 + i, day_index=1,
        )
        for i in range(seq_len)
    ]
    return Trajectory(trajectory_id=0, driver_id=driver_id, states=states)


def test_builder_returns_x1_x2():
    ms = _synthetic_ms_data()
    builder = MultiStreamContextBuilder(ms, device="cpu", seed=42)
    t_orig = _make_trajectory(driver_id=0)
    t_mod = _make_trajectory(driver_id=0)
    kw = builder.build_fidelity_kwargs(t_orig, t_mod)
    assert "x1" in kw
    assert "x2" in kw
    assert kw["x1"].shape == kw["x2"].shape


def test_same_driver_branch_invariant():
    ms = _synthetic_ms_data()
    builder = MultiStreamContextBuilder(ms, device="cpu", seed=42)
    t_orig = _make_trajectory(driver_id=0)
    t_mod = _make_trajectory(driver_id=0)
    kw = builder.build_fidelity_kwargs(t_orig, t_mod)
    if "driving_1" in kw and "driving_2" in kw:
        assert torch.equal(kw["driving_1"], kw["driving_2"])
    if "profile_1" in kw and "profile_2" in kw:
        assert torch.equal(kw["profile_1"], kw["profile_2"])
