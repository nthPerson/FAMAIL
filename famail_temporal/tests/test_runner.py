"""Tests for evaluation.runner.run_experiment (synthetic, fast)."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.runner import (
    ExperimentResult, run_experiment, _parse_override_value, _apply_config_overrides,
)


@pytest.fixture
def tiny_bundle(monkeypatch):
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=0)
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    x, y, t_block = int(ix_x[0]), int(ix_y[0]), int(ix_t[0])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    trajs = []
    for tid in range(6):
        trajs.append(Trajectory(
            trajectory_id=tid, driver_id=tid % 2,
            states=[
                TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
                TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
            ],
        ))
    from dataclasses import replace
    bundle = replace(bundle, trajectories=trajs)
    monkeypatch.setattr(
        "famail_temporal.evaluation.runner._load_bundle",
        lambda **kwargs: bundle,
    )
    return bundle


def test_parse_override_value_tries_int_then_float_then_str():
    assert _parse_override_value("42") == 42
    assert _parse_override_value("1.5") == 1.5
    assert _parse_override_value("hello") == "hello"


def test_apply_config_overrides_raises_on_unknown_key():
    with pytest.raises(KeyError, match="NOT_A_REAL_KEY"):
        _apply_config_overrides({"NOT_A_REAL_KEY": 1})


def test_apply_config_overrides_restores_on_exit():
    original = config.EPSILON_BALL
    restore_fn = _apply_config_overrides({"EPSILON_BALL": 9.9})
    assert config.EPSILON_BALL == 9.9
    restore_fn()
    assert config.EPSILON_BALL == original


def test_run_experiment_returns_result_dataclass(tiny_bundle):
    result = run_experiment(k=2, max_trajectories=6)
    assert isinstance(result, ExperimentResult)
    assert result.grid_before.shape == (*tiny_bundle.pickup_3d.shape[:2], config.T, 4)
    assert result.grid_after.shape == result.grid_before.shape
    assert len(result.histories) <= 2
    assert set(result.augmented_trajs_before.keys()) == {0, 1}
    assert set(result.augmented_trajs_after.keys())  == {0, 1}


def test_run_experiment_overrides_restore(tiny_bundle):
    original = config.MAX_ITERATIONS
    _ = run_experiment(
        k=2, max_trajectories=6,
        config_overrides={"MAX_ITERATIONS": 2},
    )
    assert config.MAX_ITERATIONS == original


def test_run_experiment_unknown_override_raises(tiny_bundle):
    with pytest.raises(KeyError):
        run_experiment(k=2, config_overrides={"NOT_REAL": 7})


def test_run_experiment_no_diagnostics_disables_history_grad_norms(tiny_bundle):
    """When diagnostics_enabled=False, per-iteration gradient-decomposition
    fields on ModificationHistory iterations should be None. (Tier C sensitivity
    grids are checked in a separate Phase 9 test.)"""
    result = run_experiment(k=2, max_trajectories=6, diagnostics_enabled=False)
    for hist in result.histories:
        for r in hist.iterations:
            assert r.grad_spatial_norm is None
            assert r.grad_causal_norm is None
            assert r.grad_fidelity_norm is None
            assert r.dominant_term is None


def test_experiment_id_format_with_name(tiny_bundle):
    result = run_experiment(k=2, max_trajectories=6, name="my-run")
    assert "my-run" in result.experiment_id
    assert result.experiment_id.startswith("2")


def test_run_experiment_records_effective_alphas_when_identity_discriminator(tiny_bundle):
    """When nn.Identity is the discriminator stub, alpha_fidelity is silently
    forced to 0.0 at objective-construction time. metrics.json must record the
    effective value so researchers can't mistake a stub run for a real one."""
    result = run_experiment(k=2, max_trajectories=6)
    # Synthetic bundle has nn.Identity discriminator -> alpha_fidelity forced to 0
    assert result.effective_alpha_fidelity == 0.0
    assert result.effective_alpha_spatial  == config.ALPHA_SPATIAL
    assert result.effective_alpha_causal   == config.ALPHA_CAUSAL


def test_augmented_trajs_after_reflects_modified_pickup_cells(tiny_bundle):
    """The augmented_trajs_after artifact must contain the modified pickup cell
    for top-k trajectories, not the original. A regression that swapped
    h.original for h.modified would silently produce identical before/after."""
    result = run_experiment(k=2, max_trajectories=6)
    if not result.histories:
        pytest.skip("no top-k modifications produced (all-zero attribution)")
    for h in result.histories:
        # Find this trajectory in augmented_trajs_after
        did = h.original.driver_id
        # The original pickup state is the last one in the trajectory
        orig_cell = h.original.pickup_cell
        mod_cell = h.modified.pickup_cell
        # If the modification actually moved the cell, the after-artifact must reflect it
        if orig_cell != mod_cell:
            # Search the after dict for this trajectory_id via last-state coords
            found_modified = False
            for traj_states in result.augmented_trajs_after.get(did, []):
                last = traj_states[-1]
                # Coords are 1-indexed on disk
                if (last[0] - 1, last[1] - 1) == mod_cell:
                    found_modified = True
                    break
            assert found_modified, (
                f"trajectory {h.original.trajectory_id} moved from "
                f"{orig_cell} to {mod_cell} but no trajectory in "
                f"augmented_trajs_after[{did}] has pickup at {mod_cell}"
            )
