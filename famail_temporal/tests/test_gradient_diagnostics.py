"""Tests for Tier A gradient decomposition in TrajectoryModifier."""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationResult
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _first_active_traj(bundle):
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    x, y, t_block = int(ix_x[0]), int(ix_y[0]), int(ix_t[0])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    return Trajectory(
        trajectory_id=0, driver_id=1,
        states=[
            TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
            TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0),
        ],
    )


def test_modification_result_has_diagnostic_fields_when_enabled():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=2, diagnostics_enabled=True,
    )
    hist = modifier.modify_single(_first_active_traj(bundle))
    for r in hist.iterations:
        assert r.grad_spatial_norm is not None
        assert r.grad_causal_norm is not None
        assert r.grad_fidelity_norm is not None
        assert r.grad_cosine_spatial_causal is not None
        assert r.dominant_term in {"spatial", "causal", "fidelity", None}
        assert isinstance(r.sign_flipped, bool)


def test_modification_result_has_none_diagnostics_when_disabled():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=2, diagnostics_enabled=False,
    )
    hist = modifier.modify_single(_first_active_traj(bundle))
    for r in hist.iterations:
        assert r.grad_spatial_norm is None
        assert r.grad_causal_norm is None
        assert r.grad_fidelity_norm is None
        assert r.grad_cosine_spatial_causal is None
        assert r.dominant_term is None


def test_decomposed_gradients_produce_same_first_step():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    obj = FAMAILObjective(bundle, alpha_spatial=0.4, alpha_causal=0.4, alpha_fidelity=0.0)
    modifier_diag = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=True,
    )
    modifier_plain = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=False,
    )
    traj = _first_active_traj(bundle)
    h_diag = modifier_diag.modify_single(traj)
    h_plain = modifier_plain.modify_single(traj)
    delta_diag = h_diag.iterations[0].cumulative_delta
    delta_plain = h_plain.iterations[0].cumulative_delta
    assert np.allclose(delta_diag, delta_plain, atol=1e-5)


def test_no_diagnostics_path_preserves_final_trajectory():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    traj = _first_active_traj(bundle)
    mod_a = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, diagnostics_enabled=True,
    )
    mod_b = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, diagnostics_enabled=False,
    )
    h_a = mod_a.modify_single(traj)
    h_b = mod_b.modify_single(traj)
    assert h_a.modified.pickup_cell == h_b.modified.pickup_cell


def test_diagnostics_default_from_config():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=3)
    obj = FAMAILObjective(bundle, alpha_spatial=0.5, alpha_causal=0.5, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=1)
    assert modifier.diagnostics_enabled is True


def test_dominant_term_none_when_all_gradients_zero():
    """At convergence or in degenerate configurations, dominant_term should
    be None rather than silently picking via dict-insertion-order tiebreak."""
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=17)
    # All three alphas zero → all gradients zero; dominant term undefined.
    obj = FAMAILObjective(bundle, alpha_spatial=0.0, alpha_causal=0.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=True,
    )
    traj = _first_active_traj(bundle)
    hist = modifier.modify_single(traj)
    # At least one iteration should have a None dominant_term since all
    # weighted norms are below the 1e-8 threshold.
    assert hist.iterations[0].dominant_term is None


def test_zero_alpha_spatial_skips_spatial_backward():
    """alpha_spatial=0 should skip the spatial backward and report zero norm."""
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=19)
    obj = FAMAILObjective(bundle, alpha_spatial=0.0, alpha_causal=1.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=True,
    )
    traj = _first_active_traj(bundle)
    hist = modifier.modify_single(traj)
    assert hist.iterations[0].grad_spatial_norm == 0.0


def test_zero_alpha_causal_skips_causal_backward():
    """alpha_causal=0 should skip the causal backward and report zero norm."""
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=21)
    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=1, diagnostics_enabled=True,
    )
    traj = _first_active_traj(bundle)
    hist = modifier.modify_single(traj)
    assert hist.iterations[0].grad_causal_norm == 0.0
