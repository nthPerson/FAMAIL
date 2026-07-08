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
    from famail_temporal.algorithm.attribution import compute_per_unit_attribution
    # seed=5 chosen because it produces a synthetic bundle with at least one
    # cell whose αᵢ < 0 (drag cell). Seeds 0-4 happen to give a uniformly
    # above-baseline distribution at this size, which would leave the top-k
    # selector with nothing to pick.
    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=5)
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    # Find a cell with strictly NEGATIVE attribution — under the 1/N-shifted
    # decomposition αᵢ < 0 marks cells dragging fairness below baseline, and
    # those are what select_top_k picks. The RNG state of the synthetic bundle
    # varies with config.T (more active cells at T=24), so "first active cell"
    # is not guaranteed to have negative attribution.
    attribution = compute_per_unit_attribution(bundle)
    gy = bundle.unit_map.grid_shape[1]
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    chosen = None
    for i in range(len(ix_x)):
        if attribution[bundle.unit_map.from_cell_time(
            int(ix_x[i]) * gy + int(ix_y[i]),
            int(ix_t[i]),
        )] < -1e-6:
            chosen = i
            break
    assert chosen is not None, (
        "synthetic bundle has no cells with negative attribution — seed unstable"
    )
    x, y, t_block = int(ix_x[chosen]), int(ix_y[chosen]), int(ix_t[chosen])
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


def test_max_iterations_override_actually_reaches_modifier(tiny_bundle, monkeypatch):
    """Regression test: --override MAX_ITERATIONS=X must be respected by the
    modifier. Before the default-arg fix, the modifier's __init__ captured
    config.MAX_ITERATIONS at module import time and ignored runtime overrides
    from _apply_config_overrides.

    We spy on TrajectoryModifier.__init__ to capture the modifier's
    max_iterations attribute as seen at construction time — that is the point
    where the bug manifests, regardless of whether the synthetic bundle happens
    to converge before hitting the cap."""
    from famail_temporal.algorithm import modifier as modifier_mod
    captured = {}
    orig_init = modifier_mod.TrajectoryModifier.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured["max_iterations"] = self.max_iterations

    monkeypatch.setattr(modifier_mod.TrajectoryModifier, "__init__", spy_init)

    run_experiment(
        k=2, max_trajectories=6,
        config_overrides={"MAX_ITERATIONS": 3},
    )
    assert captured.get("max_iterations") == 3, (
        f"modifier was constructed with max_iterations="
        f"{captured.get('max_iterations')}; override MAX_ITERATIONS=3 "
        f"was not respected at modifier construction time"
    )


def test_alpha_overrides_actually_reach_objective(tiny_bundle):
    """Regression test: --override ALPHA_* must be respected by FAMAILObjective.
    Tests alpha_spatial (alpha_fidelity is special-cased by the Identity-
    discriminator sentinel, so we probe alpha_spatial which has no such branch)."""
    result = run_experiment(
        k=2, max_trajectories=6,
        config_overrides={"ALPHA_SPATIAL": 0.77},
    )
    assert result.effective_alpha_spatial == 0.77, (
        f"expected effective_alpha_spatial=0.77, got "
        f"{result.effective_alpha_spatial}"
    )


def test_run_experiment_multiloop_records_rounds(tiny_bundle):
    """max_rounds>1 runs the engine and records per-round F_causal."""
    result = run_experiment(
        k=4, max_trajectories=None, max_drivers=None,
        max_rounds=3, round_convergence_tol=None, accept_rule="non-regression",
        epsilon_cap=2.0,
    )
    assert hasattr(result, "rounds")
    assert 1 <= len(result.rounds) <= 3
    assert all(hasattr(r, "f_causal") for r in result.rounds)


def test_run_experiment_default_is_single_round(tiny_bundle):
    """No multi-loop args ⇒ exactly one round (historical single pass)."""
    result = run_experiment(k=4)
    assert len(result.rounds) == 1


def test_run_experiment_iterative_topk_one_edit_per_round(tiny_bundle):
    """--iterative-topk maps to B=1 with max_rounds defaulting to k, so it edits
    one trajectory per round (historical behavior), not the whole batch at once."""
    result = run_experiment(k=6, iterative_topk=True)
    assert len(result.modified_trajectory_ids) >= 1
    # B=1: exactly one edit per round ⇒ #rounds == #edits.
    assert len(result.rounds) == len(result.modified_trajectory_ids)
    assert all(r.n_edited == 1 for r in result.rounds)


def test_cli_parses_multiloop_flags():
    from famail_temporal.evaluation.runner import _build_arg_parser
    args = _build_arg_parser().parse_args(
        ["-k", "10", "--max-rounds", "5", "--round-convergence-tol", "1e-4",
         "--round-patience", "2", "--epsilon-cap", "inf",
         "--accept-rule", "non-regression", "--iterative-topk-max-edits", "0"])
    assert args.max_rounds == 5
    assert args.round_convergence_tol == 1e-4
    assert args.epsilon_cap == float("inf")
    assert args.accept_rule == "non-regression"
    assert args.iterative_topk_max_edits == 0


def test_cli_parses_ste_flag():
    from famail_temporal.evaluation.runner import _build_arg_parser
    assert _build_arg_parser().parse_args(["-k", "10", "--ste"]).ste is True
    assert _build_arg_parser().parse_args(["-k", "10"]).ste is False


def test_run_experiment_ste_runs(tiny_bundle):
    result = run_experiment(k=4, use_ste=True)
    assert len(result.rounds) == 1  # default single round
    assert len(result.modified_trajectory_ids) >= 1


# ── Task 8: supply-lift runner/persistence wiring ───────────────────────────


@pytest.fixture
def lift_ready_bundle(monkeypatch):
    """Same construction as ``tiny_bundle`` (N_cells_per_block=8, seed=5 — a
    seed known to produce a strictly-negative-attribution cell), but each
    trajectory carries a real 6-state seeking tail instead of the 2-state
    stub. ``lift_candidates`` always skips trajectories with < 3 states, so
    the plain ``tiny_bundle`` fixture can never produce a lift candidate."""
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    from famail_temporal.algorithm.attribution import compute_per_unit_attribution
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    from dataclasses import replace

    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=5)
    attribution = compute_per_unit_attribution(bundle)
    gy = bundle.unit_map.grid_shape[1]
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    chosen = None
    for i in range(len(ix_x)):
        if attribution[bundle.unit_map.from_cell_time(
            int(ix_x[i]) * gy + int(ix_y[i]),
            int(ix_t[i]),
        )] < -1e-6:
            chosen = i
            break
    assert chosen is not None, (
        "synthetic bundle has no cells with negative attribution — seed unstable"
    )
    x, y, t_block = int(ix_x[chosen]), int(ix_y[chosen]), int(ix_t[chosen])
    start_hour = config.TIME_BLOCKS[t_block][1]
    tb = start_hour * 12 + 1
    trajs = []
    for tid in range(6):
        trajs.append(Trajectory(
            trajectory_id=tid, driver_id=tid % 2,
            states=[
                TrajectoryState(x_grid=x, y_grid=y, time_bucket=tb, day_index=0)
                for _ in range(6)
            ],
        ))
    bundle = replace(bundle, trajectories=trajs)
    monkeypatch.setattr(
        "famail_temporal.evaluation.runner._load_bundle",
        lambda **kwargs: bundle,
    )
    return bundle, (x, y, t_block)


def test_lift_wiring_produces_lift_edits_and_nonzero_delta_supply(
    lift_ready_bundle, monkeypatch,
):
    """A crafted supply gradient that strongly rewards moving a trajectory's
    tail toward one specific active cell must drive the runner's lift step to
    produce n_lift > 0 edits and a nonzero persisted-shape delta_supply_3d,
    end-to-end through the real modifier.modify_single(mode="lift") loop.
    supply_gradient_N is monkeypatched (selection input only) so the test is
    deterministic; the actual lift optimization runs unmocked."""
    import famail_temporal.evaluation.runner as runner_mod

    bundle, (x, y, t_block) = lift_ready_bundle
    n_units = bundle.unit_map.n_units

    # Boost a single active cell at Chebyshev distance 3 from the tail's
    # location: far enough that the OLD tail position's 5x5 lift box (half
    # width 2) never covers it, close enough that some delta within the
    # epsilon-ball (2) brings the NEW box's 5x5 window over it. Guarantees a
    # strictly positive linearized lift score for every trajectory whose tail
    # sits at (x, y, t_block) regardless of the bundle's real gradient.
    gx, gy, _ = bundle.mask_3d.shape
    boost_cell = None
    for dist in (3, 4):
        for ddx in range(-dist, dist + 1):
            for ddy in range(-dist, dist + 1):
                if max(abs(ddx), abs(ddy)) != dist:
                    continue
                cx, cy = x + ddx, y + ddy
                if 0 <= cx < gx and 0 <= cy < gy and bundle.mask_3d[cx, cy, t_block]:
                    boost_cell = (cx, cy)
                    break
            if boost_cell:
                break
        if boost_cell:
            break
    assert boost_cell is not None, "no boostable cell found — adjust bundle params"

    flat_idx = np.full(bundle.mask_3d.shape, -1, dtype=np.int64)
    flat_idx[bundle.mask_3d] = np.arange(n_units)
    grad = np.zeros(n_units, dtype=np.float64)
    grad[flat_idx[boost_cell[0], boost_cell[1], t_block]] = 1000.0

    monkeypatch.setattr(
        runner_mod, "supply_gradient_N", lambda bundle, objective: grad,
    )

    result = run_experiment(k=2, max_trajectories=6, config_overrides={"LIFT_BUDGET": 3})

    assert result.n_lift > 0
    assert result.delta_supply_3d is not None
    assert result.delta_supply_3d.shape == bundle.pickup_3d.shape
    assert np.abs(result.delta_supply_3d).sum() > 0


def test_legacy_mode_end_to_end_byte_identical(tiny_bundle, monkeypatch):
    """TAIL_LEN=0, LIFT_BUDGET=0 must reproduce the pre-supply-lift pipeline
    exactly: run the runner's new code path (run_experiment, which now always
    contains the Task 8 lift-wiring block) against a pinned pre-change call
    sequence built independently — attribution/selection/editing via
    run_editing_rounds -> modifier.modify_single(mode="trim") (the actual
    pre-Task-8 production loop; predates and is untouched by this task) ->
    build_fairness_grid, with no lift step at all. Byte-identical (== on
    floats, not approx) metrics and grids is the G1 claim."""
    monkeypatch.setattr(config, "TAIL_LEN", 0)
    monkeypatch.setattr(config, "LIFT_BUDGET", 0)

    import torch
    import torch.nn as nn
    from famail_temporal.algorithm.editing_loop import run_editing_rounds
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.evaluation.grid import build_fairness_grid
    from famail_temporal.evaluation.runner import _scalar_metrics_from_grid
    from famail_temporal.fidelity.context import MultiStreamContextBuilder

    bundle = tiny_bundle

    # ── "new" code path: the actual runner, forced onto CPU for a
    # deterministic bitwise comparison against the manual reconstruction. ──
    result = run_experiment(k=2, max_trajectories=6, device="cpu")

    # ── pinned pre-change call sequence, built independently (does not call
    # or share any state with run_experiment / the Task 8 lift block). ──
    grid_before_ref = build_fairness_grid(bundle)
    assert isinstance(bundle.discriminator, nn.Identity)
    objective_ref = FAMAILObjective(bundle, alpha_fidelity=0.0)
    ms_builder_ref = MultiStreamContextBuilder(bundle.multi_stream, device="cpu")
    modifier_ref = TrajectoryModifier(
        objective=objective_ref, bundle=bundle,
        multi_stream_builder=ms_builder_ref,
        diagnostics_enabled=False, device=torch.device("cpu"),
    )
    loop_result_ref = run_editing_rounds(
        modifier_ref, bundle,
        k=2, mode="batch", max_rounds=config.MAX_ROUNDS,
        round_convergence_tol=config.ROUND_CONVERGENCE_TOL,
        round_patience=config.ROUND_PATIENCE,
        iterative_max_edits=config.ITERATIVE_TOPK_MAX_EDITS,
        max_per_unit=None, max_per_cell=None, on_iter=None, log=None,
    )
    pickup_after_ref = modifier_ref.current_pickup_3d()
    grid_after_ref = build_fairness_grid(bundle, pickup_3d=pickup_after_ref)
    metrics_before_ref = _scalar_metrics_from_grid(grid_before_ref)
    metrics_after_ref = _scalar_metrics_from_grid(grid_after_ref)

    assert result.f_spatial_before == metrics_before_ref["f_spatial"]
    assert result.f_causal_before  == metrics_before_ref["f_causal"]
    assert result.gini_dsr_before  == metrics_before_ref["gini_dsr"]
    assert result.gini_asr_before  == metrics_before_ref["gini_asr"]
    assert result.f_spatial_after == metrics_after_ref["f_spatial"]
    assert result.f_causal_after  == metrics_after_ref["f_causal"]
    assert result.gini_dsr_after  == metrics_after_ref["gini_dsr"]
    assert result.gini_asr_after  == metrics_after_ref["gini_asr"]
    np.testing.assert_array_equal(result.grid_before, grid_before_ref)
    np.testing.assert_array_equal(result.grid_after, grid_after_ref)

    assert result.delta_supply_3d is not None
    assert result.delta_supply_3d.sum() == 0.0
    assert result.n_lift == 0
    assert len(loop_result_ref.histories) == result.n_trim


def test_persistence_roundtrip_delta_supply_and_counters(tmp_path):
    """delta_supply_3d.npz + the new metrics.json counters must round-trip
    exactly: added/removed supply_totals derived from the array's sign,
    n_trim/n_lift/n_taper_infeasible_* copied through verbatim."""
    from famail_temporal.tests.test_persistence import _fake_result
    from famail_temporal.evaluation.persistence import write
    from dataclasses import replace
    import json

    ds = np.zeros((4, 4, 2), dtype=np.float64)
    ds[0, 0, 0] = 0.5
    ds[1, 1, 1] = 0.25
    ds[2, 2, 0] = -0.2
    result = replace(
        _fake_result(),
        delta_supply_3d=ds,
        n_trim=3, n_lift=2,
        n_taper_infeasible_trim=1, n_taper_infeasible_lift=0,
    )
    out_dir = write(result, output_root=tmp_path)

    npz_path = out_dir / "delta_supply_3d.npz"
    assert npz_path.exists()
    loaded = np.load(npz_path)
    assert set(loaded.files) == {"delta_supply_3d"}
    np.testing.assert_array_equal(loaded["delta_supply_3d"], ds)

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["n_trim"] == 3
    assert metrics["n_lift"] == 2
    assert metrics["n_taper_infeasible_trim"] == 1
    assert metrics["n_taper_infeasible_lift"] == 0
    assert metrics["supply_totals"]["added"] == pytest.approx(0.75)
    assert metrics["supply_totals"]["removed"] == pytest.approx(0.2)
    assert metrics["artifact_paths"]["delta_supply_3d"] == "delta_supply_3d.npz"


def test_legacy_mode_does_not_enable_flush_denormal(tiny_bundle, monkeypatch):
    """G1 hygiene: the flush-denormal FP-environment change is guarded on
    TAIL_LEN > 0 — a TAIL_LEN=0 (legacy) invocation must never call
    torch.set_flush_denormal, so published-number reproduction runs in an
    untouched FP environment. The spy returns True without calling through,
    so the process's real FP state is never mutated by this test."""
    import torch
    calls = []
    monkeypatch.setattr(
        torch, "set_flush_denormal", lambda enabled: (calls.append(enabled), True)[1],
    )
    monkeypatch.setattr(config, "TAIL_LEN", 0)
    monkeypatch.setattr(config, "LIFT_BUDGET", 0)
    run_experiment(k=2, max_trajectories=6)
    assert calls == []


def test_trim_only_taper_mode_enables_flush_denormal_once(tiny_bundle, monkeypatch):
    """A trim-only taper ablation (TAIL_LEN=4, LIFT_BUDGET=0 — a config
    Task 11 actually runs) must still enable flush-denormal exactly once:
    the denormal-poisoning mechanism is the TRIM persist chain, which runs
    at full k with lift disabled."""
    import torch
    calls = []
    monkeypatch.setattr(
        torch, "set_flush_denormal", lambda enabled: (calls.append(enabled), True)[1],
    )
    monkeypatch.setattr(config, "LIFT_BUDGET", 0)
    result = run_experiment(k=2, max_trajectories=6)
    assert calls == [True]
    assert result.n_lift == 0  # lift stayed disabled


def test_lift_enabled_path_enables_flush_denormal_once(tiny_bundle, monkeypatch):
    """With the production defaults (TAIL_LEN=4, LIFT_BUDGET=None) the runner
    must enable flush-denormal exactly once (stall-hardening for subnormal
    float32 residuals in the demand-grid persist chains)."""
    import torch
    calls = []
    monkeypatch.setattr(
        torch, "set_flush_denormal", lambda enabled: (calls.append(enabled), True)[1],
    )
    run_experiment(k=2, max_trajectories=6)
    assert calls == [True]


def test_supply_lift_after_metrics_tolerate_float32_demand_residuals(
    tiny_bundle, monkeypatch,
):
    """Production incident #3: after all editing, the shared float32 demand
    grid can carry ~-1e-9 residuals at fully drained cells (persist -=mass
    chains vs the aggregation-time division; established at -1.86e-9 for a
    67-pickup cell), and build_fairness_grid's strict negativity check then
    rejects the whole after-grid. On the taper path (TAIL_LEN>0) the runner
    must sanitize the fetched grid so metrics_after computes; pre-fix this
    test dies with the production 'must not contain negatives' ValueError.
    The legacy path (TAIL_LEN=0) stays byte-identical-untouched — covered by
    test_legacy_mode_end_to_end_byte_identical."""
    from famail_temporal.algorithm.modifier import TrajectoryModifier

    orig = TrajectoryModifier.current_pickup_3d

    def drifted(self):
        g = orig(self)
        ix, iy, it = np.argwhere(self.bundle.mask_3d)[0]
        g[ix, iy, it] = np.float32(-1.86e-9)  # the verified drained-cell residual
        return g

    # Patching globally is safe: the in-loop consumer (compute_per_unit_
    # attribution via editing_loop) clamps demand at DEMAND_FLOOR and
    # tolerates negatives; only the after-metrics fairness grid rejects them.
    monkeypatch.setattr(TrajectoryModifier, "current_pickup_3d", drifted)

    result = run_experiment(k=2, max_trajectories=6)  # defaults: TAIL_LEN=4

    assert np.isfinite(result.f_spatial_after)
    active = ~np.isnan(result.grid_after[..., 0])
    assert active.any()  # the after-grid was actually computed


def test_persistence_skips_delta_supply_artifact_when_absent(tmp_path):
    """Legacy-shaped ExperimentResult objects (delta_supply_3d left at its
    default None — e.g. anything constructed before this task) must not grow
    a new file or new metrics.json keys, so old callers/fixtures are
    unaffected."""
    from famail_temporal.tests.test_persistence import _fake_result
    from famail_temporal.evaluation.persistence import write
    import json

    result = _fake_result()
    assert result.delta_supply_3d is None
    out_dir = write(result, output_root=tmp_path)

    assert not (out_dir / "delta_supply_3d.npz").exists()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert "supply_totals" not in metrics
    assert "n_trim" not in metrics
    assert "delta_supply_3d" not in metrics["artifact_paths"]
