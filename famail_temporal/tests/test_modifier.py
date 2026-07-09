"""Tests for algorithm.modifier TrajectoryModifier."""

import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _make_test_trajectory(driver_id=0, pickup_xy=(3, 4), time_bucket=90):
    """Make a trajectory with pickup at given coords, time_bucket in morning_peak."""
    states = [
        TrajectoryState(x_grid=0.0, y_grid=0.0,
                        time_bucket=time_bucket - 1, day_index=1),
        TrajectoryState(x_grid=float(pickup_xy[0]), y_grid=float(pickup_xy[1]),
                        time_bucket=time_bucket, day_index=1),
    ]
    return Trajectory(trajectory_id=0, driver_id=driver_id, states=states)


def _active_cell_and_bucket(bundle, active_idx=0):
    cell = bundle.unit_map.to_flat_cell(active_idx)
    t_block = bundle.unit_map.to_time_block(active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    return x, y, 1 + (start_hour * 12)


def test_modify_single_returns_history():
    """modify_single returns a ModificationHistory with iterations."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, multi_stream_builder=None,
        max_iterations=3,
    )
    # Pick an active unit
    any_active_idx = 0
    cell = bundle.unit_map.to_flat_cell(any_active_idx)
    t_block = bundle.unit_map.to_time_block(any_active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    tb = 1 + (start_hour * 12)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

    history = modifier.modify_single(traj)
    assert isinstance(history, ModificationHistory)
    assert history.total_iterations <= 3
    assert len(history.iterations) == history.total_iterations


def test_modify_single_respects_epsilon_ball():
    """After modification, pickup is within epsilon-ball of original."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle,
        max_iterations=50,
    )
    any_active_idx = 0
    cell = bundle.unit_map.to_flat_cell(any_active_idx)
    t_block = bundle.unit_map.to_time_block(any_active_idx)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    tb = 1 + (start_hour * 12)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

    history = modifier.modify_single(traj)
    orig = np.array([float(x), float(y)])
    final = np.array([
        history.modified.pickup_state.x_grid,
        history.modified.pickup_state.y_grid,
    ])
    diff = np.abs(final - orig)
    assert (diff <= config.EPSILON_BALL + 1e-5).all(), (
        f"Final pickup {final} strayed {diff} from original {orig}, "
        f"exceeding epsilon={config.EPSILON_BALL}"
    )


def test_current_pickup_3d_reflects_modifications():
    """current_pickup_3d() must return the post-modification pickup tensor as a
    numpy ndarray matching bundle.pickup_3d's shape."""
    import numpy as np
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective

    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    objective = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=objective, bundle=bundle, max_iterations=2,
    )
    before = modifier.current_pickup_3d()
    assert isinstance(before, np.ndarray)
    assert before.shape == bundle.pickup_3d.shape
    assert before.dtype == np.float32
    assert np.allclose(before, bundle.pickup_3d)

    snapshot = before.copy()
    before[0, 0, 0] = 999.0
    assert np.allclose(modifier.current_pickup_3d(), snapshot)


def test_modifier_resolves_config_at_init_not_at_import():
    """Regression: config overrides applied AFTER module import must still be
    picked up when the modifier is constructed without explicit kwargs.
    Previously, default args like `max_iterations: int = config.MAX_ITERATIONS`
    froze the value at import time."""
    from famail_temporal import config
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    objective = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0, alpha_fidelity=0.0)

    original = config.MAX_ITERATIONS
    try:
        config.MAX_ITERATIONS = 7
        modifier = TrajectoryModifier(objective=objective, bundle=bundle)
        assert modifier.max_iterations == 7, (
            f"Expected max_iterations=7 (from config mutation), got "
            f"{modifier.max_iterations} — default arg bug may have returned"
        )
    finally:
        config.MAX_ITERATIONS = original


def test_accept_rule_default_is_objective():
    """Default modifier keeps the historical objective gate."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3)
    assert modifier.accept_rule == "objective"


def test_non_regression_rejects_f_spatial_regression():
    """Under non-regression, an iterate that lifts F_causal but dips F_spatial
    below its iter-0 value is NOT persisted as best; objective rule may accept it.

    We drive this deterministically with a stub objective whose terms we control
    by iteration, so the test does not depend on bundle gradients.
    """
    import torch as _t
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    # Sequence of (f_spatial, f_causal) per iteration. iter0 = baseline.
    # iter1 improves BOTH; iter2 improves f_causal more but regresses f_spatial.
    seq = [(0.10, 0.50), (0.11, 0.55), (0.09, 0.70)]
    calls = {"i": 0}

    def fake_forward(soft_pickup_3d=None, **kw):
        i = min(calls["i"], len(seq) - 1)
        fs, fc = seq[i]
        calls["i"] += 1
        total = _t.tensor(fs + fc, requires_grad=True)
        terms = {
            "f_spatial": _t.tensor(fs),
            "f_causal": _t.tensor(fc),
            "f_fidelity": _t.tensor(0.0),
        }
        return total, terms

    # nn.Module dispatches obj(...) -> self.forward (dunder looked up on the
    # type), so override forward, NOT __call__. diagnostics_enabled=False below
    # selects the single-backward path (the decomposed path needs a real graph).
    obj.forward = fake_forward  # type: ignore[method-assign]

    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=3, patience=None,
        accept_rule="non-regression", diagnostics_enabled=False,
    )
    x, y, tb = _active_cell_and_bucket(bundle)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)
    history = modifier.modify_single(traj)
    # Best iterate must be iter1 (improves both), NOT iter2 (regresses f_spatial).
    assert history.best_iteration == 1


def test_epsilon_cap_default_equals_epsilon_ball():
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3)
    assert modifier.epsilon_cap == config.EPSILON_BALL


def test_epsilon_cap_is_respected_relative_to_original_cell():
    """modify_single keeps the pickup within epsilon_cap (L-inf) of original_cell.
    The cross-round anchor distinction (cap from the TRUE original, not the
    round-start cell) is covered at the engine level in test_editing_loop
    (test_bounded_cap_limits_total_displacement)."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=50, epsilon_cap=1.0,
    )
    x, y, tb = _active_cell_and_bucket(bundle)
    traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)
    history = modifier.modify_single(traj, original_cell=(x, y))
    s = history.modified.pickup_state
    assert abs(s.x_grid - x) <= 1.0 + 1e-5
    assert abs(s.y_grid - y) <= 1.0 + 1e-5


def test_soft_neighborhood_size_override_reaches_soft_assign(monkeypatch):
    """A runtime SOFT_NEIGHBORHOOD_SIZE override must reach SoftCellAssignment.
    Regression: the size was previously frozen by SoftCellAssignment's
    import-time default arg, so `--override SOFT_NEIGHBORHOOD_SIZE` was silently
    ignored. The modifier now resolves it from config at construction time."""
    bundle = _make_synthetic_bundle()
    m_default = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m_default.soft_assign.k == config.SOFT_NEIGHBORHOOD_SIZE // 2
    monkeypatch.setattr(config, "SOFT_NEIGHBORHOOD_SIZE", 11)
    m_wide = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m_wide.soft_assign.k == 5  # 11 // 2


def test_use_ste_default_false():
    bundle = _make_synthetic_bundle()
    m = TrajectoryModifier(
        objective=FAMAILObjective(bundle, alpha_fidelity=0.0), bundle=bundle)
    assert m.use_ste is False


def test_ste_runs_and_gradient_flows():
    """With STE on, modify_single runs end-to-end and the soft gradient still
    flows (some iteration has a nonzero gradient norm)."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=5,
                           use_ste=True)
    x, y, tb = _active_cell_and_bucket(bundle)
    h = m.modify_single(_make_test_trajectory(pickup_xy=(x, y), time_bucket=tb))
    assert isinstance(h, ModificationHistory)
    assert any(it.gradient_norm > 0 for it in h.iterations)


def test_ste_feeds_concentrated_hard_grid():
    """STE hands the objective a grid with the pickup mass concentrated in ONE
    cell (hard); the soft path spreads it over the neighborhood — so the two
    grids differ in more than a single cell of the trajectory's t_block slice."""
    import torch as _t
    bundle = _make_synthetic_bundle()
    x, y, tb = _active_cell_and_bucket(bundle)
    t_block = bundle.unit_map.to_time_block(0)

    def captured_grid(use_ste):
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        grids = []

        def rec(soft_pickup_3d=None, **kw):
            grids.append(soft_pickup_3d.detach().clone())
            return (_t.tensor(1.0, requires_grad=True),
                    {"f_spatial": _t.tensor(0.0), "f_causal": _t.tensor(0.0),
                     "f_fidelity": _t.tensor(0.0)})

        obj.forward = rec  # type: ignore[method-assign]
        # max_iterations=2 so ANNEAL_TEMPERATURE uses TAU_MAX (=1.0) at iter-0,
        # giving a spread soft distribution (TAU_MIN=0.1 would make it near-one-hot,
        # so soft≈hard and n_diff would be 0 or 1).
        m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=2,
                               patience=None, diagnostics_enabled=False,
                               use_ste=use_ste)
        m.modify_single(_make_test_trajectory(pickup_xy=(x, y), time_bucket=tb))
        return grids[0]

    soft_grid = captured_grid(False)
    ste_grid = captured_grid(True)
    n_diff = int((_t.abs(soft_grid[:, :, t_block] - ste_grid[:, :, t_block])
                  > 1e-9).sum())
    assert n_diff > 1


# ── Supply-lift mode (Task 7) ────────────────────────────────────────────
from famail_temporal.data.aggregation import (  # noqa: E402
    hour_to_block_index, time_bucket_to_hour,
)


def _king_ok(traj):
    """King-move adjacency (max(|dx|, |dy|) <= 1 between consecutive states)."""
    return all(
        max(abs(b.x_grid - a.x_grid), abs(b.y_grid - a.y_grid)) <= 1
        for a, b in zip(traj.states, traj.states[1:])
    )


def _interior_active_cell(bundle, lo=2, hi=5):
    """First active (cell, time_bucket) with both grid coords in [lo, hi].

    Keeping the pickup off the grid edge guarantees a +/-EPSILON_BALL move
    stays inside the (small) synthetic accumulator grid, so the demand persist
    (which indexes the (gx, gy)-shaped grid directly) never runs out of bounds.
    """
    gy = bundle.pickup_3d.shape[1]
    for i in range(bundle.unit_map.n_units):
        fc = bundle.unit_map.to_flat_cell(i)
        cx, cy = fc // gy, fc % gy
        if lo <= cx <= hi and lo <= cy <= hi:
            t_block = bundle.unit_map.to_time_block(i)
            _, start_hour, _ = config.TIME_BLOCKS[t_block]
            return cx, cy, 1 + (start_hour * 12)
    raise RuntimeError("no interior active cell in synthetic bundle")


def _stay_trajectory(x, y, tb, n):
    """``n`` stationary states at (x, y) sharing time_bucket ``tb`` (compliant)."""
    return Trajectory(
        trajectory_id=0, driver_id=0,
        states=[TrajectoryState(x_grid=float(x), y_grid=float(y),
                                time_bucket=tb, day_index=1) for _ in range(n)],
    )


def test_trim_mode_pickup_identical_to_legacy_and_demand_grid_unchanged(monkeypatch):
    """Trim optimization is TAIL_LEN-independent, so a TAIL_LEN=4 run (tail
    translation) and a TAIL_LEN=0 run (legacy pickup-only move) must land the
    SAME final pickup cell and produce the SAME demand grid (G1/G3). With
    alpha=1.0 the cumulative delta is integer-valued, so round-clip == int-clip.
    The TAIL_LEN=4 run accumulates hard tier-1 dS iff the tail actually moved;
    the TAIL_LEN=0 run never touches the accumulator."""
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=0)
    x, y, tb = _interior_active_cell(bundle)
    traj = _stay_trajectory(x, y, tb, n=6)

    def run(tail_len):
        monkeypatch.setattr(config, "TAIL_LEN", tail_len)
        obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                              alpha_fidelity=0.0)
        m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=5,
                               alpha=1.0, diagnostics_enabled=False)
        h = m.modify_single(traj.clone(), mode="trim")
        return m, h

    m4, h4 = run(4)
    m0, h0 = run(0)

    assert (int(h4.modified.pickup_state.x_grid),
            int(h4.modified.pickup_state.y_grid)) == \
           (int(h0.modified.pickup_state.x_grid),
            int(h0.modified.pickup_state.y_grid))
    assert np.allclose(m4.current_pickup_3d(), m0.current_pickup_3d())

    tail_moved = any(
        (int(a.x_grid), int(a.y_grid)) != (int(b.x_grid), int(b.y_grid))
        for a, b in zip(traj.states, h4.modified.states)
    )
    ds4 = np.abs(m4.current_delta_supply_3d()).sum()
    if tail_moved:
        assert ds4 > 0
    else:
        assert ds4 == 0
    # TAIL_LEN == 0 is bit-for-bit legacy: accumulator stays exactly zero.
    assert np.abs(m0.current_delta_supply_3d()).sum() == 0


@pytest.mark.parametrize("delta", [
    (1.6, 0.4), (0.4, -0.4), (-0.4, 1.6), (-1.7, -0.3), (-0.3, 2.0),
])
def test_trim_discretization_matches_legacy_cell_at_fractional_deltas(
        monkeypatch, delta):
    """G3 at FRACTIONAL step regimes (production STEP_SIZE_ALPHA=0.1 yields
    fractional cumulative deltas): taper-mode trim must deploy EXACTLY the
    pickup cell legacy deploys. Legacy = int()-truncation of
    apply_perturbation's clipped fractional position (the persist arithmetic);
    round()ing the offset diverges on negative fractional components
    (int(10-0.4)=9 vs 10+round(-0.4)=10) and on positive frac >= 0.5
    (int(10+1.6)=11 vs 10+round(1.6)=12)."""
    monkeypatch.setattr(config, "TAIL_LEN", 4)
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=0)
    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                          alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=1)
    traj = _stay_trajectory(10, 10, tb=90, n=6)  # mid-grid, king-compliant
    d = np.array(delta, dtype=np.float32)

    # Legacy deployed cell: apply_perturbation (fractional, clipped) then the
    # persist step's int() truncation — exactly modifier.py's legacy path.
    legacy = traj.apply_perturbation(d)
    legacy_cell = (int(legacy.pickup_state.x_grid),
                   int(legacy.pickup_state.y_grid))

    out = m._discretize_trim(traj, d)
    assert out is not None
    assert (int(out.pickup_state.x_grid),
            int(out.pickup_state.y_grid)) == legacy_cell
    assert _king_ok(out)


def test_lift_mode_moves_supply_toward_positive_gradient():
    """Lift's endogenous supply channel: with a stubbed objective whose supply
    gradient rewards higher-y units, lift drives the seeking tail up (+y) and
    ends with net-positive dS mass in that region — via a king-move-compliant
    edit."""
    import torch as _t
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=1)
    x, y, tb = _interior_active_cell(bundle, lo=2, hi=4)
    t_block = hour_to_block_index(time_bucket_to_hour(tb))
    gy = bundle.pickup_3d.shape[1]
    reward = _t.tensor(
        [float(bundle.unit_map.to_flat_cell(i) % gy)
         for i in range(bundle.unit_map.n_units)],
        dtype=_t.float32,
    )  # per active-unit y-coordinate — higher y = higher reward

    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                          alpha_fidelity=0.0)

    def fake_forward(soft_pickup_3d=None, delta_supply_N=None, **kw):
        if delta_supply_N is not None:
            total = (delta_supply_N * reward).sum()
        else:
            total = soft_pickup_3d.sum() * 0.0
        terms = {"f_spatial": _t.tensor(0.0), "f_causal": _t.tensor(0.0),
                 "f_fidelity": _t.tensor(0.0)}
        return total, terms

    obj.forward = fake_forward  # type: ignore[method-assign]

    traj = _stay_trajectory(x, y, tb, n=6)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=6,
                           alpha=1.0, diagnostics_enabled=False, patience=None)
    h = m.modify_single(traj, mode="lift")

    assert _king_ok(h.modified)
    assert h.modified.pickup_state.y_grid > y  # tail moved toward higher reward
    ds = m.current_delta_supply_3d()
    # New (+) presence boxes land above the removal zone → net-positive there.
    assert ds[:, y + 3:, t_block].sum() > 0


def test_lift_survives_float32_negative_epsilon_demand():
    """Regression (Task-10 production incident): after thousands of trim
    persists (chains of -= mass / += mass float32 ops on the shared demand
    grid) a cell's value can drift a few ULP below zero in exact-arithmetic
    terms — verified mechanism: a float32 cell aggregating 67 pickups, 66
    moved out by trim persists, minus the lift trajectory's own mass ends at
    -1.86e-9. compute_fspatial's strict (pickup_N < 0) check then raises
    ValueError on the FIRST lift objective call. The lift branch must sanitize
    its LOCAL demand clone (clamp min=0); the trim/legacy tensor ops stay
    byte-identical (G1) so this plants the epsilon and runs LIFT only."""
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=3)
    x, y, tb = _interior_active_cell(bundle)
    t_block = hour_to_block_index(time_bucket_to_hour(tb))
    gy = bundle.pickup_3d.shape[1]

    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                          alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3,
                           alpha=1.0, diagnostics_enabled=False)

    # Plant the drift residual at an active unit in a DIFFERENT time block —
    # guaranteed outside the lift trajectory's injection slice, exactly like a
    # far-away cell the trim phase drained before the lift phase started.
    planted = None
    for i in range(bundle.unit_map.n_units):
        t2 = bundle.unit_map.to_time_block(i)
        if t2 != t_block:
            fc = bundle.unit_map.to_flat_cell(i)
            planted = (fc // gy, fc % gy, t2)
            break
    assert planted is not None
    m._base_pickup_3d[planted[0], planted[1], planted[2]] = -1.86e-9

    traj = _stay_trajectory(x, y, tb, n=6)
    h = m.modify_single(traj, mode="lift")  # pre-fix: ValueError from spatial.py
    assert isinstance(h, ModificationHistory)
    assert len(h.iterations) > 0


def test_lift_tripwire_fires_on_large_negative_residual():
    """Companion to test_lift_survives_float32_negative_epsilon_demand: the
    lift-branch clamp (modifier.py, immediately before ``torch.clamp(base_3d,
    min=0.0)``) is preceded by a tripwire assert that ``base_3d > -1e-5``
    everywhere. The clamp is justified ONLY for ~1e-9-scale float32 ULP
    persist drift (the -1.86e-9 case above must still pass through it
    silently); a residual of -1e-4 is four orders of magnitude larger than
    any observed ULP drift and must instead be treated as an accounting bug
    and raise loudly, not be silently masked by the clamp."""
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=3)
    x, y, tb = _interior_active_cell(bundle)
    t_block = hour_to_block_index(time_bucket_to_hour(tb))
    gy = bundle.pickup_3d.shape[1]

    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                          alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=3,
                           alpha=1.0, diagnostics_enabled=False)

    # Plant a residual four orders of magnitude past the ULP floor, at an
    # active unit in a DIFFERENT time block (same placement convention as
    # the ULP-drift test above).
    planted = None
    for i in range(bundle.unit_map.n_units):
        t2 = bundle.unit_map.to_time_block(i)
        if t2 != t_block:
            fc = bundle.unit_map.to_flat_cell(i)
            planted = (fc // gy, fc % gy, t2)
            break
    assert planted is not None
    m._base_pickup_3d[planted[0], planted[1], planted[2]] = -1e-4

    traj = _stay_trajectory(x, y, tb, n=6)
    with pytest.raises(AssertionError, match="accounting bug"):
        m.modify_single(traj, mode="lift")


def test_lift_skip_on_infeasible_repair_reverts_cleanly(monkeypatch):
    """When tail repair is infeasible (apply_tail_perturbation -> None), lift
    skips the edit entirely: the shared demand grid AND the dS accumulator are
    left exactly unchanged, the skip is counted, and the returned trajectory is
    unmoved. Uses a genuine len-2 trajectory (no tail room) with the real
    objective, so the delta_supply_N gather order is exercised end-to-end."""
    bundle = _make_synthetic_bundle(N_cells_per_block=30, seed=2)
    x, y, tb = _interior_active_cell(bundle)
    traj = Trajectory(
        trajectory_id=0, driver_id=0,
        states=[TrajectoryState(float(x), float(y), tb, 1),
                TrajectoryState(float(x), float(y), tb, 1)],
    )
    obj = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=0.0,
                          alpha_fidelity=0.0)
    m = TrajectoryModifier(objective=obj, bundle=bundle, max_iterations=4,
                           alpha=1.0, diagnostics_enabled=False)
    monkeypatch.setattr(Trajectory, "apply_tail_perturbation",
                        lambda self, *a, **k: None)

    base_before = m.current_pickup_3d()
    ds_before = m.current_delta_supply_3d()
    h = m.modify_single(traj, mode="lift")

    assert np.allclose(m.current_pickup_3d(), base_before)
    assert np.allclose(m.current_delta_supply_3d(), ds_before)
    assert np.abs(m.current_delta_supply_3d()).sum() == 0
    assert m.n_taper_infeasible_lift == 1
    assert (int(h.modified.pickup_state.x_grid),
            int(h.modified.pickup_state.y_grid)) == (x, y)
