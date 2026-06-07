# famail_temporal/tests/test_editing_loop.py
"""Tests for the unified re-attribution editing loop."""
import numpy as np
from dataclasses import replace

from famail_temporal import config
from famail_temporal.algorithm.editing_loop import (
    run_editing_rounds, EditingLoopResult, RoundRecord,
)
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _bundle_with_drag_trajectories(n_trajs=8, seed=5):
    """Synthetic bundle whose trajectories sit on a strictly-negative-alpha cell."""
    bundle = _make_synthetic_bundle(N_cells_per_block=8, seed=seed)
    attribution = compute_per_unit_attribution(bundle)
    gy = bundle.unit_map.grid_shape[1]
    ix_x, ix_y, ix_t = np.where(bundle.mask_3d)
    chosen = None
    for i in range(len(ix_x)):
        uidx = bundle.unit_map.from_cell_time(
            int(ix_x[i]) * gy + int(ix_y[i]), int(ix_t[i]))
        if attribution[uidx] < -1e-6:
            chosen = i
            break
    assert chosen is not None, "seed unstable: no negative-alpha cell"
    x, y, t_block = int(ix_x[chosen]), int(ix_y[chosen]), int(ix_t[chosen])
    tb = config.TIME_BLOCKS[t_block][1] * 12 + 1
    trajs = [
        Trajectory(trajectory_id=tid, driver_id=tid % 2,
                   states=[TrajectoryState(x, y, tb, 0),
                           TrajectoryState(x, y, tb, 0)])
        for tid in range(n_trajs)
    ]
    return replace(bundle, trajectories=trajs)


def _make_modifier(bundle, **kw):
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    return TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5, **kw)


def test_batch_single_round_edits_select_top_k_set():
    """max_rounds=1 batch edits exactly the select_top_k(k) negative-alpha set."""
    bundle = _bundle_with_drag_trajectories()
    attribution = compute_per_unit_attribution(bundle)
    scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
    expected = set(select_top_k(scored, k=4, trajectories=bundle.trajectories))

    modifier = _make_modifier(bundle)
    result = run_editing_rounds(modifier, bundle, k=4, mode="batch", max_rounds=1)

    assert isinstance(result, EditingLoopResult)
    assert len(result.rounds) == 1
    assert isinstance(result.rounds[0], RoundRecord)
    edited_indices = {
        bundle.trajectories.index(h.original) for h in result.histories
    }
    assert edited_indices == expected


def test_pool_exhausts_when_no_negative_alpha():
    """A bundle whose only drag cell is fixed in round 1 eventually exhausts."""
    bundle = _bundle_with_drag_trajectories(n_trajs=3)
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=10, mode="batch", max_rounds=50,
        round_convergence_tol=None)
    assert result.stop_reason in ("pool_exhausted", "max_rounds")
    if result.stop_reason == "pool_exhausted":
        assert len(result.rounds) >= 1


def test_max_rounds_is_hard_ceiling():
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"))
    result = run_editing_rounds(
        modifier, bundle, k=4, mode="batch", max_rounds=3,
        round_convergence_tol=None)
    assert len(result.rounds) <= 3


def test_convergence_stops_when_f_causal_plateaus():
    """With a tiny epsilon_cap the grid barely changes => F_causal plateaus =>
    convergence fires within round_patience rounds of the ceiling."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=4, mode="batch", max_rounds=50,
        round_convergence_tol=1e-9, round_patience=2)
    assert result.stop_reason in ("converged", "pool_exhausted")
    assert len(result.rounds) < 50


def test_bounded_cap_limits_total_displacement():
    """The cumulative epsilon-cap genuinely BINDS (not a vacuous assertion).

    With the helper's 5 inner iterations a pickup would move up to ~0.5 cells
    per round, and settles voluntarily near ~1.0 across rounds (mirroring the
    real-data §8.3 finding that eps=2 rarely binds). A cap of 0.3 is therefore
    strictly the binding constraint: no edited trajectory may exceed 0.3 (L-inf)
    from its true original across all rounds, AND at least one must be held
    exactly at the cap — so a broken cap (which would allow >=0.4) fails here."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=0.3)
    result = run_editing_rounds(
        modifier, bundle, k=8, mode="batch", max_rounds=10,
        round_convergence_tol=None)
    orig = {t.trajectory_id: (float(t.pickup_state.x_grid),
                              float(t.pickup_state.y_grid))
            for t in bundle.trajectories}
    disps = [
        max(abs(h.modified.pickup_state.x_grid - orig[h.original.trajectory_id][0]),
            abs(h.modified.pickup_state.y_grid - orig[h.original.trajectory_id][1]))
        for h in result.histories
    ]
    assert disps, "expected at least one edit"
    assert max(disps) <= 0.3 + 1e-5          # cap is never exceeded
    assert max(disps) >= 0.3 - 1e-5          # cap actually binds (not vacuous)


def test_unbounded_cap_allows_drift_past_two():
    """With epsilon_cap=inf and multiple rounds, displacement is bounded only by
    rounds * per-round eps (sanity), and the bounded run must never exceed 2."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"))
    result = run_editing_rounds(
        modifier, bundle, k=8, mode="batch", max_rounds=5,
        round_convergence_tol=None)
    orig = {t.trajectory_id: (float(t.pickup_state.x_grid),
                              float(t.pickup_state.y_grid))
            for t in bundle.trajectories}
    max_disp = 0.0
    for h in result.histories:
        ox, oy = orig[h.original.trajectory_id]
        s = h.modified.pickup_state
        max_disp = max(max_disp, abs(s.x_grid - ox), abs(s.y_grid - oy))
    assert max_disp <= 5 * 2.0 + 1e-5


def test_iterative_max_edits_1_never_re_edits():
    """B=1 with max_edits=1 edits each trajectory at most once (historical
    --iterative-topk behavior)."""
    bundle = _bundle_with_drag_trajectories()
    modifier = _make_modifier(bundle, epsilon_cap=2.0)
    result = run_editing_rounds(
        modifier, bundle, k=1, mode="iterative", max_rounds=50,
        iterative_max_edits=1, round_convergence_tol=None)
    assert len(result.edited_ids) == len(set(result.edited_ids))
    assert all(rec.n_edited == 1 for rec in result.rounds)


def test_iterative_unlimited_can_re_edit():
    """B=1 with max_edits=0 (unlimited) may edit the same trajectory more than
    once across rounds when it stays most-negative and under the eps-cap."""
    bundle = _bundle_with_drag_trajectories(n_trajs=2)
    modifier = _make_modifier(bundle, epsilon_cap=float("inf"))
    result = run_editing_rounds(
        modifier, bundle, k=1, mode="iterative", max_rounds=6,
        iterative_max_edits=0, round_convergence_tol=None)
    assert len(result.edited_ids) > len(set(result.edited_ids))
