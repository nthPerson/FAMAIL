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
