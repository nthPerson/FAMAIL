"""Tests for the skip-on-infeasible trim post-process (Task 11a).

Fast unit tests use tiny synthetic trajectories. One (opt-in / auto-skip)
integration test is the LOAD-BEARING correctness evidence: rebuilding ΔS from
ALL histories of the real validation run must reproduce the persisted
``delta_supply_3d.npz`` bit-for-bit.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pytest

from famail_temporal.algorithm.supply import hard_delta_supply, state_presence_mass
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.analysis.filter_infeasible_trims import (
    king_ok,
    find_edit_introduced_indices,
    find_fallback_indices,
    recovered_delta_int,
    reconstruct_delta_supply_3d,
)


# --- minimal synthetic trajectory/history doubles ---------------------------

@dataclass
class _State:
    x_grid: float
    y_grid: float
    time_bucket: int = 1
    day_index: int = 0


@dataclass
class _Traj:
    states: List[_State]
    trajectory_id: str = "t"


@dataclass
class _Hist:
    original: _Traj
    modified: _Traj


def _mass_for(tb_source_bucket: int, n_hours_per_block, n_days) -> float:
    tb = hour_to_block_index(time_bucket_to_hour(tb_source_bucket))
    return state_presence_mass(n_hours_per_block, n_days, tb)


# --- king_ok / find_infeasible_indices --------------------------------------

def test_king_ok_true_for_single_cell_steps():
    t = _Traj([_State(0, 0), _State(1, 1), _State(1, 2), _State(0, 2)])
    assert king_ok(t) is True


def test_king_ok_false_for_two_cell_step():
    t = _Traj([_State(0, 0), _State(2, 0)])
    assert king_ok(t) is False


def test_king_ok_false_for_fractional_offset():
    # Legacy pickup-only fallback: raw step 1.7 violates even though int() -> 1.
    t = _Traj([_State(14.0, 39.0), _State(15.4, 37.0)])
    assert king_ok(t) is False


def test_find_infeasible_indices_picks_only_violators():
    hs = [
        _Hist(_Traj([_State(0, 0), _State(1, 1)]), _Traj([_State(0, 0), _State(1, 1)])),   # ok
        _Hist(_Traj([_State(0, 0), _State(0, 0)]), _Traj([_State(0, 0), _State(3, 0)])),   # violator
        _Hist(_Traj([_State(5, 5), _State(5, 6)]), _Traj([_State(5, 5), _State(6, 6)])),   # ok
        _Hist(_Traj([_State(2, 2), _State(2, 2)]), _Traj([_State(2, 2), _State(2.0, 4.9)])),  # violator
    ]
    assert find_edit_introduced_indices(hs) == [1, 3]


# --- edit-introduced identification (city-robust: SF raw data has ~15%
# --- baseline king-move violations from GPS gaps; those must NOT be flagged) --

def test_preexisting_violation_not_flagged():
    # SF-style raw GPS gap: the ORIGINAL already violates (0,0)->(5,0).
    # Untouched (modified identical) -> not an edit-introduced violation.
    orig = _Traj([_State(0, 0), _State(5, 0), _State(5, 1)])
    mod = _Traj([_State(0, 0), _State(5, 0), _State(5, 1)])
    assert find_edit_introduced_indices([_Hist(orig, mod)]) == []


def test_edit_introduced_violation_flagged():
    # Original fully compliant; the edit makes step 1 violating.
    orig = _Traj([_State(0, 0), _State(1, 0), _State(1, 1)])
    mod = _Traj([_State(0, 0), _State(1, 0), _State(4, 1)])
    assert find_edit_introduced_indices([_Hist(orig, mod)]) == [0]


def test_mixed_preexisting_plus_new_violation_flagged():
    # Original violates at step 0 (raw GPS gap); the edit KEEPS that violation
    # and introduces a NEW one at step 1 (was compliant) -> flagged.
    orig = _Traj([_State(0, 0), _State(5, 0), _State(5, 1)])
    mod = _Traj([_State(0, 0), _State(5, 0), _State(9, 1)])
    assert find_edit_introduced_indices([_Hist(orig, mod)]) == [0]


def test_preexisting_violation_with_compliant_edit_not_flagged():
    # Original violates at step 0; the edit only changes step 1 and keeps it
    # king-compliant -> no NEW violation -> not flagged.
    orig = _Traj([_State(0, 0), _State(5, 0), _State(5, 1)])
    mod = _Traj([_State(0, 0), _State(5, 0), _State(6, 1)])
    assert find_edit_introduced_indices([_Hist(orig, mod)]) == []


def test_preexisting_violation_changed_but_still_at_same_index_not_flagged():
    # The step-0 transition was ALREADY violating in the original; the edit
    # changes it to a different (still violating) jump at the same index —
    # not a NEW violation under the per-index edit-introduced rule.
    orig = _Traj([_State(0, 0), _State(5, 0)])
    mod = _Traj([_State(0, 0), _State(6, 0)])
    assert find_edit_introduced_indices([_Hist(orig, mod)]) == []


def test_length_mismatch_raises():
    orig = _Traj([_State(0, 0), _State(1, 0), _State(1, 1)])
    mod = _Traj([_State(0, 0), _State(1, 0)])
    with pytest.raises(ValueError):
        find_edit_introduced_indices([_Hist(orig, mod)])


def test_compliance_summary_counts():
    from famail_temporal.analysis.filter_infeasible_trims import compliance_summary
    hs = [
        # compliant everywhere
        _Hist(_Traj([_State(0, 0), _State(1, 1)]), _Traj([_State(0, 0), _State(1, 1)])),
        # pre-existing violation, untouched (absolute-noncompliant, edit-clean)
        _Hist(_Traj([_State(0, 0), _State(5, 0)]), _Traj([_State(0, 0), _State(5, 0)])),
        # edit-introduced violation
        _Hist(_Traj([_State(0, 0), _State(1, 0)]), _Traj([_State(0, 0), _State(4, 0)])),
    ]
    s = compliance_summary(hs)
    assert s["n"] == 3
    assert s["n_original_king_compliant"] == 2
    assert s["n_modified_king_compliant"] == 1
    assert s["n_edits_introducing_violations"] == 1
    assert s["edit_relative_compliance_frac"] == pytest.approx(2.0 / 3.0)


# --- replay identification (find_fallback_indices) ---------------------------
# Requires real Trajectory objects (apply_tail_perturbation), not the doubles.

def _real_traj(cells, tid="t"):
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    return Trajectory(
        trajectory_id=tid, driver_id=0,
        states=[TrajectoryState(float(x), float(y), 1, 0) for x, y in cells],
    )


def _replayable_hist(orig_cells, mod_cells, tid="t"):
    return _Hist(_real_traj(orig_cells, tid), _real_traj(mod_cells, tid))


def test_replay_flags_infeasible_repair():
    # n=2, compliant original step, pickup moved 2 cells: no tail to absorb
    # the move (anchor fixed at offset 0), so |1+2| > 1 -> repair infeasible
    # -> the modifier MUST have used the legacy fallback -> flagged.
    h = _replayable_hist([(0, 0), (1, 0)], [(0, 0), (3, 0)])
    assert find_fallback_indices([h], tail_len=4, grid_dims=(10, 10)) == [0]


def test_replay_not_flagged_when_repair_feasible():
    # 5-state stationary trajectory (steps of 0 = slack to absorb the
    # translation), pickup moved 2 cells: repair feasible -> modifier used
    # the repair, not the fallback -> not flagged. (Modified = the repaired
    # trajectory itself.)
    orig = _real_traj([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
    rep = orig.apply_tail_perturbation(np.array([2.0, 0.0]), 4, (10, 10))
    assert rep is not None
    h = _Hist(orig, rep)
    assert find_fallback_indices([h], tail_len=4, grid_dims=(10, 10)) == []


def test_replay_flags_taut_chain_even_when_tail_exists():
    # A chain of (+1) steps is taut (every step already at the king max), so
    # translating the pickup +2 is infeasible at ANY tail depth -> fallback.
    orig = _real_traj([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
    assert orig.apply_tail_perturbation(np.array([2.0, 0.0]), 4, (10, 10)) is None
    h = _replayable_hist(
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (6.0, 0)],  # legacy pickup-only move
    )
    assert find_fallback_indices([h], tail_len=4, grid_dims=(10, 10)) == [0]


def test_replay_flags_fallback_that_introduced_no_new_violation():
    # SF missed-case: the ORIGINAL's only step is already violating (GPS gap
    # (0,0)->(5,0)), so ANY repair is infeasible (l_eff==0 requires absolute
    # compliance) -> fallback, even though the legacy move (pickup 5.0->4.6,
    # int cell 5->4) alters an ALREADY-violating step and introduces no NEW
    # violation. Replay must flag it; the per-index rule must not.
    h = _replayable_hist([(0, 0), (5, 0)], [(0, 0), (4.6, 0)])
    assert find_fallback_indices([h], tail_len=4, grid_dims=(10, 10)) == [0]
    assert find_edit_introduced_indices([h]) == []


def test_replay_zero_delta_on_violating_step_is_still_fallback():
    # Unchanged trajectory whose only step violates: _discretize_trim always
    # runs apply_tail_perturbation (even with delta 0) and counts the None ->
    # replay reproduces that accounting exactly.
    h = _replayable_hist([(0, 0), (5, 0)], [(0, 0), (5, 0)])
    assert find_fallback_indices([h], tail_len=4, grid_dims=(10, 10)) == [0]


def test_recovered_delta_int_matches_both_branches():
    # Fallback branch: fractional legacy pickup int-truncates to legacy cell.
    h_fb = _replayable_hist([(0, 0), (1, 0)], [(0, 0), (3.4, 0.0)])
    assert recovered_delta_int(h_fb) == (2, 0)
    # Repair branch: integer offsets preserve the original cell arithmetic.
    orig = _real_traj([(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)])
    rep = orig.apply_tail_perturbation(np.array([2.0, 0.0]), 4, (10, 10))
    assert rep is not None
    assert recovered_delta_int(_Hist(orig, rep)) == (2, 0)


# --- reconstruct_delta_supply_3d --------------------------------------------

def test_reconstruct_unchanged_trajectory_is_zero():
    nh = np.ones(24, dtype=np.int32)
    hs = [_Hist(_Traj([_State(0, 0), _State(1, 1)]), _Traj([_State(0, 0), _State(1, 1)]))]
    out = reconstruct_delta_supply_3d(hs, nh, n_days=3, grid_shape=(10, 10, 24))
    assert np.all(out == 0.0)


def test_reconstruct_matches_direct_hard_delta_supply():
    """Only the rows whose int cell changed contribute; mass uses the ORIGINAL
    state's time block; the result equals a direct hard_delta_supply call."""
    nh = np.ones(24, dtype=np.int32)
    n_days = 3
    shape = (12, 12, 24)
    # rows 0,1 unchanged (skip); row 2 moves (5,5)->(6,6) at bucket 1 (hour0/blk0)
    orig = _Traj([_State(3, 3, 1), _State(3, 3, 1), _State(5, 5, 1)])
    mod = _Traj([_State(3, 3, 1), _State(3, 3, 1), _State(6, 6, 1)])
    out = reconstruct_delta_supply_3d([_Hist(orig, mod)], nh, n_days, shape)

    mass = _mass_for(1, nh, n_days)
    expected = hard_delta_supply([(5, 5)], [(6, 6)], [0], [mass], shape)
    np.testing.assert_allclose(out, expected)


def test_reconstruct_ints_fractional_coords_like_persist_path():
    """Fractional modified coords are int()'d exactly as the persist path does,
    so a (10.0->11.7) move lands as int cell 11, not a rounded/other value."""
    nh = np.ones(24, dtype=np.int32)
    n_days = 2
    shape = (20, 20, 24)
    orig = _Traj([_State(10.0, 10.0, 1), _State(10.0, 10.0, 1)])
    mod = _Traj([_State(10.0, 10.0, 1), _State(11.7, 9.2, 1)])
    out = reconstruct_delta_supply_3d([_Hist(orig, mod)], nh, n_days, shape)

    mass = _mass_for(1, nh, n_days)
    expected = hard_delta_supply([(10, 10)], [(11, 9)], [0], [mass], shape)
    np.testing.assert_allclose(out, expected)


def test_reconstruct_sums_across_trajectories():
    nh = np.ones(24, dtype=np.int32)
    n_days = 4
    shape = (15, 15, 24)
    h1 = _Hist(_Traj([_State(1, 1, 1), _State(2, 2, 1)]),
               _Traj([_State(1, 1, 1), _State(3, 3, 1)]))
    h2 = _Hist(_Traj([_State(8, 8, 1), _State(8, 8, 1)]),
               _Traj([_State(8, 8, 1), _State(9, 9, 1)]))
    out = reconstruct_delta_supply_3d([h1, h2], nh, n_days, shape)

    mass = _mass_for(1, nh, n_days)
    exp = (hard_delta_supply([(2, 2)], [(3, 3)], [0], [mass], shape)
           + hard_delta_supply([(8, 8)], [(9, 9)], [0], [mass], shape))
    np.testing.assert_allclose(out, exp)


# --- LOAD-BEARING integration equivalence (auto-skips without data) ----------

_REAL_DIR = Path(
    "/home/robert/FAMAIL/famail_temporal/results/"
    "2026-07-08T14-03-03_supply_lift_v1_shz_primary"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_REAL_DIR / "delta_supply_3d.npz").exists(),
    reason="real validation-run artifacts not present",
)
def test_delta_supply_reconstruction_equals_persisted_on_real_run():
    """Rebuilding ΔS from ALL real histories must reproduce the persisted
    delta_supply_3d.npz — the correctness evidence that the filtered rebuild
    can be trusted. (Slow: loads the full bundle + 10k histories.)"""
    from famail_temporal.data.loader import DataBundle

    with open(_REAL_DIR / "histories.pkl", "rb") as f:  # trusted repo artifact
        histories = pickle.load(f)
    persisted = np.load(_REAL_DIR / "delta_supply_3d.npz")["delta_supply_3d"]
    bundle = DataBundle.load()
    recon = reconstruct_delta_supply_3d(
        histories, bundle.n_hours_per_block, bundle.n_days, bundle.pickup_3d.shape,
    )
    assert np.allclose(recon, persisted, atol=1e-5, rtol=1e-4)
    assert float(np.max(np.abs(recon - persisted))) < 1e-5


@pytest.mark.skipif(
    not (_REAL_DIR / "metrics.json").exists(),
    reason="real validation-run artifacts not present",
)
def test_real_violator_count_matches_metrics():
    """On the real Shenzhen run, BOTH identifications (replay of the fallback
    decision, and the per-index edit-introduced rule) find exactly the
    metrics.json n_taper_infeasible_trim violators, on identical indices —
    the PRIMARY regression guarantee for the city-robust identification."""
    with open(_REAL_DIR / "histories.pkl", "rb") as f:  # trusted repo artifact
        histories = pickle.load(f)
    metrics = json.loads((_REAL_DIR / "metrics.json").read_text())
    n_trim = int(metrics["n_trim"])
    snap = metrics["config_snapshot"]
    fallback = find_fallback_indices(
        histories[:n_trim], int(snap["TAIL_LEN"]), tuple(snap["GRID_DIMS"]),
    )
    edit_introduced = find_edit_introduced_indices(histories)
    assert len(fallback) == int(metrics["n_taper_infeasible_trim"])
    assert edit_introduced == fallback  # Shenzhen: definitions coincide
    assert all(i < n_trim for i in edit_introduced)
