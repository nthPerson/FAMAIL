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
    find_infeasible_indices,
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
    assert find_infeasible_indices(hs) == [1, 3]


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
    """The king-move violator count on the real run equals the metrics.json
    n_taper_infeasible_trim (the exact-identification invariant the tool asserts)."""
    with open(_REAL_DIR / "histories.pkl", "rb") as f:  # trusted repo artifact
        histories = pickle.load(f)
    metrics = json.loads((_REAL_DIR / "metrics.json").read_text())
    viol = find_infeasible_indices(histories)
    assert len(viol) == int(metrics["n_taper_infeasible_trim"])
    assert all(i < int(metrics["n_trim"]) for i in viol)
