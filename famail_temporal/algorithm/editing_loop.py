# famail_temporal/algorithm/editing_loop.py
"""Unified re-attribution editing loop.

One engine for the whole family of editing schedules. Each ROUND re-attributes
against the live (post-edit) grid, selects the eligible negative-alpha set, edits
it via TrajectoryModifier.modify_single, and checks the stop rule:

- mode="batch":     edit ALL eligible negative-alpha trajectories each round
                    (against the round-start attribution); re-attribute between
                    rounds. max_rounds=1 reproduces the historical single pass.
- mode="iterative": edit the single most-negative eligible trajectory each round
                    (re-attribute every edit). The B=1 granularity.

Eligibility: alpha < 0 AND cumulative L-inf displacement from the true original
cell < epsilon_cap AND (iterative) edit-count < iterative_max_edits (0=unlimited).

F_causal of any grid == attribution.sum(); we reuse the round attribution for the
round curve and the convergence test (no extra compute).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import ModificationHistory, TrajectoryModifier
from famail_temporal.data.loader import DataBundle


@dataclass(frozen=True)
class RoundRecord:
    round_index: int          # 1-based
    n_edited: int             # edits applied this round
    f_causal: float           # global F_causal AFTER this round's edits
    delta_f_causal: float     # f_causal(this) - f_causal(previous round / baseline)
    pool_size: int            # eligible negative-alpha count at round start


@dataclass(frozen=True)
class EditingLoopResult:
    histories: List[ModificationHistory]   # one per edit (re-edits repeat the id)
    rounds: List[RoundRecord]
    stop_reason: str                       # "max_rounds"|"converged"|"pool_exhausted"
    edited_ids: List[object]               # trajectory ids edited (may repeat)
    edit_scores: List[float]               # selection-time alpha per edit (aligned w/ histories)


def _cum_disp(modified, ox: float, oy: float) -> float:
    """L-inf displacement of a modified trajectory's pickup from (ox, oy)."""
    s = modified.pickup_state
    return max(abs(float(s.x_grid) - ox), abs(float(s.y_grid) - oy))


def run_editing_rounds(
    modifier: TrajectoryModifier,
    bundle: DataBundle,
    *,
    k: int,
    mode: str = "batch",
    max_rounds: int = 1,
    round_convergence_tol: Optional[float] = None,
    round_patience: int = 2,
    iterative_max_edits: int = 1,
    max_per_unit: Optional[int] = None,
    max_per_cell: Optional[int] = None,
    on_iter: Optional[Callable[[int, object], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> EditingLoopResult:
    log = log or (lambda _msg: None)
    eps_cap = modifier.epsilon_cap

    current_trajs = list(bundle.trajectories)
    orig_pos = {t.trajectory_id: (float(t.pickup_state.x_grid),
                                  float(t.pickup_state.y_grid))
                for t in bundle.trajectories}
    cum_disp = {t.trajectory_id: 0.0 for t in bundle.trajectories}
    edit_count = {t.trajectory_id: 0 for t in bundle.trajectories}

    histories: List[ModificationHistory] = []
    rounds: List[RoundRecord] = []
    edited_ids: List[object] = []
    edit_scores: List[float] = []

    attribution = compute_per_unit_attribution(
        bundle, pickup_3d=modifier.current_pickup_3d())
    prev_fc = float(attribution.sum())
    best_fc = prev_fc
    rounds_since_improve = 0
    stop_reason = "max_rounds"

    for r in range(1, max_rounds + 1):
        scored = rank_trajectories(current_trajs, attribution, bundle.unit_map)
        # Eligibility filter (eps-cap + iterative edit-cap), preserving order.
        eligible = []
        for idx, sc in scored:
            if sc >= 0:
                break  # ascending; no more strictly-negative candidates
            tid = current_trajs[idx].trajectory_id
            if (eps_cap is not None and np.isfinite(eps_cap)
                    and cum_disp[tid] >= eps_cap - 1e-9):
                continue
            if (mode == "iterative" and iterative_max_edits > 0
                    and edit_count[tid] >= iterative_max_edits):
                continue
            eligible.append((idx, sc))

        if not eligible:
            stop_reason = "pool_exhausted"
            break

        pool_size = len(eligible)
        n_pick = k if mode == "batch" else 1
        selected = select_top_k(
            eligible, k=n_pick, trajectories=current_trajs,
            max_per_unit=max_per_unit, max_per_cell=max_per_cell,
        )
        score_by_idx = dict(eligible)  # selection-time alpha per candidate

        for idx in selected:
            traj = current_trajs[idx]
            tid = traj.trajectory_id
            h = modifier.modify_single(
                traj, on_iteration=on_iter, original_cell=orig_pos[tid])
            histories.append(h)
            edited_ids.append(tid)
            edit_scores.append(float(score_by_idx[idx]))
            current_trajs[idx] = h.modified
            edit_count[tid] += 1
            cum_disp[tid] = _cum_disp(h.modified, *orig_pos[tid])

        # Re-attribute against the post-edit grid: this is both the next round's
        # selection attribution AND this round's "after" F_causal.
        attribution = compute_per_unit_attribution(
            bundle, pickup_3d=modifier.current_pickup_3d())
        fc = float(attribution.sum())
        rounds.append(RoundRecord(
            round_index=r, n_edited=len(selected), f_causal=fc,
            delta_f_causal=fc - prev_fc, pool_size=pool_size))
        log(f"round {r}/{max_rounds}: edited={len(selected)} "
            f"F_causal={fc:.6f} (delta {fc - prev_fc:+.3e}) pool={pool_size}")
        prev_fc = fc

        if round_convergence_tol is not None:
            if fc > best_fc + round_convergence_tol:
                best_fc = fc
                rounds_since_improve = 0
            else:
                rounds_since_improve += 1
                if rounds_since_improve >= round_patience:
                    stop_reason = "converged"
                    break

    return EditingLoopResult(
        histories=histories, rounds=rounds,
        stop_reason=stop_reason, edited_ids=edited_ids, edit_scores=edit_scores)
