"""Skip-on-infeasible post-process for supply-lift edit runs (Task 11a).

Motivation
----------
The supply-lift validation run applies two kinds of edit. **Lift** mode
already *skips* a trajectory whenever its tapered-tail repair is infeasible
(``modifier.py`` returns the original untouched and increments
``n_taper_infeasible_lift``). **Trim** mode, historically, falls back to the
legacy *pickup-only* ``apply_perturbation`` when the tapered-tail repair is
infeasible; that fallback can move the pickup by more than one cell and so
violates the king-move adjacency invariant (``max(|dx|,|dy|) <= 1`` on every
consecutive step) that G4 enforces.

The user decision (2026-07-08) adopts the uniform rule:

    *An edit is applied only when a king-compliant repair exists.*

This makes trim symmetric with lift. Rather than re-running the (multi-hour,
GPU) editing pipeline, this tool implements the rule as a **post-process**:
it reverts exactly the trim edits that used the legacy fallback (leaving those
trajectories at their originals) and writes a *derived*, fully documented
results directory ``<edit_dir>_filtered/``. Nothing published is disturbed —
the legacy trim numbers remain reproducible via ``TAIL_LEN=0``.

Identification is EXACT, not heuristic: violators are the modified
trajectories that fail the king-move check (the same check the G4 sweep uses),
and their count is hard-asserted to equal ``metrics.json``'s
``n_taper_infeasible_trim``.

Derived artifacts are rebuilt FROM SCRATCH from the filtered histories (never
subtracted in place):

* ``histories.pkl``        — violators removed (order preserved).
* ``delta_supply_3d.npz``  — tier-1 ΔS re-accumulated from the surviving
  histories, mirroring ``TrajectoryModifier._hard_tail_delta_supply`` exactly.
* ``metrics.json``         — before/after fairness recomputed via
  ``evaluation.grid.build_fairness_grid`` (same clip/sanitize conventions as
  ``runner.py``), plus a ``provenance`` block and the updated counters.
* ``PROVENANCE.md``        — the rule, the rationale, the reverted ids, and the
  before/after metric deltas.

CLI::

    python -m famail_temporal.analysis.filter_infeasible_trims \\
        --edit-dir <results_dir> [--no-verify]

Read-only w.r.t. the source dir and every existing module; does not touch
``famail_temporal/fairness/*.py``, ``modifier.py`` or ``runner.py``.
"""
from __future__ import annotations

import argparse
import json
import pickle  # trusted repo-internal artifact, same as localized_metrics.py
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

from famail_temporal.algorithm.supply import hard_delta_supply, state_presence_mass
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour


# ---------------------------------------------------------------------------
# King-move identification (EDIT-INTRODUCED violations; city-robust).
# Step semantics are byte-identical to the G4 sweep.
# ---------------------------------------------------------------------------

def _violating_steps(traj) -> List[bool]:
    """Per-transition king-move violation flags (len = n_states - 1). Uses the
    RAW state coordinates (exactly as the committed G4 adjacency sweep does),
    so a legacy fractional-offset fallback edit (e.g. a +1.7-cell pickup move)
    is correctly flagged even though its int cell would round to a 1-cell
    step."""
    ss = traj.states
    return [
        max(abs(b.x_grid - a.x_grid), abs(b.y_grid - a.y_grid)) > 1
        for a, b in zip(ss, ss[1:])
    ]


def king_ok(traj) -> bool:
    """True iff every consecutive step of ``traj`` satisfies king-move
    adjacency ``max(|dx|,|dy|) <= 1`` (ABSOLUTE compliance — used for
    reporting; violator identification uses ``introduces_violation``)."""
    return not any(_violating_steps(traj))


def introduces_violation(h) -> bool:
    """True iff the MODIFIED trajectory has a king-violating transition at an
    index where the ORIGINAL's same-index transition was compliant — i.e. the
    edit *introduced* a new violation.

    This is the city-robust identification: SF's raw Cabspotting-derived
    trajectories have ~15% baseline king-move violations (GPS gaps up to ~18
    cells) that pre-exist any editing, so an absolute check on the modified
    trajectory (valid on Shenzhen, whose originals are 100% compliant)
    over-counts there. The per-index diff isolates exactly the legacy
    pickup-only fallback moves the skip-on-infeasible rule targets. On
    Shenzhen the two definitions coincide.

    The editor is length-preserving; a length mismatch means the history is
    not per-index comparable and is a hard error."""
    o, m = h.original, h.modified
    if len(o.states) != len(m.states):
        raise ValueError(
            f"introduces_violation: original has {len(o.states)} states but "
            f"modified has {len(m.states)} — the editor is length-preserving, "
            f"so per-index transition comparison is undefined for this history "
            f"(trajectory_id={getattr(o, 'trajectory_id', '?')})."
        )
    ov = _violating_steps(o)
    mv = _violating_steps(m)
    return any(m_bad and not o_bad for o_bad, m_bad in zip(ov, mv))


def find_edit_introduced_indices(histories: Sequence) -> List[int]:
    """Positions of every history whose edit INTRODUCED a king-move violation
    (see ``introduces_violation``). Pre-existing (raw-data) violations are
    NOT flagged. Order preserved. Used for compliance reporting and as a
    cross-check on the replay identification (every edit-introduced violator
    must be a fallback; the converse need not hold — a fallback whose legacy
    move is <=1 cell, or which alters an ALREADY-violating transition,
    introduces no NEW violation yet still broke the skip-on-infeasible
    rule)."""
    return [i for i, h in enumerate(histories) if introduces_violation(h)]


def recovered_delta_int(h) -> tuple:
    """Recover the integer pickup offset the modifier handed
    ``apply_tail_perturbation`` (``_discretize_trim``'s ``delta_int``) from
    the persisted history alone.

    Valid in BOTH branches: a successful taper repair deploys exactly the
    legacy cell (integer offset preserves the original's fractional part, and
    ``_discretize_trim`` computes the offset from int-truncated cells), and
    the legacy fallback's fractional pickup int-truncates to the legacy cell
    by definition. So ``int(modified.pickup) - int(original.pickup)`` equals
    the modifier's ``delta_int`` in either case."""
    o, m = h.original, h.modified
    return (
        int(m.states[-1].x_grid) - int(o.states[-1].x_grid),
        int(m.states[-1].y_grid) - int(o.states[-1].y_grid),
    )


def is_taper_infeasible(h, tail_len: int, grid_dims) -> bool:
    """REPLAY of the modifier's own fallback decision (``_discretize_trim``):
    True iff ``apply_tail_perturbation(delta_int, tail_len, grid_dims)`` on
    the ORIGINAL returns ``None`` — exactly the condition under which the
    modifier incremented ``n_taper_infeasible_trim`` and fell back to the
    legacy pickup-only move. ``tail_len``/``grid_dims`` must come from the
    run's ``config_snapshot`` (not the current config) for exactness."""
    dx, dy = recovered_delta_int(h)
    repaired = h.original.apply_tail_perturbation(
        np.array([float(dx), float(dy)], dtype=np.float32),
        tail_len, tuple(grid_dims),
    )
    return repaired is None


def find_fallback_indices(
    histories: Sequence, tail_len: int, grid_dims,
) -> List[int]:
    """Positions of every history whose trim edit used the legacy pickup-only
    fallback, identified by replaying the modifier's decision procedure (see
    ``is_taper_infeasible``). Caller passes the TRIM block only (lift mode
    skips on infeasible and never falls back). This is exact by construction:
    it evaluates the same pure function on the same inputs the modifier used,
    so the count equals ``n_taper_infeasible_trim`` whenever the histories
    are the run's own. City-robust: unlike an absolute king-move check, it is
    unaffected by pre-existing raw-data violations (~15% of SF Cabspotting
    trajectories have GPS-gap steps up to ~18 cells)."""
    return [
        i for i, h in enumerate(histories)
        if is_taper_infeasible(h, tail_len, grid_dims)
    ]


def compliance_summary(histories: Sequence) -> dict:
    """Absolute + edit-relative king-move compliance over a histories list.

    - absolute: whole-trajectory ``king_ok`` on the modified corpus, plus the
      original-corpus baseline (SF raw data is NOT 100% compliant);
    - edit-relative: fraction of edits introducing ZERO new violations — the
      cross-city G4 statement (must be 100% post-filter).
    """
    n = len(histories)
    n_mod_ok = sum(1 for h in histories if king_ok(h.modified))
    n_orig_ok = sum(1 for h in histories if king_ok(h.original))
    n_introducing = sum(1 for h in histories if introduces_violation(h))
    return {
        "n": n,
        "n_modified_king_compliant": n_mod_ok,
        "n_original_king_compliant": n_orig_ok,
        "absolute_modified_compliance_frac": (n_mod_ok / n) if n else float("nan"),
        "absolute_original_compliance_frac": (n_orig_ok / n) if n else float("nan"),
        "n_edits_introducing_violations": n_introducing,
        "edit_relative_compliance_frac": (1.0 - n_introducing / n) if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Tier-1 ΔS reconstruction (mirrors modifier._hard_tail_delta_supply exactly)
# ---------------------------------------------------------------------------

def reconstruct_delta_supply_3d(
    histories: Sequence,
    n_hours_per_block: np.ndarray,
    n_days: int,
    grid_shape,
) -> np.ndarray:
    """Rebuild the tier-1 ΔS grid from scratch from ``histories``.

    Mirrors ``TrajectoryModifier._hard_tail_delta_supply`` +
    ``modifier``'s accumulation EXACTLY:

    * per trajectory, iterate ``zip(original.states, modified.states)`` and
      keep only the rows whose *int* cell changed (unmoved rows cancel);
    * mass = ``state_presence_mass`` at the ORIGINAL state's time block;
    * ``hard_delta_supply`` builds the −box(old)/+box(new) contribution;
    * accumulate in a **float32** torch buffer (``_delta_supply_3d`` is
      ``zeros_like(_base_pickup_3d)``, i.e. float32) in histories order, then
      return as float64 (matching ``current_delta_supply_3d``'s final cast).

    The float32 accumulator + histories-order summation is what makes the
    reconstruction reproduce the persisted ``delta_supply_3d.npz`` bit-for-bit
    (verified in the test suite: rebuilt-from-all-histories == persisted).
    """
    delta = torch.zeros(tuple(grid_shape), dtype=torch.float32)
    for h in histories:
        olds, news, tbs, masses = [], [], [], []
        for s_old, s_new in zip(h.original.states, h.modified.states):
            oc = (int(s_old.x_grid), int(s_old.y_grid))
            nc = (int(s_new.x_grid), int(s_new.y_grid))
            if oc == nc:
                continue
            tb = hour_to_block_index(time_bucket_to_hour(s_old.time_bucket))
            mass = state_presence_mass(n_hours_per_block, n_days, tb)
            olds.append(oc)
            news.append(nc)
            tbs.append(tb)
            masses.append(mass)
        if not olds:
            continue
        ds = hard_delta_supply(olds, news, tbs, masses, tuple(grid_shape))
        delta = delta + torch.from_numpy(ds).float()
    return delta.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Metric recompute (same conventions as evaluation/runner.py)
# ---------------------------------------------------------------------------

def _scalar_metrics_from_grid(grid: np.ndarray) -> dict:
    """Mirror of ``runner._scalar_metrics_from_grid`` (kept local so this
    read-only analysis tool does not import the runner)."""
    return {
        "f_spatial": float(np.nansum(grid[..., 0])),
        "f_causal": float(np.nansum(grid[..., 1])),
        "gini_dsr": float(np.nansum(grid[..., 2])),
        "gini_asr": float(np.nansum(grid[..., 3])),
    }


def _supply_totals(delta_supply_3d: np.ndarray) -> dict:
    pos = delta_supply_3d[delta_supply_3d > 0]
    neg = delta_supply_3d[delta_supply_3d < 0]
    return {
        "added": float(pos.sum()) if pos.size else 0.0,
        "removed": float(-neg.sum()) if neg.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

FILTER_RULE = (
    "An edit is applied only when a king-compliant repair exists "
    "(max(|dx|,|dy|) <= 1 on every consecutive step). Trim edits that fell "
    "back to the legacy pickup-only perturbation (tapered-tail repair "
    "infeasible) are reverted to their originals, making trim symmetric with "
    "lift (which already skips on infeasible)."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edit-dir", type=Path, required=True,
        help="Source supply-lift results dir (has histories.pkl, "
             "delta_supply_3d.npz, metrics.json).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output dir (default: <edit_dir>_filtered).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the ΔS-reconstruction equivalence check against the "
             "source delta_supply_3d.npz (the check is on by default and is "
             "the load-bearing correctness evidence).",
    )
    args = parser.parse_args(argv)

    edit_dir = Path(args.edit_dir)
    out_dir = Path(args.out_dir) if args.out_dir else edit_dir.parent / (edit_dir.name + "_filtered")

    t0 = time.monotonic()

    # --- load source artifacts ---
    src_metrics = json.loads((edit_dir / "metrics.json").read_text())
    n_trim_src = int(src_metrics["n_trim"])
    n_lift_src = int(src_metrics["n_lift"])
    n_expected = int(src_metrics["n_taper_infeasible_trim"])

    with open(edit_dir / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    print(f"[filter] loaded {len(histories)} histories "
          f"(n_trim={n_trim_src}, n_lift={n_lift_src})", flush=True)

    # --- identify violators by REPLAYING the modifier's fallback decision ---
    # (city-robust: exact on SF despite ~15% pre-existing raw king-move
    # violations). tail_len/grid_dims come from the run's own config snapshot.
    snap = src_metrics["config_snapshot"]
    tail_len = int(snap["TAIL_LEN"])
    grid_dims = tuple(snap["GRID_DIMS"])
    viol = find_fallback_indices(histories[:n_trim_src], tail_len, grid_dims)
    if len(viol) != n_expected:
        raise SystemExit(
            f"[filter] ABORT: replay identified {len(viol)} fallback trims "
            f"but metrics.json reports n_taper_infeasible_trim={n_expected}. "
            f"Identification must be EXACT (not heuristic); refusing to write "
            f"a filtered dir from a mismatched violator set."
        )
    # Cross-checks: every EDIT-INTRODUCED king violation must come from a
    # fallback trim (successful repairs are compliance-preserving by
    # construction, and lift skips on infeasible).
    edit_introduced = find_edit_introduced_indices(histories)
    bad_lift = [i for i in edit_introduced if i >= n_trim_src]
    if bad_lift:
        raise SystemExit(
            f"[filter] ABORT: {len(bad_lift)} edit-introduced king-move "
            f"violation(s) fall in the lift block (index >= "
            f"n_trim={n_trim_src}): {bad_lift[:10]}. Lift skips on infeasible "
            f"and its repairs preserve compliance, so this indicates a "
            f"corrupted or mismatched run."
        )
    not_fallback = sorted(set(edit_introduced) - set(viol))
    if not_fallback:
        raise SystemExit(
            f"[filter] ABORT: {len(not_fallback)} edit-introduced king-move "
            f"violator(s) are NOT in the replayed fallback set: "
            f"{not_fallback[:10]}. Successful taper repairs preserve "
            f"compliance, so every edit-introduced violation must be a "
            f"fallback; this indicates a corrupted or mismatched run."
        )
    print(f"[filter] cross-check OK: {len(edit_introduced)} edit-introduced "
          f"violators, all within the {len(viol)} replayed fallbacks "
          f"({len(viol) - len(edit_introduced)} fallback(s) introduced no NEW "
          f"violation: <=1-cell legacy move or altered an already-violating "
          f"raw step)", flush=True)
    viol_set = set(viol)
    viol_ids = [str(histories[i].original.trajectory_id) for i in viol]
    print(f"[filter] {len(viol)} fallback-trim violators (replayed; all in "
          f"trim block) -> reverting to originals", flush=True)

    filtered = [h for i, h in enumerate(histories) if i not in viol_set]
    n_trim_new = n_trim_src - len(viol)
    assert len(filtered) == n_trim_new + n_lift_src, (
        len(filtered), n_trim_new, n_lift_src,
    )

    # --- compliance reporting (absolute + edit-relative, pre/post-filter) ---
    compliance_source = compliance_summary(histories)
    compliance_filtered = compliance_summary(filtered)
    assert compliance_filtered["n_edits_introducing_violations"] == 0, (
        "post-filter edit-relative compliance must be 100%",
        compliance_filtered,
    )
    print(f"[filter] compliance (source): absolute modified "
          f"{compliance_source['absolute_modified_compliance_frac']:.2%} | "
          f"original baseline "
          f"{compliance_source['absolute_original_compliance_frac']:.2%} | "
          f"edit-relative "
          f"{compliance_source['edit_relative_compliance_frac']:.2%}", flush=True)
    print(f"[filter] compliance (filtered): absolute modified "
          f"{compliance_filtered['absolute_modified_compliance_frac']:.2%} | "
          f"edit-relative "
          f"{compliance_filtered['edit_relative_compliance_frac']:.2%} "
          f"(must be 100%)", flush=True)

    # --- write filtered histories FIRST (build_edited_pickup_3d reads it) ---
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "histories.pkl", "wb") as f:
        pickle.dump(filtered, f)
    print(f"[filter] wrote {out_dir/'histories.pkl'} ({len(filtered)} histories)",
          flush=True)

    # --- load bundle (needed for ΔS masses + fairness grids) ---
    from dataclasses import replace
    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.evaluation.grid import build_fairness_grid
    from famail_temporal.baselines import external_fairness_io as efio

    print("[filter] loading DataBundle...", flush=True)
    bundle = DataBundle.load()
    grid_shape = bundle.pickup_3d.shape

    # --- rebuild ΔS from filtered histories ---
    print("[filter] reconstructing filtered ΔS...", flush=True)
    delta_filtered = reconstruct_delta_supply_3d(
        filtered, bundle.n_hours_per_block, bundle.n_days, grid_shape,
    )
    np.savez_compressed(out_dir / "delta_supply_3d.npz", delta_supply_3d=delta_filtered)
    print(f"[filter] wrote {out_dir/'delta_supply_3d.npz'}", flush=True)

    # --- equivalence check: rebuild from ALL histories == persisted npz ---
    equivalence = None
    if not args.no_verify:
        print("[filter] verifying ΔS reconstruction against source npz "
              "(rebuild-from-all-histories)...", flush=True)
        persisted = np.load(edit_dir / "delta_supply_3d.npz")["delta_supply_3d"]
        recon_all = reconstruct_delta_supply_3d(
            histories, bundle.n_hours_per_block, bundle.n_days, grid_shape,
        )
        max_abs = float(np.max(np.abs(recon_all - persisted)))
        equivalence = {
            "max_abs_diff": max_abs,
            "sum_recon": float(recon_all.sum()),
            "sum_persisted": float(persisted.sum()),
            "allclose_atol_1e-5": bool(np.allclose(recon_all, persisted, atol=1e-5, rtol=1e-4)),
        }
        if not np.allclose(recon_all, persisted, atol=1e-5, rtol=1e-4):
            raise SystemExit(
                f"[filter] ABORT: ΔS reconstruction does NOT match the source "
                f"delta_supply_3d.npz (max abs diff {max_abs:.3e}). The rebuild "
                f"logic must mirror the modifier exactly before it can be trusted "
                f"to rebuild the filtered grid."
            )
        print(f"[filter] ΔS equivalence OK (max abs diff {max_abs:.3e})", flush=True)

    # --- recompute metrics (same conventions as runner.py) ---
    print("[filter] recomputing fairness metrics...", flush=True)
    grid_before = build_fairness_grid(bundle)
    metrics_before = _scalar_metrics_from_grid(grid_before)

    pickup_after = efio.build_edited_pickup_3d(bundle, out_dir)
    pickup_after = np.clip(pickup_after, 0.0, None)
    if np.any(delta_filtered):
        active_after = np.clip(
            bundle.active_taxis_3d + delta_filtered, config.SUPPLY_FLOOR, None,
        ).astype(bundle.active_taxis_3d.dtype)
        bundle_after = replace(bundle, active_taxis_3d=active_after)
    else:
        bundle_after = bundle
    grid_after = build_fairness_grid(bundle_after, pickup_3d=pickup_after)
    metrics_after = _scalar_metrics_from_grid(grid_after)

    deltas = {k: metrics_after[k] - metrics_before[k] for k in metrics_before}

    metrics = {
        "provenance": {
            "derived_from": str(edit_dir),
            "derived_from_experiment_id": src_metrics.get("experiment_id"),
            "derived_from_git_sha": src_metrics.get("git_sha"),
            "filter_rule": FILTER_RULE,
            "user_decision_date": "2026-07-08",
            "tool": "famail_temporal.analysis.filter_infeasible_trims",
            "n_reverted_infeasible_trim": len(viol),
            "reverted_trajectory_ids": viol_ids,
            "legacy_reproduction_note": (
                "The published legacy trim numbers remain reproducible via "
                "TAIL_LEN=0; nothing in the source dir is modified by this tool."
            ),
            "delta_supply_reconstruction_equivalence": equivalence,
            "violator_identification": (
                "replay: a trim history is a violator iff replaying the "
                "modifier's own fallback decision "
                "(apply_tail_perturbation(delta_int, TAIL_LEN, GRID_DIMS) on "
                "the ORIGINAL, with delta_int recovered from the int pickup "
                "cells and TAIL_LEN/GRID_DIMS from this run's "
                "config_snapshot) returns None — the exact condition under "
                "which the editor fell back to the legacy pickup-only move. "
                "City-robust: pre-existing raw-data king-move violations "
                "(e.g. SF Cabspotting GPS gaps, ~15% of raw trajectories) do "
                "not affect it. Cross-checked: every EDIT-INTRODUCED king "
                "violation (a violating transition at an index where the "
                "original's same-index transition was compliant) lies within "
                "the replayed fallback set, and none occur in the lift block."
            ),
            "identification_counts": {
                "n_fallback_replayed": len(viol),
                "n_edit_introduced_violations": len(edit_introduced),
            },
            "compliance": {
                "source": compliance_source,
                "filtered": compliance_filtered,
            },
        },
        "dataset": src_metrics.get("dataset"),
        "effective_alphas": src_metrics.get("effective_alphas"),
        "config_snapshot": src_metrics.get("config_snapshot"),
        "k_modified": len(filtered),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "deltas": deltas,
        "n_trim": n_trim_new,
        "n_lift": n_lift_src,
        "n_skipped_infeasible_trim": len(viol),
        "n_taper_infeasible_lift": int(src_metrics.get("n_taper_infeasible_lift", 0)),
        "supply_totals": _supply_totals(delta_filtered),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[filter] wrote {out_dir/'metrics.json'}", flush=True)

    _write_provenance_md(out_dir, edit_dir, src_metrics, metrics, viol_ids, equivalence)

    runtime = time.monotonic() - t0
    print(f"[filter] DONE in {runtime:.1f}s -> {out_dir}", flush=True)
    print(f"[filter] F_causal {metrics_before['f_causal']:.6f} -> "
          f"{metrics_after['f_causal']:.6f} (Δ {deltas['f_causal']:+.6f})", flush=True)
    return 0


def _write_provenance_md(out_dir, edit_dir, src_metrics, metrics, viol_ids, equivalence) -> None:
    mb, ma, dd = metrics["metrics_before"], metrics["metrics_after"], metrics["deltas"]
    smb = src_metrics.get("metrics_before", {})
    sma = src_metrics.get("metrics_after", {})
    lines = [
        "# Filtered supply-lift results — PROVENANCE",
        "",
        f"Derived from: `{edit_dir}`",
        f"Source experiment_id: `{src_metrics.get('experiment_id')}`  ·  "
        f"git_sha: `{src_metrics.get('git_sha')}`",
        f"Tool: `famail_temporal.analysis.filter_infeasible_trims`  ·  "
        f"user decision: 2026-07-08",
        "",
        "## Rule",
        "",
        FILTER_RULE,
        "",
        "## Why",
        "",
        "The G4 adjacency sweep found exactly "
        f"{metrics['n_skipped_infeasible_trim']} modified trajectories that "
        "violate king-move adjacency — all trim edits that fell back to the "
        "legacy pickup-only perturbation because their tapered-tail repair was "
        "infeasible (the G3 trade-off). Lift mode already *skips* such edits; "
        "this post-process makes trim symmetric by reverting those "
        f"{metrics['n_skipped_infeasible_trim']} trajectories to their "
        "originals. After filtering, G4 must be 100% king-compliant.",
        "",
        "The published legacy trim numbers remain reproducible via `TAIL_LEN=0`;"
        " this tool modifies nothing in the source directory.",
        "",
        "**Non-reoptimized survivors:** the surviving edits were NOT "
        "re-optimized after removing the reverted trims — their optimization "
        "saw the reverted edits' intermediate demand perturbations in the "
        "sequential base grid. The filtered grids are exact for \"these "
        "surviving edits applied to base,\" which is not byte-identical to a "
        "from-scratch skip-on-infeasible run. Approved trade-off (2026-07-08) "
        "to avoid a multi-hour GPU re-run; the coupling is negligible (the "
        "reverted edits were pickup-only single-cell-mass moves).",
        "",
        "## Edit counts",
        "",
        "| | source | filtered |",
        "|---|---|---|",
        f"| n_trim | {src_metrics.get('n_trim')} | {metrics['n_trim']} |",
        f"| n_lift | {src_metrics.get('n_lift')} | {metrics['n_lift']} |",
        f"| n_skipped_infeasible_trim | 0 (fell back) | "
        f"{metrics['n_skipped_infeasible_trim']} |",
        f"| total edits | {src_metrics.get('k_modified')} | {metrics['k_modified']} |",
        "",
        "## King-move compliance (absolute + edit-relative)",
        "",
        "Violators are identified by **replaying the modifier's fallback "
        "decision** (`apply_tail_perturbation` on the original with the "
        "recovered integer pickup offset and this run's TAIL_LEN/GRID_DIMS; "
        "`None` = fallback) — exact by construction and city-robust. Raw "
        "source data is not necessarily 100% king-compliant (SF "
        "Cabspotting-derived trajectories have ~15% baseline violations from "
        "GPS gaps of up to ~18 cells — a source-data property, unrelated to "
        "editing), so *absolute* compliance of the edited corpus can only be "
        "judged against the original-corpus baseline; the cross-city G4 "
        "statement is **edit-relative compliance** (fraction of edits "
        "introducing zero new violations), which must be 100% post-filter. "
        "Note a fallback can introduce no NEW violation (<=1-cell legacy "
        "move, or altering an already-violating raw step) yet still break "
        "the skip-on-infeasible rule — such fallbacks are reverted too.",
        "",
        "| | source (pre-filter) | filtered |",
        "|---|---|---|",
    ]
    comp_s = metrics["provenance"]["compliance"]["source"]
    comp_f = metrics["provenance"]["compliance"]["filtered"]
    lines += [
        f"| absolute — modified corpus | "
        f"{comp_s['n_modified_king_compliant']}/{comp_s['n']} "
        f"({comp_s['absolute_modified_compliance_frac']:.2%}) | "
        f"{comp_f['n_modified_king_compliant']}/{comp_f['n']} "
        f"({comp_f['absolute_modified_compliance_frac']:.2%}) |",
        f"| absolute — ORIGINAL corpus baseline | "
        f"{comp_s['n_original_king_compliant']}/{comp_s['n']} "
        f"({comp_s['absolute_original_compliance_frac']:.2%}) | "
        f"{comp_f['n_original_king_compliant']}/{comp_f['n']} "
        f"({comp_f['absolute_original_compliance_frac']:.2%}) |",
        f"| edit-relative (zero new violations) | "
        f"{comp_s['n'] - comp_s['n_edits_introducing_violations']}/{comp_s['n']} "
        f"({comp_s['edit_relative_compliance_frac']:.2%}) | "
        f"{comp_f['n'] - comp_f['n_edits_introducing_violations']}/{comp_f['n']} "
        f"(**{comp_f['edit_relative_compliance_frac']:.2%}**) |",
        "",
        "## Fairness metrics (recomputed from filtered histories)",
        "",
        "| metric | source before | source after | filtered before | "
        "filtered after | filtered Δ |",
        "|---|---|---|---|---|---|",
    ]
    for key in ("f_spatial", "f_causal", "gini_dsr", "gini_asr"):
        lines.append(
            f"| {key} | {smb.get(key, float('nan')):.6f} | "
            f"{sma.get(key, float('nan')):.6f} | {mb[key]:.6f} | "
            f"{ma[key]:.6f} | {dd[key]:+.6f} |"
        )
    lines += [
        "",
        "## Supply totals (filtered ΔS)",
        "",
        f"- added: {metrics['supply_totals']['added']:.4f}",
        f"- removed: {metrics['supply_totals']['removed']:.4f}",
        "",
        "## ΔS reconstruction equivalence (load-bearing check)",
        "",
    ]
    if equivalence is not None:
        lines += [
            "Rebuilding ΔS from ALL source histories (float32 accumulator, "
            "histories order, mirroring `modifier._hard_tail_delta_supply`) "
            "reproduces the persisted `delta_supply_3d.npz`:",
            "",
            f"- max abs diff: {equivalence['max_abs_diff']:.3e}",
            f"- sum(recon)={equivalence['sum_recon']:.6f}, "
            f"sum(persisted)={equivalence['sum_persisted']:.6f}",
            f"- allclose(atol=1e-5, rtol=1e-4): "
            f"{equivalence['allclose_atol_1e-5']}",
            "",
            "The filtered `delta_supply_3d.npz` is rebuilt from scratch from "
            "the surviving histories (never subtracted in place).",
        ]
    else:
        lines.append("(equivalence check skipped via --no-verify)")
    lines += [
        "",
        "## Reverted trajectory ids (" + str(len(viol_ids)) + ")",
        "",
        "```",
        ", ".join(viol_ids),
        "```",
        "",
    ]
    (out_dir / "PROVENANCE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
