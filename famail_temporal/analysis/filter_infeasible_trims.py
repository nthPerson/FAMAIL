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
# King-move identification (byte-identical semantics to the G4 sweep)
# ---------------------------------------------------------------------------

def king_ok(traj) -> bool:
    """True iff every consecutive step of ``traj`` satisfies king-move
    adjacency ``max(|dx|,|dy|) <= 1``. Uses the RAW state coordinates (exactly
    as the committed G4 adjacency sweep does), so a legacy fractional-offset
    fallback edit (e.g. a +1.7-cell pickup move) is correctly flagged even
    though its int cell would round to a 1-cell step."""
    ss = traj.states
    return all(
        max(abs(b.x_grid - a.x_grid), abs(b.y_grid - a.y_grid)) <= 1
        for a, b in zip(ss, ss[1:])
    )


def find_infeasible_indices(histories: Sequence) -> List[int]:
    """Positions of every history whose MODIFIED trajectory violates king-move
    adjacency — i.e. the trim edits that fell back to the legacy pickup-only
    perturbation. Order preserved."""
    return [i for i, h in enumerate(histories) if not king_ok(h.modified)]


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

    # --- identify violators (exact, hard-asserted) ---
    viol = find_infeasible_indices(histories)
    if len(viol) != n_expected:
        raise SystemExit(
            f"[filter] ABORT: identified {len(viol)} king-move violators but "
            f"metrics.json reports n_taper_infeasible_trim={n_expected}. "
            f"Identification must be EXACT (not heuristic); refusing to write "
            f"a filtered dir from a mismatched violator set."
        )
    if any(i >= n_trim_src for i in viol):
        bad = [i for i in viol if i >= n_trim_src]
        raise SystemExit(
            f"[filter] ABORT: {len(bad)} violator(s) fall in the lift block "
            f"(index >= n_trim={n_trim_src}): {bad[:10]}. Lift already skips "
            f"on infeasible, so every king-move violator must be a trim "
            f"fallback."
        )
    viol_set = set(viol)
    viol_ids = [str(histories[i].original.trajectory_id) for i in viol]
    print(f"[filter] {len(viol)} infeasible-trim violators (all in trim block) "
          f"-> reverting to originals", flush=True)

    filtered = [h for i, h in enumerate(histories) if i not in viol_set]
    n_trim_new = n_trim_src - len(viol)
    assert len(filtered) == n_trim_new + n_lift_src, (
        len(filtered), n_trim_new, n_lift_src,
    )

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
