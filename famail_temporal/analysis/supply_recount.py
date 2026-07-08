"""Tier-2 distinct-count supply recount tool (gate G2 validator, Task 9).

The production supply grid (``active_taxis_3d``) and the supply-lift editor's
own accounting of a fair-supply gain use two DIFFERENT conventions:

- **Tier 1 (fraction/presence)**: the editor's ``delta_supply_3d`` treats every
  seeking STATE as contributing ``1/12`` of an hourly taxi-presence unit,
  spread over its (clipped) 5x5 box, additively — moving a state's box always
  changes S by that fraction, even if the driver was ALREADY counted at the
  destination cell/hour via some other state (own or another driver's).
- **Tier 2 (distinct-count)**: the REAL grid used everywhere else in the
  pipeline (``famail_temporal/data/source_generation/views/active_taxis.py``)
  counts each *driver* at most ONCE per (5x5 neighborhood cell, hour) — a
  driver present in the same neighborhood/hour via two different raw pings
  (or two different edited states) contributes exactly 1, not 2.

This tool recounts tier-2 supply directly from raw GPS, TWICE: once
reproducing the production ("before") grid as a sanity check, and once with
the edited run's seeking-trajectory STATES substituted in place of their
original pings (drivers' other pings, and the pickup-transition state itself,
are left untouched — the pickup only affects DEMAND, handled exactly by
``external_fairness_io.build_edited_pickup_3d``, independent of this tier-1/
tier-2 distinction). It then reports the tier1-vs-tier2 gap in both the raw
ΔS grids and in two downstream metrics (F_causal, mean(Y|D) on the
disadvantaged migrant-axis group) evaluated under each tier's AFTER-supply.

Usage::

    python -m famail_temporal.analysis.supply_recount \\
        --edit-dir <results_dir> [--city shenzhen|sf12] [--raw-dir raw_data]

Reads (all read-only; nothing in the edit dir or existing modules is
modified):
- ``<edit_dir>/histories.pkl``       — per-edit original + modified trajectories.
- ``<edit_dir>/delta_supply_3d.npz`` — the editor's own tier-1 ΔS (Task 8).
- ``raw_data/taxi_record_0*_50drivers.pkl`` — raw GPS.
- The cached ``DataBundle`` (``python -m famail_temporal.preprocess``).

Writes ``<edit_dir>/supply_recount_report.md`` + ``supply_recount.json``.

Matching an edited trajectory's states back to specific raw GPS rows
---------------------------------------------------------------------
``histories.pkl`` stores ``Trajectory`` objects (state VALUES only — no
pointer back into the raw event stream), and the per-trajectory invariant
filter (``data/source_generation/invariants.py``) can drop whole trajectories
without renumbering the survivors, so a bundle trajectory's ordinal position
cannot be trusted to locate its raw segment. This tool instead re-segments
the raw event stream itself (mirroring
``data/source_generation/views/trajectories.py``'s grouping, but keeping row
indices) and matches each ``history.original`` by exact state-VALUE-sequence
content against that driver's raw segments. This is exact except in the
(unobserved-in-the-smoke-run) case of two distinct segments for the same
driver sharing an identical state sequence, which is resolved by first-match
consumption (FIFO) and counted in the report as `n_histories` vs `n_matched`.

Standalone, read-only analysis: does not modify any existing module and does
not touch famail_temporal/fairness/*.py.
"""
from __future__ import annotations

import argparse
import json
import pickle  # trusted repo-internal artifact, same as localized_metrics.py
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Only the SF ping-source needs new plumbing (its own segmentation, grid
# quantization and active-taxis counter, see
# second_dataset/data/source_generation/sf_build.py) which the task brief
# explicitly says to defer rather than improvise. Shenzhen is the deliverable.
_SUPPORTED_CITIES = {"shenzhen"}


# ---------------------------------------------------------------------------
# Raw-GPS re-segmentation (row-index-preserving mirror of
# data/source_generation/views/trajectories.py's seeking-segment extraction)
# ---------------------------------------------------------------------------

def _segment_rows_by_driver(
    df: pd.DataFrame,
) -> Dict[str, List[Tuple[np.ndarray, Tuple[Tuple[int, int, int, int], ...]]]]:
    """Per-plate list of (row_index_array, state_value_tuple) for every
    SEEKING segment (mirrors ``_segment_is_seeking`` + ``_segment_to_trajectory``
    from ``views/trajectories.py``, but keeps the underlying df row index so
    edited states can be traced back to specific raw rows)."""
    out: Dict[str, List[Tuple[np.ndarray, Tuple[Tuple[int, int, int, int], ...]]]] = {}
    for plate, driver_df in df.groupby("plate_id", sort=False):
        segs: List[Tuple[np.ndarray, Tuple[Tuple[int, int, int, int], ...]]] = []
        for _, seg in driver_df.groupby("segment_id", sort=True):
            if len(seg) < 2 or not bool(seg.iloc[-1]["is_pickup"]):
                continue
            state_tuple = tuple(
                (int(r.x_grid), int(r.y_grid), int(r.time_bucket), int(r.day_index))
                for r in seg.itertuples(index=False)
            )
            segs.append((seg.index.to_numpy(), state_tuple))
        out[plate] = segs
    return out


def _build_seg_lookup(
    seg_rows: Dict[str, List[Tuple[np.ndarray, Tuple]]],
) -> Dict[str, Dict[Tuple, List[np.ndarray]]]:
    """plate -> {state_value_tuple: [row_index_array, ...]} (multimap; a
    driver could in principle have two segments with an identical state
    sequence — resolved by FIFO consumption in `apply_substitutions`)."""
    lookup: Dict[str, Dict[Tuple, List[np.ndarray]]] = {}
    for plate, segs in seg_rows.items():
        m: Dict[Tuple, List[np.ndarray]] = defaultdict(list)
        for row_idx, state_tuple in segs:
            m[state_tuple].append(row_idx)
        lookup[plate] = m
    return lookup


def apply_substitutions(
    df: pd.DataFrame, histories: list, idx_to_plate: Dict[int, str],
) -> Tuple[pd.DataFrame, dict]:
    """Return (df2, stats). df2 is `df` with the x_grid/y_grid of the exact
    raw rows underlying each edited trajectory's MOVED, non-terminal
    (seeking-only) states overwritten to the modified cell. The terminal
    (pickup-transition) state is intentionally left untouched here — it is a
    DEMAND event, handled by external_fairness_io.build_edited_pickup_3d.

    Mutating x_grid/y_grid IN PLACE on the matched raw row (rather than
    dropping + appending rows) is what makes the multiset semantics correct
    "for free": build_active_taxis_counts's drop_duplicates() step will keep
    the ORIGINAL (driver, x, y, hour, day) key alive if some OTHER raw ping
    (another state, possibly from a different segment) still supports it —
    exactly the "states that didn't change cell / drivers' other pings
    unchanged" requirement.
    """
    df2 = df.copy()
    seg_rows = _segment_rows_by_driver(df2)
    seg_lookup = _build_seg_lookup(seg_rows)

    n_histories = len(histories)
    n_matched = 0
    n_unmatched = 0
    n_moved_states = 0
    n_len_mismatch = 0
    unmatched_examples: List[dict] = []

    for h in histories:
        orig, mod = h.original, h.modified
        if len(orig.states) != len(mod.states):
            n_len_mismatch += 1
            continue
        plate = idx_to_plate.get(int(orig.driver_id))
        if plate is None:
            n_unmatched += 1
            continue
        key = tuple(
            (int(s.x_grid) + 1, int(s.y_grid) + 1, int(s.time_bucket), int(s.day_index))
            for s in orig.states
        )
        bucket = seg_lookup.get(plate, {}).get(key)
        if not bucket:
            n_unmatched += 1
            if len(unmatched_examples) < 10:
                unmatched_examples.append({
                    "trajectory_id": str(orig.trajectory_id),
                    "driver_id": int(orig.driver_id), "plate": plate,
                })
            continue
        row_idx = bucket.pop(0)
        n_matched += 1
        n_states = len(row_idx)
        # Exclude the terminal state (pickup-transition record) — supply-only.
        for i in range(n_states - 1):
            os_, ms_ = orig.states[i], mod.states[i]
            if (int(os_.x_grid), int(os_.y_grid)) != (int(ms_.x_grid), int(ms_.y_grid)):
                n_moved_states += 1
                ridx = row_idx[i]
                df2.at[ridx, "x_grid"] = int(ms_.x_grid) + 1
                df2.at[ridx, "y_grid"] = int(ms_.y_grid) + 1

    stats = {
        "n_histories": n_histories,
        "n_matched": n_matched,
        "n_unmatched": n_unmatched,
        "n_len_mismatch": n_len_mismatch,
        "n_moved_states": n_moved_states,
        "unmatched_examples": unmatched_examples,
    }
    return df2, stats


# ---------------------------------------------------------------------------
# Recount pipeline
# ---------------------------------------------------------------------------

def _load_driver_mapping(config) -> Dict[int, str]:
    path = config.SOURCE_DATA_DIR / "driver_index_mapping.pkl"
    # trusted repo-internal artifact, same as localized_metrics.py
    with open(path, "rb") as f:
        mapping = pickle.load(f)
    return {int(k): v for k, v in mapping.get("idx_to_plate", {}).items()}


def _load_histories(edit_dir: Path) -> list:
    # trusted repo-internal artifact, same as localized_metrics.py
    with open(edit_dir / "histories.pkl", "rb") as f:
        return pickle.load(f)


def recount_tier2(df: pd.DataFrame, n_days: int, active_taxis_view, aggregate_active_taxis):
    """Recount (48, 90, T) mean-hourly tier-2 supply from a (possibly
    edit-substituted) event-stream df, via the SAME production functions
    used to build the real grid."""
    raw_counts = active_taxis_view.build_active_taxis_counts(df)
    grid = aggregate_active_taxis(raw_counts, n_days)
    return grid, raw_counts


def _grid_compare(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    """Reproduction-error summary of `a` (recount) vs `b` (reference), over
    active-mask cells."""
    diff = (a - b)[mask]
    ref = b[mask]
    denom = np.maximum(np.abs(ref), 1e-6)
    return {
        "n_active_cells": int(mask.sum()),
        "mae": float(np.abs(diff).mean()),
        "max_abs_diff": float(np.abs(diff).max()) if diff.size else 0.0,
        "mean_relative_error": float((np.abs(diff) / denom).mean()),
        "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 1 else float("nan"),
    }


def _delta_compare(delta1: np.ndarray, delta2: np.ndarray, mask: np.ndarray) -> dict:
    """Correlation / magnitude ratio / sign-disagreement between two ΔS grids
    over active cells, plus the top-10 largest-disagreement cells."""
    d1, d2 = delta1[mask], delta2[mask]
    corr = float(np.corrcoef(d1, d2)[0, 1]) if d1.size > 1 and d1.std() > 0 and d2.std() > 0 else float("nan")
    mag1 = float(np.abs(d1).sum())
    mag2 = float(np.abs(d2).sum())
    ratio = (mag2 / mag1) if mag1 > 0 else float("nan")

    nonzero1 = d1 != 0
    n_tier1_nonzero = int(nonzero1.sum())
    sign1 = np.sign(d1[nonzero1])
    sign2 = np.sign(d2[nonzero1])
    n_agree = int((sign1 == sign2).sum())
    n_tier2_zero = int((d2[nonzero1] == 0).sum())
    n_disagree_nonzero = int(((sign1 != sign2) & (d2[nonzero1] != 0)).sum())

    ix = np.where(mask)
    abs_disagreement = np.abs(delta1 - delta2)
    flat_order = np.argsort(-abs_disagreement[mask])
    top = []
    for j in flat_order[:10]:
        x, y, t = int(ix[0][j]), int(ix[1][j]), int(ix[2][j])
        top.append({
            "x": x, "y": y, "t": t,
            "delta_tier1": float(delta1[x, y, t]),
            "delta_tier2": float(delta2[x, y, t]),
        })

    return {
        "corr": corr,
        "magnitude_tier1_sum_abs": mag1,
        "magnitude_tier2_sum_abs": mag2,
        "magnitude_ratio_tier2_over_tier1": ratio,
        "n_active_cells_tier1_nonzero": n_tier1_nonzero,
        "n_sign_agree": n_agree,
        "n_tier2_zero_where_tier1_nonzero": n_tier2_zero,
        "n_sign_disagree_both_nonzero": n_disagree_nonzero,
        "frac_sign_agree": (n_agree / n_tier1_nonzero) if n_tier1_nonzero else float("nan"),
        "top10_disagreement_cells": top,
    }


def _write_deferred_report(edit_dir: Path, city: str) -> None:
    edit_dir = Path(edit_dir)
    msg = (
        f"# Tier-2 supply recount — DEFERRED for city='{city}'\n\n"
        f"The SF ping-path (`second_dataset/data/source_generation/sf_build.py`) "
        f"uses its own segmentation (`sf_segmentation.py`), its own grid "
        f"quantization (`sf_config.grid_from_points`), and its own active-taxis "
        f"counter (`sf_grid_counts.count_active_taxis_5x5`) — none of which are "
        f"drop-in-compatible with the Shenzhen `data/source_generation` views "
        f"this tool reuses. Per the task brief, this is new plumbing (not an "
        f"analogous file read), so it is deliberately NOT implemented here.\n\n"
        f"Shenzhen (`--city shenzhen`, the default) is the deliverable and is "
        f"fully implemented in this tool.\n"
    )
    (edit_dir / "supply_recount_report.md").write_text(msg)
    (edit_dir / "supply_recount.json").write_text(json.dumps(
        {"city": city, "status": "deferred", "reason": "sf ping-path needs new plumbing"},
        indent=2,
    ))
    print(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-dir", type=Path, required=True,
                         help="Persisted editing run dir (has histories.pkl, delta_supply_3d.npz).")
    parser.add_argument("--city", default="shenzhen", choices=["shenzhen", "sf12"],
                         help="'sf12' is deliberately deferred (see module docstring).")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"),
                         help="Directory with the 3 taxi_record_0*_50drivers.pkl raw GPS files.")
    args = parser.parse_args(argv)

    edit_dir = Path(args.edit_dir)

    if args.city not in _SUPPORTED_CITIES:
        _write_deferred_report(edit_dir, args.city)
        return 0

    t0 = time.monotonic()

    # Deferred imports: famail_temporal.config resolves FAMAIL_CITY at import
    # time, so city selection must happen before any famail_temporal import.
    import os
    os.environ.setdefault("FAMAIL_CITY", args.city)
    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.data.aggregation import aggregate_active_taxis
    from famail_temporal.data.source_generation.event_stream import build_event_stream
    from famail_temporal.data.source_generation.views import active_taxis as active_taxis_view
    from famail_temporal.baselines import external_fairness as ef
    from famail_temporal.baselines import external_fairness_io as efio
    from famail_temporal.evaluation.grid import build_fairness_grid

    print("[supply_recount] loading DataBundle...", flush=True)
    bundle = DataBundle.load()
    mask = bundle.mask_3d

    print(f"[supply_recount] loading raw GPS from {args.raw_dir}...", flush=True)
    es = build_event_stream(args.raw_dir)
    n_days = bundle.n_days  # match production's aggregation divisor exactly

    print("[supply_recount] recounting tier-2 BEFORE (reproduction check)...", flush=True)
    S_tier2_before, _ = recount_tier2(es.df, n_days, active_taxis_view, aggregate_active_taxis)
    reproduction = _grid_compare(S_tier2_before, bundle.active_taxis_3d, mask)

    print(f"[supply_recount] loading histories from {edit_dir}...", flush=True)
    histories = _load_histories(edit_dir)
    idx_to_plate = _load_driver_mapping(config)

    print(f"[supply_recount] substituting {len(histories)} edited trajectories...", flush=True)
    df_after, sub_stats = apply_substitutions(es.df, histories, idx_to_plate)

    print("[supply_recount] recounting tier-2 AFTER...", flush=True)
    S_tier2_after, _ = recount_tier2(df_after, n_days, active_taxis_view, aggregate_active_taxis)

    # --- tier-1 AFTER supply (from Task 8's own persisted delta-supply) ---
    delta_path = edit_dir / "delta_supply_3d.npz"
    delta_supply_3d = np.load(delta_path)["delta_supply_3d"]
    S_tier1_after = np.clip(
        bundle.active_taxis_3d + delta_supply_3d, config.SUPPLY_FLOOR, None,
    ).astype(bundle.active_taxis_3d.dtype)

    delta_tier1 = delta_supply_3d
    delta_tier2 = S_tier2_after.astype(np.float64) - S_tier2_before.astype(np.float64)
    ds_gap = _delta_compare(delta_tier1, delta_tier2, mask)

    # --- metric-level effects: F_causal + mean(Y|D) under tier1 vs tier2 AFTER supply ---
    print("[supply_recount] computing metric-level effects...", flush=True)
    pickup_after = efio.build_edited_pickup_3d(bundle, edit_dir)

    bundle_tier1_after = replace(bundle, active_taxis_3d=S_tier1_after)
    bundle_tier2_after = replace(
        bundle, active_taxis_3d=S_tier2_after.astype(bundle.active_taxis_3d.dtype),
    )

    grid_before = build_fairness_grid(bundle)
    grid_tier1_after = build_fairness_grid(bundle_tier1_after, pickup_3d=pickup_after)
    grid_tier2_after = build_fairness_grid(bundle_tier2_after, pickup_3d=pickup_after)

    f_causal_before = float(np.nansum(grid_before[..., 1]))
    f_causal_tier1_after = float(np.nansum(grid_tier1_after[..., 1]))
    f_causal_tier2_after = float(np.nansum(grid_tier2_after[..., 1]))

    demo = efio.per_unit_demographics(bundle)
    g_unit = ef.region_extremes(demo["MigrantRatio"], disadvantaged_high=True)
    d_mask = g_unit == 1

    Y_before = efio.service_ratio_Y(bundle.pickup_3d, bundle)
    Y_tier1_after = efio.service_ratio_Y(pickup_after, bundle_tier1_after)
    Y_tier2_after = efio.service_ratio_Y(pickup_after, bundle_tier2_after)

    mean_Y_D_before = float(Y_before[d_mask].mean())
    mean_Y_D_tier1_after = float(Y_tier1_after[d_mask].mean())
    mean_Y_D_tier2_after = float(Y_tier2_after[d_mask].mean())

    runtime_s = time.monotonic() - t0

    result = {
        "city": args.city,
        "edit_dir": str(edit_dir),
        "runtime_seconds": runtime_s,
        "n_days": int(n_days),
        "substitution_stats": sub_stats,
        "sanity_check_1_reproduction_before_vs_production": reproduction,
        "sanity_check_2_tier1_vs_tier2_delta_supply_gap": ds_gap,
        "metrics": {
            "f_causal_before": f_causal_before,
            "f_causal_tier1_after": f_causal_tier1_after,
            "f_causal_tier2_after": f_causal_tier2_after,
            "delta_f_causal_tier1": f_causal_tier1_after - f_causal_before,
            "delta_f_causal_tier2": f_causal_tier2_after - f_causal_before,
            "mean_Y_D_before": mean_Y_D_before,
            "mean_Y_D_tier1_after": mean_Y_D_tier1_after,
            "mean_Y_D_tier2_after": mean_Y_D_tier2_after,
            "delta_mean_Y_D_tier1": mean_Y_D_tier1_after - mean_Y_D_before,
            "delta_mean_Y_D_tier2": mean_Y_D_tier2_after - mean_Y_D_before,
        },
        "S_tier2_before_summary": {
            "mean_active": float(S_tier2_before[mask].mean()),
            "sum_active": float(S_tier2_before[mask].sum()),
        },
        "S_tier2_after_summary": {
            "mean_active": float(S_tier2_after[mask].mean()),
            "sum_active": float(S_tier2_after[mask].sum()),
        },
    }

    (edit_dir / "supply_recount.json").write_text(json.dumps(result, indent=2))
    _write_report_md(edit_dir, result)
    print(f"[supply_recount] done in {runtime_s:.1f}s -> {edit_dir / 'supply_recount_report.md'}",
          flush=True)
    return 0


def _write_report_md(edit_dir: Path, r: dict) -> None:
    rep = r["sanity_check_1_reproduction_before_vs_production"]
    gap = r["sanity_check_2_tier1_vs_tier2_delta_supply_gap"]
    m = r["metrics"]
    sub = r["substitution_stats"]
    lines = [
        "# Tier-2 distinct-count supply recount (gate G2 validator, Task 9)",
        "",
        f"City: {r['city']} | edit_dir: `{r['edit_dir']}` | runtime: {r['runtime_seconds']:.1f}s | "
        f"n_days: {r['n_days']}",
        "",
        "## Sanity check 1 — before-recount vs production active_taxis_3d",
        "",
        f"- active cells compared: {rep['n_active_cells']}",
        f"- MAE: {rep['mae']:.6f} | max abs diff: {rep['max_abs_diff']:.6f} | "
        f"mean relative error: {rep['mean_relative_error']:.4%}",
        f"- correlation: {rep['corr']:.6f}",
        "",
        "## Substitution (edited trajectories -> raw-row match)",
        "",
        f"- n_histories: {sub['n_histories']} | n_matched: {sub['n_matched']} | "
        f"n_unmatched: {sub['n_unmatched']} | n_len_mismatch: {sub['n_len_mismatch']}",
        f"- n_moved_states substituted (supply-only, excludes the pickup state): "
        f"{sub['n_moved_states']}",
        "",
        "## Sanity check 2 — tier1 (fraction) vs tier2 (distinct-count) ΔS gap",
        "",
        f"- correlation (active cells): {gap['corr']:.6f}",
        f"- magnitude: tier1 sum|ΔS| = {gap['magnitude_tier1_sum_abs']:.4f}, "
        f"tier2 sum|ΔS| = {gap['magnitude_tier2_sum_abs']:.4f}, "
        f"ratio (tier2/tier1) = {gap['magnitude_ratio_tier2_over_tier1']:.4f}",
        f"- of {gap['n_active_cells_tier1_nonzero']} active cells where tier1 ΔS != 0: "
        f"{gap['n_sign_agree']} agree in sign "
        f"({gap['frac_sign_agree']:.2%}), {gap['n_tier2_zero_where_tier1_nonzero']} have "
        f"tier2 ΔS == 0 (distinct-count saw no NET change), "
        f"{gap['n_sign_disagree_both_nonzero']} disagree in sign (both nonzero)",
        "",
        "### Top-10 largest |tier1 - tier2| disagreement cells (x, y, t)",
        "",
        "| x | y | t | delta_tier1 | delta_tier2 |",
        "|---|---|---|---|---|",
    ]
    for c in gap["top10_disagreement_cells"]:
        lines.append(f"| {c['x']} | {c['y']} | {c['t']} | {c['delta_tier1']:.5f} | "
                      f"{c['delta_tier2']:.5f} |")
    lines += [
        "",
        "## Metric-level effect: F_causal and mean(Y|D) under tier1 vs tier2 AFTER-supply",
        "",
        "Same demand (`external_fairness_io.build_edited_pickup_3d`), same disadvantaged "
        "group (migrant-axis district extremes, `region_extremes(..., disadvantaged_high=True)`), "
        "so the ONLY thing that differs between the tier1 and tier2 columns below is which "
        "AFTER-supply grid was used.",
        "",
        "| | before | tier1 after | tier2 after | Δ tier1 | Δ tier2 |",
        "|---|---|---|---|---|---|",
        f"| F_causal | {m['f_causal_before']:.6f} | {m['f_causal_tier1_after']:.6f} | "
        f"{m['f_causal_tier2_after']:.6f} | {m['delta_f_causal_tier1']:+.6f} | "
        f"{m['delta_f_causal_tier2']:+.6f} |",
        f"| mean(Y\\|D) | {m['mean_Y_D_before']:.6f} | {m['mean_Y_D_tier1_after']:.6f} | "
        f"{m['mean_Y_D_tier2_after']:.6f} | {m['delta_mean_Y_D_tier1']:+.6f} | "
        f"{m['delta_mean_Y_D_tier2']:+.6f} |",
        "",
    ]
    (edit_dir / "supply_recount_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
