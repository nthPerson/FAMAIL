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

# "sf12" is wired via the SF ping adapter (analysis/sf_recount_adapter.py,
# D1 Task 1) + its own mirrored counting path (recount_tier2_sf /
# _build_active_taxis_counts_sf below, D1 Task 2, 2026-07-17) -- SF's grid
# transform, occupancy/seeking semantics, and supply-grid construction are
# IMPORTED or replicated verbatim from the SF source-generation pipeline,
# never re-derived (see docs/superpowers/specs/2026-07-17-d1-sf-tier2-recount-
# design.md). The substitution-replay machinery (apply_substitutions et al.)
# is city-agnostic and untouched.


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
    seg_lookup: Dict[str, Dict[Tuple, List[np.ndarray]]] | None = None,
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

    ``seg_lookup`` (optional) is a prebuilt ``plate -> {state_value_tuple:
    [row_index_array, ...]}`` lookup whose row indices point into ``df``. When
    omitted (the Shenzhen default), it is derived from ``df`` via the SZ
    transition machinery (``_segment_rows_by_driver`` / ``_build_seg_lookup``).
    sf12 injects a lookup built by ``sf_recount_adapter.build_sf_seeking_lookup``
    (SF-native 300s-gap segmentation + weekday day space) instead, because the
    SZ derivation does not reproduce the SF editor's trajectories (D1 Task 3
    diagnosis; spec addendum 2026-07-17). The matching loop below is identical
    either way -- only the source of ``seg_lookup`` differs. Only x_grid/y_grid
    are ever written, so ``df``'s day_index/hour/etc. are invariant.
    """
    df2 = df.copy()
    if seg_lookup is None:
        seg_rows = _segment_rows_by_driver(df2)
        seg_lookup = _build_seg_lookup(seg_rows)
    else:
        # Defensive per-bucket copy: the loop below consumes buckets via
        # pop(0); don't mutate the caller's prebuilt lookup.
        seg_lookup = {p: {k: list(v) for k, v in mm.items()}
                      for p, mm in seg_lookup.items()}

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
        # For SF this also holds: states[-1] sits at the pickup_cell, whose move
        # is carried by the DEMAND channel (build_edited_pickup_3d) — excluding
        # it here avoids double-counting the relocation across channels.
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


def _build_active_taxis_counts_sf(
    df: pd.DataFrame, x_grid_max: int, y_grid_max: int, k: int,
) -> Dict[Tuple[int, int, int, int], int]:
    """SF-mirrored distinct-taxi 5x5-neighborhood counter.

    Structurally the SAME expand/dedup/groupby algorithm as
    ``data.source_generation.views.active_taxis.build_active_taxis_counts``
    (active_taxis.py:14-59) -- shifting each present (plate, x, y, hour, day)
    row by every ``(dx, dy)`` in ``[-k, k] x [-k, k]``, clipping to city
    bounds, then deduping per (plate, target-cell, hour, day) is algebraically
    equivalent to the "spread a source cell's driver-set across its
    ``(2k+1)x(2k+1)`` target window" rule SF's OWN counter implements
    (``second_dataset/.../sf_grid_counts.count_active_taxis_5x5``,
    sf_grid_counts.py:36-75) -- but this function mirrors count_active_taxis_5x5's
    TWO documented divergences from the SZ counter (Task 1 adapter docstring
    "Known divergence" section + Task 2 controller adjudication, 2026-07-17):

      - NO occupancy filter: SF's counter counts a taxi present in a
        cell-hour regardless of ``passenger_indicator`` (fare status)
        (sf_grid_counts.py:47-64 has no analog of active_taxis.py:20's
        ``df["passenger_indicator"] == 0`` filter) -- so, unlike
        ``build_active_taxis_counts``, this function does NOT filter ``df``
        by occupancy before computing presence.
      - City-aware clip bounds: ``x_grid_max``/``y_grid_max`` are passed in
        (SF's grid is 32x30, ``famail_temporal/config.py:35``) instead of
        SZ's hardcoded ``data.source_generation.config.X_GRID_MAX``/
        ``Y_GRID_MAX`` (48, 90) that ``build_active_taxis_counts`` imports
        directly.

    Unlike ``count_active_taxis_5x5``, which re-derives x_grid/y_grid from
    raw lat/lon internally on every call (sf_grid_counts.py:54-57), this
    function operates on the ALREADY-quantized (x_grid, y_grid, hour,
    day_index) columns ``sf_recount_adapter.load_sf_pings`` produces --
    required so a post-substitution recount (``apply_substitutions`` mutates
    x_grid/y_grid in place on exactly this schema, supply_recount.py:181-182)
    reflects the edited cells; re-deriving from lat/lon would silently ignore
    every substitution.
    """
    if len(df) == 0:
        return {}

    present = df[["plate_id", "x_grid", "y_grid", "hour", "day_index"]].drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    pieces: List[pd.DataFrame] = []
    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            exp = present.copy()
            exp["x_grid"] = exp["x_grid"] + dx
            exp["y_grid"] = exp["y_grid"] + dy
            pieces.append(exp)
    expanded = pd.concat(pieces, ignore_index=True)

    expanded = expanded[
        (expanded["x_grid"] >= 1) & (expanded["x_grid"] <= x_grid_max)
        & (expanded["y_grid"] >= 1) & (expanded["y_grid"] <= y_grid_max)
    ]

    expanded = expanded.drop_duplicates(
        subset=["plate_id", "x_grid", "y_grid", "hour", "day_index"]
    )

    counts = (
        expanded
        .groupby(["x_grid", "y_grid", "hour", "day_index"], sort=False)
        .size()
        .reset_index(name="count")
    )

    return {
        (int(r.x_grid), int(r.y_grid), int(r.hour), int(r.day_index)): int(r.count)
        for r in counts.itertuples(index=False)
    }


def recount_tier2_sf(
    df: pd.DataFrame, n_days: int, x_grid_max: int, y_grid_max: int, k: int,
    aggregate_active_taxis,
):
    """SF counterpart of ``recount_tier2``: mirrors
    ``second_dataset/.../sf_grid_counts.count_active_taxis_5x5``'s counting
    RULE (no occupancy filter, SF clip bounds -- see
    ``_build_active_taxis_counts_sf``'s docstring) via
    ``aggregate_active_taxis`` (city-agnostic; keys off ``famail_temporal.
    config.GRID_DIMS``, already resolved to (32, 30) under
    ``FAMAIL_CITY=sf12``), instead of SZ's
    ``active_taxis_view.build_active_taxis_counts``."""
    raw_counts = _build_active_taxis_counts_sf(df, x_grid_max, y_grid_max, k)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-dir", type=Path, required=True,
                         help="Persisted editing run dir (has histories.pkl, delta_supply_3d.npz).")
    parser.add_argument("--city", default="shenzhen", choices=["shenzhen", "sf12"],
                         help="'sf12' recounts SF Cabspotting via its own mirrored counting "
                              "path (no occupancy filter, SF clip bounds; see recount_tier2_sf).")
    parser.add_argument("--raw-dir", type=Path, default=None,
                         help="Raw GPS source dir. SZ (default 'raw_data'): the 3 "
                              "taxi_record_0*_50drivers.pkl files. SF (default the full "
                              "cabspottingdata fleet dir, from which the production grid was "
                              "derived -- see sf_recount_adapter.py's 'Production grid "
                              "derivation' docstring note): Cabspotting new_*.txt traces.")
    parser.add_argument("--persist-grids", action="store_true",
                         help="Also save S_tier2_before.npz / S_tier2_after.npz "
                              "alongside the json (for the direct tier-2 supply-channel CI).")
    args = parser.parse_args(argv)

    edit_dir = Path(args.edit_dir)

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

    if args.raw_dir is None:
        # SZ default unchanged ('raw_data'); SF default is the FULL
        # cabspotting fleet dir -- sf_build.build() (sf_build.py:37-39)
        # derives the production grid from grid_from_points on the
        # UNFILTERED fleet before any driver_ids subsampling, so
        # reproducing that grid (G-repro) requires the same full-fleet
        # input (sf_recount_adapter.py "Production grid derivation" note).
        args.raw_dir = (
            Path("raw_data") if args.city == "shenzhen"
            else config.PACKAGE_ROOT / "source_data" / "second_dataset" / "cabspottingdata"
        )

    if args.city != "shenzhen":
        from famail_temporal.analysis.sf_recount_adapter import load_sf_pings
        from famail_temporal.second_dataset.data.source_generation import sf_grid_counts

    print("[supply_recount] loading DataBundle...", flush=True)
    bundle = DataBundle.load()
    mask = bundle.mask_3d

    idx_to_plate = _load_driver_mapping(config)

    print(f"[supply_recount] loading raw GPS from {args.raw_dir}...", flush=True)
    if args.city == "shenzhen":
        es_df = build_event_stream(args.raw_dir).df
    else:
        raw_sf_df = load_sf_pings(args.raw_dir)
        # Defensive: the recount clip bounds are config.GRID_DIMS, but es_df was
        # quantized by load_sf_pings via grid_from_points on raw_dir. For the
        # production grid these must agree; a smaller-fleet raw_dir yields a
        # smaller grid that would silently mismatch the clip bounds -- fail loud.
        grid_bounds = raw_sf_df.attrs.get("grid_bounds")
        assert grid_bounds == tuple(config.GRID_DIMS), (
            f"SF adapter-derived grid bounds {grid_bounds} != config.GRID_DIMS "
            f"{tuple(config.GRID_DIMS)}; raw_dir {args.raw_dir} is not the full "
            f"production fleet the sf12 grid was derived from."
        )
        # Mirrors sf_build.build()'s driver_ids filter (sf_build.py:40-41),
        # applied AFTER grid quantization (which used the full fleet above)
        # -- restrict to the plates THIS city variant's own
        # driver_index_mapping.pkl names, so counts match the production
        # subsample exactly.
        target_plates = set(idx_to_plate.values())
        es_df = raw_sf_df[raw_sf_df["plate_id"].isin(target_plates)].reset_index(drop=True)
        # Make the driver restriction explicit: every plate named in the sf12
        # driver_index_mapping.pkl must appear in the full-fleet raw pings (else
        # raw_dir is not the fleet this corpus was derived from).
        assert es_df["plate_id"].nunique() == len(target_plates), (
            f"sf12 driver restriction: {es_df['plate_id'].nunique()} distinct "
            f"plates in raw != {len(target_plates)} in driver_index_mapping.pkl"
        )

    n_days = bundle.n_days  # match production's aggregation divisor exactly

    print("[supply_recount] recounting tier-2 BEFORE (reproduction check)...", flush=True)
    if args.city == "shenzhen":
        S_tier2_before, _ = recount_tier2(es_df, n_days, active_taxis_view, aggregate_active_taxis)
    else:
        x_grid_max, y_grid_max = config.GRID_DIMS
        S_tier2_before, _ = recount_tier2_sf(
            es_df, n_days, x_grid_max, y_grid_max, sf_grid_counts.NEIGHBORHOOD_K,
            aggregate_active_taxis,
        )
    reproduction = _grid_compare(S_tier2_before, bundle.active_taxis_3d, mask)

    print(f"[supply_recount] loading histories from {edit_dir}...", flush=True)
    histories = _load_histories(edit_dir)

    print(f"[supply_recount] substituting {len(histories)} edited trajectories...", flush=True)
    if args.city == "shenzhen":
        df_after, sub_stats = apply_substitutions(es_df, histories, idx_to_plate)
    else:
        # sf12 histories were segmented by SF's own 300s-gap segmenter in
        # weekday day space; the SZ transition machinery baked into es_df does
        # not reproduce them (D1 Task 3; spec addendum). Inject an SF-native
        # match lookup whose row indices point into es_df, so substitutions land
        # on exactly the rows the SF counter counts (es_df day_index untouched).
        from famail_temporal.analysis.sf_recount_adapter import build_sf_seeking_lookup
        sf_seg_lookup = build_sf_seeking_lookup(es_df, args.raw_dir, idx_to_plate)
        df_after, sub_stats = apply_substitutions(
            es_df, histories, idx_to_plate, seg_lookup=sf_seg_lookup,
        )

    print("[supply_recount] recounting tier-2 AFTER...", flush=True)
    if args.city == "shenzhen":
        S_tier2_after, _ = recount_tier2(df_after, n_days, active_taxis_view, aggregate_active_taxis)
    else:
        S_tier2_after, _ = recount_tier2_sf(
            df_after, n_days, x_grid_max, y_grid_max, sf_grid_counts.NEIGHBORHOOD_K,
            aggregate_active_taxis,
        )

    if args.persist_grids:
        # Save the two tier-2 supply grids so the channel decomposition can
        # substitute S_tier2_after for S' in the DIRECT tier-2 supply-channel CI.
        np.savez_compressed(edit_dir / "S_tier2_before.npz", S_tier2_before=S_tier2_before)
        np.savez_compressed(edit_dir / "S_tier2_after.npz", S_tier2_after=S_tier2_after)
        print(f"[supply_recount] persisted S_tier2_before.npz / S_tier2_after.npz "
              f"to {edit_dir}", flush=True)

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
