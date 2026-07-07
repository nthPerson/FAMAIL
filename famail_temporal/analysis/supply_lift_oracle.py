"""Stage-0 supply-lift oracle (gate G0): achievable Delta mean(Y|D) ceiling.

Greedy upper bound on lifting-up via seeking-tail rerouting, BEFORE any build.
For each trajectory whose tail passes near a disadvantaged-group cell, scores
the best single discrete translation delta in [-2,2]^2 of its tail on the net
effect on mean(Y|D) (supply added/removed at 5x5 neighborhoods + the pickup's
demand mass relocated with it). Greedy best-first application up to a budget,
re-scoring against the running supply/demand grids at each step. Reports two
mass-accounting semantics: `fraction` (every moved state's mass counts) and
`distinct-seeking` (mass only counts where it changes a driver's occupancy of
a neighborhood-hour, per a static presence index built from the ORIGINAL
seeking corpus -- an approximation; exact raw-GPS recount is Task 9).

Standalone, read-only analysis: does not modify any existing module and does
not touch famail_temporal/fairness/*.py.

Run:  python -m famail_temporal.analysis.supply_lift_oracle [--budget 10000] [--tail-len 4]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from famail_temporal import config
from famail_temporal.baselines import datasets as ds
from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.data.loader import DataBundle

OUT = Path(__file__).resolve().parent / "supply_lift_oracle_out"

G0_THRESHOLD = 0.3

# All 24 discrete Chebyshev-ball translations excluding (0, 0).
DELTAS = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]


def state_mass(bundle: DataBundle, t_block: int) -> float:
    """Supply presence mass contributed by ONE seeking-tail state, spread over
    its clipped 5x5 box. 12x smaller than a pickup's demand mass in the same
    block (a presence fraction, not a discrete count)."""
    return 1.0 / (12.0 * float(bundle.n_hours_per_block[t_block]) * bundle.n_days)


def box5(x: int, y: int, gx: int, gy: int):
    """Coordinates of the clipped 5x5 (Chebyshev <=2) neighborhood around (x, y)."""
    return [(i, j) for i in range(max(0, x - 2), min(gx, x + 3))
                   for j in range(max(0, y - 2), min(gy, y + 3))]


def tail_states(traj, L: int):
    """Last min(L, len-2)+1 states (tail + pickup); anchor untouched. [] if len < 3."""
    n = len(traj.states)
    if n < 3:
        return []
    L_eff = min(L, n - 2)
    return traj.states[-(L_eff + 1):]


def evaluate_delta(states_info, driver_id, delta, S, D, group_grid, presence,
                    bundle, gx, gy):
    """Score a single candidate translation `delta` of a tail against the
    CURRENT (S, D) grids.

    `states_info` is a list of (x, y, t_block, is_pickup) for the tail states
    (untranslated / original positions). Returns:
      score_fraction, score_distinct  -- net sum of (Y' - Y) over affected
        active D-group units, NOT yet divided by N_D.
      removal, addition -- dicts {(x,y,t): mass} of the FULL (fraction-
        semantics) supply-mass moves, for use by the caller to commit the
        edit to the running grids.
      pickup_old, pickup_new, pickup_t -- the pickup's (x,y) before/after and
        its fixed time block (None if the tail has no pickup, which cannot
        happen given tail_states always ends at the pickup).

    Mass accounting:
    - `removal`/`addition`: full presence mass for every tail state (fraction
      semantics) -- the convention used to update the REAL running grids
      (models continuous presence correctly for downstream interactions).
    - `removal_distinct`/`addition_distinct`: gated by the static per-driver
      presence index. Removal at a box cell is credited only if no OTHER
      original state of this driver sits at that exact (x, y, t_block); by
      construction the box's OWN center cell (the state's un-translated
      position) always has itself in the presence set, so its removal is
      never credited under this approximation -- a deliberate conservative
      bias (distinct-seeking <= fraction, see module docstring). Addition at
      a box cell is credited only if the driver has no original state there.
    - The pickup's demand-mass relocation is NOT gated by presence (it is a
      single physical event, not a supply-presence duplicate) -- identical
      under both semantics.
    """
    removal, addition = {}, {}
    removal_distinct, addition_distinct = {}, {}
    pickup_old = pickup_new = pickup_t = None
    dx, dy = delta

    for (x, y, t, is_pickup) in states_info:
        m = state_mass(bundle, t)
        for (bx, by) in box5(x, y, gx, gy):
            key = (bx, by, t)
            removal[key] = removal.get(key, 0.0) + m
            if (bx, by) != (x, y) and key not in presence[driver_id]:
                removal_distinct[key] = removal_distinct.get(key, 0.0) + m
        nx = min(max(x + dx, 0), gx - 1)
        ny = min(max(y + dy, 0), gy - 1)
        for (bx, by) in box5(nx, ny, gx, gy):
            key = (bx, by, t)
            addition[key] = addition.get(key, 0.0) + m
            if key not in presence[driver_id]:
                addition_distinct[key] = addition_distinct.get(key, 0.0) + m
        if is_pickup:
            pickup_old, pickup_new, pickup_t = (x, y, t), (nx, ny, t), t

    keys = set(removal) | set(addition) | set(removal_distinct) | set(addition_distinct)
    if pickup_old is not None and pickup_old != pickup_new:
        keys.add(pickup_old)
        keys.add(pickup_new)

    score_fraction = 0.0
    score_distinct = 0.0
    pmass = ds.pickup_mass(bundle, pickup_t) if pickup_t is not None else 0.0
    pickup_moved = pickup_old is not None and pickup_old != pickup_new

    for k in keys:
        bx, by, t = k
        if group_grid[bx, by, t] != 1:
            continue  # only active D-group units affect the score
        base_S, base_D = S[bx, by, t], D[bx, by, t]

        dS_f = addition.get(k, 0.0) - removal.get(k, 0.0)
        dS_d = addition_distinct.get(k, 0.0) - removal_distinct.get(k, 0.0)
        S_after_f = max(base_S + dS_f, 0.0)
        S_after_d = max(base_S + dS_d, 0.0)

        D_after = base_D
        if pickup_moved:
            if k == pickup_old:
                D_after = max(base_D - pmass, config.DEMAND_FLOOR)
            elif k == pickup_new:
                D_after = base_D + pmass

        Y_before = base_S / max(base_D, config.DEMAND_FLOOR)
        Y_after_f = S_after_f / max(D_after, config.DEMAND_FLOOR)
        Y_after_d = S_after_d / max(D_after, config.DEMAND_FLOOR)
        score_fraction += Y_after_f - Y_before
        score_distinct += Y_after_d - Y_before

    return (score_fraction, score_distinct, removal, addition,
            pickup_old, pickup_new, pickup_t)


def commit_edit(S, D, bundle, removal, addition, pickup_old, pickup_new, pickup_t):
    """Apply a chosen edit's mass moves to the RUNNING (S, D) grids in place."""
    for (bx, by, t), m in removal.items():
        S[bx, by, t] = max(S[bx, by, t] - m, 0.0)
    for (bx, by, t), m in addition.items():
        S[bx, by, t] = S[bx, by, t] + m
    if pickup_old is not None and pickup_old != pickup_new:
        pmass = ds.pickup_mass(bundle, pickup_t)
        ox, oy, ot = pickup_old
        D[ox, oy, ot] = max(D[ox, oy, ot] - pmass, config.DEMAND_FLOOR)
        nx, ny, nt = pickup_new
        D[nx, ny, nt] = D[nx, ny, nt] + pmass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=10000,
                         help="Max number of edits to apply (greedy, best-first).")
    parser.add_argument("--tail-len", type=int, default=4,
                         help="Max number of pre-pickup tail states to translate.")
    parser.add_argument("--semantics", choices=["both", "fraction", "distinct"],
                         default="both",
                         help="Both mass-accounting totals are always computed; "
                              "this flag only annotates the run config in the "
                              "output (application is always gated by fraction).")
    args = parser.parse_args()

    t0 = time.monotonic()
    print("Loading DataBundle...", flush=True)
    bundle = DataBundle.load()
    gx, gy, T = bundle.mask_3d.shape
    mask = bundle.mask_3d
    print(f"  loaded in {time.monotonic() - t0:.1f}s | grid {gx}x{gy}x{T} | "
          f"{len(bundle.trajectories)} trajectories", flush=True)

    # Running supply/demand grids (fraction-semantics convention -- always full
    # presence mass; models the real physical accumulation for interactions).
    S = bundle.active_taxis_3d.astype(np.float64).copy()
    D = bundle.pickup_3d.astype(np.float64).copy()

    # --- Disadvantaged-group definition (migrant axis, district extremes) ---
    demo = io.per_unit_demographics(bundle)
    g_unit = ef.region_extremes(demo["MigrantRatio"], disadvantaged_high=True)
    N_D = int((g_unit == 1).sum())
    if N_D == 0:
        raise RuntimeError("N_D == 0: disadvantaged-group definition produced no units.")

    # Dense (gx, gy, T) group grid for O(1) (x,y,t) -> group lookup. Equivalent
    # to (and faster than) a flat-unit-index indirection: -1 for inactive/
    # excluded-middle-third units, 0 = advantaged, 1 = disadvantaged.
    group_grid = np.full((gx, gy, T), -1, dtype=np.int8)
    group_grid[mask] = g_unit

    baseline_Y = bundle.active_taxis_3d.astype(np.float64)[mask] / np.maximum(
        bundle.pickup_3d.astype(np.float64)[mask], config.DEMAND_FLOOR)
    baseline_mean_Y_D = float(baseline_Y[g_unit == 1].mean())
    print(f"  N_D (active disadvantaged units) = {N_D} | "
          f"baseline mean(Y|D) = {baseline_mean_Y_D:.4f}", flush=True)

    # --- D-group CELL set for candidate filtering (any t; from the FULL grid,
    # including inactive cells -- a broader geometric proxy than group_grid). ---
    sel = io._enriched_selected_grid()
    cell_group = ef.region_extremes(sel[:, :, 2].ravel(),
                                     disadvantaged_high=True).reshape(gx, gy)

    # Dilate D-group cells by the 5x5 (Chebyshev <=2) footprint once, for O(1)
    # "does this cell's tail pass within eps=2 of a D cell" candidate checks.
    d_bool = (cell_group == 1)
    padded = np.pad(d_bool, 2, mode="constant", constant_values=False)
    near_D = np.zeros((gx, gy), dtype=bool)
    for di in range(5):
        for dj in range(5):
            near_D |= padded[di:di + gx, dj:dj + gy]
    print(f"  D-group cells: {int(d_bool.sum())} / {gx * gy}; "
          f"near-D (dilated) cells: {int(near_D.sum())} / {gx * gy}", flush=True)

    # --- Per-driver seeking-presence index (exact (x,y,t_block), from ALL
    # states of ALL trajectories) for the distinct-seeking semantics. ---
    presence = defaultdict(set)
    for tr in bundle.trajectories:
        for s in tr.states:
            tb = hour_to_block_index(time_bucket_to_hour(s.time_bucket))
            presence[int(tr.driver_id)].add((int(s.x_grid), int(s.y_grid), tb))
    print(f"  presence index built for {len(presence)} drivers", flush=True)

    # ---------------------------------------------------------------------
    # Candidate scan: score all 24 deltas (against the ORIGINAL grids) for
    # every trajectory whose tail passes near a D-group cell; keep the best.
    # ---------------------------------------------------------------------
    t_scan = time.monotonic()
    candidates = []
    n_len_ge3 = 0
    for idx, tr in enumerate(bundle.trajectories):
        tail = tail_states(tr, args.tail_len)
        if not tail:
            continue
        n_len_ge3 += 1
        xs = [int(s.x_grid) for s in tail]
        ys = [int(s.y_grid) for s in tail]
        if not any(near_D[x, y] for x, y in zip(xs, ys)):
            continue

        tbs = [hour_to_block_index(time_bucket_to_hour(s.time_bucket)) for s in tail]
        n_tail = len(tail)
        states_info = [(xs[i], ys[i], tbs[i], i == n_tail - 1) for i in range(n_tail)]
        driver = int(tr.driver_id)

        best_score, best_delta = -np.inf, None
        for delta in DELTAS:
            sf, _sd, *_ = evaluate_delta(states_info, driver, delta, S, D,
                                          group_grid, presence, bundle, gx, gy)
            if sf > best_score:
                best_score, best_delta = sf, delta

        candidates.append({
            "traj_idx": idx, "trajectory_id": tr.trajectory_id, "driver": driver,
            "states": states_info, "delta": best_delta, "init_score": best_score,
            "tail_len": n_tail - 1,
        })
        if len(candidates) % 5000 == 0:
            print(f"  scanned {idx + 1}/{len(bundle.trajectories)} trajectories, "
                  f"{len(candidates)} candidates so far "
                  f"({time.monotonic() - t_scan:.1f}s)", flush=True)

    n_candidates = len(candidates)
    t_scan_done = time.monotonic()
    print(f"Scan done: {n_len_ge3} trajectories with len>=3, {n_candidates} "
          f"candidates (near a D-group cell), scan took "
          f"{t_scan_done - t_scan:.1f}s", flush=True)

    # ---------------------------------------------------------------------
    # Greedy apply: best-first by initial score; re-score against the RUNNING
    # grids at pop time; skip (discard, do not re-queue) if gain <= 0; stop at
    # budget applied edits.
    # ---------------------------------------------------------------------
    candidates.sort(key=lambda c: c["init_score"], reverse=True)

    total_gain_fraction = 0.0
    total_gain_distinct = 0.0
    n_applied = 0
    n_skipped = 0
    examples = []

    for c in candidates:
        if n_applied >= args.budget:
            break
        sf, sd, removal, addition, p_old, p_new, p_t = evaluate_delta(
            c["states"], c["driver"], c["delta"], S, D, group_grid, presence,
            bundle, gx, gy,
        )
        if sf <= 0:
            n_skipped += 1
            continue
        commit_edit(S, D, bundle, removal, addition, p_old, p_new, p_t)
        total_gain_fraction += sf
        total_gain_distinct += sd
        n_applied += 1
        examples.append({
            "trajectory_id": c["trajectory_id"], "driver": c["driver"],
            "tail_len": c["tail_len"], "delta": list(c["delta"]),
            "pickup_old": list(p_old) if p_old else None,
            "pickup_new": list(p_new) if p_new else None,
            "fraction_gain": sf, "distinct_gain": sd,
            "apply_rank": n_applied,
        })
        if n_applied % 2000 == 0:
            print(f"  applied {n_applied}/{args.budget} edits "
                  f"({time.monotonic() - t_scan_done:.1f}s)", flush=True)

    runtime_s = time.monotonic() - t0
    ceiling_fraction = total_gain_fraction / N_D
    ceiling_distinct_seeking = total_gain_distinct / N_D

    print(f"Applied {n_applied} edits (skipped {n_skipped}); "
          f"ceiling_fraction={ceiling_fraction:.4f}, "
          f"ceiling_distinct_seeking={ceiling_distinct_seeking:.4f} "
          f"| total runtime {runtime_s:.1f}s", flush=True)

    examples.sort(key=lambda e: e["fraction_gain"], reverse=True)
    top_examples = examples[:20]

    result = {
        "budget": args.budget, "tail_len": args.tail_len,
        "semantics_arg": args.semantics,
        "n_candidates": n_candidates, "n_applied": n_applied,
        "n_skipped": n_skipped,
        "n_trajectories_len_ge3": n_len_ge3,
        "n_trajectories_total": len(bundle.trajectories),
        "N_D": N_D,
        "ceiling_fraction": ceiling_fraction,
        "ceiling_distinct_seeking": ceiling_distinct_seeking,
        "baseline_mean_Y_D": baseline_mean_Y_D,
        "g0_threshold": G0_THRESHOLD,
        "runtime_seconds": runtime_s,
        "top_examples": top_examples,
        "notes": ("Greedy upper bound (not a true optimum): best-first over an "
                  "initial per-trajectory 24-delta scan, re-scored once against "
                  "running grids at pop time, applied if re-scored gain > 0. "
                  "Net of the pickup's relocated demand mass. distinct_seeking "
                  "is an approximation from a static original-corpus presence "
                  "index (own-cell removal always conservatively gated to zero); "
                  "exact raw-GPS recount is deferred to Task 9."),
    }

    OUT.mkdir(exist_ok=True)
    (OUT / "oracle.json").write_text(json.dumps(result, indent=1))

    gate_pass = (ceiling_fraction >= G0_THRESHOLD) or (ceiling_distinct_seeking >= G0_THRESHOLD)
    lines = [
        "# Stage-0 supply-lift oracle (gate G0)",
        "",
        f"Budget: {args.budget} edits | tail_len: {args.tail_len} | "
        f"runtime: {runtime_s:.1f}s",
        "",
        "## Headline numbers",
        "",
        f"- **baseline mean(Y|D)**: {baseline_mean_Y_D:.4f}",
        f"- **ceiling (fraction semantics)**: {ceiling_fraction:+.4f}",
        f"- **ceiling (distinct-seeking semantics)**: {ceiling_distinct_seeking:+.4f}",
        f"- **G0 threshold**: Delta mean(Y|D) >= ~+{G0_THRESHOLD}",
        f"- **Gate result**: {'PASS' if gate_pass else 'FAIL'} "
        f"(fraction {'>=':s} threshold: {ceiling_fraction >= G0_THRESHOLD}; "
        f"distinct-seeking >= threshold: {ceiling_distinct_seeking >= G0_THRESHOLD})",
        "",
        f"- n_candidates (tail near a D-group cell): {n_candidates} "
        f"(of {n_len_ge3} trajectories with len>=3, {len(bundle.trajectories)} total)",
        f"- n_applied: {n_applied} | n_skipped (re-scored gain <= 0): {n_skipped}",
        f"- N_D (fixed active disadvantaged-unit count): {N_D}",
        "",
        "## Methodology (see supply_lift_oracle.py docstring for full detail)",
        "",
        "Greedy upper bound on lifting mean(Y|D) via rigid translation of each "
        "trajectory's seeking tail (last min(tail_len, len-2) states + pickup; "
        "anchor state untouched). For each trajectory whose tail passes within "
        "Chebyshev distance 2 of a disadvantaged-group cell (migrant-ratio "
        "district extremes), the best of 24 discrete deltas is scored against "
        "the running supply/demand grids: supply mass added at the new 5x5 "
        "neighborhoods, removed at the old ones, and the pickup's demand mass "
        "relocated with it (Y = S'/max(D', 0.5)). Edits are applied best-first "
        "up to the budget, re-scoring at pop time; a candidate whose re-scored "
        "gain is <= 0 is discarded, not re-queued. Two mass-accounting "
        "semantics are reported: `fraction` (every moved state's mass counts) "
        "and `distinct-seeking` (mass counts only where it changes a driver's "
        "occupancy of a neighborhood-hour, per a static presence index -- an "
        "approximation; exact raw-GPS recount is Task 9).",
        "",
        "## Top-20 example edits (by fraction gain)",
        "",
        "Gains below are the RAW per-edit sum of (Y' - Y) over affected active "
        "D-group units (NOT yet divided by N_D); large single-edit values "
        "reflect the demand floor (0.5) nonlinearity, not an accounting bug -- "
        "the CEILING numbers above are the properly N_D-normalized totals.",
        "",
        "| rank | trajectory_id | driver | tail_len | delta | pickup old->new "
        "| fraction gain (raw) | distinct gain (raw) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in top_examples:
        lines.append(
            f"| {e['apply_rank']} | {e['trajectory_id']} | {e['driver']} | "
            f"{e['tail_len']} | {tuple(e['delta'])} | "
            f"{tuple(e['pickup_old'])} -> {tuple(e['pickup_new'])} | "
            f"{e['fraction_gain']:.5f} | {e['distinct_gain']:.5f} |"
        )
    (OUT / "oracle_report.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT / 'oracle.json'} and {OUT / 'oracle_report.md'}", flush=True)


if __name__ == "__main__":
    main()
