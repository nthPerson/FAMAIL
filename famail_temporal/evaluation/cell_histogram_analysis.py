"""
Cell-histogram analysis of a completed experiment.

Reads the artifacts saved by ``evaluation.persistence.write`` and produces:
  - A multi-panel figure summarizing origin/destination/flow/concentration/distance/α-change.
  - A JSON stats summary (machine-readable for cross-experiment comparison).
  - A stdout digest with the headline numbers.

Invoke:
    python -m famail_temporal.evaluation.cell_histogram_analysis <exp_dir>

Designed to be re-run on every new experiment so cross-run comparison stays
mechanical. See docs/TRAJECTORY_EDITING_METHODOLOGY.md §8 for the analytical
framing the visualizations are designed to support.
"""

from __future__ import annotations
import argparse
import json
import pickle as _pkl
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from famail_temporal.data.aggregation import (
    hour_to_block_index, time_bucket_to_hour,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_exp(exp_dir: Path) -> dict:
    """Load all the per-experiment artifacts we need."""
    with open(exp_dir / "histories.pkl", "rb") as f:
        histories = _pkl.load(f)
    with open(exp_dir / "grid_before.pkl", "rb") as f:
        grid_before = _pkl.load(f)
    with open(exp_dir / "grid_after.pkl", "rb") as f:
        grid_after = _pkl.load(f)
    with open(exp_dir / "metrics.json") as f:
        metrics = json.load(f)
    return {
        "histories": histories,
        "grid_before": grid_before["grid"],          # (gx, gy, T, 4)
        "grid_after":  grid_after["grid"],
        "active_mask": grid_before["active_mask"],   # (gx, gy, T)
        "metrics": metrics,
    }


def _t_block_of(traj) -> int:
    """t_block index from a trajectory's pickup state (5-min bucket → hour → block)."""
    h = time_bucket_to_hour(traj.pickup_state.time_bucket)
    return hour_to_block_index(h)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_stats(data: dict) -> dict:
    """Compute the headline statistics used by both the figure and the digest."""
    histories = data["histories"]
    grid_before = data["grid_before"]
    grid_after = data["grid_after"]
    active_mask = data["active_mask"]

    # Per-trajectory origin and destination (cell, t_block)
    orig_cells: List[Tuple[int, int]] = []
    dest_cells: List[Tuple[int, int]] = []
    orig_units: List[Tuple[int, int, int]] = []
    dest_units: List[Tuple[int, int, int]] = []
    movement_dist: List[float] = []

    for h in histories:
        ox, oy = h.original.pickup_cell
        dx, dy = h.modified.pickup_cell
        t = _t_block_of(h.original)
        orig_cells.append((ox, oy))
        dest_cells.append((dx, dy))
        orig_units.append((ox, oy, t))
        dest_units.append((dx, dy, t))
        movement_dist.append(float(np.hypot(dx - ox, dy - oy)))

    orig_cell_counts = Counter(orig_cells)
    dest_cell_counts = Counter(dest_cells)
    orig_unit_counts = Counter(orig_units)
    dest_unit_counts = Counter(dest_units)

    # α_causal before/after at destination units — the sign-flip diagnostic
    delta_alpha_at_dest: List[float] = []
    alpha_before_at_dest: List[float] = []
    alpha_after_at_dest:  List[float] = []
    for h in histories:
        dx, dy = h.modified.pickup_cell
        t = _t_block_of(h.original)
        if active_mask[dx, dy, t]:
            a_before = float(grid_before[dx, dy, t, 1])  # channel 1 = causal_attr
            a_after  = float(grid_after [dx, dy, t, 1])
            alpha_before_at_dest.append(a_before)
            alpha_after_at_dest.append(a_after)
            delta_alpha_at_dest.append(a_after - a_before)

    movement_dist = np.array(movement_dist)
    max_d = float(movement_dist.max()) if movement_dist.size else 0.0
    n_at_eps_ball = int(np.sum(np.abs(movement_dist - max_d) < 1e-6))

    pileup_per_dest_cell_hist = Counter(dest_cell_counts.values())
    pileup_per_dest_unit_hist = Counter(dest_unit_counts.values())

    stats = {
        "experiment_id": data["metrics"].get("experiment_id"),
        "config_overrides": data["metrics"].get("config_overrides", {}),
        "n_modified": len(histories),
        "n_converged": sum(1 for h in histories if h.converged),
        "metrics_before": data["metrics"]["metrics_before"],
        "metrics_after":  data["metrics"]["metrics_after"],
        "deltas":         data["metrics"]["deltas"],
        "effective_alphas": data["metrics"]["effective_alphas"],
        "n_unique_orig_cells": len(orig_cell_counts),
        "n_unique_dest_cells": len(dest_cell_counts),
        "n_unique_orig_units": len(orig_unit_counts),
        "n_unique_dest_units": len(dest_unit_counts),
        "max_trajs_per_orig_cell": max(orig_cell_counts.values()),
        "max_trajs_per_dest_cell": max(dest_cell_counts.values()),
        "max_trajs_per_orig_unit": max(orig_unit_counts.values()),
        "max_trajs_per_dest_unit": max(dest_unit_counts.values()),
        "movement_distance": {
            "mean":   float(movement_dist.mean()),
            "median": float(np.median(movement_dist)),
            "max":    float(movement_dist.max()),
            "min":    float(movement_dist.min()),
            "std":    float(movement_dist.std()),
            "n_at_max": n_at_eps_ball,
            "pct_at_max": float(n_at_eps_ball / len(movement_dist)),
        },
        "alpha_causal_at_dest": {
            "mean_before": float(np.mean(alpha_before_at_dest)) if alpha_before_at_dest else None,
            "mean_after":  float(np.mean(alpha_after_at_dest))  if alpha_after_at_dest  else None,
            "mean_delta":  float(np.mean(delta_alpha_at_dest))  if delta_alpha_at_dest  else None,
            "sign_flips_pos_to_neg": int(sum(
                1 for b, a in zip(alpha_before_at_dest, alpha_after_at_dest)
                if b > 0 and a < 0
            )),
            "sign_flips_neg_to_pos": int(sum(
                1 for b, a in zip(alpha_before_at_dest, alpha_after_at_dest)
                if b < 0 and a > 0
            )),
            "n_evaluable": len(delta_alpha_at_dest),
        },
        "top_10_dest_cells": [
            {"cell": list(c), "n_trajs": n}
            for c, n in dest_cell_counts.most_common(10)
        ],
        "top_10_dest_units": [
            {"cell_t": list(u), "n_trajs": n}
            for u, n in dest_unit_counts.most_common(10)
        ],
        "pileup_per_dest_cell_hist": {
            int(k): int(v) for k, v in sorted(pileup_per_dest_cell_hist.items())
        },
        "pileup_per_dest_unit_hist": {
            int(k): int(v) for k, v in sorted(pileup_per_dest_unit_hist.items())
        },
    }

    # Stash arrays needed for plotting (not serialized to JSON)
    stats["_raw"] = {
        "orig_cell_counts": orig_cell_counts,
        "dest_cell_counts": dest_cell_counts,
        "orig_unit_counts": orig_unit_counts,
        "dest_unit_counts": dest_unit_counts,
        "movement_dist": movement_dist,
        "alpha_before_at_dest": np.array(alpha_before_at_dest),
        "alpha_after_at_dest":  np.array(alpha_after_at_dest),
        "delta_alpha_at_dest":  np.array(delta_alpha_at_dest),
    }
    return stats


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _density_grid(counts: Counter, gx: int, gy: int) -> np.ndarray:
    """Project a Counter[(x, y)] → (gx, gy) ndarray of counts."""
    g = np.zeros((gx, gy), dtype=np.int32)
    for (x, y), n in counts.items():
        g[x, y] = n
    return g


def plot_summary(data: dict, stats: dict, out_path: Path) -> None:
    """Six-panel figure that tells the whole story of an experiment's modifier behavior."""
    grid_before = data["grid_before"]
    grid_after = data["grid_after"]
    raw = stats["_raw"]

    gx, gy = grid_before.shape[:2]
    alpha_before_xy = np.nansum(grid_before[..., 1], axis=2)  # Σ_t α_causal
    alpha_after_xy  = np.nansum(grid_after [..., 1], axis=2)
    delta_alpha_xy  = alpha_after_xy - alpha_before_xy
    v_abs_a = np.nanmax(np.abs(alpha_before_xy))
    v_abs_d = np.nanmax(np.abs(delta_alpha_xy)) if np.any(delta_alpha_xy) else v_abs_a

    orig_grid = _density_grid(raw["orig_cell_counts"], gx, gy)
    dest_grid = _density_grid(raw["dest_cell_counts"], gx, gy)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.30)

    # Panel A — origin density over α_causal_before
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(alpha_before_xy.T, origin="lower", cmap="RdBu_r",
               vmin=-v_abs_a, vmax=+v_abs_a, aspect="equal")
    xs, ys = np.where(orig_grid > 0)
    counts = orig_grid[xs, ys]
    axA.scatter(xs, ys, s=20 + 25 * np.sqrt(counts), c="black",
                edgecolors="white", linewidths=0.5, alpha=0.85, zorder=5)
    axA.set_title("A. Origin cells over Σ_t α_causal (before)\n"
                  "bubble area ∝ # trajectories with this origin")
    axA.set_xlabel("cell x");  axA.set_ylabel("cell y")

    # Panel B — destination density over α_causal_after
    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(alpha_after_xy.T, origin="lower", cmap="RdBu_r",
               vmin=-v_abs_a, vmax=+v_abs_a, aspect="equal")
    xs, ys = np.where(dest_grid > 0)
    counts = dest_grid[xs, ys]
    axB.scatter(xs, ys, s=20 + 25 * np.sqrt(counts), c="lime",
                edgecolors="black", linewidths=0.5, alpha=0.85, zorder=5)
    axB.set_title("B. Destination cells over Σ_t α_causal (after)\n"
                  "bubble area ∝ # trajectories with this destination")
    axB.set_xlabel("cell x");  axB.set_ylabel("cell y")

    # Panel C — Δ Σ_t α_causal heatmap with both clouds overlaid
    axC = fig.add_subplot(gs[0, 2])
    im = axC.imshow(delta_alpha_xy.T, origin="lower", cmap="RdBu_r",
                    vmin=-v_abs_d, vmax=+v_abs_d, aspect="equal")
    xs, ys = np.where(orig_grid > 0)
    axC.scatter(xs, ys, s=10, c="black", alpha=0.6, zorder=4, label="orig")
    xs, ys = np.where(dest_grid > 0)
    axC.scatter(xs, ys, s=10, c="lime", edgecolors="black",
                linewidths=0.3, alpha=0.7, zorder=5, label="dest")
    axC.set_title("C. ΔΣ_t α_causal (after − before)\n"
                  "blue = cells that became LESS fair globally")
    axC.set_xlabel("cell x");  axC.set_ylabel("cell y")
    axC.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=axC, fraction=0.025)

    # Panel D — cell-level pile-up histogram
    axD = fig.add_subplot(gs[1, 0])
    hist_cell = stats["pileup_per_dest_cell_hist"]
    if hist_cell:
        keys = sorted(hist_cell.keys())
        vals = [hist_cell[k] for k in keys]
        bars = axD.bar(keys, vals, color="lime", edgecolor="black", alpha=0.8)
        for b, v in zip(bars, vals):
            if v > 0:
                axD.text(b.get_x() + b.get_width() / 2, v,
                         str(v), ha="center", va="bottom", fontsize=7)
        if max(vals) > 50:
            axD.set_yscale("log")
    axD.set_xlabel("# trajectories landing on the same destination cell")
    axD.set_ylabel("# destination cells (log if max > 50)")
    axD.set_title(f"D. Cell-level pile-up at destinations\n"
                  f"max={stats['max_trajs_per_dest_cell']} trajs on a single cell, "
                  f"{stats['n_unique_dest_cells']} unique receivers")

    # Panel E — (cell, t_block)-level pile-up histogram
    axE = fig.add_subplot(gs[1, 1])
    hist_unit = stats["pileup_per_dest_unit_hist"]
    if hist_unit:
        keys = sorted(hist_unit.keys())
        vals = [hist_unit[k] for k in keys]
        bars = axE.bar(keys, vals, color="goldenrod", edgecolor="black", alpha=0.8)
        for b, v in zip(bars, vals):
            if v > 0:
                axE.text(b.get_x() + b.get_width() / 2, v,
                         str(v), ha="center", va="bottom", fontsize=7)
        if max(vals) > 50:
            axE.set_yscale("log")
    axE.set_xlabel("# trajectories landing on the same (cell, t_block)")
    axE.set_ylabel("# destination units (log if max > 50)")
    axE.set_title(f"E. Unit-level pile-up at destinations\n"
                  f"max={stats['max_trajs_per_dest_unit']} trajs on a single (cell, t)")

    # Panel F — movement distance histogram
    axF = fig.add_subplot(gs[1, 2])
    dist = raw["movement_dist"]
    if dist.size:
        bins = np.linspace(0, max(dist.max(), 1) + 0.5, 30)
        axF.hist(dist, bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
        axF.axvline(dist.mean(), color="red", linestyle="--",
                    label=f"mean = {dist.mean():.2f}")
        axF.axvline(dist.max(), color="black", linestyle=":",
                    label=f"max = {dist.max():.2f}")
    axF.set_xlabel("Δ position (cells, Euclidean)")
    axF.set_ylabel("# trajectories")
    axF.set_title(f"F. Modification distance\n"
                  f"{stats['movement_distance']['pct_at_max']:.0%} of trajectories at ε-ball boundary")
    axF.legend(fontsize=8)

    # Panel G — α_causal at destination: before vs after scatter
    axG = fig.add_subplot(gs[2, 0])
    ab = raw["alpha_before_at_dest"]
    aa = raw["alpha_after_at_dest"]
    if ab.size:
        axG.scatter(ab, aa, alpha=0.35, s=18, c="steelblue")
        lim = max(np.abs(ab).max(), np.abs(aa).max()) * 1.05 if ab.size else 1
        axG.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, alpha=0.6, label="y = x")
        axG.axhline(0, color="gray", linewidth=0.5)
        axG.axvline(0, color="gray", linewidth=0.5)
        axG.set_xlim(-lim, lim);  axG.set_ylim(-lim, lim)
    axG.set_xlabel("α_causal at destination (before)")
    axG.set_ylabel("α_causal at destination (after)")
    sf_pn = stats["alpha_causal_at_dest"]["sign_flips_pos_to_neg"]
    sf_np = stats["alpha_causal_at_dest"]["sign_flips_neg_to_pos"]
    axG.set_title(f"G. α_causal at destination units: before → after\n"
                  f"sign flips: pos→neg = {sf_pn},  neg→pos = {sf_np}")
    axG.legend(fontsize=8)

    # Panel H — top-10 destination cells, count + α-change
    axH = fig.add_subplot(gs[2, 1:])
    axH.set_title("H. Top-10 destination cells: trajectory count + α_causal change")
    axH.axis("off")
    table_rows = [["rank", "cell (x,y)", "n trajs", "Σ_t α_ca before", "Σ_t α_ca after", "Δ"]]
    for i, item in enumerate(stats["top_10_dest_cells"]):
        cx, cy = item["cell"]
        ab_v = alpha_before_xy[cx, cy]
        aa_v = alpha_after_xy[cx, cy]
        table_rows.append([
            str(i + 1),
            f"({cx}, {cy})",
            str(item["n_trajs"]),
            f"{ab_v:+.3e}",
            f"{aa_v:+.3e}",
            f"{aa_v - ab_v:+.3e}",
        ])
    table = axH.table(cellText=table_rows[1:], colLabels=table_rows[0],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)

    fig.suptitle(
        f"Cell-histogram analysis — {stats['experiment_id']}\n"
        f"n_modified={stats['n_modified']}  "
        f"unique origins={stats['n_unique_orig_cells']} cells / "
        f"{stats['n_unique_orig_units']} (cell,t)  "
        f"unique dests={stats['n_unique_dest_cells']} cells / "
        f"{stats['n_unique_dest_units']} (cell,t)  "
        f"ΔF_sp={stats['deltas']['f_spatial']:+.3e}  "
        f"ΔF_ca={stats['deltas']['f_causal']:+.3e}",
        fontsize=12, y=0.995,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

def print_digest(stats: dict) -> None:
    """Print a concise stdout summary."""
    eid = stats["experiment_id"]
    print(f"=== cell-histogram analysis: {eid} ===")
    print(f"n_modified            : {stats['n_modified']}")
    print(f"n_converged           : {stats['n_converged']}")
    print(f"effective alphas      : {stats['effective_alphas']}")
    print()
    print("=== concentration ===")
    print(f"unique origin cells   : {stats['n_unique_orig_cells']}")
    print(f"unique origin units   : {stats['n_unique_orig_units']}  (cell, t_block)")
    print(f"unique dest cells     : {stats['n_unique_dest_cells']}")
    print(f"unique dest units     : {stats['n_unique_dest_units']}  (cell, t_block)")
    print(f"max trajs / orig cell : {stats['max_trajs_per_orig_cell']}")
    print(f"max trajs / orig unit : {stats['max_trajs_per_orig_unit']}")
    print(f"max trajs / dest cell : {stats['max_trajs_per_dest_cell']}")
    print(f"max trajs / dest unit : {stats['max_trajs_per_dest_unit']}")
    print()
    print("=== movement distance (cells) ===")
    md = stats["movement_distance"]
    print(f"mean / median / max   : {md['mean']:.2f} / {md['median']:.2f} / {md['max']:.2f}")
    print(f"at ε-ball boundary    : {md['n_at_max']} / {stats['n_modified']}  ({md['pct_at_max']:.1%})")
    print()
    print("=== α_causal at destinations (sign-flip diagnostic) ===")
    a = stats["alpha_causal_at_dest"]
    if a["mean_before"] is not None:
        print(f"mean α before         : {a['mean_before']:+.3e}")
        print(f"mean α after          : {a['mean_after']:+.3e}")
        print(f"mean Δα               : {a['mean_delta']:+.3e}")
    print(f"sign flips pos→neg    : {a['sign_flips_pos_to_neg']}  (destinations that *became* unfair)")
    print(f"sign flips neg→pos    : {a['sign_flips_neg_to_pos']}  (destinations that *became* fair)")
    print()
    print("=== top-10 destination cells ===")
    print(f"{'rank':>4}  {'cell':>10}  {'n_trajs':>7}")
    for i, item in enumerate(stats["top_10_dest_cells"]):
        print(f"{i + 1:>4}  ({item['cell'][0]:>2},{item['cell'][1]:>3})  {item['n_trajs']:>7}")
    print()
    print("=== top-10 destination (cell, t_block) units ===")
    print(f"{'rank':>4}  {'cell, t':>14}  {'n_trajs':>7}")
    for i, item in enumerate(stats["top_10_dest_units"]):
        cx, cy, t = item["cell_t"]
        print(f"{i + 1:>4}  ({cx:>2},{cy:>3}, t={t:>2})  {item['n_trajs']:>7}")
    print()
    print("=== F-metric deltas (global) ===")
    print(f"ΔF_spatial            : {stats['deltas']['f_spatial']:+.3e}")
    print(f"ΔF_causal             : {stats['deltas']['f_causal']:+.3e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(exp_dir: Path) -> dict:
    """Full analysis pipeline. Returns stats dict; writes figure + JSON to exp_dir."""
    data = _load_exp(exp_dir)
    stats = compute_stats(data)
    plot_summary(data, stats, exp_dir / "cell_histogram_analysis.png")

    stats_json = {k: v for k, v in stats.items() if not k.startswith("_")}
    with open(exp_dir / "cell_histogram_summary.json", "w") as f:
        json.dump(stats_json, f, indent=2)

    print_digest(stats)
    print()
    print(f"plot saved: {exp_dir / 'cell_histogram_analysis.png'}")
    print(f"json saved: {exp_dir / 'cell_histogram_summary.json'}")
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="famail_temporal.evaluation.cell_histogram_analysis",
        description=__doc__,
    )
    p.add_argument("exp_dir", type=Path,
                   help="Path to an experiment results directory.")
    args = p.parse_args(argv)
    if not args.exp_dir.is_dir():
        raise SystemExit(f"not a directory: {args.exp_dir}")
    analyze(args.exp_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
