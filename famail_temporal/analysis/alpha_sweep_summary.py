"""Post-process the α-Pareto weight sweep into a table + Pareto scatter.

Assembles the empirical (ΔF_spatial, ΔF_causal) frontier for
``PAPER/objective-motivation/MOTIVATION.md``'s "Why these weights" from the
sweep's per-point ``*_alpha_sweep_<tag>_filtered/metrics.json`` (written by
``famail_temporal/results/alpha_sweep/driver.sh``) plus the shipped
(0.2, 0.7, 0.1) trim+lift headline as the anchor point (its numbers are read
from its own metrics.json, never recomputed). α values come from each file's
``effective_alphas`` — no hardcoded tag→α map.

Partial-mode by design: missing sweep points are reported as PENDING and the
table/figure are built from whatever exists, so the tool can be exercised
mid-sweep and re-run unchanged when the sweep completes.

Usage::

    python -m famail_temporal.analysis.alpha_sweep_summary \\
        [--results-dir famail_temporal/results] \\
        [--anchor-dir <headline_filtered_dir>] \\
        [--out famail_temporal/results/alpha_sweep/summary]

Read-only on every input; writes ``alpha_sweep_summary.md`` +
``alpha_pareto.png`` + ``alpha_sweep_summary.json`` to --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# The five sweep tags, in driver.sh POINTS order (α_fidelity fixed at 0.1).
SWEEP_TAGS: Tuple[str, ...] = (
    "s00_c90_f10", "s10_c80_f10", "s35_c55_f10", "s55_c35_f10", "s80_c10_f10",
)
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_ANCHOR_DIR = (
    DEFAULT_RESULTS_DIR / "2026-07-08T14-03-03_supply_lift_v1_shz_primary_filtered"
)


def _row_from_metrics(path: Path, tag: str, is_anchor: bool) -> Dict:
    m = json.loads(path.read_text())
    a = m["effective_alphas"]
    return {
        "tag": tag,
        "alphas": (a["alpha_spatial"], a["alpha_causal"], a["alpha_fidelity"]),
        "d_f_spatial": float(m["deltas"]["f_spatial"]),
        "d_f_causal": float(m["deltas"]["f_causal"]),
        "is_anchor": is_anchor,
        "source": str(path.parent),
    }


def load_points(
    results_dir, tags: Sequence[str] = SWEEP_TAGS, anchor_dir=None,
) -> Tuple[List[Dict], List[str]]:
    """(rows sorted by alpha_spatial, pending tags). Newest dir wins per tag."""
    results_dir = Path(results_dir)
    rows: List[Dict] = []
    pending: List[str] = []
    for tag in tags:
        candidates = sorted(results_dir.glob(f"*_alpha_sweep_{tag}_filtered"))
        candidates = [c for c in candidates if (c / "metrics.json").exists()]
        if not candidates:
            pending.append(tag)
            continue
        rows.append(_row_from_metrics(candidates[-1] / "metrics.json", tag,
                                      is_anchor=False))
    if anchor_dir is not None:
        rows.append(_row_from_metrics(Path(anchor_dir) / "metrics.json",
                                      tag="headline", is_anchor=True))
    rows.sort(key=lambda r: r["alphas"][0])
    return rows, pending


def pareto_flags(rows: Sequence[Dict]) -> List[bool]:
    """True where no other row is >= on BOTH (ΔF_spatial, ΔF_causal) and > on one."""
    flags = []
    for r in rows:
        dominated = any(
            o is not r
            and o["d_f_spatial"] >= r["d_f_spatial"]
            and o["d_f_causal"] >= r["d_f_causal"]
            and (o["d_f_spatial"] > r["d_f_spatial"]
                 or o["d_f_causal"] > r["d_f_causal"])
            for o in rows
        )
        flags.append(not dominated)
    return flags


def shipped_criterion(rows: Sequence[Dict]) -> Optional[Dict]:
    """The documented weight-selection rule: max ΔF_causal s.t. ΔF_spatial >= 0."""
    feasible = [r for r in rows if r["d_f_spatial"] >= 0.0]
    if not feasible:
        return None
    return max(feasible, key=lambda r: r["d_f_causal"])


def render_table(rows: Sequence[Dict], pending: Sequence[str]) -> str:
    flags = pareto_flags(rows)
    best = shipped_criterion(rows)
    lines = [
        "# α-sweep — empirical (ΔF_spatial, ΔF_causal) frontier "
        "(SZ primary, supply-lift editor, k=10000, +infeasible-trim filter)",
        "",
        "| α (spatial, causal, fidelity) | ΔF_causal | ΔF_spatial | Pareto | source |",
        "|---|---:|---:|:---:|---|",
    ]
    for r, on_front in zip(rows, flags):
        label = f"({r['alphas'][0]:g}, {r['alphas'][1]:g}, {r['alphas'][2]:g})"
        if r["is_anchor"]:
            label += " ★ shipped"
        lines.append(
            f"| {label} | {r['d_f_causal']:+.4f} | {r['d_f_spatial']:+.4f} "
            f"| {'✓' if on_front else '—'} | `{Path(r['source']).name}` |")
    lines.append("")
    if best is not None:
        lines.append(
            f"**Weight-selection criterion** (max ΔF_causal s.t. ΔF_spatial ≥ 0) "
            f"selects **({best['alphas'][0]:g}, {best['alphas'][1]:g}, "
            f"{best['alphas'][2]:g})**"
            + (" — the shipped configuration." if best["is_anchor"] else "."))
    if pending:
        lines.append("")
        lines.append(
            f"**PENDING sweep points (not yet on disk):** {', '.join(pending)} — "
            "re-run this tool when the driver completes.")
    return "\n".join(lines)


def pareto_figure(rows: Sequence[Dict], out_path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flags = pareto_flags(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r, on_front in zip(rows, flags):
        marker = "*" if r["is_anchor"] else ("o" if on_front else "s")
        size = 180 if r["is_anchor"] else 45
        ax.scatter(r["d_f_spatial"], r["d_f_causal"], marker=marker, s=size,
                   zorder=3)
        label = f"({r['alphas'][0]:g}, {r['alphas'][1]:g})"
        if r["is_anchor"]:
            label += " shipped"
        ax.annotate(label, (r["d_f_spatial"], r["d_f_causal"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    front = sorted((r for r, f in zip(rows, flags) if f),
                   key=lambda r: r["d_f_spatial"])
    if len(front) > 1:
        ax.plot([r["d_f_spatial"] for r in front],
                [r["d_f_causal"] for r in front],
                ls="--", lw=0.8, color="grey", zorder=2)
    ax.axhline(0.0, color="grey", lw=0.6)
    ax.axvline(0.0, color="grey", lw=0.6)
    ax.set_xlabel("ΔF_spatial (after − before)")
    ax.set_ylabel("ΔF_causal (after − before)")
    ax.set_title("α-weight sweep — empirical Pareto (SZ, supply-lift editor)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.alpha_sweep_summary")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_RESULTS_DIR / "alpha_sweep" / "summary")
    args = ap.parse_args(argv)

    rows, pending = load_points(args.results_dir, anchor_dir=args.anchor_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "alpha_sweep_summary.md").write_text(render_table(rows, pending))
    pareto_figure(rows, args.out / "alpha_pareto.png")
    (args.out / "alpha_sweep_summary.json").write_text(json.dumps(
        {"rows": rows, "pending": pending}, indent=2, default=str))
    print(f"[alpha_sweep_summary] wrote {args.out} "
          f"({len(rows)} rows, {len(pending)} pending)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
