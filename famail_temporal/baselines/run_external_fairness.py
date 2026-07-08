"""Compute + report external fairness metrics before/after edit. See
docs/superpowers/plans/2026-07-02-external-fairness-metrics.md."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from famail_temporal.baselines import external_fairness as ef
from famail_temporal.baselines import external_fairness_io as io

GROUPINGS: Tuple[str, str] = ("district_extremes", "median_split")


def _groups_for(values: np.ndarray, axis: str, grouping: str) -> np.ndarray:
    high = io.DISADVANTAGED_HIGH[axis]
    if grouping == "district_extremes":
        return ef.region_extremes(values, disadvantaged_high=high)
    if grouping == "median_split":
        return ef.median_split(values, disadvantaged_high=high)
    raise ValueError(f"unknown grouping {grouping!r}")


def assemble_results(
    Y_before: np.ndarray, Y_after: np.ndarray,
    demo: Dict[str, np.ndarray], seed: int = 0, B: int = 1000,
) -> dict:
    regions = ef.regions_from_values([demo[a] for a in io.EQUITY_AXES])
    specs: List[Tuple[str, object, np.ndarray]] = [("theil", ef.theil_index, regions)]
    metrics: Dict[str, dict] = {}
    for axis in io.EQUITY_AXES:
        metrics[axis] = {}
        for g in GROUPINGS:
            groups = _groups_for(demo[axis], axis, g)
            metrics[axis][g] = {
                "group_sizes": {
                    "n_disadvantaged": int((groups == 1).sum()),
                    "n_advantaged": int((groups == 0).sum()),
                    "n_excluded": int((groups == -1).sum()),
                },
                "supply_demand_ratio": {
                    "before": ef.supply_demand_ratio(Y_before, groups),
                    "after": ef.supply_demand_ratio(Y_after, groups),
                    "delta_gap": ef.sdr_gap(Y_after, groups) - ef.sdr_gap(Y_before, groups),
                },
                "demographic_parity": {
                    "before": ef.demographic_parity(Y_before, groups),
                    "after": ef.demographic_parity(Y_after, groups),
                    "delta": (ef.demographic_parity(Y_after, groups)
                              - ef.demographic_parity(Y_before, groups)),
                },
                "disparate_impact": {
                    "before": ef.disparate_impact(Y_before, groups),
                    "after": ef.disparate_impact(Y_after, groups),
                    "delta": (ef.disparate_impact(Y_after, groups)
                              - ef.disparate_impact(Y_before, groups)),
                },
            }
            specs.append((f"dp::{axis}::{g}", ef.demographic_parity, groups))
            specs.append((f"di::{axis}::{g}", ef.disparate_impact, groups))
            specs.append((f"sdrgap::{axis}::{g}", ef.sdr_gap, groups))

    boot = ef.paired_bootstrap(Y_before, Y_after, specs, B=B, seed=seed)

    # attach CIs
    theil_before = ef.theil_index(Y_before, regions)
    theil_after = ef.theil_index(Y_after, regions)
    result = {
        "theil": {
            "before": theil_before, "after": theil_after,
            "delta": theil_after - theil_before,
            "delta_ci": boot["theil"]["delta"],
            "n_dropped": boot["theil"]["n_dropped"],
        },
        "metrics": metrics,
    }
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = metrics[axis][g]
            e["demographic_parity"]["delta_ci"] = boot[f"dp::{axis}::{g}"]["delta"]
            e["disparate_impact"]["delta_ci"] = boot[f"di::{axis}::{g}"]["delta"]
            e["supply_demand_ratio"]["gap_ci"] = boot[f"sdrgap::{axis}::{g}"]["delta"]
    return result


def _fmt(x) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def _fmt_ci(ci) -> str:
    lo, hi = ci
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def write_json(result: dict, out_dir, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, **result}
    path = out_dir / "external_fairness.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def render_markdown(result: dict, meta: dict) -> str:
    lines: List[str] = [f"# External fairness — {meta.get('dataset','')}", ""]
    lines.append(f"**Edit dir:** `{meta.get('edit_dir','')}`  ·  "
                 f"**B:** {meta.get('B','')}  ·  **seed:** {meta.get('seed','')}")
    lines.append("")
    t = result["theil"]
    lines.append("## Theil index (between-region, on Y)")
    lines.append("| Before | After | Delta | Δ 95% CI |")
    lines.append("|---:|---:|---:|---:|")
    lines.append(f"| {_fmt(t['before'])} | {_fmt(t['after'])} | "
                 f"{t['delta']:+.4f} | {_fmt_ci(t['delta_ci'])} |")
    lines.append("")
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = result["metrics"][axis][g]
            gs = e["group_sizes"]
            lines.append(f"## {axis} — {g}  "
                         f"(D={gs['n_disadvantaged']}, A={gs['n_advantaged']}, "
                         f"excl={gs['n_excluded']})")
            lines.append("| Metric | Before | After | Delta | Δ 95% CI |")
            lines.append("|---|---:|---:|---:|---:|")
            dp = e["demographic_parity"]
            di = e["disparate_impact"]
            sd = e["supply_demand_ratio"]
            sd_d_before = sd["before"]["mean_disadvantaged"]
            sd_d_after = sd["after"]["mean_disadvantaged"]
            sd_a_before = sd["before"]["mean_advantaged"]
            sd_a_after = sd["after"]["mean_advantaged"]
            lines.append(f"| Supply/demand ratio (disadvantaged) | {_fmt(sd_d_before)} | "
                         f"{_fmt(sd_d_after)} | {sd_d_after - sd_d_before:+.4f} | "
                         f"— |")
            lines.append(f"| Supply/demand ratio (advantaged) | {_fmt(sd_a_before)} | "
                         f"{_fmt(sd_a_after)} | {sd_a_after - sd_a_before:+.4f} | "
                         f"— |")
            lines.append(f"| Demographic parity | {_fmt(dp['before'])} | "
                         f"{_fmt(dp['after'])} | {dp['delta']:+.4f} | "
                         f"{_fmt_ci(dp['delta_ci'])} |")
            lines.append(f"| Disparate impact | {_fmt(di['before'])} | "
                         f"{_fmt(di['after'])} | {di['delta']:+.4f} | "
                         f"{_fmt_ci(di['delta_ci'])} |")
            lines.append("")
    return "\n".join(lines)


def render_combined_table(named_results: List[Tuple[str, dict]]) -> str:
    lines = ["# External fairness — cross-dataset comparison", "",
             "| Dataset | Theil Δ | DP Δ (migrant/extremes) | DI Δ (migrant/extremes) |",
             "|---|---:|---:|---:|"]
    for label, res in named_results:
        e = res["metrics"]["MigrantRatio"]["district_extremes"]
        lines.append(f"| {label} | {res['theil']['delta']:+.4f} | "
                     f"{e['demographic_parity']['delta']:+.4f} | "
                     f"{e['disparate_impact']['delta']:+.4f} |")
    return "\n".join(lines)


def write_figure(result: dict, out_dir, meta: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, float, float, float]] = []
    for axis in io.EQUITY_AXES:
        for g in GROUPINGS:
            e = result["metrics"][axis][g]
            for mname, key in (("DP", "demographic_parity"),
                               ("DI", "disparate_impact")):
                d = e[key]
                lo, hi = d["delta_ci"]
                rows.append((f"{mname} {axis[:6]}/{g[:4]}", d["delta"], lo, hi))
    rows.append(("Theil", result["theil"]["delta"],
                 *result["theil"]["delta_ci"]))

    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    # Percentile-bootstrap CIs are not guaranteed to strictly bracket the
    # full-sample point estimate, so clip error-bar half-widths at 0 to
    # avoid negative xerr values (which errorbar would draw backwards).
    los = [max(0.0, r[1] - r[2]) for r in rows]
    his = [max(0.0, r[3] - r[1]) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1))
    ax.errorbar(deltas, y, xerr=[los, his], fmt="o", capsize=3)
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Δ (after − before)")
    ax.set_title(f"External fairness Δ — {meta.get('dataset','')}")
    fig.tight_layout()
    path = out_dir / "external_fairness_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


import argparse
from famail_temporal import config
from famail_temporal.data.loader import DataBundle


def _run_one(edit_dir: Path, dataset: str, out_dir: Path,
             seed: int, B: int, delta_supply_path: Path | None = None) -> dict:
    bundle = DataBundle.load()
    Y_before = io.service_ratio_Y(bundle.pickup_3d, bundle)
    after_pickup = io.build_edited_pickup_3d(bundle, edit_dir)
    # Supply-lift edits persist delta_supply_3d.npz; when given, the AFTER
    # side's supply is S' = clip(S_base + delta_supply_3d, SUPPLY_FLOOR,
    # None) instead of the bundle's frozen (pre-edit) active_taxis_3d. The
    # BEFORE side always uses S_base (bundle.active_taxis_3d, via the
    # supply_3d=None default), matching the "before" convention elsewhere
    # in this harness (see supply_recount.py's S_tier1_after).
    supply_after = None
    if delta_supply_path is not None:
        delta_supply_3d = np.load(delta_supply_path)["delta_supply_3d"]
        supply_after = np.clip(
            bundle.active_taxis_3d + delta_supply_3d, config.SUPPLY_FLOOR, None,
        )
    Y_after = io.service_ratio_Y(after_pickup, bundle, supply_3d=supply_after)
    demo = io.per_unit_demographics(bundle)
    result = assemble_results(Y_before, Y_after, demo, seed=seed, B=B)
    meta = {"dataset": dataset, "city": config.CITY, "edit_dir": str(edit_dir),
            "seed": seed, "B": B, "n_active": int(bundle.mask_3d.sum())}
    write_json(result, out_dir, meta)
    (out_dir / "report.md").write_text(render_markdown(result, meta))
    write_figure(result, out_dir, meta)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.baselines.run_external_fairness")
    ap.add_argument("--edit-dir", type=Path, required=False,
                    help="Results dir with histories.pkl for the edit")
    ap.add_argument("--dataset", default=None,
                    help="Label for outputs (e.g. shenzhen-primary, sf12)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--delta-supply", type=Path, default=None,
                    help="Path to the edit's delta_supply_3d.npz; when given, the "
                         "AFTER-side supply is clip(S_base + delta_supply_3d, "
                         "SUPPLY_FLOOR, None) instead of S_base (BEFORE unaffected)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--combine", nargs="+", type=Path, default=None,
                    help="external_fairness.json paths to combine into one table")
    args = ap.parse_args(argv)

    if args.combine:
        named = []
        for p in args.combine:
            payload = json.loads(Path(p).read_text())
            named.append((payload["meta"].get("dataset", str(p)), payload))
        out = args.out_dir or Path(config.PACKAGE_ROOT) / "baselines" / \
            "external_fairness" / "results"
        out.mkdir(parents=True, exist_ok=True)
        (out / "combined.md").write_text(render_combined_table(named))
        print(f"wrote {out / 'combined.md'}")
        return 0

    if not args.edit_dir:
        ap.error("--edit-dir is required (unless --combine)")
    dataset = args.dataset or f"{config.CITY}"
    out_dir = args.out_dir or (Path(config.PACKAGE_ROOT) / "baselines" /
                               "external_fairness" / "results" / dataset)
    _run_one(args.edit_dir, dataset, out_dir, args.seed, args.bootstrap,
             delta_supply_path=args.delta_supply)
    print(f"wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
