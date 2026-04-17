"""Report generator: reads {output_dir} from disk and writes report.md."""

from __future__ import annotations
import csv
import json
from pathlib import Path


def render(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    metrics = json.loads((output_dir / "metrics.json").read_text())

    lines: list[str] = []
    _header(lines, metrics)
    _config_table(lines, metrics)
    _dataset_summary(lines, metrics)
    _fairness_table(lines, metrics)
    _convergence_summary(lines, metrics)
    if metrics.get("diagnostics_enabled"):
        _diagnostics_summary(lines, metrics)
    _top_k_table(lines, output_dir / "trajectories.csv")
    _key_findings(lines, metrics)
    _artifact_index(lines, metrics)

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def _header(lines, m):
    lines.append(f"# Experiment Report - `{m['experiment_id']}`\n")
    lines.append(f"- **Timestamp (UTC):** {m['timestamp_utc']}")
    lines.append(f"- **Git SHA:** `{m['git_sha']}`"
                 + ("  **(dirty)**" if m.get("git_dirty") else ""))
    lines.append(f"- **Command line:** `{m['command_line']}`")
    lines.append("")


def _config_table(lines, m):
    lines.append("## Config\n")
    lines.append("| Param | Value |")
    lines.append("|---|---|")
    overridden = set(m.get("config_overrides", {}).keys())
    for k, v in sorted(m["config_snapshot"].items()):
        key_cell = f"**{k}**" if k in overridden else k
        val_cell = f"**{v}**"  if k in overridden else str(v)
        lines.append(f"| {key_cell} | {val_cell} |")
    lines.append("")


def _dataset_summary(lines, m):
    ds = m.get("dataset", {})
    lines.append("## Dataset\n")
    lines.append("| n_trajectories | n_drivers | n_active_units | k_modified |")
    lines.append("|---|---|---|---|")
    lines.append(f"| {ds.get('n_trajectories', '-')} | {ds.get('n_drivers', '-')} "
                 f"| {ds.get('n_active_units', '-')} | {m.get('k_modified', '-')} |")
    lines.append("")


def _fairness_table(lines, m):
    mb = m["metrics_before"]; ma = m["metrics_after"]; d = m["deltas"]
    def _arrow(delta):
        if delta > 1e-6:  return "up"
        if delta < -1e-6: return "down"
        return "-"
    lines.append("## Fairness\n")
    lines.append("| Metric | Before | After | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k in ("f_spatial", "f_causal", "gini_dsr", "gini_asr"):
        lines.append(f"| `{k}` | {mb[k]:.4f} | {ma[k]:.4f} | "
                     f"{d[k]:+.4f} {_arrow(d[k])} |")
    lines.append("")


def _convergence_summary(lines, m):
    cs = m.get("convergence_summary", {})
    total = cs.get("n_converged", 0) + cs.get("n_max_iter", 0)
    lines.append("## Convergence\n")
    lines.append(f"- Converged: {cs.get('n_converged')} / {total}")
    lines.append(f"- Mean total iterations: {cs.get('mean_total_iterations', 0.0):.2f}")
    lines.append(f"- Mean final gradient norm: {cs.get('mean_final_grad_norm', 0.0):.4f}")
    lines.append("")


def _diagnostics_summary(lines, m):
    ds = m.get("diagnostics_summary")
    lines.append("## Gradient diagnostics\n")
    if ds is None:
        lines.append("_No diagnostics captured (top-k was empty or histories produced no iterations)._\n")
        return
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k in ("mean_grad_spatial_norm", "mean_grad_causal_norm", "mean_grad_fidelity_norm",
              "mean_cos_spatial_causal", "mean_cos_fairness_fidelity",
              "frac_iters_spatial_dominant", "frac_iters_causal_dominant",
              "frac_iters_fidelity_dominant"):
        v = ds.get(k)
        lines.append(f"| `{k}` | {'' if v is None else f'{v:.4f}'} |")
    lines.append("")


def _top_k_table(lines, csv_path: Path):
    lines.append("## Top 10 modified trajectories\n")
    if not csv_path.exists():
        lines.append("_No trajectories._\n")
        return
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        lines.append("_No trajectories._\n")
        return
    cols = ["rank", "trajectory_id", "driver_id",
            "original_pickup_cell_x", "original_pickup_cell_y",
            "modified_pickup_cell_x", "modified_pickup_cell_y",
            "delta_x", "delta_y",
            "converged", "total_iterations",
            "initial_objective", "final_objective"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows[:10]:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    lines.append("")


def _key_findings(lines, m):
    d = m["deltas"]
    findings: list[str] = []
    if d["f_spatial"] > 0:
        findings.append(f"F_spatial improved by {d['f_spatial']:+.4f}.")
    elif d["f_spatial"] < 0:
        findings.append(f"F_spatial regressed by {d['f_spatial']:+.4f}.")
    if d["f_causal"] > 0:
        findings.append(f"F_causal improved by {d['f_causal']:+.4f}.")
    elif d["f_causal"] < 0:
        findings.append(f"F_causal regressed by {d['f_causal']:+.4f}.")
    if abs(d["gini_asr"]) < 1e-6:
        findings.append("ASR Gini unchanged - only pickups are modified by the framework.")
    if m.get("diagnostics_enabled") and m.get("diagnostics_summary"):
        ds = m["diagnostics_summary"]
        fracs = [
            ("spatial", ds.get("frac_iters_spatial_dominant") or 0.0),
            ("causal",  ds.get("frac_iters_causal_dominant")  or 0.0),
            ("fidelity",ds.get("frac_iters_fidelity_dominant")or 0.0),
        ]
        # Skip the dominance finding when every fraction is effectively zero —
        # this happens at convergence or with fully-zero gradients, and a
        # "dominant at 0.0%" bullet is misleading.
        if max(f for _, f in fracs) >= 1e-4:
            dom = max(fracs, key=lambda kv: kv[1])
            findings.append(f"Dominant gradient term: `{dom[0]}` in {dom[1]:.1%} of iterations.")
    lines.append("## Key findings\n")
    if not findings:
        lines.append("_No notable findings._\n")
    else:
        for f in findings:
            lines.append(f"- {f}")
    lines.append("")


def _artifact_index(lines, m):
    lines.append("## Artifacts\n")
    paths = m.get("artifact_paths", {})
    sizes = m.get("file_sizes_bytes", {})
    lines.append("| Artifact | Path | Size (bytes) |")
    lines.append("|---|---|---:|")
    for name, path in sorted(paths.items()):
        lines.append(f"| {name} | `{path}` | {sizes.get(name, '-')} |")
    lines.append("")
