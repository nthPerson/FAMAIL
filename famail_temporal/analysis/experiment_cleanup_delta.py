"""E22 (experiment-level) — dirty-vs-clean robustness comparison.

Reads the headline numbers from each experiment's DIRTY (pre-cleanup) and CLEAN
(post-cleanup) result dirs and emits a side-by-side table showing every pillar of
the argument survives the stuck-GPS data cleanup. Read-only over existing result
JSONs (no source_data access, no recompute).

Schemas handled (per the runners):
  - L1-v2:   level1_v2_metrics.json -> sources.{src}.{f_causal,f_spatial}  (scalars)
  - L2:      level2_metrics.json    -> per_source.{src}.{metric}.mean ; paired.f_causal.{other}.{mean,wilcoxon_p}
  - wbc:     sweep.json             -> per_arm.{arm}.{metric}.mean ; paired_vs_raw.f_causal.{arm}.{mean,wilcoxon_p}
  - variance: aggregate.json        -> {b0,famail}.{metric}.mean ; paired_delta.f_causal.{mean,std}
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from famail_temporal.analysis._io import read_json


def _g(d, *keys, default=None):
    """Nested get; returns default if any key is missing."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def l1v2_summary(metrics: dict) -> dict:
    """Per-source scalar F_causal/F_spatial (works for both single-seed dirty and
    the clean canonical metrics.json, both of which carry scalar sources)."""
    out = {}
    for src in ("raw", "edited", "bc", "gan"):
        s = _g(metrics, "sources", src, default={})
        out[src] = {"f_causal": s.get("f_causal"), "f_spatial": s.get("f_spatial")}
    return out


def l2_summary(metrics: dict) -> dict:
    """Per-source mean F_causal + the edited-vs-raw paired delta."""
    out = {"per_source": {}}
    for src in ("raw", "edited", "bcgen", "gangen"):
        out["per_source"][src] = {
            "f_causal": _g(metrics, "per_source", src, "f_causal", "mean"),
            "f_spatial": _g(metrics, "per_source", src, "f_spatial", "mean"),
        }
    # paired baseline is "edited"; the edited-vs-raw contrast lives under paired.f_causal.raw
    out["edited_vs_raw"] = {
        "delta": _g(metrics, "paired", "f_causal", "raw", "mean"),
        "wilcoxon_p": _g(metrics, "paired", "f_causal", "raw", "wilcoxon_p"),
    }
    return out


def wbc_summary(sweep: dict) -> dict:
    """Per-arm mean F_causal + the paired-vs-raw delta/p for each weighted arm."""
    out = {"per_arm": {}, "paired_vs_raw": {}}
    per_arm = sweep.get("per_arm", {})
    for arm in per_arm:
        out["per_arm"][arm] = {"f_causal": _g(per_arm, arm, "f_causal", "mean")}
    pvr = _g(sweep, "paired_vs_raw", "f_causal", default={})
    for arm in pvr:
        out["paired_vs_raw"][arm] = {
            "delta": _g(pvr, arm, "mean"), "wilcoxon_p": _g(pvr, arm, "wilcoxon_p"),
        }
    return out


def variance_summary(agg: dict) -> dict:
    return {
        "b0_f_causal": _g(agg, "b0", "f_causal", "mean"),
        "famail_f_causal": _g(agg, "famail", "f_causal", "mean"),
        "paired_delta_f_causal": {
            "mean": _g(agg, "paired_delta", "f_causal", "mean"),
            "std": _g(agg, "paired_delta", "f_causal", "std"),
        },
    }


def build_comparison(dirty: dict, clean: dict) -> dict:
    """dirty/clean are {experiment: parsed-json}. Returns the structured comparison."""
    cmp = {}
    if "l1v2" in dirty and "l1v2" in clean:
        cmp["l1v2"] = {"dirty": l1v2_summary(dirty["l1v2"]),
                       "clean": l1v2_summary(clean["l1v2"])}
    if "l2" in dirty and "l2" in clean:
        cmp["l2"] = {"dirty": l2_summary(dirty["l2"]),
                     "clean": l2_summary(clean["l2"])}
    if "wbc" in dirty and "wbc" in clean:
        cmp["wbc"] = {"dirty": wbc_summary(dirty["wbc"]),
                      "clean": wbc_summary(clean["wbc"])}
    if "variance" in dirty and "variance" in clean:
        cmp["variance"] = {"dirty": variance_summary(dirty["variance"]),
                           "clean": variance_summary(clean["variance"])}
    return cmp


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def render_markdown(cmp: dict) -> str:
    L = ["# Experiment-level cleanup robustness (E22): dirty vs clean", ""]
    if "l1v2" in cmp:
        L += ["## L1-v2 data quality — per-source F_causal (edited should stay fairest faithful)", "",
              "| source | dirty F_causal | clean F_causal |", "|---|---|---|"]
        for src in ("raw", "edited", "bc", "gan"):
            d = _g(cmp, "l1v2", "dirty", src, "f_causal"); c = _g(cmp, "l1v2", "clean", src, "f_causal")
            L.append(f"| {src} | {_fmt(d)} | {_fmt(c)} |")
        L.append("")
    if "l2" in cmp:
        d = _g(cmp, "l2", "dirty", "edited_vs_raw", default={}); c = _g(cmp, "l2", "clean", "edited_vs_raw", default={})
        L += ["## L2 vanilla transfer — edited−raw paired Δ F_causal (should stay ~0, n.s.)", "",
              "| | Δ F_causal | wilcoxon p |", "|---|---|---|",
              f"| dirty | {_fmt(d.get('delta'))} | {d.get('wilcoxon_p')} |",
              f"| clean | {_fmt(c.get('delta'))} | {c.get('wilcoxon_p')} |", ""]
    if "wbc" in cmp:
        L += ["## weighted-BC — edited_wN paired Δ F_causal vs raw (should stay significant + dose-responsive)", "",
              "| arm | dirty Δ (p) | clean Δ (p) |", "|---|---|---|"]
        arms = sorted(set(list(_g(cmp, "wbc", "dirty", "paired_vs_raw", default={})) +
                          list(_g(cmp, "wbc", "clean", "paired_vs_raw", default={}))))
        for arm in arms:
            d = _g(cmp, "wbc", "dirty", "paired_vs_raw", arm, default={})
            c = _g(cmp, "wbc", "clean", "paired_vs_raw", arm, default={})
            ds = f"{_fmt(d.get('delta'))} (p={d.get('wilcoxon_p')})" if d else "— (new)"
            cs = f"{_fmt(c.get('delta'))} (p={c.get('wilcoxon_p')})" if c else "—"
            L.append(f"| {arm} | {ds} | {cs} |")
        L.append("")
    if "variance" in cmp:
        d = cmp["variance"]["dirty"]; c = cmp["variance"]["clean"]
        L += ["## variance (model-level) — b0 vs FAMAIL F_causal (should stay ≈equal)", "",
              "| | b0 | famail | paired Δ |", "|---|---|---|---|",
              f"| dirty | {_fmt(d['b0_f_causal'])} | {_fmt(d['famail_f_causal'])} | {_fmt(_g(d,'paired_delta_f_causal','mean'))} |",
              f"| clean | {_fmt(c['b0_f_causal'])} | {_fmt(c['famail_f_causal'])} | {_fmt(_g(c,'paired_delta_f_causal','mean'))} |", ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="famail_temporal.analysis.experiment_cleanup_delta")
    ap.add_argument("--dirty-l1v2", required=True)
    ap.add_argument("--clean-l1v2", required=True)
    ap.add_argument("--dirty-l2", required=True)
    ap.add_argument("--clean-l2", required=True)
    ap.add_argument("--dirty-wbc", required=True)
    ap.add_argument("--clean-wbc", required=True)
    ap.add_argument("--dirty-variance", required=True)
    ap.add_argument("--clean-variance", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)

    dirty = {"l1v2": read_json(Path(a.dirty_l1v2) / "level1_v2_metrics.json"),
             "l2": read_json(Path(a.dirty_l2) / "level2_metrics.json"),
             "wbc": read_json(Path(a.dirty_wbc) / "sweep.json"),
             "variance": read_json(Path(a.dirty_variance) / "aggregate.json")}
    clean = {"l1v2": read_json(Path(a.clean_l1v2) / "level1_v2_metrics.json"),
             "l2": read_json(Path(a.clean_l2) / "level2_metrics.json"),
             "wbc": read_json(Path(a.clean_wbc) / "sweep.json"),
             "variance": read_json(Path(a.clean_variance) / "aggregate.json")}
    cmp = build_comparison(dirty, clean)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_cleanup_delta.json").write_text(json.dumps(cmp, indent=2, default=float))
    (out / "experiment_cleanup_delta.md").write_text(render_markdown(cmp))
    print(f"wrote {out/'experiment_cleanup_delta.json'}")
    print(f"wrote {out/'experiment_cleanup_delta.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
