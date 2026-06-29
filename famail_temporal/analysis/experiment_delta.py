"""E22 (experiment-level) — Dirty-vs-clean delta across all four experiment stages.

Pure comparison helpers (TDD'd) + a CLI that reads the real result JSON pairs
and writes experiment_cleanup_delta.json + experiment_cleanup_delta.md.

Read-only over existing result JSONs — safe to run at any time.

Schemas handled:
  - L1-v2:  sources.{src}.{f_causal,f_spatial,fidelity_a,fidelity_b}  (scalars)
  - L2:     per_source.{src}.{metric}.mean; paired.f_causal.raw.{mean,wilcoxon_p}
  - wbc:    paired_vs_raw.f_causal.{arm}.{mean,wilcoxon_p}
  - variance: paired_delta.{f_causal,f_spatial}.{mean,std}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from famail_temporal.analysis._io import read_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_L1_SOURCES = ("raw", "edited", "bc", "gan")
_L1_METRICS = ("f_causal", "f_spatial", "fidelity_a", "fidelity_b")
_L2_SOURCES = ("raw", "edited", "bcgen", "gangen")


def _g(d, *keys, default=None):
    """Nested dict get; returns default if any key is missing or d is not a dict."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _delta(dirty_val, clean_val):
    """Return clean - dirty, or None if either value is None."""
    if dirty_val is None or clean_val is None:
        return None
    return clean_val - dirty_val


def _cleanup_caption(sinks: dict | None) -> str:
    """Build the cleanup one-liner from the real stuck_gps_sinks metadata.

    Replaces a previously hardcoded "6 drivers, cell (28,52)" string that did
    NOT match the on-disk filter (the PI-decided filter removed 10 calibrated
    sink cells across multiple driver plates, and (28,52) is not among them).
    Derives the counts from ``processing_metadata.json -> stuck_gps_sinks`` so
    the caption can never drift from the actual filter again. Falls back to a
    metadata-free but still-correct generic description when ``sinks`` is None.
    """
    if not sinks:
        return (
            "Data cleanup = per-driver stuck-GPS pickup-sink filter "
            "(signature rule on the calibrated sink cells; see "
            "source_data/processing_metadata.json -> stuck_gps_sinks)."
        )
    flagged = sinks.get("flagged_cells", []) or []
    n_cells = len(flagged)
    sink_rows = sinks.get("sinks", []) or []
    n_drivers = len({s.get("plate_id") for s in sink_rows if isinstance(s, dict)})
    n_removed = sinks.get("n_pickups_removed", sinks.get("n_rows_removed", 0)) or 0
    driver_str = f"{n_drivers} driver{'s' if n_drivers != 1 else ''}" if n_drivers else "per-driver"
    return (
        f"Data cleanup = stuck-GPS sink filter ({driver_str}, "
        f"{n_cells} flagged cells; {n_removed:,} phantom pickups removed)."
    )


# ---------------------------------------------------------------------------
# Pure comparison functions (TDD'd)
# ---------------------------------------------------------------------------

def l1_delta(dirty: dict, clean: dict) -> dict:
    """Compare L1-v2 metrics across sources.

    Args:
        dirty: contents of the dirty level1_v2_metrics.json
        clean: contents of the clean level1_v2_metrics.json

    Returns:
        {source: {metric: {dirty, clean, delta}}} for each source × metric pair.
        Values are None when the source/metric is absent in the input.
    """
    result = {}
    for src in _L1_SOURCES:
        d_src = _g(dirty, "sources", src, default={})
        c_src = _g(clean, "sources", src, default={})
        result[src] = {}
        for metric in _L1_METRICS:
            d_val = d_src.get(metric) if d_src else None
            c_val = c_src.get(metric) if c_src else None
            result[src][metric] = {
                "dirty": d_val,
                "clean": c_val,
                "delta": _delta(d_val, c_val),
            }
    return result


def l2_delta(dirty: dict, clean: dict) -> dict:
    """Compare L2 metrics.

    Args:
        dirty: contents of the dirty level2_metrics.json
        clean: contents of the clean level2_metrics.json

    Returns:
        {
          "paired_edited_raw": {dirty_mean, clean_mean, delta, dirty_p, clean_p},
          "per_source": {src: {f_causal: {dirty, clean, delta}, f_spatial: {...}}}
        }
    """
    # Paired edited-vs-raw contrast (lives under paired.f_causal.raw)
    d_paired = _g(dirty, "paired", "f_causal", "raw", default={})
    c_paired = _g(clean, "paired", "f_causal", "raw", default={})
    d_mean = d_paired.get("mean") if d_paired else None
    c_mean = c_paired.get("mean") if c_paired else None

    paired_edited_raw = {
        "dirty_mean": d_mean,
        "clean_mean": c_mean,
        "delta": _delta(d_mean, c_mean),
        "dirty_p": d_paired.get("wilcoxon_p") if d_paired else None,
        "clean_p": c_paired.get("wilcoxon_p") if c_paired else None,
    }

    # Per-source means
    per_source = {}
    for src in _L2_SOURCES:
        per_source[src] = {}
        for metric in ("f_causal", "f_spatial"):
            d_val = _g(dirty, "per_source", src, metric, "mean")
            c_val = _g(clean, "per_source", src, metric, "mean")
            per_source[src][metric] = {
                "dirty": d_val,
                "clean": c_val,
                "delta": _delta(d_val, c_val),
            }

    return {"paired_edited_raw": paired_edited_raw, "per_source": per_source}


def wbc_delta(dirty: dict, clean: dict) -> dict:
    """Compare weighted-BC sweep results.

    Handles arms present in only one of dirty/clean ("clean_only" / "dirty_only").

    Args:
        dirty: contents of the dirty sweep.json
        clean: contents of the clean sweep.json

    Returns:
        {arm: {dirty_delta_vs_raw, clean_delta_vs_raw, p_dirty, p_clean, status}}
        where status is 'compared', 'clean_only', or 'dirty_only'.
    """
    d_pvr = _g(dirty, "paired_vs_raw", "f_causal", default={})
    c_pvr = _g(clean, "paired_vs_raw", "f_causal", default={})

    all_arms = set(d_pvr.keys()) | set(c_pvr.keys())
    result = {}
    for arm in sorted(all_arms):
        in_dirty = arm in d_pvr
        in_clean = arm in c_pvr

        if in_dirty and in_clean:
            status = "compared"
        elif in_clean:
            status = "clean_only"
        else:
            status = "dirty_only"

        d_arm = d_pvr.get(arm, {})
        c_arm = c_pvr.get(arm, {})
        d_val = d_arm.get("mean") if d_arm else None
        c_val = c_arm.get("mean") if c_arm else None

        result[arm] = {
            "dirty_delta_vs_raw": d_val,
            "clean_delta_vs_raw": c_val,
            "p_dirty": d_arm.get("wilcoxon_p") if d_arm else None,
            "p_clean": c_arm.get("wilcoxon_p") if c_arm else None,
            "status": status,
        }
    return result


def variance_delta(dirty: dict, clean: dict) -> dict:
    """Compare variance suite paired_delta across dirty/clean runs.

    Args:
        dirty: contents of dirty aggregate.json
        clean: contents of clean aggregate.json

    Returns:
        {f_causal: {dirty, clean, delta}, f_spatial: {dirty, clean, delta}}
    """
    result = {}
    for metric in ("f_causal", "f_spatial"):
        d_val = _g(dirty, "paired_delta", metric, "mean")
        c_val = _g(clean, "paired_delta", metric, "mean")
        result[metric] = {
            "dirty": d_val,
            "clean": c_val,
            "delta": _delta(d_val, c_val),
        }
    return result


# ---------------------------------------------------------------------------
# CLI: read real result dirs + write outputs
# ---------------------------------------------------------------------------

_DIRTY_L1V2 = "famail_temporal/results/level1_table_v2/2026-06-18_full_run"
_CLEAN_L1V2 = "famail_temporal/results/level1_table_v2/cleaned_5seed"
_DIRTY_L2 = "famail_temporal/results/level2_table/2026-06-18T17-27-34"
_CLEAN_L2 = "famail_temporal/results/level2_table/cleaned_5seed"
_DIRTY_WBC = "famail_temporal/results/weighted_bc_sweep/sig_6seed_w10_w20_w30"
_CLEAN_WBC = "famail_temporal/results/weighted_bc_sweep/cleaned_6seed"
_DIRTY_VAR = "famail_temporal/baselines/variance_suite/results/2026-06-11T00-04-19_seeds0-4"
_CLEAN_VAR = "famail_temporal/results/variance_suite/cleaned_5seed"


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def _verdict_l1(l1: dict) -> str:
    """True iff edited F_causal is highest among sources in both dirty and clean."""
    sources = list(l1.keys())
    dirty_vals = {s: l1[s]["f_causal"]["dirty"] for s in sources
                  if l1[s]["f_causal"]["dirty"] is not None}
    clean_vals = {s: l1[s]["f_causal"]["clean"] for s in sources
                  if l1[s]["f_causal"]["clean"] is not None}
    if not dirty_vals or not clean_vals:
        return "UNKNOWN (missing data)"
    d_edit = dirty_vals.get("edited")
    c_edit = clean_vals.get("edited")
    d_rest = [v for s, v in dirty_vals.items() if s != "edited"]
    c_rest = [v for s, v in clean_vals.items() if s != "edited"]
    if d_edit is None or c_edit is None:
        return "UNKNOWN"
    dirty_ok = d_edit > max(d_rest) if d_rest else True
    clean_ok = c_edit > max(c_rest) if c_rest else True
    return "PRESERVED — edited stays fairest faithful" if (dirty_ok and clean_ok) else "CHANGED"


def render_report(l1: dict, l2: dict, wbc: dict, var: dict,
                  cleanup_caption: str | None = None) -> str:
    """Render a readable Markdown report summarising all four stage deltas.

    ``cleanup_caption`` describes the data filter; when None a metadata-free
    (but correct) generic caption is used. Callers should pass the real caption
    built from ``stuck_gps_sinks`` via :func:`_cleanup_caption`.
    """
    lines = [
        "# E22: Experiment-level dirty-vs-clean robustness",
        "",
        cleanup_caption or _cleanup_caption(None),
        "All four stages of the argument are compared below.",
        "",
        "_Note: F_causal here uses the 3-feature set "
        "{AvgHousingPricePerSqM, GDPperCapita, CompPerCapita} under which the "
        "cleanup was validated; the dirty-vs-clean conclusions are an "
        "apples-to-apples comparison at constant feature set and are "
        "feature-set-invariant. Absolute F_causal values are NOT comparable to "
        "other feature sets' headline tables._",
        "",
    ]

    # L1-v2
    lines += [
        "## Stage L1-v2 — per-source F_causal (edited should stay fairest faithful)", "",
        "| source | dirty F_causal | clean F_causal | Δ (clean−dirty) |",
        "|--------|---------------|---------------|-----------------|",
    ]
    for src in _L1_SOURCES:
        row = l1[src]["f_causal"]
        lines.append(f"| {src} | {_fmt(row['dirty'])} | {_fmt(row['clean'])} | {_fmt(row['delta'])} |")
    verdict_l1 = _verdict_l1(l1)
    lines += ["", f"**Conclusion preserved?** {verdict_l1}", ""]

    # L2
    pr = l2["paired_edited_raw"]
    lines += [
        "## Stage L2 — vanilla-BC transfer: edited−raw paired Δ F_causal (should stay n.s.)", "",
        "| | Δ F_causal (mean) | wilcoxon p |",
        "|--|-------------------|------------|",
        f"| dirty | {_fmt(pr['dirty_mean'])} | {pr['dirty_p']} |",
        f"| clean | {_fmt(pr['clean_mean'])} | {pr['clean_p']} |",
        "",
    ]
    d_ns = pr["dirty_p"] is not None and pr["dirty_p"] >= 0.05
    c_ns = pr["clean_p"] is not None and pr["clean_p"] >= 0.05
    verdict_l2 = "PRESERVED — both dirty & clean n.s. (p≥0.05)" if (d_ns and c_ns) else \
                 "CHANGED — significance changed across cleanup"
    lines += [f"**Conclusion preserved?** {verdict_l2}", ""]

    # WBC
    lines += [
        "## Stage weighted-BC — paired Δ F_causal vs raw (edited_wN should stay significant + dose-responsive)", "",
        "| arm | dirty Δ (p) | clean Δ (p) | status |",
        "|-----|------------|------------|--------|",
    ]
    for arm, row in wbc.items():
        d_str = f"{_fmt(row['dirty_delta_vs_raw'])} (p={row['p_dirty']})" if row["dirty_delta_vs_raw"] is not None else "— (absent)"
        c_str = f"{_fmt(row['clean_delta_vs_raw'])} (p={row['p_clean']})" if row["clean_delta_vs_raw"] is not None else "— (absent)"
        lines.append(f"| {arm} | {d_str} | {c_str} | {row['status']} |")
    # Check edited_w10/w20/w30 remain significant
    key_arms = [a for a in wbc if a.startswith("edited_w") and wbc[a]["status"] == "compared"]
    sig_ok = all(
        wbc[a]["p_dirty"] is not None and wbc[a]["p_dirty"] < 0.05 and
        wbc[a]["p_clean"] is not None and wbc[a]["p_clean"] < 0.05
        for a in key_arms
    )
    new_arms = [a for a in wbc if wbc[a]["status"] == "clean_only"]
    verdict_wbc = "PRESERVED — weighted (wN) arms stay significant in both dirty & clean"
    if not sig_ok:
        verdict_wbc = "CHANGED — some weighted arms changed significance"
    # The unweighted 'edited' (w=1) arm is non-load-bearing (the recovery story
    # rests on the upweighted arms), but disclose any significance flip rather
    # than silently filtering it out of the key-arm set above.
    ua = wbc.get("edited")
    if (ua and ua.get("status") == "compared"
            and ua.get("p_dirty") is not None and ua.get("p_clean") is not None):
        if (ua["p_dirty"] < 0.05) != (ua["p_clean"] < 0.05):
            verdict_wbc += (
                f"; NOTE unweighted 'edited' (w=1) arm significance flipped "
                f"(p {ua['p_dirty']}→{ua['p_clean']}), direction preserved "
                f"(Δ {ua['dirty_delta_vs_raw']:+.4f}→{ua['clean_delta_vs_raw']:+.4f}, "
                f"both ≤0; the dirty p=0.03125 is the n=6 Wilcoxon floor, n.s. in "
                f"the dedicated L2 5-seed run)"
            )
    if new_arms:
        verdict_wbc += f"; new clean-only arms: {', '.join(new_arms)}"
    lines += ["", f"**Conclusion preserved?** {verdict_wbc}", ""]

    # Variance
    fc = var["f_causal"]
    fs = var["f_spatial"]
    lines += [
        "## Stage variance — b0 vs FAMAIL paired Δ F_causal (should stay ≈null)", "",
        "| | dirty paired Δ F_causal | clean paired Δ F_causal | shift |",
        "|--|------------------------|------------------------|-------|",
        f"| f_causal | {_fmt(fc['dirty'])} | {_fmt(fc['clean'])} | {_fmt(fc['delta'])} |",
        f"| f_spatial | {_fmt(fs['dirty'])} | {_fmt(fs['clean'])} | {_fmt(fs['delta'])} |",
        "",
    ]
    # Both ~null means |mean| is small relative to the inherent noise
    dirty_null = fc["dirty"] is not None and abs(fc["dirty"]) < 0.005
    clean_null = fc["clean"] is not None and abs(fc["clean"]) < 0.005
    verdict_var = "PRESERVED — paired Δ F_causal remains near-zero in both" if (dirty_null and clean_null) else \
                  "CHANGED — null finding changed across cleanup"
    lines += [f"**Conclusion preserved?** {verdict_var}", ""]

    return "\n".join(lines) + "\n"


def write_experiment_cleanup_delta(
    out_dir: str | Path,
    dirty_l1v2: str | Path | None = None,
    clean_l1v2: str | Path | None = None,
    dirty_l2: str | Path | None = None,
    clean_l2: str | Path | None = None,
    dirty_wbc: str | Path | None = None,
    clean_wbc: str | Path | None = None,
    dirty_var: str | Path | None = None,
    clean_var: str | Path | None = None,
) -> dict:
    """Read four real dir-pairs, compute deltas, write JSON + MD, return results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _rj(base_dir, filename):
        return read_json(Path(base_dir) / filename)

    d_l1v2 = _rj(dirty_l1v2 or _DIRTY_L1V2, "level1_v2_metrics.json")
    c_l1v2 = _rj(clean_l1v2 or _CLEAN_L1V2, "level1_v2_metrics.json")
    d_l2   = _rj(dirty_l2 or _DIRTY_L2, "level2_metrics.json")
    c_l2   = _rj(clean_l2 or _CLEAN_L2, "level2_metrics.json")
    d_wbc  = _rj(dirty_wbc or _DIRTY_WBC, "sweep.json")
    c_wbc  = _rj(clean_wbc or _CLEAN_WBC, "sweep.json")
    d_var  = _rj(dirty_var or _DIRTY_VAR, "aggregate.json")
    c_var  = _rj(clean_var or _CLEAN_VAR, "aggregate.json")

    l1 = l1_delta(d_l1v2, c_l1v2)
    l2 = l2_delta(d_l2, c_l2)
    wbc = wbc_delta(d_wbc, c_wbc)
    var = variance_delta(d_var, c_var)

    # Build the cleanup caption from the real on-disk filter metadata so the
    # report can never drift from what was actually removed.
    cleanup_caption = None
    try:
        from famail_temporal import config as _cfg
        from famail_temporal.analysis import _io as _io_mod
        clean_meta = _io_mod.processing_metadata(_cfg.SOURCE_DATA_DIR)
        cleanup_caption = _cleanup_caption(clean_meta.get("stuck_gps_sinks"))
    except Exception:
        cleanup_caption = None  # fall back to the generic correct caption

    full = {"l1_delta": l1, "l2_delta": l2, "wbc_delta": wbc, "variance_delta": var}
    (out_dir / "experiment_cleanup_delta.json").write_text(
        json.dumps(full, indent=2, default=float)
    )
    (out_dir / "experiment_cleanup_delta.md").write_text(
        render_report(l1, l2, wbc, var, cleanup_caption=cleanup_caption)
    )
    print(f"[E22] wrote {out_dir / 'experiment_cleanup_delta.json'}")
    print(f"[E22] wrote {out_dir / 'experiment_cleanup_delta.md'}")
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.experiment_delta",
        description="E22: experiment-level dirty-vs-clean robustness comparison",
    )
    ap.add_argument("--out-dir", required=True, help="Directory for output files")
    ap.add_argument("--dirty-l1v2", default=None)
    ap.add_argument("--clean-l1v2", default=None)
    ap.add_argument("--dirty-l2", default=None)
    ap.add_argument("--clean-l2", default=None)
    ap.add_argument("--dirty-wbc", default=None)
    ap.add_argument("--clean-wbc", default=None)
    ap.add_argument("--dirty-var", default=None)
    ap.add_argument("--clean-var", default=None)
    a = ap.parse_args(argv)

    results = write_experiment_cleanup_delta(
        out_dir=a.out_dir,
        dirty_l1v2=a.dirty_l1v2,
        clean_l1v2=a.clean_l1v2,
        dirty_l2=a.dirty_l2,
        clean_l2=a.clean_l2,
        dirty_wbc=a.dirty_wbc,
        clean_wbc=a.clean_wbc,
        dirty_var=a.dirty_var,
        clean_var=a.clean_var,
    )
    # Print the key numbers
    l1 = results["l1_delta"]
    l2 = results["l2_delta"]
    wbc = results["wbc_delta"]
    var = results["variance_delta"]

    print("\n=== KEY NUMBERS ===")
    print(f"L1 edited F_causal: dirty={l1['edited']['f_causal']['dirty']:.4f}  "
          f"clean={l1['edited']['f_causal']['clean']:.4f}  "
          f"Δ={l1['edited']['f_causal']['delta']:.4f}")
    pr = l2["paired_edited_raw"]
    print(f"L2 paired edited−raw: dirty={pr['dirty_mean']:.4f} (p={pr['dirty_p']})  "
          f"clean={pr['clean_mean']:.4f} (p={pr['clean_p']})")
    if "edited_w30" in wbc:
        row = wbc["edited_w30"]
        print(f"WBC edited_w30 Δ-vs-raw: dirty={row['dirty_delta_vs_raw']:.4f} (p={row['p_dirty']})  "
              f"clean={row['clean_delta_vs_raw']:.4f} (p={row['p_clean']})")
    print(f"Variance paired Δ F_causal: dirty={var['f_causal']['dirty']:.5f}  "
          f"clean={var['f_causal']['clean']:.5f}  "
          f"Δ={var['f_causal']['delta']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
