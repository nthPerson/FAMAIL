"""replot_training_curves.py — Regenerate training-curve PNGs from CSV files.

Pure CSV → PNG: does NOT retrain or read any model.

Usage:
    python -m famail_temporal.baselines.replot_training_curves \\
        --curves-dir <dir> [--out-dir <dir>] [--smooth-window N]
        [--dpi N] [--yscale {linear,symlog}]
        [--clip-lo-pct F] [--clip-hi-pct F]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend set)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def read_series_csv(path) -> List[float]:
    """Read a 'step,loss' CSV (skip the header) -> list of loss floats (in row order)."""
    path = Path(path)
    values: List[float] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            values.append(float(row[1]))
    return values


def rolling_mean(values: List[float], window: int) -> List[float]:
    """Centered-ish moving average, SAME length as input.

    window<=1 returns a copy.
    For index i, average values[max(0,i-window//2) : i+window//2+1].
    No NaNs introduced.
    """
    if window <= 1:
        return list(values)
    half = window // 2
    result: List[float] = []
    n = len(values)
    for i in range(n):
        lo = max(0, i - half)
        hi = i + half + 1
        chunk = values[lo:hi]
        result.append(sum(chunk) / len(chunk))
    return result


def robust_ylim(
    values: List[float],
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
    margin: float = 0.05,
) -> Optional[Tuple[float, float]]:
    """(y_lo, y_hi) from lo/hi percentiles, expanded by margin of the span on each side.

    Returns None if <2 finite values or degenerate (lo==hi).
    Uses numpy.percentile on the finite values only.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return None
    arr = np.array(finite, dtype=float)
    y_lo = float(np.percentile(arr, lo_pct))
    y_hi = float(np.percentile(arr, hi_pct))
    if y_lo == y_hi:
        return None
    span = y_hi - y_lo
    return (y_lo - margin * span, y_hi + margin * span)


def clip_report(values: List[float], y_lo: float, y_hi: float) -> dict:
    """Count points outside [y_lo, y_hi] and find the global max.

    Returns {'n_clipped': int, 'max': float, 'argmax': int}.
    """
    n_clipped = sum(1 for v in values if v < y_lo or v > y_hi)
    if values:
        argmax = int(np.argmax(values))
        max_val = float(values[argmax])
    else:
        argmax = 0
        max_val = float("nan")
    return {"n_clipped": n_clipped, "max": max_val, "argmax": argmax}


def default_window(n: int) -> int:
    """A sensible smoothing window for a series of length n.

    max(1, n // 200), capped at 500.
    """
    return min(500, max(1, n // 200))


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_smoothed(
    title: str,
    series_map: Dict[str, List[float]],
    out_png,
    *,
    smooth_window: Optional[int] = None,
    dpi: int = 150,
) -> Path:
    """One figure with faint raw + bold smoothed lines for each named series.

    Returns the path to the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    for name, vals in series_map.items():
        n = len(vals)
        steps = list(range(n))
        w = smooth_window if smooth_window is not None else default_window(n)
        smoothed = rolling_mean(vals, w)
        ax.plot(steps, vals, alpha=0.25, linewidth=0.6, label=f"{name} (raw)")
        ax.plot(steps, smoothed, linewidth=1.5, label=f"{name} (smooth)")

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def render_adversarial(
    g_values: List[float],
    d_values: List[float],
    out_png,
    *,
    smooth_window: Optional[int] = None,
    dpi: int = 150,
    yscale: str = "linear",
    clip_lo_pct: float = 1.0,
    clip_hi_pct: float = 99.0,
) -> Path:
    """Two stacked panels (generator top, discriminator bottom) sharing the x-axis.

    Each panel: faint raw line + bold rolling-mean.
    In linear mode: robust_ylim clipping with spike annotation.
    In symlog mode: ax.set_yscale('symlog'), no clipping.
    Returns the path to the written PNG.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # NOT sharex: the generator and critic update on different cadences (g every
    # n_critic batches, d every batch), so their step indices are different
    # counters. A shared x-axis would leave the shorter g panel mostly empty and
    # falsely imply a common timeline. Each panel scales to its own update count.
    fig, (ax_g, ax_d) = plt.subplots(
        2, 1, figsize=(11, 7), constrained_layout=True
    )

    for ax, vals, panel_title, xlabel in [
        (ax_g, g_values, "Generator (adversarial) loss", "Generator update step"),
        (ax_d, d_values, "Critic / discriminator loss", "Critic update step"),
    ]:
        n = len(vals)
        steps = list(range(n))
        w = smooth_window if smooth_window is not None else default_window(n)
        smoothed = rolling_mean(vals, w)

        ax.plot(steps, vals, alpha=0.25, linewidth=0.6, color="tab:blue", label="raw")
        ax.plot(steps, smoothed, linewidth=1.5, color="tab:blue", label="smooth")
        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Loss")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

        if yscale == "symlog":
            ax.set_yscale("symlog")
        else:
            ylim = robust_ylim(vals, lo_pct=clip_lo_pct, hi_pct=clip_hi_pct)
            if ylim is not None:
                ax.set_ylim(*ylim)
                report = clip_report(vals, ylim[0], ylim[1])
                if report["n_clipped"] > 0:
                    msg = (
                        f"{report['n_clipped']} spikes clipped"
                        f" (max {report['max']:.3g} @ step {report['argmax']})"
                    )
                    ax.text(
                        0.01,
                        0.97,
                        msg,
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=7,
                        color="darkred",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7),
                    )

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Regenerate training-curve PNGs from exported CSV files."
    )
    p.add_argument("--curves-dir", required=True, help="Directory containing curve CSVs.")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: same as curves-dir).",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=None,
        help="Smoothing window size (default: per-series default_window).",
    )
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI (default: 150).")
    p.add_argument(
        "--yscale",
        choices=["linear", "symlog"],
        default="linear",
        help="Y-axis scale for adversarial plot (default: linear).",
    )
    p.add_argument(
        "--clip-lo-pct",
        type=float,
        default=1.0,
        help="Lower percentile for y-axis clipping (default: 1.0).",
    )
    p.add_argument(
        "--clip-hi-pct",
        type=float,
        default=99.0,
        help="Upper percentile for y-axis clipping (default: 99.0).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """CLI entry point. Returns 0 on success, 1 if no recognized CSVs found."""
    args = _parse_args(argv)
    curves_dir = Path(args.curves_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else curves_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sw = args.smooth_window
    dpi = args.dpi
    yscale = args.yscale
    clip_lo = args.clip_lo_pct
    clip_hi = args.clip_hi_pct

    written: List[Path] = []

    # --- BC MLE ---
    bc_batch = curves_dir / "bc_mle_batch.csv"
    bc_epoch = curves_dir / "bc_mle_epoch.csv"
    if bc_batch.exists():
        vals = read_series_csv(bc_batch)
        out = render_smoothed(
            "BC MLE Loss", {"bc_mle_batch": vals}, out_dir / "bc_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)
    elif bc_epoch.exists():
        vals = read_series_csv(bc_epoch)
        out = render_smoothed(
            "BC MLE Loss", {"bc_mle_epoch": vals}, out_dir / "bc_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)

    # --- GAN MLE ---
    gan_mle_batch = curves_dir / "gan_mle_batch.csv"
    gan_mle_epoch = curves_dir / "gan_mle_epoch.csv"
    if gan_mle_batch.exists():
        vals = read_series_csv(gan_mle_batch)
        out = render_smoothed(
            "GAN MLE Loss", {"gan_mle_batch": vals}, out_dir / "gan_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)
    elif gan_mle_epoch.exists():
        vals = read_series_csv(gan_mle_epoch)
        out = render_smoothed(
            "GAN MLE Loss", {"gan_mle_epoch": vals}, out_dir / "gan_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)

    # --- GAN Adversarial ---
    adv_g_batch = curves_dir / "gan_adv_g_batch.csv"
    adv_d_batch = curves_dir / "gan_adv_d_batch.csv"
    adv_g_epoch = curves_dir / "gan_adv_g_epoch.csv"
    adv_d_epoch = curves_dir / "gan_adv_d_epoch.csv"

    if adv_g_batch.exists() and adv_d_batch.exists():
        g_vals = read_series_csv(adv_g_batch)
        d_vals = read_series_csv(adv_d_batch)
        out = render_adversarial(
            g_vals, d_vals, out_dir / "gan_adversarial.png",
            smooth_window=sw, dpi=dpi, yscale=yscale,
            clip_lo_pct=clip_lo, clip_hi_pct=clip_hi,
        )
        print(f"Written: {out}")
        written.append(out)
    elif adv_g_epoch.exists() and adv_d_epoch.exists():
        g_vals = read_series_csv(adv_g_epoch)
        d_vals = read_series_csv(adv_d_epoch)
        out = render_adversarial(
            g_vals, d_vals, out_dir / "gan_adversarial.png",
            smooth_window=sw, dpi=dpi, yscale=yscale,
            clip_lo_pct=clip_lo, clip_hi_pct=clip_hi,
        )
        print(f"Written: {out}")
        written.append(out)

    # --- B0 seed variance ---
    b0_seed_csvs = sorted(curves_dir.glob("b0_seed*_mle.csv"))
    if b0_seed_csvs:
        series: Dict[str, List[float]] = {}
        for p in b0_seed_csvs:
            series[p.stem] = read_series_csv(p)
        out = render_smoothed(
            "B0 MLE Loss", series, out_dir / "b0_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)

    # --- FAMAIL seed variance ---
    famail_seed_csvs = sorted(curves_dir.glob("famail_seed*_mle.csv"))
    if famail_seed_csvs:
        series = {}
        for p in famail_seed_csvs:
            series[p.stem] = read_series_csv(p)
        out = render_smoothed(
            "FAMAIL MLE Loss", series, out_dir / "famail_mle.png",
            smooth_window=sw, dpi=dpi
        )
        print(f"Written: {out}")
        written.append(out)

    if not written:
        print(
            f"No recognized CSVs found in {curves_dir}. "
            "Expected one or more of: bc_mle_batch.csv, bc_mle_epoch.csv, "
            "gan_mle_batch.csv, gan_mle_epoch.csv, gan_adv_g_batch.csv, "
            "gan_adv_d_batch.csv, gan_adv_g_epoch.csv, gan_adv_d_epoch.csv, "
            "b0_seed*_mle.csv, famail_seed*_mle.csv"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
