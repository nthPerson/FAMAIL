#!/usr/bin/env python3
"""Generate TikZ coordinates for the teaser dose-response panel from the
committed a10 artifacts. Prints the coordinate blocks to paste into
teaser-c4.tex / teaser-c3.tex (panel b/c). Keeps the figure's plotted
positions provenance-true instead of hand-computed.

Usage: python3 gen_coords.py   (from anywhere; paths are repo-absolute)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STATS = REPO / "PAPER/supply-lift/data/a10/shz_a10_weighted_bc_paired_stats.json"

# plot geometry (cm), must match the .tex files
X0, XSPAN, WMAX = 1.1, 7.1, 32          # x(w) = X0 + w * XSPAN/WMAX
VLO, VHI = -0.004, 0.032                # value range mapped onto the plot
PLOT_H = 2.6                            # cm
Y_BOTTOM_C4 = -7.5                      # panel-b bottom edge (v = VLO)
Y_BOTTOM_C3 = -8.74                     # same panel shifted down in c3

SCALE = PLOT_H / (VHI - VLO)            # cm per fairness unit


def x(w):
    return X0 + w * XSPAN / WMAX


def y(v, y_bottom):
    return y_bottom + (v - VLO) * SCALE


def fmt(val):
    return f"{val:.3f}"


stats = json.loads(STATS.read_text())["f_causal"]
series = {
    "edited": [("edited", 1), ("edited_w10", 10), ("edited_w20", 20),
               ("edited_w30", 30)],
    "most_fair": [("most_fair_w10", 10), ("most_fair_w20", 20),
                  ("most_fair_w30", 30)],
    "random": [("random_w10", 10), ("random_w30", 30)],
}

for name, y_bottom in (("c4", Y_BOTTOM_C4), ("c3", Y_BOTTOM_C3)):
    print(f"%% ===== {name} (y_bottom = {y_bottom}) =====")
    print("% y refs: " + ", ".join(
        f"v={v} -> {fmt(y(v, y_bottom))}" for v in (0.0, 0.01, 0.02, 0.03)))
    for label, keys in series.items():
        pts = [(x(w), y(stats[k]["mean"], y_bottom)) for k, w in keys]
        print(f"% {label}: means " + ", ".join(
            f"{stats[k]['mean']:+.4f}" for k, _ in keys))
        print("  line:  " + " -- ".join(f"({fmt(px)},{fmt(py)})"
                                        for px, py in pts))
        print("  marks: " + ", ".join(f"{fmt(px)}/{fmt(py)}"
                                      for px, py in pts))
        if label == "edited":
            whisk = [(x(w), y(stats[k]["t_ci"][0], y_bottom),
                      y(stats[k]["t_ci"][1], y_bottom)) for k, w in keys]
            print("  CIs:   " + ", ".join(
                f"{fmt(px)}/{fmt(lo)}/{fmt(hi)}" for px, lo, hi in whisk))
    print()
