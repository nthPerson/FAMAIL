#!/usr/bin/env python3
"""Visualize SF supply / demand / demographics concentration on the 32x30 grid.

Reads the raw SF source_data counts (true totals, not mean-hourly) and renders
north-up heatmaps so we can see how supply and demand are spatially concentrated
-- the crux of the F_causal supply/demand regime finding (docs/SF_PHASE3_RESULTS.md).

Run:  python famail_temporal/docs/sf_supply_demand_heatmap.py
"""
from __future__ import annotations

import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.normpath(os.path.join(_HERE, "..", "source_data", "second_dataset", "sf_source"))
OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "results", "sf_diagnostics"))
os.makedirs(OUT_DIR, exist_ok=True)
GX, GY = 32, 30


def _load(name):
    # Safe: these pickles are our own SF source_data, generated locally by
    # sf_build.py (same trust boundary as the rest of the pipeline), not
    # untrusted input.
    with open(os.path.join(SD, name), "rb") as f:
        return pickle.load(f)


# --- Total demand (pickups) and dropoffs per cell ---------------------------
pd_counts = _load("pickup_dropoff_counts.pkl")
demand = np.zeros((GX, GY))
dropoff = np.zeros((GX, GY))
for (x, y, tb, day), (p, d) in pd_counts.items():
    demand[x - 1, y - 1] += p
    dropoff[x - 1, y - 1] += d

# --- Total supply exposure (sum of 5x5 distinct-taxi counts) per cell --------
active = _load("active_taxis_5x5_hourly.pkl")["data"]
supply = np.zeros((GX, GY))
for (x, y, hour, day), c in active.items():
    supply[x - 1, y - 1] += c

# --- Demographics -----------------------------------------------------------
demo = _load("cell_demographics.pkl")
grid = demo["demographics_grid"]
names = list(demo["feature_names"])

# --- Concentration stats (the numbers behind the regime finding) ------------
def conc(a, label):
    flat = a.flatten()
    total = flat.sum()
    nz = (flat > 0).sum()
    top = np.sort(flat)[::-1]
    share10 = top[:10].sum() / total if total else 0
    print(f"{label:16s} total={total:,.0f}  nonzero cells={nz}/{GX*GY} "
          f"({100*nz/(GX*GY):.0f}%)  top-10 cells hold {100*share10:.0f}% of total")

print("=== SF concentration ===")
conc(demand, "demand (pickups)")
conc(supply, "supply exposure")
dsr = np.where(supply > 0, demand / np.maximum(supply, 1e-9), np.nan)
print(f"DSR=demand/supply  median(active)={np.nanmedian(dsr):.4f}  "
      f"max={np.nanmax(dsr):.3f}  (supply >> demand => low DSR everywhere)")

# --- Plot -------------------------------------------------------------------
def panel(ax, A, title, *, log=False, cmap="viridis"):
    M = np.ma.masked_invalid(A)
    if log:
        M = np.ma.masked_less_equal(M, 0)
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap,
                       norm=LogNorm(vmin=max(M.min(), 1), vmax=M.max()))
    else:
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("lon idx (W→E)"); ax.set_ylabel("lat idx (S→N)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panel(axes[0, 0], demand, "Demand: total pickups/cell (log)", log=True, cmap="magma")
panel(axes[0, 1], supply, "Supply: 5x5 distinct-taxi exposure/cell (log)", log=True, cmap="viridis")
panel(axes[0, 2], dsr, "DSR = demand / supply", cmap="cividis")
panel(axes[1, 0], grid[..., names.index("AvgHousingPricePerSqM")], "Housing (median value $)", cmap="plasma")
panel(axes[1, 1], grid[..., names.index("CompPerCapita")], "Comp (per-capita income $)", cmap="plasma")
panel(axes[1, 2], grid[..., names.index("MigrantRatio")], "Migrant (foreign-born share)", cmap="plasma")
fig.suptitle("SF Cabspotting — supply / demand / demographics on the 32x30 (0.01°) grid  (north up)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(OUT_DIR, "sf_supply_demand.png")
fig.savefig(out, dpi=110)
print(f"\nwrote {out}")
