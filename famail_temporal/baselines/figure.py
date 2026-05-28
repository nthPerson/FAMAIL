"""Render the data-level fairness x retention Pareto figure."""
from __future__ import annotations
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt

from famail_temporal.baselines.pareto import ParetoPoint


def plot_pareto(
    points: List[ParetoPoint], path: Path, metric: str = "f_causal",
) -> None:
    """Scatter retention (x) vs fairness `metric` (y), filter points joined
    into a curve, raw and edit drawn as standout markers."""
    fig, ax = plt.subplots(figsize=(7, 5))

    filt = sorted(
        [p for p in points if p.label.startswith("filter@")],
        key=lambda p: p.retention,
    )
    if filt:
        ax.plot(
            [p.retention for p in filt], [getattr(p, metric) for p in filt],
            "-o", color="#dc2626", label="B2 filter", zorder=2,
        )
    for p in points:
        if p.label == "raw":
            ax.scatter([p.retention], [getattr(p, metric)], s=90,
                       color="#1e3a5f", label="B0 raw", zorder=3)
        elif p.label == "edit":
            ax.scatter([p.retention], [getattr(p, metric)], s=140,
                       color="#047857", marker="*", label="FAMAIL edit", zorder=4)

    ax.set_xlabel("Data retention (fraction of corpus kept)")
    ax.set_ylabel(f"{metric}  (1 = fairest)")
    ax.set_title("Data-level fairness x retention")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
