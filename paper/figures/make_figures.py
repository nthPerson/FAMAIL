"""Generate the manuscript's two committed figures.

  1. extended_frontier.pdf  — the weight-sensitivity figure (fig:alpha-pareto):
     panel A = the (dF_spatial, dF_demo (key d_f_causal)) plane across the six alpha points;
     panel B = the lift-up channels vs alpha_spatial (monotone decline), with
     filled markers = 95% CI excludes 0, open = n.s.
     Data: PAPER/objective-motivation/weight-sensitivity/extended_frontier.json
     (committed, provenance in its _src field). Nothing is hard-coded here.

  2. method_overview.pdf — a DRAFT three-panel schematic for fig:overview
     (the service gap / trim / lift). Placeholder quality by design: the
     production Figure 1 is being made separately (see PAPER/figures/figure-1.md);
     this keeps the manuscript compiling with a real, honest figure meanwhile.

Print-first design: identity is carried by marker shape + line style + direct
labels (grayscale-safe); one blue accent marks the adopted configuration and
is redundant with shape. Fonts sized for a 3.33in ACM column.

Run:  python paper/figures/make_figures.py   (from the repo root)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "PAPER/objective-motivation/weight-sensitivity/extended_frontier.json"
OUT = Path(__file__).resolve().parent

INK = "#1a1a1a"
GRAY = "#6f6f6f"
FAINT = "#c9c9c9"
ACCENT = "#0f62fe"  # adopted point; always redundant with shape/label

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "font.family": "serif", "pdf.fonttype": 42,
})


def frontier_figure() -> None:
    d = json.loads(DATA.read_text())
    pts = d["points"]
    adopted = d["adopted"]
    a = [p["alpha_sp"] for p in pts]

    # 3.75in tall (~270pt): the earlier 4.35in figure + caption left the left
    # column of its page unable to fill (render-QA R7); both panels have slack.
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(3.35, 3.75))
    LABEL_BBOX = dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85)

    # ---- Panel A: the optimized-metric plane -------------------------------
    for p in pts:
        is_adopted = p["alpha_sp"] == adopted
        axA.scatter(p["d_f_spatial"], p["d_f_causal"],
                    s=34 if is_adopted else 20,
                    marker="*" if is_adopted else "o",
                    facecolor=ACCENT if is_adopted else "none",
                    edgecolor=ACCENT if is_adopted else INK,
                    linewidth=0.8, zorder=3)
    # Per-point label placement (render-QA R8: the cluster near x~0.006 needs
    # explicit, non-colliding offsets; the adopted point gets ONE merged label).
    offs = {0.0: (-4, -8), 0.1: (-4, 6), 0.2: (4, -8), 0.35: (4, 3),
            0.55: (4, 3), 0.8: (4, 3)}
    for p in pts:
        is_adopted = p["alpha_sp"] == adopted
        label = (f"({p['alpha_sp']:g}, {p['alpha_ca']:g})" if not is_adopted
                 else f"({p['alpha_sp']:g}, {p['alpha_ca']:g}) — adopted")
        axA.annotate(label, (p["d_f_spatial"], p["d_f_causal"]),
                     textcoords="offset points",
                     xytext=offs.get(p["alpha_sp"], (4, 3)),
                     ha="right" if (is_adopted or p["alpha_sp"] == 0.0) else "left",
                     fontsize=6.2, color=ACCENT if is_adopted else INK,
                     fontstyle="italic" if is_adopted else "normal",
                     bbox=LABEL_BBOX, zorder=4)
    axA.set_xlabel(r"$\Delta F_\mathrm{spatial}$")
    axA.set_ylabel(r"$\Delta F_\mathrm{demo}$")
    axA.set_title(r"A · optimized metrics: $\Delta F_\mathrm{demo}$ is flat", loc="left")
    axA.grid(True, color=FAINT, linewidth=0.4, alpha=0.6)
    axA.set_axisbelow(True)
    pad = 0.0012
    axA.set_ylim(min(p["d_f_causal"] for p in pts) - pad,
                 max(p["d_f_causal"] for p in pts) + pad)

    # ---- Panel B: the lift-up channels (design-targeted ring) --------------
    series = [
        ("total_yd", "total_sig", r"total $\Delta\,\mathrm{mean}(Y\,|\,\mathrm{disadv.})$",
         "-", "o", GRAY),
        ("supply_t2", "t2_sig", "supply, distinct-taxi", "--", "s", INK),
        ("supply_t1", "t1_sig", "supply, fractional-presence", ":", "^", INK),
    ]
    for key, sig_key, label, ls, mk, color in series:
        ys = [p[key] for p in pts]
        axB.plot(a, ys, ls, color=color, linewidth=1.1, zorder=2)
        for p in pts:
            filled = p[sig_key]
            axB.scatter(p["alpha_sp"], p[key], marker=mk, s=18,
                        facecolor=color if filled else "white",
                        edgecolor=color, linewidth=0.8, zorder=3)
    # direct labels at explicit clear positions (the series converge near zero
    # at alpha_sp = 0.8, so end-labeling fails; left side is well separated)
    axB.annotate(r"total $\Delta\,\mathrm{mean}(Y\,|\,\mathrm{disadv.})$",
                 (0.035, 0.0745), fontsize=6.2, color=GRAY, ha="left",
                 bbox=LABEL_BBOX, zorder=4)
    axB.annotate("supply, distinct-taxi", (0.15, 0.043),
                 fontsize=6.2, color=INK, ha="left", bbox=LABEL_BBOX, zorder=4)
    axB.annotate("supply, fractional-presence", (0.03, 0.004), fontsize=6.2, color=INK,
                 ha="left", bbox=LABEL_BBOX, zorder=4)
    axB.axhline(0, color=GRAY, linewidth=0.6)
    axB.axvline(adopted, color=ACCENT, linewidth=0.8, alpha=0.9,
                linestyle=(0, (2, 2)))
    axB.annotate("adopted", (adopted, -0.006), fontsize=6.2,
                 color=ACCENT, fontstyle="italic",
                 textcoords="offset points", xytext=(3, 0))
    axB.set_xlabel(r"$\alpha_\mathrm{spatial}$")
    axB.set_ylabel(r"$\Delta$ on $\mathrm{mean}(Y\,|\,\mathrm{disadv.})$")
    axB.set_title("B · the lift-up declines with spatial weight", loc="left")
    axB.grid(True, color=FAINT, linewidth=0.4, alpha=0.6)
    axB.set_axisbelow(True)
    # significance key (filled vs open), kept out of the data area
    axB.scatter([], [], marker="o", facecolor=INK, edgecolor=INK, s=18,
                label="CI excludes 0")
    axB.scatter([], [], marker="o", facecolor="white", edgecolor=INK, s=18,
                label="n.s.")
    axB.legend(loc="upper right", frameon=False, handletextpad=0.3,
               borderaxespad=0.2)

    fig.tight_layout(h_pad=1.4)
    fig.savefig(OUT / "extended_frontier.pdf")
    plt.close(fig)


def _city_grid(ax, title: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_xticks(range(11))
    ax.set_yticks(range(9))
    ax.grid(True, color=FAINT, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelbottom=False, labelleft=False)
    for s in ax.spines.values():
        s.set_color(GRAY)
        s.set_linewidth(0.6)
    ax.set_title(title, loc="left", fontsize=8)


# Fixed, seed-free scatter geometry for the schematic (pure illustration).
TAXIS_ADV = [(1.4, 5.6), (2.2, 6.4), (2.9, 5.2), (1.8, 4.6), (3.3, 6.1),
             (2.6, 4.2), (1.2, 6.6), (3.6, 5.0), (2.0, 5.5), (3.0, 6.7)]
PICKUPS_ADV = [(1.9, 5.9), (2.7, 5.7), (2.4, 4.9), (3.2, 5.6)]
PICKUPS_DIS = [(7.4, 2.1), (8.2, 1.6), (7.9, 2.8), (8.6, 2.3), (7.1, 1.4),
               (8.9, 1.1), (7.6, 3.1), (8.4, 3.3)]
TAXI_DIS = [(7.8, 1.9)]
LIFT_PATH = [(4.6, 6.9), (5.4, 6.2), (6.1, 5.4), (6.7, 4.5)]
LIFT_BEND = [(7.2, 3.6), (7.7, 2.9), (8.1, 2.2)]


def _marks(ax, taxis, pickups):
    for x, y in taxis:
        ax.scatter(x, y, marker="s", s=12, facecolor="white", edgecolor=INK,
                   linewidth=0.7, zorder=3)
    for x, y in pickups:
        ax.scatter(x, y, marker="x", s=16, color=GRAY, linewidth=0.9, zorder=3)


def overview_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))

    # Panel 1 — the service gap
    ax = axes[0]
    _city_grid(ax, "1 · the service gap")
    _marks(ax, TAXIS_ADV, PICKUPS_ADV)
    _marks(ax, TAXI_DIS, PICKUPS_DIS)
    ax.annotate("over-served", (2.3, 7.1), fontsize=6.5, color=INK, ha="center")
    ax.annotate("under-served", (8.0, 4.0), fontsize=6.5, color=INK, ha="center")
    ax.scatter([], [], marker="s", s=12, facecolor="white", edgecolor=INK,
               label="taxi presence")
    ax.scatter([], [], marker="x", s=16, color=GRAY, label="pickup")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.2,
              borderaxespad=0.2)

    # Panel 2 — trim (demand relocation out of the hotspot)
    ax = axes[1]
    _city_grid(ax, "2 · trim: relocate excess pickups")
    _marks(ax, TAXIS_ADV, PICKUPS_ADV[2:])
    _marks(ax, TAXI_DIS, PICKUPS_DIS)
    for (x0, y0), (dx, dy) in [(PICKUPS_ADV[0], (1.5, 0.6)), (PICKUPS_ADV[1], (1.4, -0.8))]:
        ax.scatter(x0, y0, marker="x", s=16, color=FAINT, linewidth=0.9, zorder=2)
        ax.add_patch(FancyArrowPatch((x0, y0), (x0 + dx, y0 + dy),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.9, color=INK, zorder=4))
        ax.scatter(x0 + dx, y0 + dy, marker="x", s=16, color=GRAY,
                   linewidth=0.9, zorder=3)
    ax.annotate("gap closes from the top:\nunder-served untouched", (8.0, 6.7),
                fontsize=6.2, color=GRAY, ha="center")

    # Panel 3 — lift (tail reroute into the value-of-presence glow)
    ax = axes[2]
    _city_grid(ax, "3 · lift: reroute seeking tails")
    glow = Rectangle((6.6, 0.6), 2.9, 3.2, facecolor=ACCENT, alpha=0.12,
                     edgecolor="none", zorder=1)
    ax.add_patch(glow)
    ax.annotate("value of added\npresence (supply\ngradient)", (1.9, 1.9),
                fontsize=6.2, color=ACCENT, ha="center")
    ax.add_patch(FancyArrowPatch((3.4, 1.9), (6.4, 1.9), arrowstyle="-|>",
                                 mutation_scale=6, linewidth=0.6, color=ACCENT,
                                 alpha=0.7, zorder=2))
    _marks(ax, TAXI_DIS, PICKUPS_DIS[:5])
    xs, ys = zip(*LIFT_PATH)
    ax.plot(xs, ys, "-", color=INK, linewidth=1.0, zorder=3)
    bent = [LIFT_PATH[-1]] + LIFT_BEND
    bx, by = zip(*bent)
    ax.plot(bx, by, "--", color=ACCENT, linewidth=1.1, zorder=3)
    ax.add_patch(FancyArrowPatch(LIFT_BEND[-2], LIFT_BEND[-1], arrowstyle="-|>",
                                 mutation_scale=8, linewidth=1.1, color=ACCENT,
                                 zorder=4))
    ax.scatter(*LIFT_BEND[-1], marker="x", s=18, color=ACCENT, linewidth=1.1,
               zorder=4)
    ax.annotate("seeking tail\n(tapered reroute)", (3.3, 6.6), fontsize=6.2,
                color=INK, ha="center")

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "method_overview.pdf")
    plt.close(fig)


def dose_response_figure() -> None:
    """dose_response.pdf — downstream transfer vs upweighting dose (fig:dose).

    Replaces the dense per-dose paragraph in the Downstream Transfer subsection
    (Dr. Zhang, 2026-07-26: "A dose-response figure for downstream transfer
    would be much clearer than presenting all the values in a long paragraph").

    Everything is read from the committed sweep outputs; nothing is hard-coded.
    Two source directories are needed and BOTH are required: the primary sweep
    stops at w30, and the w40/w50 saturation points that make the knee visible
    live in the dose-extension suite.

    Honest-gap note: the random-subset control was never run at w20, so its line
    carries a real gap there. It is drawn as a break rather than interpolated,
    and the caption states the omission.

    All arms are n=6 paired seeds. The w30 flagship is additionally run at n=12
    (+0.0297) and the uniform-weight null at n=12 is +0.0016; both are different
    quantities from the n=6 points plotted here and are reported in the text.
    """
    base = ROOT / "famail_temporal/results/weighted_bc_sweep"
    srcs = ["alpha_sweep_s10_c80_f10_filtered_6seed", "alpha_sweep_s10_dose_ext_6seed"]

    arms: dict[str, dict[int, float]] = {"edited": {}, "most_fair": {}, "random": {}}
    vanilla = None
    for s in srcs:
        stats = json.loads((base / s / "paired_stats.json").read_text())
        block = stats.get("f_causal", stats)
        for key, val in block.items():
            if not isinstance(val, dict):
                continue
            mean = val.get("mean_delta", val.get("mean"))
            if mean is None:
                continue
            if key == "edited":                      # uniform weight == the null
                vanilla = mean
                continue
            arm, _, w = key.rpartition("_w")
            if arm in arms and w.isdigit():
                arms[arm][int(w)] = mean

    # w=1 anchors the edited curve at the uniform-weight null: the whole claim is
    # that this point is flat and the curve climbs away from it.
    if vanilla is not None:
        arms["edited"][1] = vanilla

    # 2026-07-26: aspect ratio cut from 2.00/3.35 = 0.60 to 1.45/3.35 = 0.43.
    # The graphic is drawn at \linewidth, so RENDERED HEIGHT is set by the aspect
    # ratio, not by figsize in inches — shrinking both axes equally would have
    # changed nothing on the page. Tight y-limits below recover the vertical
    # resolution the shorter box would otherwise cost.
    fig, ax = plt.subplots(figsize=(3.35, 1.45))
    ax.axhline(0.0, color=FAINT, linewidth=0.6, zorder=1)

    style = {
        "edited":    dict(label="edited (FATE)", marker="o", ls="-",
                          color=ACCENT, lw=1.2, ms=3.4, zorder=4),
        "most_fair": dict(label="most-fair control", marker="s", ls="--",
                          color=INK, lw=0.9, ms=3.0, zorder=3),
        "random":    dict(label="random control", marker="^", ls=":",
                          color=GRAY, lw=0.9, ms=3.0, zorder=2),
    }
    for arm, st in style.items():
        got = arms[arm]
        # Plot over the union grid with NaN at un-run doses, so a dose that was
        # never run reads as a BREAK in the line instead of a straight segment
        # implying an interpolated value (the random arm has no w20).
        grid = sorted(set(got) | ({10, 20, 30, 40, 50} if arm != "edited" else set()))
        ax.plot(grid, [got.get(w, float("nan")) for w in grid], **st)
        missing = [w for w in grid if w not in got]
        if missing:
            print(f"  note: {arm} has no data at w={missing} — drawn as a gap")

    # The uniform-weight null is the point the argument turns on. It is marked
    # with an OPEN marker (redundant with position, so it survives grayscale) and
    # named in the caption rather than in-axes: at 3.35in the only free space
    # around w=1 is against the y-axis, where the label clipped.
    if vanilla is not None:
        ax.scatter([1], [vanilla], s=30, marker="o", facecolor="white",
                   edgecolor=ACCENT, linewidth=0.9, zorder=5)

    ax.set_xlabel(r"upweighting factor $w$ on the edited demonstrations")
    ax.set_ylabel(r"paired $\Delta F_{\mathrm{demo}}$")
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(7.5)
    ax.yaxis.label.set_size(7.5)
    # Pad the data range by 8% rather than matplotlib's default 5% + legend
    # collision: the legend is what forced extra headroom before.
    vals = [v for d in arms.values() for v in d.values()]
    lo, hi = min(vals + [0.0]), max(vals)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.legend(frameon=False, loc="upper left", handlelength=1.8,
              borderaxespad=0.1, fontsize=6.5, labelspacing=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "dose_response.pdf")
    plt.close(fig)

    flat = {a: {w: round(v, 4) for w, v in sorted(d.items())} for a, d in arms.items()}
    print(f"dose_response.pdf plotted values: {flat}")


if __name__ == "__main__":
    frontier_figure()
    overview_figure()
    dose_response_figure()
    print(f"wrote {OUT / 'extended_frontier.pdf'}, {OUT / 'method_overview.pdf'} "
          f"and {OUT / 'dose_response.pdf'}")
