"""Headline figures for the FAMAIL paper (4-feature cleaned results).

Read-only over committed result JSONs + matplotlib (Agg). This module NEVER
touches source_data/cache, never calls DataBundle.load(), and never runs a
runner -- it is a pure file-read + plotting driver.

Figures (all written to ``results/analysis/figures_4feat/``):

  1. fig_dose_response.png    -- THE headline: edit vs select vs random, the
                                 upweight dose-response of Delta F_causal vs raw.
  2. fig_l1_data_quality.png  -- L1 per-source data quality (F_causal/F_spatial
                                 + Fidelity-A; GAN disqualified by Fidelity-B).
  3. fig_l2_negative_transfer -- vanilla BC averages it away; upweighting recovers.
  4. fig_fidb_components.png  -- Fidelity-B component breakdown (E9/E36).
  5. fig_feature_robustness   -- two feature sets: scale shifts, directional
                                 conclusions hold (null rows marked as nulls).

Pure, unit-tested helper: :func:`t_ci_from_values`.

CLI::

    python -m famail_temporal.analysis.paper_figures [--results-root PATH] [--out-dir PATH]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from famail_temporal.analysis._io import read_json

# ---------------------------------------------------------------------------
# Result-file locations (relative to a results root, default famail_temporal/results)
# ---------------------------------------------------------------------------
REL_SWEEP_4FEAT = "weighted_bc_sweep/cleaned_4feat_6seed/sweep.json"
REL_L1_4FEAT = "level1_table_v2/cleaned_4feat_5seed/level1_v2_multiseed.json"
REL_L2_4FEAT = "level2_table/cleaned_4feat_5seed/level2_metrics.json"
REL_VAR_4FEAT = "variance_suite/cleaned_4feat_5seed/aggregate.json"

REL_SWEEP_3FEAT = "weighted_bc_sweep/cleaned_6seed/sweep.json"
REL_L1_3FEAT = "level1_table_v2/cleaned_5seed/level1_v2_multiseed.json"
REL_L2_3FEAT = "level2_table/cleaned_5seed/level2_metrics.json"
REL_VAR_3FEAT = "variance_suite/cleaned_5seed/aggregate.json"

# PRIMARY equity set {housing, comp, migrant} ("hcm").
REL_SWEEP_HCM = "weighted_bc_sweep/cleaned_hcm_6seed/sweep.json"
REL_L1_HCM = "level1_table_v2/cleaned_hcm_5seed/level1_v2_multiseed.json"
REL_L2_HCM = "level2_table/cleaned_hcm_5seed/level2_metrics.json"
REL_VAR_HCM = "variance_suite/cleaned_hcm_5seed/aggregate.json"

# feat key -> (sweep, l1, l2, var) relative paths.
_FEAT_RELS = {
    "4feat": (REL_SWEEP_4FEAT, REL_L1_4FEAT, REL_L2_4FEAT, REL_VAR_4FEAT),
    "3feat": (REL_SWEEP_3FEAT, REL_L1_3FEAT, REL_L2_3FEAT, REL_VAR_3FEAT),
    "hcm": (REL_SWEEP_HCM, REL_L1_HCM, REL_L2_HCM, REL_VAR_HCM),
}

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"
DEFAULT_OUT_DIR = DEFAULT_RESULTS_ROOT / "analysis" / "figures_4feat"

# ---------------------------------------------------------------------------
# Style: one shared rcParams setup + colorblind-safe palette.
# Palette = Wong (2011) colorblind-safe set.
# ---------------------------------------------------------------------------
_WONG = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
}

# Stable semantic role -> color mapping used across figures.
COLORS = {
    "edited": _WONG["blue"],       # FAMAIL editing (the method)
    "most_fair": _WONG["orange"],  # selection baseline
    "random": _WONG["grey"],       # placebo
    "raw": _WONG["black"],
    "bc": _WONG["green"],
    "gan": _WONG["vermillion"],
    "3feat": _WONG["skyblue"],
    "4feat": _WONG["blue"],
}


def setup_style() -> None:
    """Apply one consistent matplotlib style. Forces the Agg backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "errorbar.capsize": 3,
        }
    )


# ---------------------------------------------------------------------------
# Pure helper (unit-tested)
# ---------------------------------------------------------------------------
def t_ci_from_values(values, confidence: float = 0.95):
    """Two-sided t confidence interval of the MEAN of ``values``.

    Returns ``(lo, hi)``. With fewer than two finite values the interval is
    undefined and ``(nan, nan)`` is returned. The half-width uses the Student-t
    quantile with ``n - 1`` degrees of freedom and the sample standard
    deviation (ddof=1), matching :func:`famail_temporal.baselines._enrich.t_ci`.

    Parameters
    ----------
    values : iterable of float
    confidence : float, default 0.95
        Central probability mass of the interval (0 < confidence < 1).
    """
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")
    from scipy.stats import t

    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1)) / math.sqrt(n)
    half = sem * float(t.ppf(0.5 + confidence / 2.0, n - 1))
    return (mean - half, mean + half)


def _ci_halfwidth(values, confidence: float = 0.95) -> float:
    """Symmetric half-width of the t-CI of the mean (for error bars). nan-safe."""
    lo, hi = t_ci_from_values(values, confidence)
    if math.isnan(lo) or math.isnan(hi):
        return float("nan")
    return (hi - lo) / 2.0


def _ci_half_from_std(std, n, confidence: float = 0.95) -> float:
    """t-CI half-width of the mean from a summary (std, n). nan-safe.

    Used when only aggregate {mean, std, n} is available (e.g. the variance
    suite's paired_delta) rather than per-seed values.
    """
    try:
        std = float(std)
        n = int(n)
    except (TypeError, ValueError):
        return float("nan")
    if n < 2 or not math.isfinite(std):
        return float("nan")
    from scipy.stats import t

    sem = std / math.sqrt(n)
    return sem * float(t.ppf(0.5 + confidence / 2.0, n - 1))


def _p_stars(p) -> str:
    """Significance marker. '*' p<=0.05, '**' p<=0.01, 'n.s.' otherwise / None."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "n.s."


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------
class ResultBundle:
    """Holds the four result JSONs for one feature-count variant."""

    def __init__(self, sweep: dict, l1: dict, l2: dict, var: dict):
        self.sweep = sweep
        self.l1 = l1
        self.l2 = l2
        self.var = var

    @classmethod
    def load(cls, root, *, feat: str) -> "ResultBundle":
        root = Path(root)
        try:
            rels = _FEAT_RELS[feat]
        except KeyError:
            raise ValueError(f"unknown feat variant: {feat!r}") from None
        return cls(*(read_json(root / r) for r in rels))


# ---------------------------------------------------------------------------
# Figure 1 -- dose-response (THE headline)
# ---------------------------------------------------------------------------
def fig_dose_response(bundle: ResultBundle, out_path: Path) -> Path:
    """Delta F_causal vs raw as a function of upweight dose, three series.

    x : upweight dose. w=1 is the no-upweight baseline (the ``edited`` arm,
        plotted only on the editing series at Delta~=0). w in {10,20,30}.
    y : paired mean Delta F_causal vs raw, with t-CI(n=6) error bars from
        per-seed diffs.
    series : edited_wN (FAMAIL, solid), most_fair_wN (selection, dashed),
             random_wN (placebo, dotted; only 10/30).
    Wilcoxon-p significance marked above each point.
    """
    import matplotlib.pyplot as plt

    pc = bundle.sweep["paired_vs_raw"]["f_causal"]

    def series(prefix, weights):
        xs, ys, errs, ps = [], [], [], []
        for w in weights:
            arm = "edited" if (prefix == "edited" and w == 1) else f"{prefix}_w{int(w)}"
            d = pc[arm]
            xs.append(w)
            ys.append(d["mean"])
            errs.append(_ci_halfwidth(d["diffs"]))
            ps.append(d.get("wilcoxon_p"))
        return xs, ys, errs, ps

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0.0, color=_WONG["grey"], lw=1.0, ls="-", zorder=0)

    specs = [
        ("edited", [1, 10, 20, 30], "-", "o", COLORS["edited"], "FAMAIL edit (upweighted)"),
        ("most_fair", [10, 20, 30], "--", "s", COLORS["most_fair"], "Most-fair selection"),
        ("random", [10, 30], ":", "^", COLORS["random"], "Random placebo"),
    ]
    for prefix, weights, ls, marker, color, label in specs:
        xs, ys, errs, ps = series(prefix, weights)
        ax.errorbar(
            xs, ys, yerr=errs, ls=ls, marker=marker, color=color,
            label=label, markersize=6, zorder=3,
        )
        is_random = prefix == "random"
        for x, y, e, p in zip(xs, ys, errs, ps):
            star = _p_stars(p)
            if not star:
                continue
            ehw = e if math.isfinite(e) else 0.0
            if star == "n.s.":
                # Place the (always-n.s.) random placebo's label BELOW its point
                # so it never occludes the significant most-fair '*' that shares
                # the same x at w=10/w=30. Non-random n.s. labels go above, small.
                if is_random:
                    ax.annotate(star, (x, y - ehw), textcoords="offset points",
                                xytext=(0, -9), ha="center", va="top",
                                fontsize=8, color=color, zorder=4)
                else:
                    ax.annotate(star, (x, y + ehw), textcoords="offset points",
                                xytext=(0, 6), ha="center", va="bottom",
                                fontsize=8, color=color, zorder=4)
            else:
                # Significant markers: larger + bold in the series color, drawn
                # last (high zorder) so a real effect is never hidden under a
                # null label that happens to share the same x.
                ax.annotate(star, (x, y + ehw), textcoords="offset points",
                            xytext=(0, 6), ha="center", va="bottom",
                            fontsize=13, fontweight="bold", color=color, zorder=6)

    ax.set_xticks([1, 10, 20, 30])
    ax.set_xticklabels(["1\n(no upweight)", "10", "20", "30"])
    ax.set_xlabel("Upweight factor on the edited demonstrations")
    ax.set_ylabel(r"$\Delta\,F_{\mathrm{causal}}$ vs raw (paired, mean $\pm$ 95% CI)")
    ax.set_title(
        "Editing dominates selection and placebo under upweighting\n"
        r"($*\ p\leq 0.05$, $**\ p\leq 0.01$, Wilcoxon; $n=6$ seeds)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2 -- L1 per-source data quality
# ---------------------------------------------------------------------------
def fig_l1_data_quality(bundle: ResultBundle, out_path: Path) -> Path:
    """Per-source (raw/edited/bc/gan) data-quality panel.

    Left  : grouped bars of F_causal and F_spatial (fairness; higher = fairer)
            with std error bars from the multiseed ``values``. ``edited`` marked
            as the fairest faithful source.
    Right : Fidelity-A (identity faithfulness) + Fidelity-B annotation; GAN is
            distributionally disqualified by its huge Fidelity-B.
    """
    import matplotlib.pyplot as plt

    ps = bundle.l1["per_source"]
    sources = ["raw", "edited", "bc", "gan"]
    labels = {"raw": "raw", "edited": "edited\n(FAMAIL)", "bc": "BC-gen", "gan": "GAN-gen"}
    bar_colors = [COLORS[s] for s in sources]

    def mean_std(src, metric):
        m = ps[src][metric]
        vals = m["values"]
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.4))
    x = np.arange(len(sources))
    width = 0.38

    # Left: F_causal & F_spatial grouped bars.
    fc = [mean_std(s, "f_causal") for s in sources]
    fsp = [mean_std(s, "f_spatial") for s in sources]
    axL.bar(
        x - width / 2, [m for m, _ in fc], width,
        yerr=[s for _, s in fc], color=bar_colors, edgecolor="black", lw=0.5,
        label=r"$F_{\mathrm{causal}}$",
    )
    axL.bar(
        x + width / 2, [m for m, _ in fsp], width,
        yerr=[s for _, s in fsp], color=bar_colors, edgecolor="black", lw=0.5,
        hatch="///", alpha=0.85, label=r"$F_{\mathrm{spatial}}$",
    )
    axL.set_xticks(x)
    axL.set_xticklabels([labels[s] for s in sources])
    axL.set_ylabel("Fairness (1 = fairest)")
    axL.set_title("Data-quality fairness per source\n(solid $F_{causal}$, hatched $F_{spatial}$)")
    # Mark edited as fairest-faithful.
    ed_fc = mean_std("edited", "f_causal")[0]
    axL.annotate(
        "fairest\nfaithful", xy=(1 - width / 2, ed_fc),
        xytext=(1 - width / 2, ed_fc + 0.06), ha="center", fontsize=8,
        arrowprops=dict(arrowstyle="->", color=COLORS["edited"]),
        color=COLORS["edited"],
    )
    axL.set_ylim(0, max(m for m, _ in fc) * 1.25)
    axL.legend(loc="upper right")

    # Right: Fidelity-A (identity faithfulness, higher = better separation).
    fa = [mean_std(s, "fidelity_a") for s in sources]
    fb = [mean_std(s, "fidelity_b") for s in sources]
    axR.bar(
        x, [m for m, _ in fa], 0.55,
        yerr=[s for _, s in fa], color=bar_colors, edgecolor="black", lw=0.5,
    )
    axR.set_xticks(x)
    axR.set_xticklabels([labels[s] for s in sources])
    axR.set_ylabel("Fidelity-A (identity faithfulness)")
    axR.set_title(
        "Identity faithfulness per source\n"
        "(zero-baseline: all sources within 0.006 — edited is faithful)"
    )
    # Zero baseline (matches the left panel). A truncated [0.80,0.86] window
    # exaggerated the <0.006 spread ~17x and made edited (the thesis bar) read
    # as least faithful; on a 0-1 axis the four sources are correctly ~identical.
    axR.set_ylim(0, 1.0)
    for xi, (m, _), (fbm, _fbs) in zip(x, fa, fb):
        axR.annotate(
            f"Fid-B\n{fbm:.3f}", xy=(xi, m), xytext=(xi, m + 0.02),
            ha="center", va="bottom", fontsize=7.5,
            color=(COLORS["gan"] if fbm > 0.1 else "black"),
        )
    # Call out GAN disqualification.
    gan_fb = mean_std("gan", "fidelity_b")[0]
    axR.annotate(
        f"GAN distributionally\ndisqualified (Fid-B={gan_fb:.2f})",
        xy=(3, fa[3][0]), xytext=(2.0, 0.55), ha="center", fontsize=8,
        color=COLORS["gan"],
        arrowprops=dict(arrowstyle="->", color=COLORS["gan"]),
    )

    fig.suptitle("L1: edited data is the fairest identity-faithful source", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3 -- L2 negative transfer vs weighted-BC recovery
# ---------------------------------------------------------------------------
def fig_l2_negative_transfer(bundle: ResultBundle, out_path: Path) -> Path:
    """Vanilla BC averages the edit away; upweighting recovers it.

    Point A : L2 vanilla edited-vs-raw paired Delta F_causal (baseline=edited,
              so ``paired.f_causal.raw.diffs`` = edited - raw), t-CI(n=5),
              crossing zero (n.s.).
    Point B : weighted-BC recovery, edited_w30 paired Delta from the sweep,
              t-CI(n=6), well above zero (significant).
    """
    import matplotlib.pyplot as plt

    l2 = bundle.l2["paired"]["f_causal"]["raw"]  # edited - raw, vanilla BC transfer
    l2_mean = l2["mean"]
    l2_err = _ci_halfwidth(l2["diffs"])
    l2_p = l2.get("wilcoxon_p")

    w30 = bundle.sweep["paired_vs_raw"]["f_causal"]["edited_w30"]
    w30_mean = w30["mean"]
    w30_err = _ci_halfwidth(w30["diffs"])
    w30_p = w30.get("wilcoxon_p")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(0.0, color=_WONG["grey"], lw=1.0, zorder=0)

    xs = [0, 1]
    means = [l2_mean, w30_mean]
    errs = [l2_err, w30_err]
    pts_colors = [COLORS["random"], COLORS["edited"]]
    plabels = [_p_stars(l2_p), _p_stars(w30_p)]

    for x, m, e, c, pl in zip(xs, means, errs, pts_colors, plabels):
        ax.errorbar([x], [m], yerr=[e], marker="o", markersize=10, color=c, zorder=3)
        # Label below-left so it never collides with the title band above.
        ax.annotate(
            f"{m:+.4f} ({pl})", (x, m), textcoords="offset points",
            xytext=(0, -22), ha="center", va="top", fontsize=9, color=c,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["vanilla BC\n(edited demos,\nw=1)", "weighted BC\n(edited demos,\nw=30)"]
    )
    ax.set_xlim(-0.6, 1.6)
    top = max(m + e for m, e in zip(means, errs))
    ax.set_ylim(min(means) - 0.006, top + 0.004)
    ax.set_ylabel(r"$\Delta\,F_{\mathrm{causal}}$ vs raw (paired, mean $\pm$ 95% CI)")
    ax.set_title(
        "Vanilla BC averages the edit away;\nupweighting recovers it"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4 -- Fidelity-B component breakdown (E9/E36)
# ---------------------------------------------------------------------------
def fig_fidb_components(bundle: ResultBundle, out_path: Path) -> Path:
    """Per-source Fidelity-B component breakdown.

    Sources edited/bc/gan; the six Fid-B components as grouped bars (mean over
    seeds, std error bars). Story: editing concentrates its distributional shift
    in terminal_cell (relocated pickups, ~0.55) — the next-largest component
    (net_disp ~0.13) is ~4x smaller. Only ``length`` (~0.01) is genuinely low;
    coverage (~0.09) and RoG (~0.10) are mid-range (comparable to GAN's
    net_disp/RoG), and edited net_disp slightly exceeds GAN's — so the claim is
    shift-concentrated-in-terminal_cell, NOT shape preservation per se. GAN
    inflates the shape components (length/coverage/RoG) on top of that.
    """
    import matplotlib.pyplot as plt

    comps = [
        "length", "mean_displacement", "coverage",
        "radius_of_gyration", "net_displacement", "terminal_cell",
    ]
    comp_labels = ["length", "mean\ndisp", "coverage", "radius\ngyr", "net\ndisp", "terminal\ncell"]
    sources = ["edited", "bc", "gan"]
    src_labels = {"edited": "edited (FAMAIL)", "bc": "BC-gen", "gan": "GAN-gen"}

    ps = bundle.l1["per_source"]

    def comp_mean_std(src, comp):
        vals = ps[src]["fidelity_b_per_component"][comp]["values"]
        return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(comps))
    width = 0.26
    offsets = {"edited": -width, "bc": 0.0, "gan": width}

    for src in sources:
        ms = [comp_mean_std(src, c) for c in comps]
        ax.bar(
            x + offsets[src], [m for m, _ in ms], width,
            yerr=[s for _, s in ms], color=COLORS[src], edgecolor="black", lw=0.4,
            label=src_labels[src],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Fidelity-B component (JS divergence; lower = closer to raw)")
    ax.set_title(
        "Editing concentrates its distributional shift in terminal-cell\n"
        "(relocated pickups); GAN-gen distorts every shape component"
    )
    ax.legend(loc="upper left")
    # Annotate the editing story on terminal_cell.
    ed_term = comp_mean_std("edited", "terminal_cell")[0]
    ax.annotate(
        "edit signal:\nrelocated pickups",
        xy=(5 + offsets["edited"], ed_term),
        xytext=(4.0, ed_term + 0.012), ha="center", fontsize=8,
        color=COLORS["edited"],
        arrowprops=dict(arrowstyle="->", color=COLORS["edited"]),
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure 5 -- 3-feature vs 4-feature demographic robustness
# ---------------------------------------------------------------------------
def _robustness_numbers(bundle: ResultBundle) -> dict:
    """Extract the four headline numbers (value, CI half-width, flags) per set.

    Each entry is {"val", "err", "deterministic", "null"}. ``err`` is a 95%
    t-CI half-width (nan if not estimable); ``deterministic`` marks data-level
    gaps with no sampling variance (the L1 edited-raw gap is a static rescore,
    std=0 by construction, so no CI applies); ``null`` marks rows whose CI
    straddles zero (so the figure does not present a null as a conclusion).
    """
    # Model-level variance-suite paired delta (b0 vs FAMAIL) — NOT the editor
    # before->after delta. This row is a known model-level null.
    var_fc = bundle.var["paired_delta"]["f_causal"]
    var_err = _ci_half_from_std(var_fc.get("std"), var_fc.get("n", 5))

    l1_gap = (
        bundle.l1["per_source"]["edited"]["f_causal"]["mean"]
        - bundle.l1["per_source"]["raw"]["f_causal"]["mean"]
    )

    w30 = bundle.sweep["paired_vs_raw"]["f_causal"]["edited_w30"]
    w30_err = _ci_halfwidth(w30["diffs"])

    l2 = bundle.l2["paired"]["f_causal"]["raw"]  # edited - raw, vanilla BC
    l2_err = _ci_halfwidth(l2["diffs"])

    def _entry(val, err, deterministic=False):
        is_null = (not deterministic) and math.isfinite(err) and abs(val) < err
        return {"val": val, "err": err, "deterministic": deterministic, "null": is_null}

    return {
        "model_level_delta": _entry(var_fc["mean"], var_err),
        "l1_edited_minus_raw": _entry(l1_gap, 0.0, deterministic=True),
        "weighted_bc_w30": _entry(w30["mean"], w30_err),
        "l2_edited_minus_raw": _entry(l2["mean"], l2_err),
    }


# Per-set plot style for the robustness dumbbell. The PRIMARY set is drawn
# black/diamond so it reads as the headline; the sensitivity sets are the
# blue family. Each tuple: (legend label, color, marker, value-label placement).
_ROBUSTNESS_STYLE = {
    "hcm":   ("{housing,comp,migrant} (PRIMARY)", _WONG["black"], "D", "above"),
    "3feat": ("{housing,gdp,comp}", COLORS["3feat"], "o", "below"),
    "4feat": ("{housing,comp,migrant,logpop}", COLORS["4feat"], "s", "above2"),
}


def fig_feature_robustness(sets, out_path: Path) -> Path:
    """Dumbbell comparison of the four headline numbers across feature sets.

    ``sets`` is an ordered list of ``(feat_key, ResultBundle)`` pairs (2 or 3
    entries); the first is treated as the PRIMARY/reference. Story: the absolute
    F_causal scale shifts with the demographic feature set, but the DIRECTIONAL
    conclusions hold. Two of the four rows (model-level variance-suite delta and
    vanilla-BC L2 transfer) are nulls by design and are marked as such — they are
    nulls that reproduce, not conclusions that hold.
    """
    import matplotlib.pyplot as plt

    nums = [(fk, _robustness_numbers(b)) for fk, b in sets]

    keys = ["model_level_delta", "l1_edited_minus_raw", "weighted_bc_w30", "l2_edited_minus_raw"]
    nice = {
        "model_level_delta": r"Model-level $\Delta F_{causal}$" + "\n(variance suite, b0 vs FAMAIL; null)",
        "l1_edited_minus_raw": "L1 edited$-$raw\n" + r"$F_{causal}$ gap (deterministic)",
        "weighted_bc_w30": r"Weighted-BC $\Delta F_{causal}$" + "\n(edited, w=30)",
        "l2_edited_minus_raw": "L2 edited$-$raw\n(vanilla BC; null)",
    }
    y = np.arange(len(keys))[::-1]  # top-to-bottom in listed order

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axvline(0.0, color=_WONG["grey"], lw=1.0, zorder=0)

    def _xerr(entry):
        e = entry["err"]
        return 0.0 if (not math.isfinite(e)) else e

    # Pure-vertical stagger so labels never collide horizontally with an
    # adjacent set's marker (the sets share a row but sit at different x).
    place = {"above": dict(xytext=(0, 11), ha="center", va="bottom"),
             "below": dict(xytext=(0, -13), ha="center", va="top"),
             "above2": dict(xytext=(0, 26), ha="center", va="bottom")}

    for yi, k in zip(y, keys):
        vals = [(fk, n[k]) for fk, n in nums]
        xs = [e["val"] for _, e in vals]
        # connect the spread of points on this row
        ax.plot([min(xs), max(xs)], [yi, yi], color=_WONG["grey"], lw=1.2, ls=":", zorder=1)
        any_null = False
        for fk, e in vals:
            label_txt, color, marker, placement = _ROBUSTNESS_STYLE[fk]
            primary = (fk == nums[0][0])
            ax.errorbar([e["val"]], [yi], xerr=[_xerr(e)], fmt=marker, color=color,
                        markersize=10 if primary else 8,
                        markeredgecolor="black" if primary else color,
                        markeredgewidth=1.0 if primary else 0.0,
                        zorder=4 if primary else 3,
                        label=label_txt if k == keys[0] else None)
            ax.annotate(f"{e['val']:+.4f}", (e["val"], yi), textcoords="offset points",
                        fontsize=7.5, color=color, fontweight="bold" if primary else "normal",
                        **place[placement])
            any_null = any_null or e["null"]
        if any_null:
            ax.annotate("null (CI ∋ 0)", (0.0, yi), textcoords="offset points",
                        xytext=(6, 0), ha="left", va="center", fontsize=7,
                        color=_WONG["grey"], style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([nice[k] for k in keys])
    ax.set_xlabel(r"$\Delta\,F_{\mathrm{causal}}$ (paired/gap, mean $\pm$ 95% CI)")
    ax.set_title(
        "Demographic robustness: scale shifts; directional conclusions hold across feature sets\n"
        "(model-level & vanilla-BC rows are null/within noise; edit targeting Jaccard ≥ 0.92, housing-retaining family)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT),
        help="Root of the results tree (default: famail_temporal/results).",
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR),
        help="Output directory for PNGs (default: results/analysis/figures_4feat).",
    )
    parser.add_argument(
        "--feat", default="4feat", choices=sorted(_FEAT_RELS),
        help="Primary feature set for the four single-set figures.",
    )
    parser.add_argument(
        "--compare-feat", default="3feat",
        help="Comma-separated feature sets for the cross-set robustness dumbbell "
             "(e.g. '3feat,4feat'), or 'none' to skip. The --feat set leads.",
    )
    args = parser.parse_args(argv)

    root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    primary = ResultBundle.load(root, feat=args.feat)

    written = []
    written.append(fig_dose_response(primary, out_dir / "fig_dose_response.png"))
    written.append(fig_l1_data_quality(primary, out_dir / "fig_l1_data_quality.png"))
    written.append(fig_l2_negative_transfer(primary, out_dir / "fig_l2_negative_transfer.png"))
    written.append(fig_fidb_components(primary, out_dir / "fig_fidb_components.png"))

    # Cross-set robustness dumbbell: the --feat set plus every --compare-feat set
    # whose results are on disk (PRIMARY leads). Skips missing sets gracefully.
    compare = [] if args.compare_feat == "none" else [c for c in args.compare_feat.split(",") if c]
    sets = [(args.feat, primary)]
    for fk in compare:
        if fk == args.feat:
            continue
        try:
            sets.append((fk, ResultBundle.load(root, feat=fk)))
        except FileNotFoundError:
            print(f"[skip] robustness comparison set {fk!r}: results not found")
    if len(sets) >= 2:
        written.append(fig_feature_robustness(sets, out_dir / "fig_feature_robustness.png"))

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
