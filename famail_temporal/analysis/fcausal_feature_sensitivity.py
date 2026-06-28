"""Feature-set sensitivity analysis for F_causal (paper-defensibility deliverable).

Question answered
-----------------
Are FAMAIL's F_causal conclusions — the headline value *and* which cells the
editor targets — robust to *which* demographic variables enter the causal
fairness metric?

Method
------
F_causal = R'(I − H_demo)R / R'MR = 1 − r²_demo, where

    R = Y − g₀(D),   Y = S / max(D, DEMAND_FLOOR)

is the Stage-1 demand residual that does **not** depend on the demographic
choice. Only H_demo — the projection onto [intercept | z-scored chosen
features] — changes with the feature set. So this analysis:

  1. Builds R **once**, exactly as production does (see
     ``evaluation.export_fairness_attributions._scalar_F_metrics`` and
     ``fairness.causal.compute_fcausal_from_compact``), on the shipped
     cleaned active mask / unit ordering (from cache/). R, M, and the
     active set are held fixed across every subset.
  2. For each candidate demographic subset, re-derives the per-unit
     demographic matrix via the **real** ``enrich_demographics`` + the
     canonical ``unit_map`` ordering, then calls the **real**
     ``precompute_hat_matrices`` → ``compute_fcausal_compact`` /
     ``per_cell_fairness_attribution_causal``. No reimplementation of the
     metric — only the inputs (the chosen demographic columns) change.

Everything is computed IN MEMORY. This module never mutates
``config.DEMOGRAPHIC_FEATURES``, ``source_data/``, or ``cache/``.

Sanity gate
-----------
Before trusting any sweep result, the baseline subset
{AvgHousingPricePerSqM, GDPperCapita, CompPerCapita} is recomputed and must
match the editor's measured before-edit value F_causal = 0.8069 to ~1e-3
(``results/2026-06-26T12-32-59_..._cleaned/metrics.json`` →
``metrics_before.f_causal``). It is checked **twice**: once with the cached
hat matrices straight from disk (proves the R-construction path), and once
with hat matrices rebuilt from scratch through ``enrich_demographics`` (proves
the from-scratch demographic path that the sweep relies on). If either fails,
the run aborts.

CLI
---
    python -m famail_temporal.analysis.fcausal_feature_sensitivity
        [--out-dir DIR] [--top-k N] [--no-gate-abort]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from famail_temporal import config
from famail_temporal.data.active_mask import UnitIndexMap
from famail_temporal.data.cache_io import load_artifact, load_raw
from famail_temporal.data.demographics import enrich_demographics
from famail_temporal.fairness.causal import per_cell_fairness_attribution_causal
from famail_temporal.fairness.hat_matrices import (
    compute_fcausal_compact,
    hat_matrices_to_torch,
    precompute_hat_matrices,
)

# Editor-measured before-edit F_causal on the CLEANED data (the value the
# whole metric must reproduce). Source:
# results/2026-06-26T12-32-59_k-10000_causal_emphasis_no-dedup_cleaned/
#   metrics.json -> metrics_before.f_causal
EXPECTED_BASELINE_FCAUSAL: float = 0.8069279193878174
GATE_TOL: float = 1e-3

# Edited-N: the number of trajectories the editor moved on the cleaned data
# (modified_trajectory_ids.json). The top-K most-unfair-cell overlap uses
# K = this value — "does the editor target the same cells under a different
# feature set?" measured at the budget the editor actually spends.
DEFAULT_TOP_K: int = 2293

# Default output location (gitignored — results/* is ignored).
DEFAULT_OUT_DIR = (
    config.PACKAGE_ROOT / "results" / "analysis" / "fcausal_feature_sensitivity"
)

# Candidate subsets. Each is (label, [feature_names]). Feature names must
# resolve against the enriched demographic name list (raw + derived); any that
# don't are skipped and noted.
CANDIDATE_SUBSETS: List[Tuple[str, List[str]]] = [
    ("baseline", ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita"]),
    ("+popdensity",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita", "LogPopDensity"]),
    ("+migrant",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita", "MigrantRatio"]),
    ("drop_gdp", ["AvgHousingPricePerSqM", "CompPerCapita"]),
    ("drop_comp", ["AvgHousingPricePerSqM", "GDPperCapita"]),
    ("drop_housing", ["GDPperCapita", "CompPerCapita"]),
    ("logs", ["LogHousingPrice", "LogGDP", "LogCompensation"]),
    ("broad5",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita",
      "LogPopDensity", "MigrantRatio"]),
]

BASELINE_LABEL = "baseline"


# ---------------------------------------------------------------------------
# Data assembly (in-memory; reuses real production functions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _SensitivityInputs:
    """Everything the sweep needs, built once from the cleaned cache.

    R is the fixed Stage-1 residual; ``unit_demo_all`` is the per-unit matrix
    of *all* candidate demographic features in canonical unit order; the rest
    are bookkeeping.
    """
    R: torch.Tensor                  # (N,) fixed residual, float32 (prod dtype)
    D_clamped: torch.Tensor          # (N,) demand input to precompute_hat
    unit_demo_all: np.ndarray        # (N, n_all_feats) per-unit demographics
    all_feature_names: List[str]     # column names for unit_demo_all
    n_units: int
    cached_hat: Dict[str, np.ndarray]  # shipped hat matrices (baseline subset)


def _build_inputs() -> _SensitivityInputs:
    """Assemble R + the all-feature per-unit demographic matrix from cache.

    Mirrors ``preprocess.run`` (Phases 1, 5, 6) and the production R-build in
    ``export_fairness_attributions._scalar_F_metrics`` exactly. Reads only;
    never writes cache/ or source_data/.
    """
    # --- cached active-set artifacts (the shipped CLEANED grid) ---
    pickup_3d = load_artifact("pickup_counts")
    active_taxis_3d = load_artifact("active_taxis")
    mask_3d = load_artifact("active_mask")
    unit_map: UnitIndexMap = load_artifact("unit_index_map")
    g0_func = load_artifact("g0_power_basis")
    cached_hat = load_artifact("hat_matrices", include_features=True)

    # --- fixed residual R (production path, verbatim) ---
    pickup_N = torch.from_numpy(pickup_3d[mask_3d]).float()
    active_N = torch.from_numpy(active_taxis_3d[mask_3d]).float()
    D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
    g0_D = torch.from_numpy(
        np.asarray(g0_func(D_clamped.numpy()), dtype=np.float32)
    )
    # R = Y - g0(D); compute_fcausal_from_compact does this internally, but the
    # sweep needs R explicitly for the per-cell attribution call.
    Y = active_N / D_clamped
    R = Y - g0_D

    # --- per-unit demographics for ALL candidate features ---
    # Re-derive via the real enrich_demographics, then index through the
    # canonical unit ordering exactly as preprocess Phase 5 does.
    demographics_raw = load_raw("cell_demographics.pkl")
    demographics_grid = demographics_raw["demographics_grid"]
    demo_feature_names = list(demographics_raw["feature_names"])
    demographics_grid, all_feature_names = enrich_demographics(
        demographics_grid, demo_feature_names
    )
    gy = unit_map.grid_shape[1]
    n_feats_all = demographics_grid.shape[-1]
    demo_flat = demographics_grid.reshape(-1, n_feats_all)  # (gx*gy, n_feats)

    n_units = unit_map.n_units
    unit_demo_all = np.empty((n_units, n_feats_all), dtype=np.float64)
    for i in range(n_units):
        flat_cell = unit_map.to_flat_cell(i)
        unit_demo_all[i] = demo_flat[flat_cell]

    # Cross-check: flat_cell encoding here must match grid reshape encoding.
    # to_flat_cell returns x*gy + y; demo_flat row index is x*gy + y. Consistent.
    assert gy == demographics_grid.shape[1], (
        f"grid_shape gy={gy} disagrees with demographics grid "
        f"y-dim={demographics_grid.shape[1]}"
    )

    return _SensitivityInputs(
        R=R,
        D_clamped=D_clamped,
        unit_demo_all=unit_demo_all,
        all_feature_names=all_feature_names,
        n_units=n_units,
        cached_hat=cached_hat,
    )


# ---------------------------------------------------------------------------
# Per-subset computation (real production functions)
# ---------------------------------------------------------------------------

def _vifs(demo_subset: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Variance-inflation factors for the chosen (un-standardized) features.

    VIF_j = 1 / (1 − R²_j) where R²_j regresses feature j on the other
    features (with intercept). Standardization does not change VIF, so this is
    computed directly on the raw subset columns. Returns +inf on perfect
    collinearity.
    """
    demo = np.asarray(demo_subset, dtype=np.float64)
    n, p = demo.shape
    out: Dict[str, float] = {}
    if p == 1:
        # A single predictor has no other regressors → VIF is 1 by definition.
        return {feature_names[0]: 1.0}
    for j in range(p):
        y = demo[:, j]
        others = np.delete(demo, j, axis=1)
        X = np.column_stack([np.ones(n), others])
        # Least squares; guard rank deficiency.
        beta, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-300:
            out[feature_names[j]] = float("inf")
            continue
        r2 = 1.0 - ss_res / ss_tot
        out[feature_names[j]] = (
            float("inf") if (1.0 - r2) < 1e-12 else float(1.0 / (1.0 - r2))
        )
    return out


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (ties averaged). No scipy dependency needed."""
    ar = _rankdata_average(a)
    br = _rankdata_average(b)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = np.sqrt(np.sum(ar * ar) * np.sum(br * br))
    if denom < 1e-300:
        return float("nan")
    return float(np.sum(ar * br) / denom)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Rank with average ties (matches scipy.stats.rankdata default)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


@dataclass
class SubsetResult:
    label: str
    features: List[str]
    skipped: bool = False
    skip_reason: Optional[str] = None
    f_causal: Optional[float] = None
    vifs: Dict[str, float] = field(default_factory=dict)
    # Robustness vs baseline (None for the baseline row itself):
    topk_jaccard: Optional[float] = None
    spearman_alpha: Optional[float] = None
    # Per-cell attribution, kept in-memory for overlap computation.
    alpha: Optional[np.ndarray] = field(default=None, repr=False)


def _compute_subset(
    label: str,
    features: List[str],
    inputs: _SensitivityInputs,
) -> SubsetResult:
    """Recompute F_causal + per-cell αᵢ + VIFs for one subset via real funcs."""
    name_to_idx = {n: i for i, n in enumerate(inputs.all_feature_names)}
    missing = [f for f in features if f not in name_to_idx]
    if missing:
        return SubsetResult(
            label=label, features=features, skipped=True,
            skip_reason=f"features not in enriched demographics: {missing}",
        )

    col_idx = [name_to_idx[f] for f in features]
    demo_subset = inputs.unit_demo_all[:, col_idx]  # (N, p), float64

    if not np.all(np.isfinite(demo_subset)):
        # The shipped active mask was built on baseline-feature finiteness; a
        # different subset could in principle introduce NaN (cell finite for
        # baseline feats but not for the new one). Surface, don't crash.
        n_bad = int(np.sum(~np.isfinite(demo_subset).all(axis=1)))
        return SubsetResult(
            label=label, features=features, skipped=True,
            skip_reason=(
                f"{n_bad} active units have non-finite values for this subset "
                f"(active mask was calibrated on the baseline features)"
            ),
        )

    vifs = _vifs(demo_subset, features)

    # Real production hat-matrix builder + compact F_causal. precompute does
    # its own standardization, rank check, and zero-variance preflight; if a
    # subset is degenerate it raises — we capture that as a skip.
    try:
        hat = precompute_hat_matrices(
            demands=inputs.D_clamped.numpy().astype(np.float64),
            demographic_features=demo_subset,
            feature_names=list(features),
        )
    except (ValueError, AssertionError, RuntimeError) as e:
        return SubsetResult(
            label=label, features=features, skipped=True,
            skip_reason=f"precompute_hat_matrices rejected subset: {e}",
            vifs=vifs,
        )

    tensors = hat_matrices_to_torch(hat)
    f_causal = float(
        compute_fcausal_compact(inputs.R, tensors["X_demo"], tensors["XtX_inv"])
    )
    alpha = per_cell_fairness_attribution_causal(
        inputs.R, tensors["X_demo"], tensors["XtX_inv"]
    ).numpy()

    return SubsetResult(
        label=label, features=features, skipped=False,
        f_causal=f_causal, vifs=vifs, alpha=alpha,
    )


def _topk_unfair_indices(alpha: np.ndarray, k: int) -> np.ndarray:
    """Indices of the K *most unfair* cells = K smallest (most negative) αᵢ.

    Per the attribution sign convention (positive = more fair, negative = drags
    fairness below baseline / priority for modification), the most-unfair cells
    are the smallest αᵢ. These are exactly the cells the editor prioritises.
    """
    k = min(k, len(alpha))
    return np.argsort(alpha, kind="mergesort")[:k]


def _jaccard(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    sa, sb = set(idx_a.tolist()), set(idx_b.tolist())
    union = len(sa | sb)
    if union == 0:
        return float("nan")
    return len(sa & sb) / union


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class SensitivityReport:
    gate_passed: bool
    gate_detail: Dict[str, float]
    top_k: int
    n_units: int
    results: List[SubsetResult]


def run_sensitivity(top_k: int = DEFAULT_TOP_K) -> SensitivityReport:
    """Run the full sweep in memory. Does not write anything."""
    inputs = _build_inputs()

    # --- SANITY GATE ---------------------------------------------------------
    # (a) cached hat matrices straight from disk → proves the R path.
    cached_tensors = hat_matrices_to_torch(inputs.cached_hat)
    f_cached = float(
        compute_fcausal_compact(
            inputs.R, cached_tensors["X_demo"], cached_tensors["XtX_inv"]
        )
    )
    # (b) from-scratch rebuild of the baseline subset → proves the demo path.
    baseline_features = dict(CANDIDATE_SUBSETS)[BASELINE_LABEL]
    baseline_scratch = _compute_subset(
        BASELINE_LABEL, baseline_features, inputs
    )
    f_scratch = baseline_scratch.f_causal if not baseline_scratch.skipped else None

    gate_detail = {
        "expected": EXPECTED_BASELINE_FCAUSAL,
        "recomputed_cached_hat": f_cached,
        "recomputed_from_scratch": (
            float(f_scratch) if f_scratch is not None else float("nan")
        ),
        "tol": GATE_TOL,
        "abs_err_cached": abs(f_cached - EXPECTED_BASELINE_FCAUSAL),
        "abs_err_scratch": (
            abs(f_scratch - EXPECTED_BASELINE_FCAUSAL)
            if f_scratch is not None else float("nan")
        ),
    }
    gate_passed = (
        gate_detail["abs_err_cached"] <= GATE_TOL
        and f_scratch is not None
        and gate_detail["abs_err_scratch"] <= GATE_TOL
    )

    # --- SWEEP ---------------------------------------------------------------
    results: List[SubsetResult] = []
    for label, feats in CANDIDATE_SUBSETS:
        if label == BASELINE_LABEL:
            res = baseline_scratch  # reuse the gate computation
        else:
            res = _compute_subset(label, feats, inputs)
        results.append(res)

    # Baseline top-K reference (for overlap). Only if baseline computed.
    baseline_res = next(r for r in results if r.label == BASELINE_LABEL)
    if not baseline_res.skipped and baseline_res.alpha is not None:
        base_topk = _topk_unfair_indices(baseline_res.alpha, top_k)
        for r in results:
            if r.skipped or r.alpha is None:
                continue
            if r.label == BASELINE_LABEL:
                r.topk_jaccard = 1.0
                r.spearman_alpha = 1.0
                continue
            r_topk = _topk_unfair_indices(r.alpha, top_k)
            r.topk_jaccard = _jaccard(base_topk, r_topk)
            r.spearman_alpha = _spearman(baseline_res.alpha, r.alpha)

    return SensitivityReport(
        gate_passed=gate_passed,
        gate_detail=gate_detail,
        top_k=top_k,
        n_units=inputs.n_units,
        results=results,
    )


# ---------------------------------------------------------------------------
# Verdict + serialization
# ---------------------------------------------------------------------------

def _verdict(report: SensitivityReport) -> Dict[str, object]:
    """Derive a ROBUST/FRAGILE verdict from the computed (non-baseline) subsets.

    Robust iff the F_causal spread across all *defensible* recomputed subsets is
    small AND every recomputed subset keeps high editor-targeting agreement
    (top-K Jaccard and Spearman). Thresholds are conservative and reported so a
    reader can re-judge.
    """
    computed = [r for r in report.results if not r.skipped and r.f_causal is not None]
    fcausals = [r.f_causal for r in computed]
    f_spread = (max(fcausals) - min(fcausals)) if fcausals else float("nan")

    non_base = [r for r in computed if r.label != BASELINE_LABEL]
    jaccards = [r.topk_jaccard for r in non_base if r.topk_jaccard is not None]
    spearmans = [r.spearman_alpha for r in non_base if r.spearman_alpha is not None]
    min_jaccard = min(jaccards) if jaccards else float("nan")
    min_spearman = min(spearmans) if spearmans else float("nan")

    # Thresholds (documented, conservative).
    F_SPREAD_ROBUST = 0.05     # absolute F_causal range across subsets
    JACCARD_ROBUST = 0.60      # min top-K overlap with baseline
    SPEARMAN_ROBUST = 0.90     # min per-cell α rank-correlation

    robust = (
        f_spread <= F_SPREAD_ROBUST
        and (not jaccards or min_jaccard >= JACCARD_ROBUST)
        and (not spearmans or min_spearman >= SPEARMAN_ROBUST)
    )

    # Identify the worst offenders for the justification line.
    worst_jaccard = min(non_base, key=lambda r: r.topk_jaccard) if jaccards else None
    return {
        "verdict": "ROBUST" if robust else "FRAGILE",
        "f_causal_spread": f_spread,
        "min_topk_jaccard": min_jaccard,
        "min_spearman_alpha": min_spearman,
        "thresholds": {
            "f_spread_robust": F_SPREAD_ROBUST,
            "jaccard_robust": JACCARD_ROBUST,
            "spearman_robust": SPEARMAN_ROBUST,
        },
        "worst_jaccard_subset": (
            worst_jaccard.label if worst_jaccard is not None else None
        ),
        "worst_jaccard_value": (
            worst_jaccard.topk_jaccard if worst_jaccard is not None else None
        ),
    }


def _to_json(report: SensitivityReport) -> Dict[str, object]:
    verdict = _verdict(report)
    rows = []
    for r in report.results:
        rows.append({
            "label": r.label,
            "features": r.features,
            "skipped": r.skipped,
            "skip_reason": r.skip_reason,
            "f_causal": r.f_causal,
            "vifs": r.vifs,
            "topk_jaccard": r.topk_jaccard,
            "spearman_alpha": r.spearman_alpha,
        })
    return {
        "metric": "F_causal feature-set sensitivity (cleaned data, before-edit)",
        "expected_baseline_f_causal": EXPECTED_BASELINE_FCAUSAL,
        "sanity_gate": {
            "passed": report.gate_passed,
            **report.gate_detail,
        },
        "top_k": report.top_k,
        "n_units": report.n_units,
        "subsets": rows,
        "verdict": verdict,
    }


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):  # NaN
        return "nan"
    if isinstance(x, float) and x == float("inf"):
        return "inf"
    return f"{x:.{nd}f}"


def _to_markdown(report: SensitivityReport) -> str:
    payload = _to_json(report)
    gate = payload["sanity_gate"]
    verdict = payload["verdict"]
    lines: List[str] = []
    lines.append("# F_causal feature-set sensitivity analysis (cleaned data)")
    lines.append("")
    lines.append(
        "Recomputes the **before-edit** F_causal and its per-cell attribution "
        "for alternative demographic feature subsets, reusing the production "
        "compute path (`precompute_hat_matrices` → `compute_fcausal_compact` / "
        "`per_cell_fairness_attribution_causal`). The Stage-1 residual R, the "
        "centering M, and the active set are held **fixed**; only the "
        "demographic projection H_demo changes."
    )
    lines.append("")
    lines.append("## Sanity gate")
    lines.append("")
    lines.append(f"- expected (editor before-edit): **{_fmt(gate['expected'], 6)}**")
    lines.append(
        f"- recomputed (cached hat matrices): "
        f"**{_fmt(gate['recomputed_cached_hat'], 6)}** "
        f"(|Δ| = {_fmt(gate['abs_err_cached'], 2 + 6)})"
    )
    lines.append(
        f"- recomputed (from-scratch demographics): "
        f"**{_fmt(gate['recomputed_from_scratch'], 6)}** "
        f"(|Δ| = {_fmt(gate['abs_err_scratch'], 2 + 6)})"
    )
    lines.append(f"- tolerance: {_fmt(gate['tol'], 6)}")
    lines.append(
        f"- **GATE {'PASSED' if gate['passed'] else 'FAILED'}**"
    )
    lines.append("")
    lines.append("## F_causal per subset")
    lines.append("")
    lines.append(
        f"Top-K overlap uses K = {payload['top_k']} (edited-N); N_units = "
        f"{payload['n_units']}. Jaccard/Spearman are vs the baseline subset."
    )
    lines.append("")
    lines.append(
        "| subset | features | F_causal | top-K Jaccard | Spearman α | "
        "max VIF | status |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in report.results:
        feats = ", ".join(r.features)
        if r.skipped:
            status = f"SKIPPED: {r.skip_reason}"
            lines.append(
                f"| {r.label} | {feats} | — | — | — | "
                f"{_fmt(max(r.vifs.values()) if r.vifs else None, 2)} | {status} |"
            )
            continue
        max_vif = max(r.vifs.values()) if r.vifs else None
        lines.append(
            f"| {r.label} | {feats} | {_fmt(r.f_causal, 4)} | "
            f"{_fmt(r.topk_jaccard, 4)} | {_fmt(r.spearman_alpha, 4)} | "
            f"{_fmt(max_vif, 2)} | ok |"
        )
    lines.append("")
    lines.append("### Per-feature VIFs")
    lines.append("")
    for r in report.results:
        if r.skipped and not r.vifs:
            continue
        vif_str = ", ".join(f"{k}={_fmt(v, 2)}" for k, v in r.vifs.items())
        lines.append(f"- **{r.label}**: {vif_str}")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{verdict['verdict']}**")
    lines.append("")
    lines.append(
        f"- F_causal spread across recomputed subsets: "
        f"{_fmt(verdict['f_causal_spread'], 4)} "
        f"(robust threshold ≤ {verdict['thresholds']['f_spread_robust']})"
    )
    lines.append(
        f"- min top-K Jaccard vs baseline: {_fmt(verdict['min_topk_jaccard'], 4)} "
        f"(robust threshold ≥ {verdict['thresholds']['jaccard_robust']}) "
        f"[worst: {verdict['worst_jaccard_subset']} = "
        f"{_fmt(verdict['worst_jaccard_value'], 4)}]"
    )
    lines.append(
        f"- min Spearman α vs baseline: {_fmt(verdict['min_spearman_alpha'], 4)} "
        f"(robust threshold ≥ {verdict['thresholds']['spearman_robust']})"
    )
    lines.append("")
    return "\n".join(lines)


def write_report(report: SensitivityReport, out_dir: Path) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fcausal_feature_sensitivity.json"
    md_path = out_dir / "fcausal_feature_sensitivity.md"
    json_path.write_text(json.dumps(_to_json(report), indent=2, default=str))
    md_path.write_text(_to_markdown(report))
    return json_path, md_path


# ===========================================================================
# DEEP FEATURE-SELECTION ANALYSIS (PI extension)
# ---------------------------------------------------------------------------
# Goal: decide whether to ADOPT a better demographic feature set before the
# paper re-run. A "valid" feature is RELEVANT (plausible driver of where taxis
# serve), EFFECTIVE (explains demand-residual variance → lowers F_causal), and
# LOW co-linearity (VIF < 10), subject to ≤5 features (only 10 districts of DOF).
#
# All of this reuses the same fixed-R, real-production compute path as the
# sweep above. No config / source_data / cache mutation.
# ===========================================================================

# Base set the project currently ships.
BASE3: List[str] = ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita"]

# VIF / DOF policy.
VIF_LIMIT: float = 10.0          # PI's acceptable-collinearity ceiling
MAX_FEATURES: int = 5            # only 10 districts → keep DOF sane

# Candidate pool for the marginal table + correlation/VIF matrix. Grouped by
# the distinct demographic AXES so we can see which axes are independent:
#   SES/housing | income | population-structure | density | other
CANDIDATE_POOL: List[str] = [
    # SES / housing
    "AvgHousingPricePerSqM", "LogHousingPrice",
    # income
    "GDPperCapita", "CompPerCapita", "EmployeeCompensation100MYuan",
    "GDPin10000Yuan", "LogGDP", "LogCompensation",
    # population structure
    "MigrantRatio", "NonRegisteredPermanentPop10k", "SexRatio100",
    # density
    "PopDensityPerKm2", "LogPopDensity",
    # other plausibly-relevant scale axes
    "YearEndPermanentPop10k", "AvgEmployedPersons",
]

# Axis membership (for "spans distinct axes" reasoning in the set search).
FEATURE_AXIS: Dict[str, str] = {
    "AvgHousingPricePerSqM": "housing", "LogHousingPrice": "housing",
    "GDPperCapita": "income", "CompPerCapita": "income",
    "EmployeeCompensation100MYuan": "income", "GDPin10000Yuan": "income",
    "LogGDP": "income", "LogCompensation": "income",
    "MigrantRatio": "pop_structure",
    "NonRegisteredPermanentPop10k": "pop_structure", "SexRatio100": "pop_structure",
    "PopDensityPerKm2": "density", "LogPopDensity": "density",
    "YearEndPermanentPop10k": "scale", "AvgEmployedPersons": "scale",
}

# Curated feature SETS to evaluate (sizes 3–5), emphasizing distinct-axis spans.
# Includes the PI's requested sets plus our own picks. The first entry is the
# current baseline (reference for Jaccard/Spearman).
CURATED_SETS: List[Tuple[str, List[str]]] = [
    ("baseline_h-g-c", BASE3),
    # swap one income feature for a migrant axis (size 3, distinct axes)
    ("h-g-migrant", ["AvgHousingPricePerSqM", "GDPperCapita", "MigrantRatio"]),
    ("h-c-migrant", ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]),
    # add migrant to the base-3 (size 4) — the "just add migrant" set
    ("h-g-c-migrant",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita", "MigrantRatio"]),
    # add a density axis to base-3 (size 4)
    ("h-g-c-logpop",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita", "LogPopDensity"]),
    # drop a redundant income feature, add migrant + density (size 4, 4 axes)
    ("h-c-migrant-logpop",
     ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio", "LogPopDensity"]),
    ("h-g-migrant-logpop",
     ["AvgHousingPricePerSqM", "GDPperCapita", "MigrantRatio", "LogPopDensity"]),
    # our picks: span 4 axes with a single income feature (lowest expected VIF)
    ("h-g-migrant-sexratio",
     ["AvgHousingPricePerSqM", "GDPperCapita", "MigrantRatio", "SexRatio100"]),
    # 5-axis full span (housing+income+migrant+density+sex), 1 income only
    ("h-g-migrant-logpop-sexratio",
     ["AvgHousingPricePerSqM", "GDPperCapita", "MigrantRatio",
      "LogPopDensity", "SexRatio100"]),
    # the "just enrich base-3 broadly" set (PI flagged as high-VIF earlier)
    ("h-g-c-migrant-logpop",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita",
      "MigrantRatio", "LogPopDensity"]),
    # raw NonRegistered population instead of the ratio (avoids GDP-ratio collinearity?)
    ("h-g-c-nonreg",
     ["AvgHousingPricePerSqM", "GDPperCapita", "CompPerCapita",
      "NonRegisteredPermanentPop10k"]),
]


def _pairwise_corr(
    demo: np.ndarray, names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Pearson correlation matrix over the (un-standardized) feature columns."""
    C = np.corrcoef(demo, rowvar=False)
    out: Dict[str, Dict[str, float]] = {}
    for i, ni in enumerate(names):
        out[ni] = {nj: float(C[i, j]) for j, nj in enumerate(names)}
    return out


def _max_abs_offdiag_corr(demo: np.ndarray) -> float:
    """Largest |pairwise Pearson r| among the columns (0 if a single column)."""
    if demo.shape[1] < 2:
        return 0.0
    C = np.corrcoef(demo, rowvar=False)
    p = C.shape[0]
    m = 0.0
    for i in range(p):
        for j in range(i + 1, p):
            m = max(m, abs(float(C[i, j])))
    return m


def _redundant_pairs(
    corr: Dict[str, Dict[str, float]], names: List[str], thresh: float = 0.95
) -> List[Tuple[str, str, float]]:
    """Near-perfectly-collinear feature pairs (|r| ≥ thresh) within the pool.

    These are why the whole-pool VIF is infinite: e.g. AvgHousingPricePerSqM ≈
    LogHousingPrice, GDPin10000Yuan ≈ LogGDP. Listed so the reader understands
    the pool contains alternative encodings of the same axis, not 15 independent
    signals — VIF must be judged per *candidate SET*, not over the raw pool.
    """
    out: List[Tuple[str, str, float]] = []
    for i, ni in enumerate(names):
        for nj in names[i + 1:]:
            r = corr[ni][nj]
            if abs(r) >= thresh:
                out.append((ni, nj, float(r)))
    return out


def _marginal_contribution_table(
    inputs: _SensitivityInputs, base3_fcausal: float
) -> List[dict]:
    """For each pool feature NOT in the base-3, add it to {housing,gdp,comp}.

    Reports ΔF_causal (signed; negative = lowers F_causal = captures more
    unfairness) and the resulting max VIF. Ranks independent-signal vs redundant.
    """
    rows: List[dict] = []
    for feat in CANDIDATE_POOL:
        if feat in BASE3:
            continue
        set_feats = BASE3 + [feat]
        res = _compute_subset(f"base3+{feat}", set_feats, inputs)
        if res.skipped:
            rows.append({
                "feature": feat,
                "axis": FEATURE_AXIS.get(feat, "?"),
                "skipped": True,
                "skip_reason": res.skip_reason,
            })
            continue
        max_vif = max(res.vifs.values()) if res.vifs else float("nan")
        rows.append({
            "feature": feat,
            "axis": FEATURE_AXIS.get(feat, "?"),
            "skipped": False,
            "f_causal": res.f_causal,
            "delta_f_causal": res.f_causal - base3_fcausal,
            "max_vif": max_vif,
            "added_feature_vif": res.vifs.get(feat, float("nan")),
        })
    # Rank by how much it LOWERS F_causal (most negative ΔF first).
    rows.sort(key=lambda r: (r.get("skipped", False), r.get("delta_f_causal", 0.0)))
    return rows


def _set_search_table(
    inputs: _SensitivityInputs,
    base_alpha: np.ndarray,
    top_k: int,
) -> List[dict]:
    """Evaluate each curated set: F_causal, max VIF, max|r|, Jaccard, Spearman."""
    base_topk = _topk_unfair_indices(base_alpha, top_k)
    name_to_idx = {n: i for i, n in enumerate(inputs.all_feature_names)}
    rows: List[dict] = []
    for label, feats in CURATED_SETS:
        res = _compute_subset(label, feats, inputs)
        axes = sorted({FEATURE_AXIS.get(f, "?") for f in feats})
        if res.skipped:
            rows.append({
                "set": label, "features": feats, "axes": axes,
                "n_features": len(feats), "skipped": True,
                "skip_reason": res.skip_reason,
            })
            continue
        col_idx = [name_to_idx[f] for f in feats]
        demo_subset = inputs.unit_demo_all[:, col_idx]
        max_r = _max_abs_offdiag_corr(demo_subset)
        max_vif = max(res.vifs.values()) if res.vifs else float("nan")
        if res.label == "baseline_h-g-c":
            jacc, spear = 1.0, 1.0
        else:
            r_topk = _topk_unfair_indices(res.alpha, top_k)
            jacc = _jaccard(base_topk, r_topk)
            spear = _spearman(base_alpha, res.alpha)
        rows.append({
            "set": label, "features": feats, "axes": axes,
            "n_features": len(feats), "skipped": False,
            "f_causal": res.f_causal,
            "max_vif": max_vif,
            "max_abs_corr": max_r,
            "topk_jaccard": jacc,
            "spearman_alpha": spear,
            "vifs": res.vifs,
        })
    return rows


def _pareto_and_verdicts(
    set_rows: List[dict], base_fcausal: float
) -> Tuple[List[dict], dict]:
    """Per-set verdict + Pareto-domination vs the current base-3.

    Verdict policy (adversarial about VIF and the 10-district DOF limit):
      - HIGH-VIF/UNSTABLE  : max VIF ≥ VIF_LIMIT, or >MAX_FEATURES features,
                             or Spearman α < 0.80 (targeting reshuffled).
      - ROBUST-AND-BETTER  : low VIF, ≤MAX_FEATURES, AND captures materially
                             more unfairness (F_causal lower than base by a
                             margin > 0.01) AND preserves targeting
                             (Jaccard ≥ 0.60 and Spearman ≥ 0.80).
      - ROBUST-EQUIVALENT  : low VIF + ≤MAX_FEATURES + stable targeting but
                             not a material F_causal improvement.
    "Dominates base-3" = acceptable VIF (<limit) AND ≤MAX_FEATURES AND captures
    more unfairness (lower F_causal) AND keeps Jaccard ≥ 0.60.
    """
    MATERIAL_DELTA = 0.01
    JACCARD_OK = 0.60
    SPEARMAN_OK = 0.80
    out_rows: List[dict] = []
    dominators: List[str] = []
    for r in set_rows:
        if r.get("skipped"):
            r2 = dict(r); r2["verdict"] = "SKIPPED"; out_rows.append(r2); continue
        max_vif = r["max_vif"]
        nfeat = r["n_features"]
        jacc = r["topk_jaccard"]
        spear = r["spearman_alpha"]
        delta = r["f_causal"] - base_fcausal  # negative = captures more unfairness
        acceptable_vif = max_vif < VIF_LIMIT
        size_ok = nfeat <= MAX_FEATURES
        targeting_ok = (jacc >= JACCARD_OK) and (spear >= SPEARMAN_OK)
        captures_more = delta < -MATERIAL_DELTA

        if (not acceptable_vif) or (not size_ok) or (spear < SPEARMAN_OK):
            verdict = "HIGH-VIF/UNSTABLE"
        elif acceptable_vif and size_ok and captures_more and targeting_ok:
            verdict = "ROBUST-AND-BETTER"
        elif acceptable_vif and size_ok and targeting_ok:
            verdict = "ROBUST-EQUIVALENT"
        else:
            verdict = "HIGH-VIF/UNSTABLE"

        dominates_base = (
            acceptable_vif and size_ok and captures_more and (jacc >= JACCARD_OK)
            and r["set"] != "baseline_h-g-c"
        )
        if dominates_base:
            dominators.append(r["set"])
        r2 = dict(r)
        r2["delta_f_causal"] = delta
        r2["acceptable_vif"] = acceptable_vif
        r2["dominates_base3"] = dominates_base
        r2["verdict"] = verdict
        out_rows.append(r2)

    # Pareto frontier on (F_causal lower = better, max_vif lower = better),
    # restricted to acceptable sets (VIF<limit, ≤MAX_FEATURES).
    feasible = [r for r in out_rows
                if not r.get("skipped") and r["max_vif"] < VIF_LIMIT
                and r["n_features"] <= MAX_FEATURES]
    pareto: List[str] = []
    for r in feasible:
        dominated = False
        for s in feasible:
            if s is r:
                continue
            # s dominates r if s is ≤ on both objectives and < on at least one.
            if (s["f_causal"] <= r["f_causal"] and s["max_vif"] <= r["max_vif"]
                    and (s["f_causal"] < r["f_causal"] or s["max_vif"] < r["max_vif"])):
                dominated = True
                break
        if not dominated:
            pareto.append(r["set"])

    summary = {
        "vif_limit": VIF_LIMIT,
        "max_features": MAX_FEATURES,
        "base3_f_causal": base_fcausal,
        "sets_dominating_base3": dominators,
        "pareto_frontier_sets": pareto,
        "any_low_vif_pop_axis_beats_base3": _migrant_axis_winner(out_rows),
    }
    return out_rows, summary


def _migrant_axis_winner(out_rows: List[dict]) -> Optional[str]:
    """Best set that (a) includes a pop-structure axis, (b) VIF<limit, ≤5 feats,
    (c) dominates base-3 on captured unfairness with preserved targeting.

    Returns the dominating set's label with the LOWEST F_causal, or None.
    """
    cands = []
    for r in out_rows:
        if r.get("skipped"):
            continue
        axes = r.get("axes", [])
        if "pop_structure" not in axes:
            continue
        if r.get("dominates_base3"):
            cands.append(r)
    if not cands:
        return None
    best = min(cands, key=lambda r: r["f_causal"])
    return best["set"]


@dataclass
class FeatureSelectionReport:
    base3_f_causal: float
    top_k: int
    n_units: int
    marginal_rows: List[dict]
    corr_matrix: Dict[str, Dict[str, float]]
    pool_vifs: Dict[str, float]
    pool_features_used: List[str]
    redundant_pairs: List[Tuple[str, str, float]]
    set_rows: List[dict]
    pareto_summary: dict


def run_feature_selection(top_k: int = DEFAULT_TOP_K) -> FeatureSelectionReport:
    """Run the deep feature-selection analysis in memory (no writes)."""
    inputs = _build_inputs()

    # Baseline reference (fixed-R production path).
    base_res = _compute_subset("baseline_h-g-c", BASE3, inputs)
    assert not base_res.skipped, "base-3 must compute"
    base_fcausal = base_res.f_causal
    base_alpha = base_res.alpha

    # 1) marginal contribution table
    marginal_rows = _marginal_contribution_table(inputs, base_fcausal)

    # 2) corr + VIF matrix over the candidate pool (only features that exist)
    name_to_idx = {n: i for i, n in enumerate(inputs.all_feature_names)}
    pool_used = [f for f in CANDIDATE_POOL if f in name_to_idx]
    pool_idx = [name_to_idx[f] for f in pool_used]
    pool_demo = inputs.unit_demo_all[:, pool_idx]
    corr_matrix = _pairwise_corr(pool_demo, pool_used)
    pool_vifs = _vifs(pool_demo, pool_used)
    redundant_pairs = _redundant_pairs(corr_matrix, pool_used)

    # 3) curated set search
    set_rows = _set_search_table(inputs, base_alpha, top_k)

    # 4 + 5) pareto + per-set verdicts
    set_rows, pareto_summary = _pareto_and_verdicts(set_rows, base_fcausal)

    return FeatureSelectionReport(
        base3_f_causal=base_fcausal,
        top_k=top_k,
        n_units=inputs.n_units,
        marginal_rows=marginal_rows,
        corr_matrix=corr_matrix,
        pool_vifs=pool_vifs,
        pool_features_used=pool_used,
        redundant_pairs=redundant_pairs,
        set_rows=set_rows,
        pareto_summary=pareto_summary,
    )


def _fs_to_json(fs: FeatureSelectionReport) -> dict:
    return {
        "analysis": "F_causal demographic feature-SELECTION (cleaned, before-edit)",
        "base3_f_causal": fs.base3_f_causal,
        "top_k": fs.top_k,
        "n_units": fs.n_units,
        "vif_limit": VIF_LIMIT,
        "max_features": MAX_FEATURES,
        "marginal_contribution": fs.marginal_rows,
        "candidate_pool": fs.pool_features_used,
        "pool_vifs": fs.pool_vifs,
        "pool_vif_note": (
            "Whole-pool VIFs are inf by design: the pool contains alternative "
            "encodings of the same axis (e.g. AvgHousingPricePerSqM≈LogHousingPrice, "
            "GDPin10000Yuan≈LogGDP), which makes the pooled design singular. Judge "
            "collinearity per candidate SET (set_search.max_vif), not over the pool."
        ),
        "redundant_pairs": [
            {"a": a, "b": b, "r": r} for (a, b, r) in fs.redundant_pairs
        ],
        "correlation_matrix": fs.corr_matrix,
        "set_search": fs.set_rows,
        "pareto_summary": fs.pareto_summary,
    }


def _fs_to_markdown(fs: FeatureSelectionReport) -> str:
    L: List[str] = []
    L.append("# F_causal demographic feature-SELECTION analysis (cleaned data)")
    L.append("")
    L.append(
        f"Base-3 {{housing, gdp, comp}} F_causal = **{_fmt(fs.base3_f_causal, 4)}** "
        f"(before-edit). Lower F_causal = more demographic-driven unfairness "
        f"captured. Policy: VIF < {VIF_LIMIT:.0f}, ≤ {MAX_FEATURES} features "
        f"(10-district DOF). Jaccard/Spearman are top-{fs.top_k} most-unfair "
        f"cells vs the current base-3 (does the editor target the same cells?)."
    )
    L.append("")

    # 1) Marginal table
    L.append("## 1. Marginal contribution (each feature added to base-3)")
    L.append("")
    L.append("Sorted by ΔF_causal (most negative = adds most independent unfairness signal).")
    L.append("")
    L.append("| + feature | axis | F_causal | ΔF_causal | added-feat VIF | set max VIF |")
    L.append("|---|---|---|---|---|---|")
    for r in fs.marginal_rows:
        if r.get("skipped"):
            L.append(f"| {r['feature']} | {r['axis']} | — | — | — | SKIPPED: {r['skip_reason']} |")
            continue
        L.append(
            f"| {r['feature']} | {r['axis']} | {_fmt(r['f_causal'], 4)} | "
            f"{_fmt(r['delta_f_causal'], 4)} | {_fmt(r['added_feature_vif'], 2)} | "
            f"{_fmt(r['max_vif'], 2)} |"
        )
    L.append("")

    # 2) VIF + corr matrix
    L.append("## 2. Candidate-pool VIF + pairwise correlation")
    L.append("")
    L.append(
        "Whole-pool VIFs are **inf by design** — the pool carries alternative "
        "encodings of the same axis (housing≈loghousing, gdp≈loggdp, "
        "comp≈logcomp≈employed), so the pooled design is singular. **Judge "
        "collinearity per candidate SET (§3 max VIF), not over the raw pool.** "
        "The near-perfectly-collinear pairs that cause this:"
    )
    L.append("")
    if fs.redundant_pairs:
        L.append("| feature A | feature B | r |")
        L.append("|---|---|---|")
        for a, b, r in fs.redundant_pairs:
            L.append(f"| {a} | {b} | {r:+.3f} |")
    else:
        L.append("- (none with |r| ≥ 0.95)")
    L.append("")
    L.append("### Pairwise |Pearson r| (upper triangle; flagged if |r| ≥ 0.8)")
    L.append("")
    names = fs.pool_features_used
    header = "| feature | " + " | ".join(n[:10] for n in names) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(names) + 1))
    for ni in names:
        cells = []
        for nj in names:
            r = fs.corr_matrix[ni][nj]
            cells.append(f"{r:+.2f}" if ni != nj else "—")
        L.append(f"| {ni} | " + " | ".join(cells) + " |")
    L.append("")

    # 3) Set search
    L.append("## 3. Curated feature-SET search (sizes 3–5, distinct axes)")
    L.append("")
    L.append(
        "| set | features | axes | n | F_causal | max VIF | max |r| | "
        "Jaccard | Spearman α | verdict |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in fs.set_rows:
        if r.get("skipped"):
            L.append(
                f"| {r['set']} | {', '.join(r['features'])} | "
                f"{','.join(r['axes'])} | {r['n_features']} | — | — | — | — | — | "
                f"SKIPPED |"
            )
            continue
        L.append(
            f"| {r['set']} | {', '.join(r['features'])} | {','.join(r['axes'])} | "
            f"{r['n_features']} | {_fmt(r['f_causal'], 4)} | {_fmt(r['max_vif'], 2)} | "
            f"{_fmt(r['max_abs_corr'], 2)} | {_fmt(r['topk_jaccard'], 4)} | "
            f"{_fmt(r['spearman_alpha'], 4)} | {r['verdict']} |"
        )
    L.append("")

    # 4) Pareto
    ps = fs.pareto_summary
    L.append("## 4. Pareto view (lower F_causal × lower VIF; VIF<10, ≤5 feats)")
    L.append("")
    L.append(f"- Pareto-frontier sets: {ps['pareto_frontier_sets'] or '(none)'}")
    L.append(f"- Sets that DOMINATE base-3 (more unfairness, VIF<10, Jaccard≥0.6): "
             f"{ps['sets_dominating_base3'] or '(none)'}")
    L.append(
        f"- Best low-VIF set with a population/migrant axis that beats base-3: "
        f"**{ps['any_low_vif_pop_axis_beats_base3'] or 'NONE'}**"
    )
    L.append("")

    # 5) Verdicts narrative
    L.append("## 5. Per-set verdicts")
    L.append("")
    for r in fs.set_rows:
        if r.get("skipped"):
            L.append(f"- **{r['set']}**: SKIPPED ({r['skip_reason']})")
            continue
        L.append(
            f"- **{r['set']}** → {r['verdict']}  "
            f"(F_causal {_fmt(r['f_causal'], 4)}, ΔF {_fmt(r['delta_f_causal'], 4)}, "
            f"maxVIF {_fmt(r['max_vif'], 2)}, Jaccard {_fmt(r['topk_jaccard'], 3)}, "
            f"Spearman {_fmt(r['spearman_alpha'], 3)})"
        )
    L.append("")
    return "\n".join(L)


def write_feature_selection(
    fs: FeatureSelectionReport, out_dir: Path
) -> Tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fcausal_feature_selection.json"
    md_path = out_dir / "fcausal_feature_selection.md"
    json_path.write_text(json.dumps(_fs_to_json(fs), indent=2, default=str))
    md_path.write_text(_fs_to_markdown(fs))
    return json_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.fcausal_feature_sensitivity",
        description=(
            "Feature-set sensitivity + selection analysis for F_causal on the "
            "cleaned data. Read-only: never mutates config.DEMOGRAPHIC_FEATURES, "
            "source_data/, or cache/."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="K for the most-unfair-cell overlap (default: edited-N).")
    ap.add_argument(
        "--no-gate-abort", action="store_true",
        help="Write the report even if the sanity gate fails (default: abort).",
    )
    ap.add_argument(
        "--selection", action="store_true",
        help="Also run the deep feature-SELECTION analysis (marginal table, "
             "corr/VIF matrix, set search, Pareto + verdicts).",
    )
    ap.add_argument(
        "--selection-only", action="store_true",
        help="Run ONLY the deep feature-selection analysis (still gated).",
    )
    args = ap.parse_args(argv)

    report = run_sensitivity(top_k=args.top_k)

    gate = report.gate_detail
    print(
        f"[sanity-gate] expected={gate['expected']:.6f}  "
        f"cached={gate['recomputed_cached_hat']:.6f}  "
        f"scratch={gate['recomputed_from_scratch']:.6f}  "
        f"=> {'PASS' if report.gate_passed else 'FAIL'}",
        flush=True,
    )
    if not report.gate_passed and not args.no_gate_abort:
        print(
            "[sanity-gate] ABORT: recomputed baseline F_causal disagrees with "
            "the editor's measured value beyond tolerance; not writing report. "
            "Re-run with --no-gate-abort to override.",
            flush=True,
        )
        return 2

    if not args.selection_only:
        json_path, md_path = write_report(report, args.out_dir)
        print(f"[write] {json_path}")
        print(f"[write] {md_path}")
        verdict = _verdict(report)
        print(
            f"[verdict] {verdict['verdict']}  "
            f"F_spread={verdict['f_causal_spread']:.4f}  "
            f"min_jaccard={verdict['min_topk_jaccard']:.4f}  "
            f"min_spearman={verdict['min_spearman_alpha']:.4f}",
            flush=True,
        )

    if args.selection or args.selection_only:
        fs = run_feature_selection(top_k=args.top_k)
        fs_json, fs_md = write_feature_selection(fs, args.out_dir)
        print(f"[write] {fs_json}")
        print(f"[write] {fs_md}")
        ps = fs.pareto_summary
        print(
            f"[selection] base3 F_causal={fs.base3_f_causal:.4f}  "
            f"pareto={ps['pareto_frontier_sets']}  "
            f"dominators={ps['sets_dominating_base3']}  "
            f"best_pop_axis_winner={ps['any_low_vif_pop_axis_beats_base3']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
