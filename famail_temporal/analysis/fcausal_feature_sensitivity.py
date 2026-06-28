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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="famail_temporal.analysis.fcausal_feature_sensitivity",
        description=(
            "Feature-set sensitivity analysis for F_causal on the cleaned data. "
            "Read-only: never mutates config.DEMOGRAPHIC_FEATURES, source_data/, "
            "or cache/."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help="K for the most-unfair-cell overlap (default: edited-N).")
    ap.add_argument(
        "--no-gate-abort", action="store_true",
        help="Write the report even if the sanity gate fails (default: abort).",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
