"""Persistence layer for ExperimentResult.

Writes an ExperimentResult to a timestamped output directory as a set of
artifacts (metrics.json, trajectories.csv, per_unit_attribution.csv,
grid_before.pkl, grid_after.pkl, augmented_trajs_before.pkl[.gz],
augmented_trajs_after.pkl[.gz], modified_trajectory_ids.json, histories.pkl,
and optional gradient_sensitivity_{before,after}.pkl).

Note: pickle is used here for structural compatibility with the existing
passenger_seeking_trajs_45-800.pkl dataset. This is a documented requirement
in the evaluation framework spec.
"""

from __future__ import annotations
import csv
import datetime as _dt
import gzip
import json
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from famail_temporal import config
from famail_temporal.evaluation.runner import ExperimentResult


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def _command_line() -> str:
    return shlex.join(sys.argv)


def _gzip_threshold_bytes() -> int:
    return 500 * 1024 * 1024


def _conditional_gzip_pickle(obj: Any, path: Path) -> Path:
    data = pickle.dumps(obj, protocol=4)
    if len(data) > _gzip_threshold_bytes():
        gz_path = path.with_suffix(".pkl.gz")
        with gzip.open(gz_path, "wb") as f:
            f.write(data)
        return gz_path
    path.write_bytes(data)
    return path


def _grid_payload(grid: np.ndarray, active_mask: np.ndarray) -> dict:
    return {
        "grid": grid,
        "channel_names": ["spatial_attr", "causal_attr", "gini_decomp_dsr", "gini_decomp_asr"],
        "time_blocks": list(config.TIME_BLOCKS),
        "active_mask": active_mask,
    }


def _sensitivity_payload(grid: np.ndarray, active_mask: np.ndarray) -> dict:
    return {
        "grid": grid,
        "channel_names": ["dF_spatial_dp", "dF_causal_dp"],
        "time_blocks": list(config.TIME_BLOCKS),
        "active_mask": active_mask,
    }


def _diagnostics_summary(result: ExperimentResult) -> dict | None:
    if not result.diagnostics_enabled or not result.histories:
        return None
    all_iters = [r for h in result.histories for r in h.iterations]
    if not all_iters:
        return None
    def _mean(attr):
        vals = [getattr(r, attr) for r in all_iters if getattr(r, attr) is not None]
        return float(np.mean(vals)) if vals else None
    dom = [r.dominant_term for r in all_iters if r.dominant_term is not None]
    total = len(dom) or 1
    return {
        "mean_grad_spatial_norm":       _mean("grad_spatial_norm"),
        "mean_grad_causal_norm":        _mean("grad_causal_norm"),
        "mean_grad_fidelity_norm":      _mean("grad_fidelity_norm"),
        "mean_cos_spatial_causal":      _mean("grad_cosine_spatial_causal"),
        "mean_cos_fairness_fidelity":   _mean("grad_cosine_fairness_fidelity"),
        "frac_iters_spatial_dominant":  dom.count("spatial") / total,
        "frac_iters_causal_dominant":   dom.count("causal") / total,
        "frac_iters_fidelity_dominant": dom.count("fidelity") / total,
    }


def _convergence_summary(result: ExperimentResult) -> dict:
    if not result.histories:
        return {"n_converged": 0, "n_max_iter": 0,
                "mean_total_iterations": 0.0, "mean_final_grad_norm": 0.0}
    n_conv = sum(1 for h in result.histories if h.converged)
    n_max = len(result.histories) - n_conv
    total_iters = [h.total_iterations for h in result.histories]
    finals = [h.iterations[-1].gradient_norm for h in result.histories if h.iterations]
    return {
        "n_converged": n_conv,
        "n_max_iter": n_max,
        "mean_total_iterations": float(np.mean(total_iters)) if total_iters else 0.0,
        "mean_final_grad_norm": float(np.mean(finals)) if finals else 0.0,
    }


def _write_trajectories_csv(result: ExperimentResult, path: Path) -> None:
    from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
    headers = [
        "trajectory_id", "driver_id",
        "original_pickup_cell_x", "original_pickup_cell_y",
        "modified_pickup_cell_x", "modified_pickup_cell_y",
        "pickup_t_block", "delta_x", "delta_y",
        "attribution_score", "rank",
        "converged", "total_iterations",
        "initial_objective", "final_objective",
        "f_spatial_initial", "f_spatial_final",
        "f_causal_initial", "f_causal_final",
        "f_fidelity_initial", "f_fidelity_final",
        "mean_grad_spatial_norm", "mean_grad_causal_norm", "mean_grad_fidelity_norm",
        "frac_iters_spatial_dominant", "frac_iters_causal_dominant", "frac_iters_fidelity_dominant",
        "mean_cos_spatial_causal", "mean_cos_fairness_fidelity",
        "sign_flip_rate",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rank, (h, score) in enumerate(zip(result.histories, result.top_k_scores), start=1):
            orig = h.original.pickup_cell
            modc = h.modified.pickup_cell
            tb = hour_to_block_index(time_bucket_to_hour(h.original.pickup_state.time_bucket))
            iters = h.iterations
            def _first(attr):
                return getattr(iters[0], attr) if iters else 0.0
            def _last(attr):
                return getattr(iters[-1], attr) if iters else 0.0
            def _mean_none(attr):
                vals = [getattr(r, attr) for r in iters if getattr(r, attr) is not None]
                return float(np.mean(vals)) if vals else ""
            def _frac(term):
                if not iters or iters[0].dominant_term is None:
                    return ""
                return sum(1 for r in iters if r.dominant_term == term) / len(iters)
            sign_flip_rate = (
                sum(1 for r in iters if r.sign_flipped) / len(iters)
                if iters and iters[0].sign_flipped is not None else ""
            )
            writer.writerow([
                h.original.trajectory_id, h.original.driver_id,
                orig[0], orig[1], modc[0], modc[1],
                tb, modc[0] - orig[0], modc[1] - orig[1],
                score, rank,
                h.converged, h.total_iterations,
                _first("objective_value"), _last("objective_value"),
                _first("f_spatial"),       _last("f_spatial"),
                _first("f_causal"),        _last("f_causal"),
                _first("f_fidelity"),      _last("f_fidelity"),
                _mean_none("grad_spatial_norm"),
                _mean_none("grad_causal_norm"),
                _mean_none("grad_fidelity_norm"),
                _frac("spatial"), _frac("causal"), _frac("fidelity"),
                _mean_none("grad_cosine_spatial_causal"),
                _mean_none("grad_cosine_fairness_fidelity"),
                sign_flip_rate,
            ])


def _write_per_unit_attribution_csv(result: ExperimentResult, path: Path, mask_3d: np.ndarray) -> None:
    """Per-cell attribution CSV.

    Columns are the four channels of the fairness grid (spatial αᵢ, causal
    αᵢ, gini_dsr_contrib, gini_asr_contrib) before and after modification.
    αᵢ values sum to F (the fairness metric); positive = above-baseline
    fairness contribution, negative = drags fairness down. See
    ``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``.
    """
    ix_x, ix_y, ix_t = np.where(mask_3d)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "unit_idx", "cell_x", "cell_y", "t_block", "flat_cell_id",
            "spatial_attr_before", "spatial_attr_after",
            "causal_attr_before",  "causal_attr_after",
            "gini_dsr_contrib_before", "gini_dsr_contrib_after",
            "gini_asr_contrib_before", "gini_asr_contrib_after",
        ])
        for i, (x, y, t) in enumerate(zip(ix_x, ix_y, ix_t)):
            writer.writerow([
                i, int(x), int(y), int(t), int(x) * config.GRID_DIMS[1] + int(y),
                float(result.grid_before[x, y, t, 0]),
                float(result.grid_after [x, y, t, 0]),
                float(result.grid_before[x, y, t, 1]),
                float(result.grid_after [x, y, t, 1]),
                float(result.grid_before[x, y, t, 2]),
                float(result.grid_after [x, y, t, 2]),
                float(result.grid_before[x, y, t, 3]),
                float(result.grid_after [x, y, t, 3]),
            ])


def _coerce_json(v: Any) -> Any:
    """Recursively coerce values to JSON-native types.

    Lists and tuples both become JSON arrays (lossy for tuple-vs-list; acceptable
    since this dict is for human inspection, not round-tripping). Dicts recurse
    through values (keys are stringified by json.dumps itself). Any other type
    falls back to str(v) as a last resort. This is the single source of JSON
    coercion for the metrics.json payload — json.dumps is called WITHOUT default=.
    """
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_coerce_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _coerce_json(val) for k, val in v.items()}
    return str(v)


def write(result: ExperimentResult, output_root: Path, bundle=None) -> Path:
    """Serialize an ExperimentResult to {output_root}/{experiment_id}/ and return that path.

    Write ordering: all other artifacts first, then metrics.json LAST. This means
    readers (e.g. Phase 8's report generator) can treat metrics.json as a
    completion sentinel — a run directory is considered complete iff metrics.json
    exists. Partial directories (due to crash mid-write) lack metrics.json and
    should be skipped or re-generated.

    The active_mask used in artifact payloads is taken from bundle.mask_3d when
    bundle is provided; otherwise it is reconstructed from NaN pattern of
    grid_before[..., 0]. Both paths produce equivalent masks under the current
    build_fairness_grid invariant (inactive cells are NaN, active cells are
    finite) but the bundle path is preferred when available.
    """
    output_root = Path(output_root)
    out_dir = output_root / result.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    active_mask = bundle.mask_3d if bundle is not None else ~np.isnan(result.grid_before[..., 0])

    artifact_paths: Dict[str, str] = {}
    file_sizes: Dict[str, int] = {}

    for name, grid in [("grid_before", result.grid_before),
                       ("grid_after",  result.grid_after)]:
        path = out_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(_grid_payload(grid, active_mask), f, protocol=4)
        artifact_paths[name] = path.name
        file_sizes[name] = path.stat().st_size

    for name, obj in [("augmented_trajs_before", result.augmented_trajs_before),
                      ("augmented_trajs_after",  result.augmented_trajs_after)]:
        base = out_dir / f"{name}.pkl"
        written = _conditional_gzip_pickle(obj, base)
        artifact_paths[name] = written.name
        file_sizes[name] = written.stat().st_size

    mod_ids_payload = {
        "modified_trajectory_ids": list(result.modified_trajectory_ids),
        "original_pickup_cells": {
            str(h.original.trajectory_id): list(h.original.pickup_cell)
            for h in result.histories
        },
        "modified_pickup_cells": {
            str(h.original.trajectory_id): list(h.modified.pickup_cell)
            for h in result.histories
        },
    }
    path = out_dir / "modified_trajectory_ids.json"
    path.write_text(json.dumps(mod_ids_payload, indent=2))
    artifact_paths["modified_trajectory_ids"] = path.name
    file_sizes["modified_trajectory_ids"] = path.stat().st_size

    path = out_dir / "histories.pkl"
    with open(path, "wb") as f:
        pickle.dump(result.histories, f, protocol=4)
    artifact_paths["histories"] = path.name
    file_sizes["histories"] = path.stat().st_size

    if result.diagnostics_enabled and result.gradient_sensitivity_before is not None:
        for name, grid in [
            ("gradient_sensitivity_before", result.gradient_sensitivity_before),
            ("gradient_sensitivity_after",  result.gradient_sensitivity_after),
        ]:
            path = out_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(_sensitivity_payload(grid, active_mask), f, protocol=4)
            artifact_paths[name] = path.name
            file_sizes[name] = path.stat().st_size

    path = out_dir / "trajectories.csv"
    _write_trajectories_csv(result, path)
    artifact_paths["trajectories_csv"] = path.name
    file_sizes["trajectories_csv"] = path.stat().st_size

    path = out_dir / "per_unit_attribution.csv"
    _write_per_unit_attribution_csv(result, path, active_mask)
    artifact_paths["per_unit_attribution_csv"] = path.name
    file_sizes["per_unit_attribution_csv"] = path.stat().st_size

    metrics = {
        "experiment_id": result.experiment_id,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "command_line": _command_line(),
        "config_snapshot": {k: _coerce_json(v) for k, v in result.config_snapshot.items()},
        "config_overrides": {k: _coerce_json(v) for k, v in result.config_overrides.items()},
        "diagnostics_enabled": result.diagnostics_enabled,
        "effective_alphas": {
            "alpha_spatial":  result.effective_alpha_spatial,
            "alpha_causal":   result.effective_alpha_causal,
            "alpha_fidelity": result.effective_alpha_fidelity,
        },
        "dataset": {
            "n_trajectories": sum(len(v) for v in result.augmented_trajs_before.values()),
            "n_drivers": len(result.augmented_trajs_before),
            "n_active_units": int(np.sum(active_mask)),
        },
        "k_modified": len(result.histories),
        "metrics_before": {
            "f_spatial": result.f_spatial_before, "f_causal": result.f_causal_before,
            "gini_dsr":  result.gini_dsr_before,  "gini_asr":  result.gini_asr_before,
        },
        "metrics_after": {
            "f_spatial": result.f_spatial_after,  "f_causal": result.f_causal_after,
            "gini_dsr":  result.gini_dsr_after,   "gini_asr":  result.gini_asr_after,
        },
        "deltas": {
            "f_spatial": result.f_spatial_after - result.f_spatial_before,
            "f_causal":  result.f_causal_after  - result.f_causal_before,
            "gini_dsr":  result.gini_dsr_after  - result.gini_dsr_before,
            "gini_asr":  result.gini_asr_after  - result.gini_asr_before,
        },
        "convergence_summary": _convergence_summary(result),
        "diagnostics_summary": _diagnostics_summary(result),
        "artifact_paths": artifact_paths,
        "file_sizes_bytes": file_sizes,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_coerce_json(metrics), indent=2)
    )
    return out_dir
