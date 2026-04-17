"""Experiment runner: orchestrates the full FAMAIL pipeline."""

from __future__ import annotations
import argparse
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch.nn as nn

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.fidelity.context import MultiStreamContextBuilder


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    config_snapshot: dict
    config_overrides: dict
    diagnostics_enabled: bool

    f_spatial_before: float
    f_spatial_after: float
    f_causal_before: float
    f_causal_after: float
    gini_dsr_before: float
    gini_dsr_after: float
    gini_asr_before: float
    gini_asr_after: float

    grid_before: np.ndarray
    grid_after: np.ndarray
    per_unit_attribution_before: np.ndarray
    per_unit_attribution_signed_before: np.ndarray

    gradient_sensitivity_before: Optional[np.ndarray]
    gradient_sensitivity_after: Optional[np.ndarray]

    modified_trajectory_ids: List[int]
    histories: List[ModificationHistory]
    top_k_scores: List[float]

    augmented_trajs_before: Dict[int, list]
    augmented_trajs_after: Dict[int, list]


def _parse_override_value(s: str) -> Any:
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_config_overrides(overrides: Dict[str, Any]):
    if overrides is None:
        return lambda: None
    for key in overrides:
        if not hasattr(config, key):
            raise KeyError(
                f"Unknown config override '{key}'. Only existing config.* "
                f"attributes can be overridden."
            )
    originals: Dict[str, Any] = {}
    for key, value in overrides.items():
        originals[key] = getattr(config, key)
        setattr(config, key, value)

    def restore():
        for key, value in originals.items():
            setattr(config, key, value)

    return restore


def _load_bundle(max_trajectories: Optional[int], max_drivers: Optional[int]) -> DataBundle:
    return DataBundle.load(
        max_trajectories=max_trajectories, max_drivers=max_drivers,
    )


_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name).strip("-")


def _generate_experiment_id(name: Optional[str]) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if name:
        return f"{timestamp}_{_slugify(name)}"
    return timestamp


def _scalar_metrics_from_grid(grid: np.ndarray) -> dict:
    return {
        "f_spatial": float(1.0 - np.nansum(grid[..., 0])),
        "f_causal":  float(1.0 - np.nansum(grid[..., 1])),
        "gini_dsr":  float(np.nansum(grid[..., 2])),
        "gini_asr":  float(np.nansum(grid[..., 3])),
    }


def run_experiment(
    config_overrides: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    output_root: Optional[Path] = None,
    max_trajectories: Optional[int] = None,
    max_drivers: Optional[int] = None,
    k: int = 100,
    diagnostics_enabled: bool = True,
) -> ExperimentResult:
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")

    restore_config = _apply_config_overrides(config_overrides or {})
    try:
        experiment_id = _generate_experiment_id(name)
        bundle = _load_bundle(max_trajectories=max_trajectories, max_drivers=max_drivers)

        grid_before = build_fairness_grid(bundle)
        metrics_before = _scalar_metrics_from_grid(grid_before)
        augmented_before = augment_trajectories(bundle.trajectories, grid_before)
        attr_unsigned, attr_signed = compute_per_unit_attribution(bundle)

        scored = rank_trajectories(bundle.trajectories, attr_unsigned, bundle.unit_map)
        if k > len(scored):
            raise ValueError(
                f"k={k} exceeds ranked trajectory count {len(scored)}. "
                f"Reduce k or widen max_trajectories."
            )
        top_k_indices = select_top_k(scored, k=k)
        if not top_k_indices:
            raise ValueError(
                "Top-k is empty - no trajectories with strictly positive "
                "attribution were found. Inspect per_unit_attribution_before; "
                "if all zeros, demographics carry no signal on this bundle."
            )
        top_k_scores = [scored[i][1] for i in range(len(top_k_indices))]
        top_k_trajs = [bundle.trajectories[i] for i in top_k_indices]

        # When no trained discriminator is available the bundle carries an
        # nn.Identity() placeholder, which cannot handle the fidelity call
        # signature. In that case drop the fidelity term from the objective
        # so the pipeline still runs end-to-end (useful for synthetic tests
        # and for environments without a checkpoint).
        if isinstance(bundle.discriminator, nn.Identity):
            objective = FAMAILObjective(bundle, alpha_fidelity=0.0)
        else:
            objective = FAMAILObjective(bundle)
        ms_builder = MultiStreamContextBuilder(bundle.multi_stream)
        modifier = TrajectoryModifier(
            objective=objective, bundle=bundle,
            multi_stream_builder=ms_builder,
            diagnostics_enabled=diagnostics_enabled,
        )
        histories = modifier.modify_batch(top_k_trajs)

        pickup_after = modifier.current_pickup_3d()
        grid_after = build_fairness_grid(bundle, pickup_3d=pickup_after)
        metrics_after = _scalar_metrics_from_grid(grid_after)

        modified_by_tid = {h.original.trajectory_id: h.modified for h in histories}
        trajs_after = [
            modified_by_tid.get(t.trajectory_id, t) for t in bundle.trajectories
        ]
        augmented_after = augment_trajectories(trajs_after, grid_after)

        snapshot = {
            key: getattr(config, key) for key in dir(config)
            if key.isupper() and not key.startswith("_")
        }

        return ExperimentResult(
            experiment_id=experiment_id,
            config_snapshot=snapshot,
            config_overrides=dict(config_overrides or {}),
            diagnostics_enabled=diagnostics_enabled,
            f_spatial_before=metrics_before["f_spatial"],
            f_spatial_after=metrics_after["f_spatial"],
            f_causal_before=metrics_before["f_causal"],
            f_causal_after=metrics_after["f_causal"],
            gini_dsr_before=metrics_before["gini_dsr"],
            gini_dsr_after=metrics_after["gini_dsr"],
            gini_asr_before=metrics_before["gini_asr"],
            gini_asr_after=metrics_after["gini_asr"],
            grid_before=grid_before,
            grid_after=grid_after,
            per_unit_attribution_before=attr_unsigned,
            per_unit_attribution_signed_before=attr_signed,
            gradient_sensitivity_before=None,
            gradient_sensitivity_after=None,
            modified_trajectory_ids=[h.original.trajectory_id for h in histories],
            histories=histories,
            top_k_scores=top_k_scores,
            augmented_trajs_before=augmented_before,
            augmented_trajs_after=augmented_after,
        )
    finally:
        restore_config()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="famail_temporal.evaluation.runner")
    p.add_argument("--name", default=None)
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--max-drivers", type=int, default=None)
    p.add_argument("-k", type=int, default=100)
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--override", action="append", default=[],
                   help="KEY=VALUE override (repeatable)")
    return p


def _parse_cli_overrides(raw: list[str]) -> dict:
    out: Dict[str, Any] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(f"Invalid --override entry '{entry}', expected KEY=VALUE")
        k, v = entry.split("=", 1)
        out[k] = _parse_override_value(v)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_cli_overrides(args.override)
    result = run_experiment(
        config_overrides=overrides,
        name=args.name,
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
        k=args.k,
        diagnostics_enabled=not args.no_diagnostics,
    )
    print(f"[runner] experiment_id = {result.experiment_id}")
    print(f"[runner]   F_spatial: {result.f_spatial_before:.4f} -> {result.f_spatial_after:.4f}")
    print(f"[runner]   F_causal:  {result.f_causal_before:.4f} -> {result.f_causal_after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
