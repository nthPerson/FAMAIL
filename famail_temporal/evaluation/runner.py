"""Experiment runner: orchestrates the full FAMAIL pipeline."""

from __future__ import annotations
import argparse
import datetime as _dt
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories, select_top_k,
)
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.data.aggregation import hour_to_block_index, time_bucket_to_hour
from famail_temporal.data.loader import DataBundle
from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.fidelity.context import MultiStreamContextBuilder

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    _tqdm = None  # type: ignore


def _log(t0: float, msg: str) -> None:
    """One-line phase marker: ``[runner +HH:MM:SS] msg``.

    All phase prints go through this so the timestamp format stays uniform and
    elapsed time is continuous from ``t0`` (typically set at CLI start).
    """
    dt = time.monotonic() - t0
    h, rem = divmod(int(dt), 3600)
    m, s = divmod(rem, 60)
    print(f"[runner +{h:02d}:{m:02d}:{s:02d}] {msg}", flush=True)


def _resolve_device(arg: str) -> torch.device:
    """Resolve a CLI ``--device`` argument to a concrete torch.device.

    Accepts ``auto`` (CUDA when available, else CPU), ``cpu``, or any
    torch-recognized CUDA string (``cuda``, ``cuda:0``, …). Failing to find
    CUDA when explicitly requested raises — silently falling back to CPU
    would hide a configuration bug.
    """
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cpu":
        return torch.device("cpu")
    if arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={arg} requested but torch.cuda.is_available() is False"
            )
        return torch.device(arg)
    raise ValueError(
        f"Invalid --device {arg!r}; expected 'auto', 'cpu', or 'cuda[:idx]'"
    )


def _iterative_topk_modify(
    modifier: TrajectoryModifier,
    bundle: "DataBundle",  # noqa: F821
    k: int,
    t0: float,
    max_per_unit: Optional[int] = None,
    max_per_cell: Optional[int] = None,
) -> tuple[List[ModificationHistory], List[float]]:
    """Iterative top-k with re-attribution.

    Instead of selecting the entire top-k subset against the initial
    attribution and then modifying each in sequence (which produces
    trajectory-level interference when the top-k cluster geographically),
    this loop alternates: re-attribute → re-rank → pick best remaining →
    modify → repeat. Each new selection sees the true post-modification
    state via ``modifier.current_pickup_3d()``.

    Designed as a mitigation for the trajectory-interference failure mode
    documented in TRAJECTORY_EDITING_METHODOLOGY.md §8.1: when many top-k
    trajectories share a pickup cell, locally-optimal moves into the
    same destination cell cumulatively oversupply that cell, dragging
    global F_causal in the wrong direction.

    Returns ``(histories, scores)`` where ``scores[i]`` is the
    attribution score of the trajectory at the moment it was selected
    (i.e., the score AFTER all prior modifications).
    """
    histories: List[ModificationHistory] = []
    scores: List[float] = []
    modified_ids: set = set()
    # Cumulative pool counters for the per-(cell, t_block) and per-cell
    # diversity budgets. These constrain which trajectory is picked each
    # round, mirroring the batch-mode select_top_k(... max_per_unit, ...).
    unit_counts: dict = {}
    cell_counts: dict = {}

    max_iters = modifier.max_iterations
    total_iter_units = k * max_iters
    use_tqdm = _TQDM_AVAILABLE and sys.stderr.isatty()

    bar = None
    if use_tqdm:
        state = {"round": 0, "n_conv": 0}
        bar = _tqdm(
            total=total_iter_units, desc="iter-topk", unit="iter",
            leave=True, dynamic_ncols=True,
        )

        def _on_iter(_it_idx: int, _result) -> None:
            bar.update(1)
            bar.set_postfix(
                round=f"{state['round']}/{k}",
                conv=state["n_conv"],
                refresh=False,
            )
        on_iter_cb = _on_iter
    else:
        # Non-TTY fallback: log every ~5% of total
        step_fallback = max(1, total_iter_units // 20)
        seen_state = {"iters_seen": 0, "n_conv": 0}

        def _on_iter_fallback(_it_idx: int, _result) -> None:
            seen_state["iters_seen"] += 1
            if (seen_state["iters_seen"] % step_fallback == 0
                    or seen_state["iters_seen"] == total_iter_units):
                print(
                    f"[runner]   iter-topk: {seen_state['iters_seen']}/"
                    f"{total_iter_units} iters  "
                    f"converged_so_far={seen_state['n_conv']}",
                    flush=True,
                )
        on_iter_cb = _on_iter_fallback

    try:
        for round_i in range(1, k + 1):
            if use_tqdm:
                state["round"] = round_i

            # Re-attribute against current modifier state. This is the
            # core of M2: each new pick sees prior modifications' effects.
            pickup_now = modifier.current_pickup_3d()
            attribution = compute_per_unit_attribution(
                bundle, pickup_3d=pickup_now,
            )

            # Re-rank remaining trajectories. select_top_k's strict-αᵢ<0
            # filter is enforced manually here so we can also exclude
            # already-modified ids, enforce per-unit/per-cell budgets on
            # the cumulative pool, and short-circuit when no candidates
            # remain.
            scored = rank_trajectories(
                bundle.trajectories, attribution, bundle.unit_map,
            )
            best_idx, best_score = -1, 0.0
            for idx, sc in scored:
                if sc >= 0:
                    break  # ascending order — first non-negative ends the search
                traj = bundle.trajectories[idx]
                if traj.trajectory_id in modified_ids:
                    continue
                if max_per_unit is not None or max_per_cell is not None:
                    cx, cy = traj.pickup_cell
                    t_block = hour_to_block_index(
                        time_bucket_to_hour(traj.pickup_state.time_bucket)
                    )
                    unit_key = (cx, cy, t_block)
                    cell_key = (cx, cy)
                    if (max_per_unit is not None
                            and unit_counts.get(unit_key, 0) >= max_per_unit):
                        continue
                    if (max_per_cell is not None
                            and cell_counts.get(cell_key, 0) >= max_per_cell):
                        continue
                best_idx, best_score = idx, sc
                break

            if best_idx < 0:
                _log(
                    t0,
                    f"iter-topk: terminating early at round {round_i}/{k} — "
                    f"no negative-α candidates remain (modified={len(modified_ids)})",
                )
                # Advance bar to total so the progress display closes cleanly
                if use_tqdm:
                    remaining_iters = total_iter_units - bar.n
                    if remaining_iters > 0:
                        bar.update(remaining_iters)
                break

            traj = bundle.trajectories[best_idx]
            modified_ids.add(traj.trajectory_id)
            if max_per_unit is not None or max_per_cell is not None:
                cx, cy = traj.pickup_cell
                t_block = hour_to_block_index(
                    time_bucket_to_hour(traj.pickup_state.time_bucket)
                )
                unit_counts[(cx, cy, t_block)] = (
                    unit_counts.get((cx, cy, t_block), 0) + 1
                )
                cell_counts[(cx, cy)] = cell_counts.get((cx, cy), 0) + 1
            h = modifier.modify_single(traj, on_iteration=on_iter_cb)

            # Account for early-converged iters
            unused = max_iters - len(h.iterations)
            if use_tqdm:
                if unused > 0:
                    bar.update(unused)
            else:
                seen_state["iters_seen"] += unused

            histories.append(h)
            scores.append(best_score)
            if h.converged:
                if use_tqdm:
                    state["n_conv"] += 1
                else:
                    seen_state["n_conv"] += 1
    finally:
        if bar is not None:
            bar.close()

    return histories, scores


def _modify_with_progress(
    modifier: TrajectoryModifier,
    trajectories: List[Any],
) -> List[ModificationHistory]:
    """Run ``modify_single`` on each trajectory with an iteration-level progress bar.

    The bar's total is ``len(trajectories) * max_iterations`` (iter-units) so it
    ticks every ST-iFGSM step rather than waiting for an entire trajectory to
    finish. Per-trajectory work can be 5-15 seconds × ``max_iterations``, so a
    per-trajectory bar would sit at 0/n for many minutes; this bar updates once
    every ~10 sec. Early-converged trajectories advance the bar by the unused
    iters so total ETA stays honest.

    Postfix shows ``traj=N/K conv=C`` — current trajectory index and the
    running count of converged trajectories (primary signal for "is the
    optimizer actually finding minima or just hitting the iter cap").

    Falls back to throttled plain-print every ~5% when stderr is not a TTY or
    tqdm is unavailable.
    """
    n_trajs = len(trajectories)
    max_iters = modifier.max_iterations
    total_iter_units = n_trajs * max_iters
    use_tqdm = _TQDM_AVAILABLE and sys.stderr.isatty()

    if use_tqdm:
        state = {"current_traj": 0, "n_conv": 0}
        bar = _tqdm(
            total=total_iter_units, desc="modifying", unit="iter",
            leave=True, dynamic_ncols=True,
        )

        def _on_iter(_it_idx: int, _result) -> None:
            bar.update(1)
            bar.set_postfix(
                traj=f"{state['current_traj']}/{n_trajs}",
                conv=state["n_conv"],
                refresh=False,
            )

        histories: List[ModificationHistory] = []
        try:
            for i, t in enumerate(trajectories, start=1):
                state["current_traj"] = i
                bar.set_postfix(
                    traj=f"{state['current_traj']}/{n_trajs}",
                    conv=state["n_conv"],
                    refresh=False,
                )
                h = modifier.modify_single(t, on_iteration=_on_iter)
                # Advance bar past any unused iters when convergence broke early
                unused = max_iters - len(h.iterations)
                if unused > 0:
                    bar.update(unused)
                histories.append(h)
                if h.converged:
                    state["n_conv"] += 1
        finally:
            bar.close()
        return histories

    # Non-TTY / no-tqdm fallback: print iter-units progress every ~5% of total
    step = max(1, total_iter_units // 20)
    iters_seen = 0
    n_conv = 0
    histories = []

    def _on_iter_fallback(_it_idx: int, _result) -> None:
        nonlocal iters_seen
        iters_seen += 1
        if iters_seen % step == 0 or iters_seen == total_iter_units:
            print(
                f"[runner]   modifying: {iters_seen}/{total_iter_units} iters  "
                f"converged_so_far={n_conv}",
                flush=True,
            )

    for t in trajectories:
        h = modifier.modify_single(t, on_iteration=_on_iter_fallback)
        # Account for early-converged iters in the fallback print path too
        unused = max_iters - len(h.iterations)
        iters_seen += unused
        histories.append(h)
        if h.converged:
            n_conv += 1
    return histories


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    config_snapshot: dict
    config_overrides: dict
    diagnostics_enabled: bool

    # Effective alpha values used at runtime (may differ from config.ALPHA_*
    # when a discriminator stub forces alpha_fidelity=0.0 at construction time).
    effective_alpha_spatial: float
    effective_alpha_causal: float
    effective_alpha_fidelity: float

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
    # Canonical per-cell fairness attribution (sums to F_causal). Sign:
    # positive = above-baseline fairness contribution; negative = drags below.
    # See docs/FAIRNESS_DECOMPOSITION_FORMULATION.md for the formulation.
    per_cell_fairness_attribution: np.ndarray

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
    """Aggregate the per-cell fairness grid into the scalar evaluation metrics.

    Convention (canonical across the project):

    - ``f_spatial`` and ``f_causal`` are **fairness** values in [0, 1].
      **1 = maximally fair, 0 = least fair.** They are direct sums of
      ``grid[..., 0]`` and ``grid[..., 1]``, which are the
      ``per_cell_fairness_attribution_*`` outputs whose Σ equals the
      mathematical F (see ``fairness/spatial.py`` and ``fairness/causal.py``
      for the formulations).
    - ``gini_dsr`` and ``gini_asr`` are **Gini coefficients** (unfairness
      values) in [0, 1]. **0 = perfectly equal, 1 = maximally unequal.**
      These are the only fields in metrics.json that report the
      unfairness direction; everything labeled ``F_*`` reports fairness.

    Historical note (sign-convention erratum, 2026-05-14): prior to this
    revision, ``f_spatial`` and ``f_causal`` were stored as ``1 - F``
    (i.e. the unfairness values) due to an inverted formula here. All
    experiments before that date have metrics.json files with the
    inverted convention and should not be compared directly to
    post-fix runs. See ``docs/TRAJECTORY_EDITING_METHODOLOGY.md`` §8.0.
    """
    return {
        "f_spatial": float(np.nansum(grid[..., 0])),
        "f_causal":  float(np.nansum(grid[..., 1])),
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
    diagnostics_enabled: bool = False,
    t0: Optional[float] = None,
    device: Optional[torch.device | str] = None,
    patience: Optional[int] = None,
    convergence_tol: Optional[float] = None,
    iterative_topk: bool = False,
    max_per_unit: Optional[int] = None,
    max_per_cell: Optional[int] = None,
) -> ExperimentResult:
    if k <= 0:
        raise ValueError(f"k must be > 0; got {k}")
    if t0 is None:
        t0 = time.monotonic()
    # Resolve device: explicit > auto-detect. The modifier hot path (objective
    # forward + discriminator + multi-stream context) all land here. The
    # pre-loop (grid_before, sensitivity_before, augment) and post-loop
    # artifacts stay on CPU because they're called once each and operate on
    # the numpy bundle data.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device) if not isinstance(device, torch.device) else device

    restore_config = _apply_config_overrides(config_overrides or {})
    try:
        experiment_id = _generate_experiment_id(name)
        _log(t0, f"experiment_id = {experiment_id}")
        _log(
            t0,
            f"loading data bundle (max_trajectories={max_trajectories}, "
            f"max_drivers={max_drivers}, diagnostics={diagnostics_enabled})...",
        )
        bundle = _load_bundle(max_trajectories=max_trajectories, max_drivers=max_drivers)
        _log(
            t0,
            f"bundle loaded: n_trajectories={len(bundle.trajectories)}  "
            f"n_active_units={bundle.unit_map.n_units}  "
            f"grid={tuple(bundle.unit_map.grid_shape)} T={bundle.pickup_3d.shape[2]}",
        )

        _log(t0, "building fairness grid (before)...")
        grid_before = build_fairness_grid(bundle)
        if diagnostics_enabled:
            _log(t0, "computing gradient sensitivity (before)...")
            from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
            sensitivity_before = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
        else:
            sensitivity_before = None
        metrics_before = _scalar_metrics_from_grid(grid_before)
        _log(
            t0,
            f"metrics_before: F_spatial={metrics_before['f_spatial']:.6f}  "
            f"F_causal={metrics_before['f_causal']:.6f}",
        )
        _log(t0, f"augmenting {len(bundle.trajectories)} trajectories (before)...")
        augmented_before = augment_trajectories(bundle.trajectories, grid_before)

        _log(t0, "computing per-unit attribution...")
        attribution = compute_per_unit_attribution(bundle)
        _log(
            t0,
            f"attribution: range=[{attribution.min():.3e}, {attribution.max():.3e}]  "
            f"frac_negative={(attribution < 0).mean():.3f}",
        )

        _log(t0, f"ranking {len(bundle.trajectories)} trajectories...")
        scored = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
        if k > len(scored):
            raise ValueError(
                f"k={k} exceeds ranked trajectory count {len(scored)}. "
                f"Reduce k or widen max_trajectories."
            )
        if iterative_topk:
            # Iterative top-k mode: select the first trajectory now (to
            # validate that at least one negative-αᵢ candidate exists), but
            # defer the remaining k−1 picks to ``_iterative_topk_modify``,
            # which re-attributes after each modification. ``top_k_indices``,
            # ``top_k_scores``, and ``top_k_trajs`` are left empty here —
            # the helper fills them dynamically.
            top_k_indices: List[int] = []
            top_k_scores: List[float] = []
            top_k_trajs: List = []
            first_neg = next(
                ((i, s) for i, s in scored if s < 0), (None, None),
            )
            if first_neg[0] is None:
                raise ValueError(
                    "Top-k is empty - no trajectories with strictly negative "
                    "attribution were found at the initial state."
                )
            _log(
                t0,
                f"iterative top-k enabled: re-attribute after each "
                f"modification (k={k} target rounds)",
            )
        else:
            top_k_indices = select_top_k(
                scored, k=k,
                trajectories=bundle.trajectories,
                max_per_unit=max_per_unit,
                max_per_cell=max_per_cell,
            )
            if not top_k_indices:
                raise ValueError(
                    "Top-k is empty - no trajectories with strictly negative "
                    "attribution were found. Under the F-decomposition convention, "
                    "negative αᵢ marks cells dragging fairness below baseline. "
                    "If no such cells exist, the audit set is uniformly fair "
                    "(unusual; check that the active mask is populated and "
                    "demographics carry signal)."
                )
            top_k_scores = [scored[i][1] for i in range(len(top_k_indices))]
            top_k_trajs = [bundle.trajectories[i] for i in top_k_indices]
            div_str = ""
            if max_per_unit is not None or max_per_cell is not None:
                div_str = (
                    f"  (diversity: max_per_unit={max_per_unit}, "
                    f"max_per_cell={max_per_cell})"
                )
            _log(
                t0,
                f"selected top-k: {len(top_k_indices)}/{k} requested  "
                f"(score range [{top_k_scores[0]:.3e}, {top_k_scores[-1]:.3e}])"
                f"{div_str}",
            )

        # When no trained discriminator is available the bundle carries an
        # nn.Identity() placeholder, which cannot handle the fidelity call
        # signature. In that case drop the fidelity term from the objective
        # so the pipeline still runs end-to-end (useful for synthetic tests
        # and for environments without a checkpoint).
        if isinstance(bundle.discriminator, nn.Identity):
            objective = FAMAILObjective(bundle, alpha_fidelity=0.0)
        else:
            objective = FAMAILObjective(bundle)
        # Move objective (mask_3d, dropoff_N, X_demo, … + the discriminator
        # sub-module via .to() cascade) plus the bundle's discriminator
        # reference to the target device. The bundle is frozen but its
        # discriminator nn.Module is mutable, so .to() works in place.
        objective = objective.to(device)
        bundle.discriminator.to(device)
        _log(t0, f"objective + discriminator on device={device}")
        effective_alphas = (
            objective.alpha_spatial,
            objective.alpha_causal,
            objective.alpha_fidelity,
        )
        ms_builder = MultiStreamContextBuilder(
            bundle.multi_stream, device=str(device),
        )
        # CLI ``--patience -1`` is the explicit "disable early stop" sentinel
        # (argparse can't natively express ``None`` for an int flag without
        # nargs='?'). Map it to patience=None for the modifier.
        resolved_patience = patience
        if resolved_patience is not None and resolved_patience < 0:
            resolved_patience = None
        modifier = TrajectoryModifier(
            objective=objective, bundle=bundle,
            multi_stream_builder=ms_builder,
            diagnostics_enabled=diagnostics_enabled,
            device=device,
            patience=resolved_patience,
            convergence_tol=convergence_tol,
        )
        patience_str = (
            "off" if modifier.patience is None else str(modifier.patience)
        )
        mode_str = "iterative-topk" if iterative_topk else "batch-topk"
        n_to_modify = k if iterative_topk else len(top_k_trajs)
        _log(
            t0,
            f"modifying {n_to_modify} trajectories  "
            f"(mode={mode_str}, max_iters={modifier.max_iterations}, "
            f"patience={patience_str}, "
            f"conv_tol={modifier.convergence_tol:.2e}, "
            f"alphas=(sp={effective_alphas[0]:.2f}, "
            f"ca={effective_alphas[1]:.2f}, fi={effective_alphas[2]:.2f}))",
        )
        if iterative_topk:
            histories, top_k_scores = _iterative_topk_modify(
                modifier, bundle, k, t0,
                max_per_unit=max_per_unit,
                max_per_cell=max_per_cell,
            )
            top_k_indices = [
                bundle.trajectories.index(h.original) for h in histories
            ] if histories else []
            top_k_trajs = [h.original for h in histories]
        else:
            histories = _modify_with_progress(modifier, top_k_trajs)
        n_converged = sum(1 for h in histories if h.converged)
        mean_total_iters = (
            np.mean([h.total_iterations for h in histories]) if histories else 0.0
        )
        best_iters = [h.best_iteration for h in histories if h.best_iteration >= 0]
        mean_best_iter = np.mean(best_iters) if best_iters else 0.0
        _log(
            t0,
            f"modification done: converged={n_converged}/{len(histories)}  "
            f"mean_iters={mean_total_iters:.1f}  "
            f"mean_best_iter={mean_best_iter:.1f}",
        )

        pickup_after = modifier.current_pickup_3d()
        _log(t0, "building fairness grid (after)...")
        grid_after = build_fairness_grid(bundle, pickup_3d=pickup_after)
        if diagnostics_enabled:
            _log(t0, "computing gradient sensitivity (after)...")
            sensitivity_after = compute_gradient_sensitivity(bundle, pickup_after)
        else:
            sensitivity_after = None
        metrics_after = _scalar_metrics_from_grid(grid_after)
        _log(
            t0,
            f"metrics_after:  F_spatial={metrics_after['f_spatial']:.6f}  "
            f"F_causal={metrics_after['f_causal']:.6f}  "
            f"(ΔF_sp={metrics_after['f_spatial']-metrics_before['f_spatial']:+.3e}, "
            f"ΔF_ca={metrics_after['f_causal']-metrics_before['f_causal']:+.3e})",
        )

        modified_by_tid = {h.original.trajectory_id: h.modified for h in histories}
        trajs_after = [
            modified_by_tid.get(t.trajectory_id, t) for t in bundle.trajectories
        ]
        _log(t0, f"augmenting {len(trajs_after)} trajectories (after)...")
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
            effective_alpha_spatial=effective_alphas[0],
            effective_alpha_causal=effective_alphas[1],
            effective_alpha_fidelity=effective_alphas[2],
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
            per_cell_fairness_attribution=attribution,
            gradient_sensitivity_before=sensitivity_before,
            gradient_sensitivity_after=sensitivity_after,
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
    p.add_argument(
        "--diagnostics", action="store_true",
        help="Enable Tier A gradient decomposition and Tier C sensitivity "
             "grids. Costs ~3x per-iteration backward time. Off by default; "
             "set this flag when you want diagnostic artifacts.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Torch device for the modifier hot path: 'auto' (cuda if "
             "available, else cpu), 'cpu', or 'cuda[:idx]'. Default: auto.",
    )
    p.add_argument(
        "--patience", type=int, default=None,
        help="Patience-based early-stop: terminate a trajectory's "
             "optimization when its best objective hasn't improved by more "
             "than --convergence-tol for N consecutive iterations. Pass "
             "-1 to disable early-stop entirely (always run "
             "MAX_ITERATIONS). Default: config.PATIENCE.",
    )
    p.add_argument(
        "--convergence-tol", type=float, default=None,
        help="Minimum objective improvement that counts as an improvement "
             "for the patience-based early-stop. Should be set above the "
             "metric's numerical noise floor; the F-metric reductions "
             "are float64 internally so 1e-6 is well above noise. "
             "Default: config.CONVERGENCE_TOL.",
    )
    p.add_argument(
        "--iterative-topk", action="store_true",
        help="Iterative top-k with re-attribution: instead of selecting the "
             "entire top-k subset against the initial attribution and "
             "modifying them in sequence, re-attribute after each "
             "modification and pick the most-negative remaining trajectory "
             "for the next round. Mitigates trajectory-level interference "
             "when the initial top-k clusters geographically (see "
             "TRAJECTORY_EDITING_METHODOLOGY.md §8.1). Costs k extra "
             "attribution computations.",
    )
    p.add_argument(
        "--max-per-unit", type=int, default=None,
        help="Maximum trajectories selected from any single (pickup_cell, "
             "t_block) unit. Default: None (no cap). Setting --max-per-unit "
             "1 enforces unit-distinct selection — every selected trajectory "
             "has a different (cell, t_block) origin. Recommended for "
             "production k values where one POI/(cell, t) might otherwise "
             "dominate selection (see TRAJECTORY_EDITING_METHODOLOGY.md §8.3).",
    )
    p.add_argument(
        "--max-per-cell", type=int, default=None,
        help="Maximum trajectories selected from any single pickup cell "
             "(across all time blocks). Default: None (no cap). More "
             "aggressive than --max-per-unit; use when you want full "
             "cell-distinct selection.",
    )
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
    t0 = time.monotonic()
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_cli_overrides(args.override)
    device = _resolve_device(args.device)
    result = run_experiment(
        config_overrides=overrides,
        name=args.name,
        max_trajectories=args.max_trajectories,
        max_drivers=args.max_drivers,
        k=args.k,
        diagnostics_enabled=args.diagnostics,
        t0=t0,
        device=device,
        patience=args.patience,
        convergence_tol=args.convergence_tol,
        iterative_topk=args.iterative_topk,
        max_per_unit=args.max_per_unit,
        max_per_cell=args.max_per_cell,
    )
    from famail_temporal.evaluation.persistence import write
    from famail_temporal.evaluation.report import render
    output_root = Path(config.PACKAGE_ROOT) / "results"
    _log(t0, "writing artifacts to disk...")
    out_dir = write(result, output_root=output_root)
    _log(t0, "rendering report.md...")
    render(out_dir)
    _log(t0, f"experiment_id = {result.experiment_id}")
    _log(t0, f"results_dir  = {out_dir}")
    _log(t0, f"report       = {out_dir / 'report.md'}")
    _log(
        t0,
        f"F_spatial: {result.f_spatial_before:.4f} -> {result.f_spatial_after:.4f}",
    )
    _log(
        t0,
        f"F_causal:  {result.f_causal_before:.4f} -> {result.f_causal_after:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
