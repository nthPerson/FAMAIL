"""Experiment runner: orchestrates the full FAMAIL pipeline."""

from __future__ import annotations
import argparse
import datetime as _dt
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution, rank_trajectories,
)
from famail_temporal.algorithm.editing_loop import run_editing_rounds, RoundRecord
from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.supply import (
    supply_gradient_N, lift_candidates, assemble_edit_plan,
)
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

    rounds: List[RoundRecord] = field(default_factory=list)
    # E6: every trajectory's selection αᵢ (ascending), for the attribution
    # distribution figure. Optional so synthetic constructions stay valid.
    all_trajectory_scores: Optional[np.ndarray] = None

    # Supply-lift editing (Task 8). ``delta_supply_3d`` is the modifier's
    # accumulated tier-1 ΔS grid (None for anything constructed before this
    # field existed, e.g. pre-existing test fixtures — persistence treats
    # that as "no supply-lift data to write", see persistence.write). Under
    # TAIL_LEN=0/LIFT_BUDGET=0 (G1) this is an all-zero array, never None,
    # since run_experiment always populates it from the live modifier.
    delta_supply_3d: Optional[np.ndarray] = None
    n_trim: int = 0
    n_lift: int = 0
    n_taper_infeasible_trim: int = 0
    n_taper_infeasible_lift: int = 0


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
    max_rounds: Optional[int] = None,
    round_convergence_tol: Optional[float] = None,
    round_patience: Optional[int] = None,
    epsilon_cap: Optional[float] = None,
    accept_rule: Optional[str] = None,
    iterative_topk_max_edits: Optional[int] = None,
    use_ste: Optional[bool] = None,
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
        if not any(s < 0 for _, s in scored):
            raise ValueError(
                "No trajectories with strictly negative attribution were found. "
                "Under the F-decomposition convention, negative αᵢ marks cells "
                "dragging fairness below baseline; if none exist the audit set is "
                "uniformly fair (check the active mask / demographics carry signal)."
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
            accept_rule=accept_rule,
            epsilon_cap=epsilon_cap,
            use_ste=use_ste,
        )
        # Resolve outer-loop knobs. Historical --iterative-topk did up to k
        # single-edit rounds (B=1), stopping at pool-exhaustion — so iterative
        # mode defaults max_rounds to k. Batch defaults to config.MAX_ROUNDS (1).
        if max_rounds is not None:
            resolved_max_rounds = max_rounds
        elif iterative_topk:
            resolved_max_rounds = k
        else:
            resolved_max_rounds = config.MAX_ROUNDS
        resolved_round_patience = (
            config.ROUND_PATIENCE if round_patience is None else round_patience
        )
        resolved_round_tol = (
            config.ROUND_CONVERGENCE_TOL if round_convergence_tol is None
            else round_convergence_tol
        )
        resolved_max_edits = (
            config.ITERATIVE_TOPK_MAX_EDITS if iterative_topk_max_edits is None
            else iterative_topk_max_edits
        )
        if resolved_round_tol is not None and resolved_max_rounds <= 1:
            _log(t0, "WARNING: round-convergence-tol set but max-rounds<=1; "
                     "running a single pass. Raise --max-rounds for convergence mode.")

        # Lift enablement for the lift-selection block after the trim loop.
        # False includes the G1 legacy configuration (TAIL_LEN=0 or
        # LIFT_BUDGET=0): such invocations run zero lift compute.
        lift_enabled = config.TAIL_LEN > 0 and config.LIFT_BUDGET != 0
        if config.TAIL_LEN > 0:
            # Subnormal residuals from float32 persist chains (-=mass/+=mass on
            # the shared demand grid) cause pathological 10-100x CPU backward
            # slowdowns; flush-to-zero eliminates them at no accuracy cost at
            # our magnitudes (observed: one edit stalling 25+ min/iteration
            # after ~2250 normal edits in the k=10000 validation run).
            #
            # Scope: guarded on TAIL_LEN > 0 (ANY taper-mode run), not on
            # lift_enabled — the denormal-poisoning mechanism is the TRIM
            # persist chain, which runs at full k in a trim-only ablation
            # (TAIL_LEN=4, LIFT_BUDGET=0). TAIL_LEN=0 legacy runs keep the
            # historical FP environment untouched for bit-reproduction of
            # published numbers (G1).
            #
            # WARNING (process-global side effect): set_flush_denormal
            # mutates process-wide FPU state (FTZ/DAZ bits in MXCSR) that
            # PERSISTS after run_experiment returns — a taper-mode call
            # followed by a legacy call in the SAME process would leave FTZ
            # enabled for the legacy call. Current guarantee: the runner CLI
            # is one-shot per process, so production runs are unaffected;
            # library callers mixing taper and legacy modes in one process
            # beware.
            ftz_supported = torch.set_flush_denormal(True)
            if ftz_supported:
                _log(t0, "flush-denormal enabled (subnormal float32 residuals "
                         "from persist chains cause pathological CPU backward "
                         "slowdowns)")
            else:
                _log(t0, "WARNING: torch.set_flush_denormal(True) not supported "
                         "on this platform — subnormal-residual slowdowns "
                         "remain possible")

        # Per-edit progress heartbeat for the trim loop. run_editing_rounds'
        # on_iter is forwarded to modify_single's on_iteration, which fires
        # with (iteration_index, ModificationResult) on EVERY inner ST-iFGSM
        # step (see editing_loop.py:126 / modifier.py:739) — there is no
        # per-edit hook in that signature, so iteration_index == 0 is used as
        # the "new edit started" marker and per-100-edit wall time + the
        # iteration's f_causal are logged from what the callback does provide.
        # Pure instrumentation: touches no algorithmic state.
        _progress = {"edit_idx": 0, "t_mark": time.monotonic()}

        def _trim_progress(it_idx: int, rec) -> None:
            if it_idx != 0:
                return
            _progress["edit_idx"] += 1
            if _progress["edit_idx"] % 100 == 0:
                now = time.monotonic()
                _log(t0, f"trim progress: edit {_progress['edit_idx']} "
                         f"(+{now - _progress['t_mark']:.1f}s /100 edits) "
                         f"f_causal={rec.f_causal:.6f}")
                _progress["t_mark"] = now

        _log(t0, f"editing loop: mode={'iterative' if iterative_topk else 'batch'} "
                 f"max_rounds={resolved_max_rounds} eps_cap={modifier.epsilon_cap} "
                 f"accept={modifier.accept_rule} round_tol={resolved_round_tol}")
        loop_result = run_editing_rounds(
            modifier, bundle,
            k=k,
            mode="iterative" if iterative_topk else "batch",
            max_rounds=resolved_max_rounds,
            round_convergence_tol=resolved_round_tol,
            round_patience=resolved_round_patience,
            iterative_max_edits=resolved_max_edits,
            max_per_unit=max_per_unit,
            max_per_cell=max_per_cell,
            on_iter=_trim_progress,
            log=lambda msg: _log(t0, msg),
        )
        _log(t0, "trim phase complete")
        histories = loop_result.histories
        rounds = loop_result.rounds
        # Selection-time αᵢ per edit (aligned with histories) — persistence
        # writes this per trajectory, so it must carry real scores.
        top_k_scores = loop_result.edit_scores
        _log(t0, f"editing loop done: {len(histories)} edits over "
                 f"{len(rounds)} round(s), stop={loop_result.stop_reason}")
        n_trim = len(histories)

        # ── Supply-lift selection (Task 8) ──────────────────────────────
        # Runs AFTER the trim rounds (which are untouched above) so trim
        # keeps sole claim on its budget/eligibility machinery. Entirely
        # skipped — no supply_gradient_N / lift_candidates call at all — when
        # lift is disabled (TAIL_LEN<=0 or LIFT_BUDGET==0); this is exactly
        # the G1 legacy configuration, so the legacy path pays zero extra
        # compute and carries zero extra risk.
        #
        # Pass-ordering assumption (load-bearing, not incidental): every lift
        # edit below runs after ALL trim edits from run_editing_rounds have
        # already persisted. The lift branch in modifier.py sanitizes the
        # shared demand grid (clamps ULP-negative float32 persist residuals)
        # before it reads it; the trim path does not. Reordering this block
        # ahead of the trim loop, or interleaving trim and lift edits, would
        # let a trim read an unsanitized grid and risk a crash in
        # compute_fspatial. See assemble_edit_plan's docstring (supply.py)
        # and the invariant comment at the modifier's lift clamp.
        lift_histories: List[ModificationHistory] = []
        lift_scores: List[float] = []
        if lift_enabled:
            _log(t0, "lift phase starting")
            _log(t0, "computing supply-gradient attribution for lift selection...")
            # supply_gradient_N (Task 4, algorithm/supply.py — out of scope
            # to modify here) always builds its internal leaf/pickup tensors
            # on CPU, so it needs the objective's buffers on CPU too. It runs
            # once per experiment (not the per-trajectory hot path), so the
            # temporary shuttle costs one extra .to() round trip — same
            # one-shot-CPU convention already used above for grid_before /
            # sensitivity_before / augment. nn.Module.to() mutates in place
            # and returns self, so `objective` is moved back before any
            # lift edit runs (the modifier's optimization loop needs it on
            # the configured device).
            objective.to("cpu")
            try:
                grad_N = supply_gradient_N(bundle, objective)
            finally:
                objective.to(device)
            lift_scored = lift_candidates(
                bundle, grad_N, tail_len=config.TAIL_LEN, epsilon=config.EPSILON_BALL,
            )
            # trim_indices: positions (into bundle.trajectories) of every
            # trajectory the trim rounds edited, deduped and order-preserved
            # (iterative mode may re-edit the same trajectory across rounds).
            # These give assemble_edit_plan its trim-precedence set and the
            # default lift_budget = k - n_trim fill.
            id_to_idx = {t.trajectory_id: i for i, t in enumerate(bundle.trajectories)}
            trim_indices = list(dict.fromkeys(
                id_to_idx[tid] for tid in loop_result.edited_ids
            ))
            plan = assemble_edit_plan(
                trim_indices, lift_scored, k_total=k, lift_budget=config.LIFT_BUDGET,
            )
            lift_score_by_idx = dict(lift_scored)
            orig_pos_all = {
                t.trajectory_id: (float(t.pickup_state.x_grid), float(t.pickup_state.y_grid))
                for t in bundle.trajectories
            }
            lift_t_mark = time.monotonic()
            for idx, mode in plan:
                if mode != "lift":
                    continue
                # idx is guaranteed disjoint from trim_indices (assemble_edit_plan
                # dedupes), so bundle.trajectories[idx] is still the pristine
                # original — no trim edit ever touched it.
                traj = bundle.trajectories[idx]
                h = modifier.modify_single(
                    traj, mode="lift", original_cell=orig_pos_all[traj.trajectory_id],
                )
                lift_histories.append(h)
                lift_scores.append(float(lift_score_by_idx[idx]))
                if len(lift_histories) % 100 == 0:
                    now = time.monotonic()
                    _log(t0, f"lift progress: edit {len(lift_histories)} "
                             f"(+{now - lift_t_mark:.1f}s /100 edits) "
                             f"n_taper_infeasible_lift="
                             f"{modifier.n_taper_infeasible_lift}")
                    lift_t_mark = now
            _log(
                t0,
                f"lift step done: {len(lift_histories)} lift edits "
                f"(n_taper_infeasible_lift={modifier.n_taper_infeasible_lift})",
            )

        histories = histories + lift_histories
        top_k_scores = top_k_scores + lift_scores
        n_lift = len(lift_histories)

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
        if config.TAIL_LEN > 0:
            # Taper-mode runs fully drain far more demand cells than legacy
            # trim ever did (lift relocates pickup demand too, and lift
            # selection deliberately clusters near under-served regions). A
            # fully drained float32 cell can rest a few ULP below zero
            # (established mechanism: 67 aggregated pickups minus 67 persist
            # -=mass subtractions = -1.86e-9), and the fairness stack's
            # strict negativity check then rejects the whole after-grid.
            # Clamp negatives to exact 0 at THE single fetch point so every
            # downstream consumer (grid_after, sensitivity_after, persisted
            # artifacts) sees the sanitized grid — downstream evaluators
            # already treat |v| < 1e-6 as zero (legacy float32 drift).
            # Legacy (TAIL_LEN=0) keeps the raw grid byte-identical (G1):
            # verified that the pre-branch pipeline passes the raw drifted
            # grid straight through and has simply never drained an
            # unluckily-rounded cell.
            pickup_after = np.clip(pickup_after, 0.0, None)
        # delta_supply_3d is always materialized (a zero array when neither
        # trim-taper nor lift moved any supply — e.g. G1) so the "no ΔS
        # happened" case is a real, checkable value, not an absent field.
        delta_supply_3d = modifier.current_delta_supply_3d()
        _log(t0, "building fairness grid (after)...")
        if np.any(delta_supply_3d):
            # Build a bundle whose active_taxis_3d reflects the endogenous
            # supply change, then reuse build_fairness_grid's existing,
            # unmodified math (grid.py is frozen for this task) so
            # f_spatial/f_causal/gini_dsr/gini_asr all come out mutually
            # consistent under the new supply. Same clip convention as the
            # objective's delta_supply_N path (config.SUPPLY_FLOOR).
            active_taxis_after = np.clip(
                bundle.active_taxis_3d + delta_supply_3d, config.SUPPLY_FLOOR, None,
            ).astype(bundle.active_taxis_3d.dtype)
            bundle_for_after = replace(bundle, active_taxis_3d=active_taxis_after)
        else:
            bundle_for_after = bundle
        grid_after = build_fairness_grid(bundle_for_after, pickup_3d=pickup_after)
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
            rounds=rounds,
            all_trajectory_scores=np.asarray([s for _, s in scored], dtype=np.float32),
            delta_supply_3d=delta_supply_3d,
            n_trim=n_trim,
            n_lift=n_lift,
            n_taper_infeasible_trim=modifier.n_taper_infeasible_trim,
            n_taper_infeasible_lift=modifier.n_taper_infeasible_lift,
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
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Outer re-attribution rounds (hard ceiling; also the "
                        "convergence-mode safety cap). Default config.MAX_ROUNDS "
                        "(1 = single pass).")
    p.add_argument("--round-convergence-tol", type=float, default=None,
                   help="Enable convergence stop: halt when best round F_causal "
                        "has not improved by more than this for --round-patience "
                        "rounds. Default config.ROUND_CONVERGENCE_TOL (off).")
    p.add_argument("--round-patience", type=int, default=None,
                   help="Outer-loop patience (rounds). Default config.ROUND_PATIENCE.")
    p.add_argument("--epsilon-cap", type=float, default=None,
                   help="Cumulative L-inf displacement cap from each trajectory's "
                        "true original cell, across rounds. Pass 'inf' for "
                        "unbounded per-round-epsilon stacking. Default "
                        "config.EPSILON_CAP (=EPSILON_BALL, 2.0).")
    p.add_argument("--accept-rule", choices=["objective", "non-regression"],
                   default=None,
                   help="Inner acceptance gate. 'non-regression' requires each "
                        "persisted edit to improve F_causal and not regress "
                        "F_spatial. Default config.ACCEPT_RULE ('objective').")
    p.add_argument("--iterative-topk-max-edits", type=int, default=None,
                   help="Max edits per trajectory in --iterative-topk mode "
                        "(0 = unlimited). Default config.ITERATIVE_TOPK_MAX_EDITS (1).")
    p.add_argument("--ste", action="store_true",
                   help="Straight-through hard-metric editing: optimize/select/gate "
                        "on the realizable hard grid (forward=hard, grad=soft). "
                        "Off by default (config.STE_ENABLED).")
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
        max_rounds=args.max_rounds,
        round_convergence_tol=args.round_convergence_tol,
        round_patience=args.round_patience,
        epsilon_cap=args.epsilon_cap,
        accept_rule=args.accept_rule,
        iterative_topk_max_edits=args.iterative_topk_max_edits,
        use_ste=(True if args.ste else None),
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
