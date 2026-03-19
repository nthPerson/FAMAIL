"""
Experiment runner for the FAMAIL Experiment & Analysis Framework.

Orchestrates the full trajectory modification pipeline:
  Phase 1: Attribution scoring + trajectory selection
  Phase 2: ST-iFGSM modification with per-trajectory instrumentation
  Global: Cumulative fairness tracking with periodic snapshots

Imports from trajectory_modification/ and objective_function/ — no code duplication.
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiment_framework.experiment_config import ExperimentConfig, SweepConfig
from experiment_framework.experiment_result import (
    ExperimentResult,
    TrajectoryResult,
    IterationRecord,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrate a FAMAIL experiment: data load -> Phase 1 -> Phase 2 -> save.

    Usage:
        config = ExperimentConfig(top_k=20, alpha_spatial=0.5)
        runner = ExperimentRunner(config)
        result = runner.run()
        run_dir = result.save()
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        config.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ExperimentResult:
        """Execute the full experiment pipeline."""
        start = time.time()
        cfg = self.config
        device = cfg.resolve_device()

        logger.info(f"Starting experiment '{cfg.experiment_name}' on {device}")

        # 1. Load data
        bundle = self._load_data()
        trajectories = bundle.trajectories
        n_drivers = len(set(t.driver_id for t in trajectories))
        logger.info(f"Loaded {len(trajectories)} trajectories from {n_drivers} drivers")

        # 2. Initialize components
        objective, discriminator_adapter, modifier, metrics, ms_context = (
            self._initialize_components(bundle, device)
        )

        # 3. Baseline global snapshot
        initial_snapshot = self._snapshot_to_dict(metrics.compute_snapshot())
        logger.info(f"Baseline: Gini={initial_snapshot['gini']:.4f}, "
                    f"F_causal={initial_snapshot['f_causal']:.4f}")

        # 4. Phase 1: Attribution
        attribution_scores, selected_indices = self._run_attribution(
            bundle, trajectories
        )
        logger.info(f"Phase 1: selected {len(selected_indices)} trajectories for modification")

        # 5. Phase 2: Modification loop
        trajectory_results, global_snapshots = self._run_modification(
            trajectories, modifier, metrics, selected_indices, device,
        )

        # 6. Final snapshot
        final_snapshot = self._snapshot_to_dict(metrics.compute_snapshot())
        duration = time.time() - start

        logger.info(f"Complete in {duration:.1f}s. "
                    f"Final Gini={final_snapshot['gini']:.4f}, "
                    f"F_causal={final_snapshot['f_causal']:.4f}")

        return ExperimentResult(
            config=cfg.to_dict(),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'),
            duration_seconds=duration,
            attribution_scores=attribution_scores,
            selected_indices=selected_indices,
            trajectory_results=trajectory_results,
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            global_snapshots=global_snapshots,
        )

    def run_sweep(self, sweep: SweepConfig) -> List[ExperimentResult]:
        """Run a parameter sweep, returning results for each configuration."""
        configs = sweep.generate_configs()
        results = []
        for i, cfg in enumerate(configs):
            logger.info(f"[{i+1}/{len(configs)}] Running: {cfg.experiment_name}")
            runner = ExperimentRunner(cfg)
            result = runner.run()
            result.save(cfg.output_dir)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load DataBundle with all required data."""
        from trajectory_modification import DataBundle

        return DataBundle.load_default(
            max_trajectories=self.config.max_trajectories,
            estimate_g_from_data=True,
            aggregation='mean',
            load_multi_stream=True,
        )

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _initialize_components(self, bundle, device: str):
        """Initialize all pipeline components.

        Mirrors the setup logic from dashboard.py _execute_modification().
        Returns: (objective, discriminator_adapter, modifier, metrics, ms_context)
        """
        from trajectory_modification import (
            TrajectoryModifier, FAMAILObjective, GlobalMetrics,
        )
        from trajectory_modification.discriminator_adapter import DiscriminatorAdapter

        cfg = self.config

        # --- Discriminator ---
        discriminator = None
        discriminator_adapter = None
        if cfg.use_discriminator:
            workspace_root = Path(__file__).parent.parent
            checkpoint_path = workspace_root / cfg.discriminator_checkpoint
            if checkpoint_path.exists():
                discriminator_adapter = DiscriminatorAdapter(
                    checkpoint_path=str(checkpoint_path), device=device,
                )
                discriminator = discriminator_adapter.model
                logger.info(f"Discriminator loaded: {getattr(discriminator_adapter, 'model_version', 'v1')}")
            else:
                logger.warning(f"Discriminator not found: {checkpoint_path}")

        # --- Objective function ---
        objective = FAMAILObjective(
            alpha_spatial=cfg.alpha_spatial,
            alpha_causal=cfg.alpha_causal,
            alpha_fidelity=cfg.alpha_fidelity,
            g_function=bundle.g_function,
            discriminator=discriminator,
            causal_formulation=cfg.causal_formulation,
            hat_matrices=bundle.hat_matrices,
            g0_power_basis_func=bundle.g0_power_basis_func,
            active_cell_indices=bundle.active_cell_indices,
        )

        # --- Multi-stream context (V3) ---
        ms_context = None
        if (bundle.ms_driving_trajs is not None
                and discriminator_adapter is not None
                and getattr(discriminator_adapter, 'model_version', 'v1') == 'v3'):
            from trajectory_modification.multi_stream_context import MultiStreamContextBuilder
            ms_context = MultiStreamContextBuilder(
                driving_trajs=bundle.ms_driving_trajs,
                seeking_trajs=bundle.ms_seeking_trajs or {},
                profile_features=bundle.ms_profile_features or {},
                seeking_days=bundle.ms_seeking_days,
                driving_days=bundle.ms_driving_days,
                fill_strategy=cfg.seeking_fill_strategy,
                device=device,
            )
            logger.info("V3 multi-stream context built")

        # --- Modifier ---
        modifier = TrajectoryModifier(
            objective_fn=objective,
            alpha=cfg.alpha,
            epsilon=cfg.epsilon,
            max_iterations=cfg.max_iterations,
            convergence_threshold=cfg.convergence_threshold,
            gradient_mode='soft_cell',
            temperature=cfg.temperature,
            temperature_annealing=cfg.temperature_annealing,
            tau_max=cfg.tau_max,
            tau_min=cfg.tau_min,
            multi_stream_context=ms_context,
        )

        # --- Set global state ---
        causal_demand = bundle.causal_demand_grid
        causal_supply = bundle.causal_supply_grid

        modifier.set_global_state(
            pickup_counts=torch.tensor(bundle.pickup_grid, device=device, dtype=torch.float32),
            dropoff_counts=torch.tensor(bundle.dropoff_grid, device=device, dtype=torch.float32),
            active_taxis=torch.tensor(bundle.active_taxis_grid, device=device, dtype=torch.float32),
            causal_demand=(
                torch.tensor(causal_demand, device=device, dtype=torch.float32)
                if causal_demand is not None else None
            ),
            causal_supply=(
                torch.tensor(causal_supply, device=device, dtype=torch.float32)
                if causal_supply is not None else None
            ),
        )

        # --- Global metrics ---
        metrics = GlobalMetrics(
            g_function=bundle.g_function,
            alpha_weights=(cfg.alpha_spatial, cfg.alpha_causal, cfg.alpha_fidelity),
            hat_matrices=bundle.hat_matrices,
            active_cell_indices=bundle.active_cell_indices,
            g0_power_basis_func=bundle.g0_power_basis_func,
        )
        metrics.initialize_from_data(
            bundle.pickup_grid,
            bundle.dropoff_grid,
            bundle.active_taxis_grid,
        )

        return objective, discriminator_adapter, modifier, metrics, ms_context

    # ------------------------------------------------------------------
    # Phase 1: Attribution
    # ------------------------------------------------------------------

    def _run_attribution(self, bundle, trajectories) -> Tuple[List[Dict], List[int]]:
        """Compute attribution scores and select top-k trajectories.

        Uses the attribution_integration module for baseline DCD.
        For Option B, extends with hat-matrix demographic projection.
        """
        cfg = self.config

        # Convert Trajectory objects to dicts for attribution_integration
        traj_dicts = []
        for i, traj in enumerate(trajectories):
            pickup = traj.pickup_cell
            # dropoff is first state (start of seeking trajectory)
            first_state = traj.states[0]
            dropoff = (int(first_state.x_grid), int(first_state.y_grid))
            traj_dicts.append({
                'trajectory_id': getattr(traj, 'trajectory_id', f'traj_{i}'),
                'pickup_cell': pickup,
                'dropoff_cell': dropoff,
                'index': i,
            })

        # Use baseline attribution from attribution_integration
        try:
            from objective_function.dashboards.components.attribution_integration import (
                compute_combined_attribution,
                select_trajectories_for_modification,
            )

            result = compute_combined_attribution(
                trajectories=traj_dicts,
                pickup_counts=bundle.pickup_grid,
                dropoff_counts=bundle.dropoff_grid,
                supply_counts=bundle.causal_supply_grid if bundle.causal_supply_grid is not None else bundle.pickup_grid,
                g_function=bundle.g_function,
                lis_weight=cfg.lis_weight,
                dcd_weight=cfg.dcd_weight,
                normalize=True,
            )

            # Select top-k
            method_map = {'top_k': 'top_n', 'diverse': 'diverse'}
            method = method_map.get(cfg.selection_method, 'top_n')
            selected = select_trajectories_for_modification(
                result, n_trajectories=cfg.top_k, selection_method=method,
            )

            # Map back to trajectory indices
            # Attribution scores use trajectory_id from our dicts
            selected_ids = {s.trajectory_id for s in selected}
            selected_indices = [
                i for i, td in enumerate(traj_dicts) if td['trajectory_id'] in selected_ids
            ]

            # Build attribution score dicts for output
            attribution_scores = []
            for s in result.trajectory_scores:
                idx = next(
                    (i for i, td in enumerate(traj_dicts) if td['trajectory_id'] == s.trajectory_id),
                    None,
                )
                attribution_scores.append({
                    'index': idx,
                    'trajectory_id': str(s.trajectory_id),
                    'lis_score': s.lis_score,
                    'dcd_score': s.dcd_score,
                    'combined_score': s.combined_score,
                    'pickup_cell': list(s.pickup_cell) if s.pickup_cell else None,
                })

            # Sort by combined_score descending
            attribution_scores.sort(key=lambda x: x['combined_score'], reverse=True)

        except ImportError:
            logger.warning("Attribution integration not available. Using random selection.")
            np.random.seed(cfg.random_seed)
            n = min(cfg.top_k, len(trajectories))
            selected_indices = np.random.choice(len(trajectories), size=n, replace=False).tolist()
            attribution_scores = [
                {'index': i, 'trajectory_id': str(i), 'combined_score': 0.0}
                for i in selected_indices
            ]

        return attribution_scores, selected_indices

    # ------------------------------------------------------------------
    # Phase 2: Modification loop
    # ------------------------------------------------------------------

    def _run_modification(
        self,
        trajectories,
        modifier,
        metrics,
        selected_indices: List[int],
        device: str,
    ) -> Tuple[List[TrajectoryResult], List[Dict]]:
        """Run ST-iFGSM on selected trajectories with global tracking."""
        cfg = self.config
        trajectory_results = []
        global_snapshots = []

        for rank, traj_idx in enumerate(selected_indices):
            traj = trajectories[traj_idx]
            old_cell = traj.pickup_cell

            logger.info(
                f"  [{rank+1}/{len(selected_indices)}] Modifying trajectory {traj_idx} "
                f"(driver {traj.driver_id}, cell {old_cell})"
            )

            # Run modification
            if cfg.record_gradient_decomposition:
                tr = self._modify_single_with_decomposition(
                    traj, traj_idx, modifier, device,
                )
            else:
                tr = self._modify_single_standard(traj, traj_idx, modifier)

            trajectory_results.append(tr)

            # Update global metrics
            new_cell = tuple(tr.modified_pickup)
            fidelity = tr.final_f_fidelity
            metrics.update_pickup(old_cell, new_cell, fidelity)

            # Periodic global snapshot
            n_done = rank + 1
            if n_done % cfg.snapshot_every_n == 0 or n_done == len(selected_indices):
                snapshot = metrics.compute_snapshot()
                global_snapshots.append({
                    'after_n_trajectories': n_done,
                    **self._snapshot_to_dict(snapshot),
                })

        return trajectory_results, global_snapshots

    def _modify_single_standard(self, traj, traj_idx: int, modifier) -> TrajectoryResult:
        """Modify a trajectory using the existing TrajectoryModifier.modify_single()."""
        history = modifier.modify_single(traj)

        # Extract per-iteration records
        iterations = []
        for i, it in enumerate(history.iterations):
            iterations.append(IterationRecord(
                iteration=i,
                objective=float(it.objective_value),
                f_spatial=float(it.f_spatial),
                f_causal=float(it.f_causal),
                f_fidelity=float(it.f_fidelity),
                gradient_norm=float(it.gradient_norm),
                perturbation=it.perturbation.tolist() if hasattr(it.perturbation, 'tolist') else list(it.perturbation),
            ))

        # Final term values
        last = history.iterations[-1] if history.iterations else None
        modified = history.modified
        orig_cell = traj.pickup_cell
        mod_cell = modified.pickup_cell

        dx = mod_cell[0] - orig_cell[0]
        dy = mod_cell[1] - orig_cell[1]
        perturbation_mag = float(np.sqrt(dx**2 + dy**2))

        return TrajectoryResult(
            trajectory_index=traj_idx,
            trajectory_id=str(getattr(traj, 'trajectory_id', traj_idx)),
            driver_id=int(traj.driver_id),
            original_pickup=list(orig_cell),
            modified_pickup=list(mod_cell),
            converged=history.converged,
            total_iterations=history.total_iterations,
            final_objective=float(history.final_objective),
            final_f_spatial=float(last.f_spatial) if last else 0.0,
            final_f_causal=float(last.f_causal) if last else 0.0,
            final_f_fidelity=float(last.f_fidelity) if last else 0.0,
            perturbation_magnitude=perturbation_mag,
            iterations=iterations,
        )

    def _modify_single_with_decomposition(
        self, traj, traj_idx: int, modifier, device: str,
    ) -> TrajectoryResult:
        """Modify with per-term gradient decomposition.

        Delegates to the GradientDecomposer for the inner loop.
        Falls back to standard modification if decomposition is not available.
        """
        try:
            from experiment_framework.gradient_decomposition import (
                run_modification_with_decomposition,
            )
            return run_modification_with_decomposition(
                traj, traj_idx, modifier, device, self.config,
            )
        except ImportError:
            logger.warning("Gradient decomposition module not available. Using standard modification.")
            return self._modify_single_standard(traj, traj_idx, modifier)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_to_dict(snapshot) -> Dict[str, float]:
        """Convert a FairnessSnapshot to a plain dict."""
        return {
            'gini': float(snapshot.gini_coefficient),
            'f_spatial': float(snapshot.f_spatial),
            'f_causal': float(snapshot.f_causal),
            'f_fidelity': float(snapshot.f_fidelity),
            'combined': float(snapshot.combined_objective),
            'num_modified': int(snapshot.num_trajectories_modified),
        }
