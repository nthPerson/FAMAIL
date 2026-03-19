"""
Compare V2 and V3 discriminators in the FAMAIL trajectory modification pipeline.

Runs identical trajectory modifications with V2 (single-stream) and V3 (multi-stream)
discriminators, then compares fidelity scores, convergence, objective values, and
spatial fairness gains.

Includes an ablation: V3 without multi-stream context (zero-padded) vs V3 with
full context, to isolate the effect of providing real driving/profile data.

Usage:
    python -m trajectory_modification.compare_discriminators \
        --v2-checkpoint discriminator/model/checkpoints/.../best.pt \
        --v3-checkpoint checkpoints/20260316_223817/best.pt \
        --n-trajectories 50 \
        --output-dir comparison_results/
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class RunMetrics:
    """Metrics from a single modification run."""
    label: str
    n_trajectories: int = 0
    fidelity_scores: List[float] = field(default_factory=list)
    objective_values: List[float] = field(default_factory=list)
    spatial_values: List[float] = field(default_factory=list)
    causal_values: List[float] = field(default_factory=list)
    convergence_iters: List[int] = field(default_factory=list)
    perturbation_magnitudes: List[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        def stats(arr):
            a = np.array(arr)
            if len(a) == 0:
                return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "median": float(np.median(a)),
            }

        return {
            "label": self.label,
            "n_trajectories": self.n_trajectories,
            "elapsed_seconds": self.elapsed_seconds,
            "fidelity": stats(self.fidelity_scores),
            "objective": stats(self.objective_values),
            "spatial": stats(self.spatial_values),
            "causal": stats(self.causal_values),
            "convergence_iters": stats(self.convergence_iters),
            "perturbation": stats(self.perturbation_magnitudes),
        }


@dataclass
class ComparisonConfig:
    """Configuration for discriminator comparison."""
    v2_checkpoint: str = ""
    v3_checkpoint: str = "checkpoints/20260316_223817/best.pt"
    n_trajectories: int = 50
    max_iterations: int = 50
    alpha_spatial: float = 0.33
    alpha_causal: float = 0.33
    alpha_fidelity: float = 0.34
    epsilon: float = 2.0
    alpha_step: float = 0.1
    gradient_mode: str = "heuristic"
    seed: int = 42
    output_dir: str = "comparison_results"
    run_v3_ablation: bool = True  # V3 without multi-stream context


class DiscriminatorComparison:
    """Run identical trajectory modification with V2 and V3 discriminators."""

    def __init__(self, config: ComparisonConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> Dict[str, RunMetrics]:
        """Execute comparison runs.

        Returns dict with keys:
            'v2': RunMetrics from V2 run
            'v3': RunMetrics from V3 with multi-stream
            'v3_no_context': RunMetrics from V3 without multi-stream (ablation)
        """
        from trajectory_modification import (
            DataBundle, TrajectoryModifier, FAMAILObjective,
            DiscriminatorAdapter,
        )
        from trajectory_modification.multi_stream_context import MultiStreamContextBuilder

        print(f"Loading data (max {self.config.n_trajectories} trajectories)...")
        bundle = DataBundle.load_default(
            max_trajectories=self.config.n_trajectories,
            estimate_g_from_data=True,
            aggregation='mean',
            load_multi_stream=True,
        )
        trajectories = bundle.trajectories
        print(f"  Loaded {len(trajectories)} trajectories from "
              f"{len(set(t.driver_id for t in trajectories))} drivers")

        results = {}

        # --- V2 run ---
        if self.config.v2_checkpoint and Path(self.config.v2_checkpoint).exists():
            print(f"\n{'='*60}")
            print(f"Running V2 ({self.config.v2_checkpoint})")
            print(f"{'='*60}")
            adapter_v2 = DiscriminatorAdapter(
                checkpoint_path=self.config.v2_checkpoint, device=self.device
            )
            results['v2'] = self._run_modification(
                trajectories, bundle, adapter_v2.model,
                multi_stream_context=None, label="V2 single-stream"
            )
        else:
            print(f"Skipping V2 (checkpoint not found: {self.config.v2_checkpoint})")

        # --- V3 with multi-stream context ---
        if Path(self.config.v3_checkpoint).exists():
            print(f"\n{'='*60}")
            print(f"Running V3 with multi-stream context ({self.config.v3_checkpoint})")
            print(f"{'='*60}")
            adapter_v3 = DiscriminatorAdapter(
                checkpoint_path=self.config.v3_checkpoint, device=self.device
            )

            ms_context = None
            if bundle.ms_driving_trajs is not None:
                ms_context = MultiStreamContextBuilder(
                    driving_trajs=bundle.ms_driving_trajs,
                    seeking_trajs=bundle.ms_seeking_trajs,
                    profile_features=bundle.ms_profile_features,
                    seeking_days=bundle.ms_seeking_days,
                    driving_days=bundle.ms_driving_days,
                    device=self.device,
                    seed=self.config.seed,
                )
                print(f"  Multi-stream context: {len(bundle.ms_driving_trajs)} drivers")

            results['v3'] = self._run_modification(
                trajectories, bundle, adapter_v3.model,
                multi_stream_context=ms_context, label="V3 multi-stream"
            )

            # --- V3 ablation (no multi-stream context) ---
            if self.config.run_v3_ablation:
                print(f"\n{'='*60}")
                print(f"Running V3 WITHOUT multi-stream context (ablation)")
                print(f"{'='*60}")
                results['v3_no_context'] = self._run_modification(
                    trajectories, bundle, adapter_v3.model,
                    multi_stream_context=None, label="V3 no-context (ablation)"
                )
        else:
            print(f"Skipping V3 (checkpoint not found: {self.config.v3_checkpoint})")

        return results

    def _run_modification(
        self,
        trajectories: list,
        bundle: 'DataBundle',
        discriminator: 'torch.nn.Module',
        multi_stream_context: Optional['MultiStreamContextBuilder'],
        label: str,
    ) -> RunMetrics:
        """Run modification pipeline with given discriminator."""
        from trajectory_modification import TrajectoryModifier, FAMAILObjective

        metrics = RunMetrics(label=label, n_trajectories=len(trajectories))

        objective = FAMAILObjective(
            alpha_spatial=self.config.alpha_spatial,
            alpha_causal=self.config.alpha_causal,
            alpha_fidelity=self.config.alpha_fidelity,
            g_function=bundle.g_function,
            discriminator=discriminator,
        )

        modifier = TrajectoryModifier(
            objective_fn=objective,
            alpha=self.config.alpha_step,
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_iterations,
            gradient_mode=self.config.gradient_mode,
            multi_stream_context=multi_stream_context,
        )

        # Set global state
        modifier.set_global_state(
            pickup_counts=torch.tensor(bundle.pickup_grid, device=self.device, dtype=torch.float32),
            dropoff_counts=torch.tensor(bundle.dropoff_grid, device=self.device, dtype=torch.float32),
            active_taxis=torch.tensor(bundle.active_taxis_grid, device=self.device, dtype=torch.float32),
            causal_demand=torch.tensor(bundle.causal_demand_grid, device=self.device, dtype=torch.float32) if bundle.causal_demand_grid is not None else None,
            causal_supply=torch.tensor(bundle.causal_supply_grid, device=self.device, dtype=torch.float32) if bundle.causal_supply_grid is not None else None,
        )

        start = time.time()
        for i, traj in enumerate(trajectories):
            try:
                history = modifier.modify_single(traj)

                if history.iterations:
                    last = history.iterations[-1]
                    metrics.fidelity_scores.append(last.f_fidelity)
                    metrics.objective_values.append(last.objective_value)
                    metrics.spatial_values.append(last.f_spatial)
                    metrics.causal_values.append(last.f_causal)
                    metrics.convergence_iters.append(history.total_iterations)

                    # Perturbation magnitude
                    orig = traj.states[-1]
                    mod = history.modified.states[-1]
                    delta = np.sqrt(
                        (orig.x_grid - mod.x_grid)**2 + (orig.y_grid - mod.y_grid)**2
                    )
                    metrics.perturbation_magnitudes.append(float(delta))

                if (i + 1) % 10 == 0 or i == 0:
                    fid = metrics.fidelity_scores[-1] if metrics.fidelity_scores else 0
                    obj = metrics.objective_values[-1] if metrics.objective_values else 0
                    print(f"  [{i+1}/{len(trajectories)}] fidelity={fid:.4f} obj={obj:.4f}")

            except Exception as e:
                print(f"  [{i+1}] ERROR: {e}")

        metrics.elapsed_seconds = time.time() - start
        summary = metrics.summary()
        print(f"\n  {label} summary:")
        print(f"    Fidelity:    mean={summary['fidelity']['mean']:.4f} "
              f"std={summary['fidelity']['std']:.4f}")
        print(f"    Objective:   mean={summary['objective']['mean']:.4f}")
        print(f"    Spatial:     mean={summary['spatial']['mean']:.4f}")
        print(f"    Convergence: mean={summary['convergence_iters']['mean']:.1f} iters")
        print(f"    Perturbation: mean={summary['perturbation']['mean']:.3f} cells")
        print(f"    Time: {metrics.elapsed_seconds:.1f}s")

        return metrics

    def generate_report(self, results: Dict[str, RunMetrics]) -> str:
        """Generate markdown comparison report."""
        lines = ["# Discriminator Comparison Report\n"]
        lines.append(f"**Config**: {self.config.n_trajectories} trajectories, "
                      f"ε={self.config.epsilon}, α_step={self.config.alpha_step}, "
                      f"max_iter={self.config.max_iterations}, "
                      f"gradient_mode={self.config.gradient_mode}\n")

        # Summary table
        labels = list(results.keys())
        metrics_list = [results[k].summary() for k in labels]

        lines.append("## Summary\n")
        lines.append("| Metric | " + " | ".join(m["label"] for m in metrics_list) + " |")
        lines.append("|" + "---|" * (len(metrics_list) + 1))

        rows = [
            ("Fidelity (mean)", "fidelity", "mean"),
            ("Fidelity (std)", "fidelity", "std"),
            ("Objective (mean)", "objective", "mean"),
            ("Spatial (mean)", "spatial", "mean"),
            ("Causal (mean)", "causal", "mean"),
            ("Convergence (mean iters)", "convergence_iters", "mean"),
            ("Perturbation (mean cells)", "perturbation", "mean"),
            ("Time (seconds)", None, None),
        ]

        for row_name, metric_key, stat_key in rows:
            vals = []
            for m in metrics_list:
                if metric_key is None:
                    vals.append(f"{m.get('elapsed_seconds', 0):.1f}")
                else:
                    vals.append(f"{m[metric_key][stat_key]:.4f}")
            lines.append(f"| {row_name} | " + " | ".join(vals) + " |")

        lines.append("")

        # Ablation analysis
        if 'v3' in results and 'v3_no_context' in results:
            v3 = results['v3'].summary()
            v3nc = results['v3_no_context'].summary()
            lines.append("## V3 Ablation: Multi-Stream Context Effect\n")
            fid_delta = v3['fidelity']['mean'] - v3nc['fidelity']['mean']
            std_delta = v3['fidelity']['std'] - v3nc['fidelity']['std']
            lines.append(f"- Fidelity change with context: {fid_delta:+.4f} (mean), {std_delta:+.4f} (std)")
            obj_delta = v3['objective']['mean'] - v3nc['objective']['mean']
            lines.append(f"- Objective change with context: {obj_delta:+.4f}")
            lines.append("")
            if abs(fid_delta) < 0.01:
                lines.append("**Interpretation**: Minimal effect — multi-stream context does not "
                             "significantly change fidelity scores. The driving/profile streams "
                             "may not be contributing meaningful signal.\n")
            elif fid_delta > 0:
                lines.append("**Interpretation**: Multi-stream context INCREASES fidelity scores, "
                             "indicating the discriminator is more confident that modified "
                             "trajectories match the driver when given full context.\n")
            else:
                lines.append("**Interpretation**: Multi-stream context DECREASES fidelity scores, "
                             "suggesting the discriminator is more discriminating with full "
                             "context — it may be detecting modification artifacts more effectively.\n")

        return "\n".join(lines)

    def save_results(
        self,
        results: Dict[str, RunMetrics],
        output_dir: Optional[str] = None,
    ) -> Path:
        """Save comparison results and report to disk."""
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save metrics JSON
        metrics_data = {k: v.summary() for k, v in results.items()}
        metrics_path = out / "comparison_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        # Save report
        report = self.generate_report(results)
        report_path = out / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report)

        # Save config
        config_path = out / "comparison_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"\nResults saved to {out}/")
        print(f"  - {metrics_path.name}")
        print(f"  - {report_path.name}")
        print(f"  - {config_path.name}")

        return out


def main():
    parser = argparse.ArgumentParser(description="Compare V2 vs V3 discriminators")
    parser.add_argument("--v2-checkpoint", type=str, default="",
                        help="Path to V2 checkpoint (skip if empty)")
    parser.add_argument("--v3-checkpoint", type=str,
                        default="checkpoints/20260316_223817/best.pt")
    parser.add_argument("--n-trajectories", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=2.0)
    parser.add_argument("--gradient-mode", type=str, default="heuristic",
                        choices=["heuristic", "soft_cell"])
    parser.add_argument("--output-dir", type=str, default="comparison_results")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip V3 ablation (without multi-stream)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ComparisonConfig(
        v2_checkpoint=args.v2_checkpoint,
        v3_checkpoint=args.v3_checkpoint,
        n_trajectories=args.n_trajectories,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        gradient_mode=args.gradient_mode,
        output_dir=args.output_dir,
        run_v3_ablation=not args.no_ablation,
        seed=args.seed,
    )

    comparison = DiscriminatorComparison(config)
    results = comparison.run()
    comparison.save_results(results)

    # Print report
    print("\n" + comparison.generate_report(results))


if __name__ == "__main__":
    main()
