"""
Result types and serialization for the FAMAIL Experiment & Analysis Framework.

Defines the data structures that capture experiment outputs at every level:
per-iteration gradient decomposition, per-trajectory modification traces,
and global fairness evolution. Includes JSON/CSV serialization and
auto-generated markdown report.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-iteration data
# ---------------------------------------------------------------------------

@dataclass
class GradientDecomposition:
    """Per-term gradient vectors at a single ST-iFGSM iteration.

    Each gradient is a 2D vector [dx, dy] representing the partial derivative
    of the corresponding objective term w.r.t. the pickup position.

    The fractions indicate what share of the total gradient magnitude
    each term contributes. alignment_spatial_causal is the cosine similarity
    between the spatial and causal gradient vectors (positive = cooperating,
    negative = conflicting).
    """

    grad_spatial: List[float]       # [dx, dy]
    grad_causal: List[float]        # [dx, dy]
    grad_fidelity: List[float]      # [dx, dy]
    grad_combined: List[float]      # [dx, dy]
    spatial_fraction: float         # ||grad_spatial|| / total_norm
    causal_fraction: float
    fidelity_fraction: float
    alignment_spatial_causal: float  # cosine similarity [-1, 1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GradientDecomposition':
        return cls(**d)


@dataclass
class IterationRecord:
    """Metrics captured at a single ST-iFGSM iteration."""

    iteration: int
    objective: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    gradient_norm: float
    perturbation: List[float]  # [x, y] cumulative delta from original
    gradient_decomposition: Optional[GradientDecomposition] = None
    debug_dict: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = {
            'iteration': self.iteration,
            'objective': self.objective,
            'f_spatial': self.f_spatial,
            'f_causal': self.f_causal,
            'f_fidelity': self.f_fidelity,
            'gradient_norm': self.gradient_norm,
            'perturbation': self.perturbation,
        }
        if self.gradient_decomposition is not None:
            d['gradient_decomposition'] = self.gradient_decomposition.to_dict()
        # debug_dict intentionally excluded from serialization (too large)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'IterationRecord':
        gd = d.pop('gradient_decomposition', None)
        return cls(
            gradient_decomposition=GradientDecomposition.from_dict(gd) if gd else None,
            **d,
        )


# ---------------------------------------------------------------------------
# Per-trajectory data
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryResult:
    """Complete modification result for a single trajectory."""

    trajectory_index: int
    trajectory_id: str
    driver_id: int
    original_pickup: List[int]    # [x, y] grid coords
    modified_pickup: List[int]    # [x, y] grid coords
    converged: bool
    total_iterations: int
    final_objective: float
    final_f_spatial: float
    final_f_causal: float
    final_f_fidelity: float
    perturbation_magnitude: float
    iterations: List[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'trajectory_index': self.trajectory_index,
            'trajectory_id': self.trajectory_id,
            'driver_id': self.driver_id,
            'original_pickup': self.original_pickup,
            'modified_pickup': self.modified_pickup,
            'converged': self.converged,
            'total_iterations': self.total_iterations,
            'final_objective': self.final_objective,
            'final_f_spatial': self.final_f_spatial,
            'final_f_causal': self.final_f_causal,
            'final_f_fidelity': self.final_f_fidelity,
            'perturbation_magnitude': self.perturbation_magnitude,
            'iterations': [it.to_dict() for it in self.iterations],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'TrajectoryResult':
        d = d.copy()
        d['iterations'] = [IterationRecord.from_dict(it) for it in d.get('iterations', [])]
        return cls(**d)

    @property
    def perturbation_vector(self) -> List[float]:
        """The final cumulative perturbation [dx, dy]."""
        if self.iterations:
            return self.iterations[-1].perturbation
        return [0.0, 0.0]


# ---------------------------------------------------------------------------
# Full experiment result
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Complete output from a single experiment run.

    Contains everything needed to reconstruct the experiment: configuration,
    attribution scores, per-trajectory modification results, and global
    fairness snapshots taken at intervals during modification.
    """

    config: Dict[str, Any]  # ExperimentConfig.to_dict()
    timestamp: str
    duration_seconds: float
    attribution_scores: List[Dict[str, Any]]
    selected_indices: List[int]
    trajectory_results: List[TrajectoryResult]
    initial_snapshot: Dict[str, float]
    final_snapshot: Dict[str, float]
    global_snapshots: List[Dict[str, Any]]  # [{after_n, gini, f_spatial, ...}]

    def to_dict(self) -> dict:
        return {
            'metadata': {
                'timestamp': self.timestamp,
                'duration_seconds': self.duration_seconds,
            },
            'config': self.config,
            'summary': self._build_summary(),
            'attribution_scores': self.attribution_scores,
            'selected_indices': self.selected_indices,
            'global_snapshots': self.global_snapshots,
            'trajectory_results': [tr.to_dict() for tr in self.trajectory_results],
            'initial_snapshot': self.initial_snapshot,
            'final_snapshot': self.final_snapshot,
        }

    def _build_summary(self) -> dict:
        """Build high-level summary from results."""
        n_converged = sum(1 for tr in self.trajectory_results if tr.converged)
        mean_iters = (
            np.mean([tr.total_iterations for tr in self.trajectory_results])
            if self.trajectory_results else 0
        )
        mean_perturbation = (
            np.mean([tr.perturbation_magnitude for tr in self.trajectory_results])
            if self.trajectory_results else 0
        )

        improvement = {}
        for key in ('gini', 'f_spatial', 'f_causal', 'f_fidelity', 'combined'):
            initial_val = self.initial_snapshot.get(key, 0)
            final_val = self.final_snapshot.get(key, 0)
            improvement[f'delta_{key}'] = final_val - initial_val

        return {
            'initial': self.initial_snapshot,
            'final': self.final_snapshot,
            'improvement': improvement,
            'n_trajectories_modified': len(self.trajectory_results),
            'n_converged': n_converged,
            'mean_iterations': float(mean_iters),
            'mean_perturbation_magnitude': float(mean_perturbation),
        }

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def save(self, output_dir: Optional[str] = None) -> str:
        """Save all results to a timestamped run directory.

        Returns the path to the created run directory.
        """
        base = output_dir or self.config.get('output_dir', 'experiment_results')
        name = self.config.get('experiment_name', 'default')
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(base) / f"{ts}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Full results JSON
        with open(run_dir / 'results.json', 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

        # Config JSON
        with open(run_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2, default=_json_default)

        # Summary JSON
        with open(run_dir / 'summary.json', 'w') as f:
            json.dump(self._build_summary(), f, indent=2, default=_json_default)

        # CSV files
        self._write_trajectories_csv(run_dir / 'trajectories.csv')
        self._write_iterations_csv(run_dir / 'iterations.csv')
        self._write_global_snapshots_csv(run_dir / 'global_snapshots.csv')
        self._write_attribution_csv(run_dir / 'attribution_scores.csv')

        # Report
        report = self.generate_report()
        with open(run_dir / 'report.md', 'w') as f:
            f.write(report)

        return str(run_dir)

    @classmethod
    def load(cls, run_dir: str) -> 'ExperimentResult':
        """Load results from a previously saved run directory."""
        run_path = Path(run_dir)
        with open(run_path / 'results.json') as f:
            data = json.load(f)

        return cls(
            config=data['config'],
            timestamp=data['metadata']['timestamp'],
            duration_seconds=data['metadata']['duration_seconds'],
            attribution_scores=data.get('attribution_scores', []),
            selected_indices=data.get('selected_indices', []),
            trajectory_results=[
                TrajectoryResult.from_dict(tr)
                for tr in data.get('trajectory_results', [])
            ],
            initial_snapshot=data.get('initial_snapshot', {}),
            final_snapshot=data.get('final_snapshot', {}),
            global_snapshots=data.get('global_snapshots', []),
        )

    # -----------------------------------------------------------------------
    # CSV writers
    # -----------------------------------------------------------------------

    def _write_trajectories_csv(self, path: Path) -> None:
        """One row per modified trajectory."""
        headers = [
            'trajectory_index', 'trajectory_id', 'driver_id',
            'orig_x', 'orig_y', 'mod_x', 'mod_y',
            'converged', 'n_iterations', 'final_objective',
            'f_spatial', 'f_causal', 'f_fidelity',
            'perturbation_x', 'perturbation_y', 'perturbation_magnitude',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for tr in self.trajectory_results:
                pv = tr.perturbation_vector
                writer.writerow([
                    tr.trajectory_index, tr.trajectory_id, tr.driver_id,
                    tr.original_pickup[0], tr.original_pickup[1],
                    tr.modified_pickup[0], tr.modified_pickup[1],
                    int(tr.converged), tr.total_iterations, f'{tr.final_objective:.6f}',
                    f'{tr.final_f_spatial:.6f}', f'{tr.final_f_causal:.6f}',
                    f'{tr.final_f_fidelity:.6f}',
                    f'{pv[0]:.4f}', f'{pv[1]:.4f}', f'{tr.perturbation_magnitude:.4f}',
                ])

    def _write_iterations_csv(self, path: Path) -> None:
        """One row per iteration per trajectory."""
        headers = [
            'trajectory_index', 'iteration', 'objective',
            'f_spatial', 'f_causal', 'f_fidelity', 'gradient_norm',
            'perturbation_x', 'perturbation_y',
            'grad_spatial_x', 'grad_spatial_y',
            'grad_causal_x', 'grad_causal_y',
            'grad_fidelity_x', 'grad_fidelity_y',
            'spatial_fraction', 'causal_fraction', 'fidelity_fraction',
            'alignment_spatial_causal',
        ]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for tr in self.trajectory_results:
                for it in tr.iterations:
                    row = [
                        tr.trajectory_index, it.iteration,
                        f'{it.objective:.6f}',
                        f'{it.f_spatial:.6f}', f'{it.f_causal:.6f}',
                        f'{it.f_fidelity:.6f}', f'{it.gradient_norm:.6f}',
                        f'{it.perturbation[0]:.4f}', f'{it.perturbation[1]:.4f}',
                    ]
                    gd = it.gradient_decomposition
                    if gd:
                        row.extend([
                            f'{gd.grad_spatial[0]:.6f}', f'{gd.grad_spatial[1]:.6f}',
                            f'{gd.grad_causal[0]:.6f}', f'{gd.grad_causal[1]:.6f}',
                            f'{gd.grad_fidelity[0]:.6f}', f'{gd.grad_fidelity[1]:.6f}',
                            f'{gd.spatial_fraction:.4f}', f'{gd.causal_fraction:.4f}',
                            f'{gd.fidelity_fraction:.4f}',
                            f'{gd.alignment_spatial_causal:.4f}',
                        ])
                    else:
                        row.extend([''] * 10)
                    writer.writerow(row)

    def _write_global_snapshots_csv(self, path: Path) -> None:
        """One row per global metrics snapshot."""
        headers = ['after_n_trajectories', 'gini', 'f_spatial', 'f_causal',
                    'f_fidelity', 'combined']
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for snap in self.global_snapshots:
                writer.writerow([
                    snap.get('after_n_trajectories', ''),
                    f'{snap.get("gini", 0):.6f}',
                    f'{snap.get("f_spatial", 0):.6f}',
                    f'{snap.get("f_causal", 0):.6f}',
                    f'{snap.get("f_fidelity", 0):.6f}',
                    f'{snap.get("combined", 0):.6f}',
                ])

    def _write_attribution_csv(self, path: Path) -> None:
        """One row per attributed trajectory."""
        if not self.attribution_scores:
            return
        headers = list(self.attribution_scores[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in self.attribution_scores:
                writer.writerow(row)

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def generate_report(self) -> str:
        """Auto-generate a markdown analysis report."""
        summary = self._build_summary()
        cfg = self.config
        lines = []

        lines.append(f"# Experiment Report: {cfg.get('experiment_name', 'default')}")
        lines.append("")
        if cfg.get('description'):
            lines.append(f"> {cfg['description']}")
            lines.append("")
        lines.append(f"**Timestamp**: {self.timestamp}")
        lines.append(f"**Duration**: {self.duration_seconds:.1f}s")
        lines.append("")

        # Configuration summary
        lines.append("## Configuration")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        key_params = [
            ('top_k', 'Trajectories to modify'),
            ('alpha', 'ST-iFGSM step size'),
            ('epsilon', 'Max perturbation'),
            ('max_iterations', 'Max iterations'),
            ('alpha_spatial', 'Weight: spatial'),
            ('alpha_causal', 'Weight: causal'),
            ('alpha_fidelity', 'Weight: fidelity'),
            ('causal_formulation', 'Causal formulation'),
            ('discriminator_checkpoint', 'Discriminator'),
            ('record_gradient_decomposition', 'Gradient decomposition'),
        ]
        for key, label in key_params:
            val = cfg.get(key, 'N/A')
            lines.append(f"| {label} | {val} |")
        lines.append("")

        # Before/after metrics
        lines.append("## Global Fairness Metrics")
        lines.append("")
        lines.append("| Metric | Before | After | Delta |")
        lines.append("|--------|--------|-------|-------|")
        for key in ('gini', 'f_spatial', 'f_causal', 'f_fidelity', 'combined'):
            before = summary['initial'].get(key, 0)
            after = summary['final'].get(key, 0)
            delta = summary['improvement'].get(f'delta_{key}', 0)
            sign = '+' if delta >= 0 else ''
            lines.append(f"| {key} | {before:.4f} | {after:.4f} | {sign}{delta:.4f} |")
        lines.append("")

        # Modification summary
        lines.append("## Modification Summary")
        lines.append("")
        lines.append(f"- **Trajectories modified**: {summary['n_trajectories_modified']}")
        lines.append(f"- **Converged**: {summary['n_converged']}")
        lines.append(f"- **Mean iterations**: {summary['mean_iterations']:.1f}")
        lines.append(f"- **Mean perturbation**: {summary['mean_perturbation_magnitude']:.2f} grid cells")
        lines.append("")

        # Top trajectories by impact
        if self.trajectory_results:
            lines.append("## Top 10 Most Impactful Trajectories")
            lines.append("")
            lines.append("| Index | Driver | Orig Cell | Mod Cell | Perturbation | Objective |")
            lines.append("|-------|--------|-----------|----------|-------------|-----------|")
            sorted_trs = sorted(
                self.trajectory_results,
                key=lambda tr: tr.final_objective,
                reverse=True,
            )
            for tr in sorted_trs[:10]:
                orig = f"({tr.original_pickup[0]}, {tr.original_pickup[1]})"
                mod = f"({tr.modified_pickup[0]}, {tr.modified_pickup[1]})"
                lines.append(
                    f"| {tr.trajectory_index} | {tr.driver_id} | {orig} | {mod} "
                    f"| {tr.perturbation_magnitude:.2f} | {tr.final_objective:.4f} |"
                )
            lines.append("")

        # Gradient decomposition summary (if recorded)
        has_decomp = any(
            it.gradient_decomposition is not None
            for tr in self.trajectory_results
            for it in tr.iterations
        )
        if has_decomp:
            lines.append("## Gradient Decomposition Summary")
            lines.append("")
            spatial_fracs, causal_fracs, fidelity_fracs, alignments = [], [], [], []
            for tr in self.trajectory_results:
                for it in tr.iterations:
                    gd = it.gradient_decomposition
                    if gd:
                        spatial_fracs.append(gd.spatial_fraction)
                        causal_fracs.append(gd.causal_fraction)
                        fidelity_fracs.append(gd.fidelity_fraction)
                        alignments.append(gd.alignment_spatial_causal)

            lines.append("| Metric | Mean | Std |")
            lines.append("|--------|------|-----|")
            lines.append(f"| Spatial fraction | {np.mean(spatial_fracs):.3f} | {np.std(spatial_fracs):.3f} |")
            lines.append(f"| Causal fraction | {np.mean(causal_fracs):.3f} | {np.std(causal_fracs):.3f} |")
            lines.append(f"| Fidelity fraction | {np.mean(fidelity_fracs):.3f} | {np.std(fidelity_fracs):.3f} |")
            lines.append(f"| Spatial-causal alignment | {np.mean(alignments):.3f} | {np.std(alignments):.3f} |")
            lines.append("")

            mean_align = np.mean(alignments)
            if mean_align > 0.5:
                lines.append("**Interpretation**: Spatial and causal terms are largely **cooperating** — "
                             "both push modifications in similar directions.")
            elif mean_align < -0.5:
                lines.append("**Interpretation**: Spatial and causal terms are largely **conflicting** — "
                             "they pull modifications in opposing directions.")
            else:
                lines.append("**Interpretation**: Spatial and causal terms have **mixed** alignment — "
                             "sometimes cooperating, sometimes conflicting.")
            lines.append("")

        # Convergence statistics
        lines.append("## Convergence Statistics")
        lines.append("")
        iter_counts = [tr.total_iterations for tr in self.trajectory_results]
        if iter_counts:
            lines.append(f"- **Min iterations**: {min(iter_counts)}")
            lines.append(f"- **Max iterations**: {max(iter_counts)}")
            lines.append(f"- **Median iterations**: {np.median(iter_counts):.0f}")
            pct_converged = summary['n_converged'] / len(self.trajectory_results) * 100
            lines.append(f"- **Convergence rate**: {pct_converged:.0f}%")
        lines.append("")

        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
