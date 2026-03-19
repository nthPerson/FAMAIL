"""
FAMAIL Experiment & Analysis Framework.

Provides systematic experiment execution, per-term gradient decomposition,
structured output (JSON/CSV), and analysis tools for the FAMAIL trajectory
modification algorithm.

Usage:
    from experiment_framework import ExperimentConfig, ExperimentRunner, ExperimentResult

    config = ExperimentConfig(top_k=20, alpha_spatial=0.5, alpha_causal=0.3, alpha_fidelity=0.2)
    runner = ExperimentRunner(config)
    result = runner.run()
    run_dir = result.save()
"""

from experiment_framework.experiment_config import ExperimentConfig, SweepConfig
from experiment_framework.experiment_result import (
    ExperimentResult,
    TrajectoryResult,
    IterationRecord,
    GradientDecomposition,
)
from experiment_framework.experiment_runner import ExperimentRunner

__all__ = [
    'ExperimentConfig',
    'SweepConfig',
    'ExperimentRunner',
    'ExperimentResult',
    'TrajectoryResult',
    'IterationRecord',
    'GradientDecomposition',
]
