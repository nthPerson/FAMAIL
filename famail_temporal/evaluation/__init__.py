"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.augment import augment_trajectories
from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.runner import ExperimentResult, run_experiment

__all__ = [
    "ExperimentResult",
    "augment_trajectories",
    "build_fairness_grid",
    "compute_gradient_sensitivity",
    "run_experiment",
]
