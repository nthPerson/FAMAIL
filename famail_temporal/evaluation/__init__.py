"""Evaluation framework: runs the FAMAIL pipeline and produces reproducible artifacts."""

from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.evaluation.augment import augment_trajectories

__all__ = ["build_fairness_grid", "augment_trajectories"]
