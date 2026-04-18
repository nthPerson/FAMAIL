"""Algorithm orchestration — objective, modifier, attribution, soft assignment."""

from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution,
    rank_trajectories,
    select_top_k,
)
from famail_temporal.algorithm.modifier import (
    TrajectoryModifier,
    ModificationResult,
    ModificationHistory,
)
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.algorithm.soft_cell_assignment import (
    SoftCellAssignment,
    inject_soft_counts_into_3d,
)

__all__ = [
    "FAMAILObjective",
    "TrajectoryModifier", "ModificationResult", "ModificationHistory",
    "SoftCellAssignment", "inject_soft_counts_into_3d",
    "compute_per_unit_attribution", "rank_trajectories", "select_top_k",
]
