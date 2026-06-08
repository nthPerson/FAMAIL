"""Training-corpus variant builders for the model-level baselines.

FAMAIL trains the shared generator on the EDITED corpus (pickups relocated by a
persisted ST-iFGSM editing run, ε=2); B2 trains on a FILTERED corpus (top-K
most-unfair trajectories removed). Both reuse the same DataBundle for fairness
scoring (scoring reads pickup_3d/mask_3d/hat_matrices, never the trajectory
list), so only the *training* trajectory list changes.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Union

from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.trajectory import Trajectory
from famail_temporal.baselines.datasets import rank_unfair_trajectory_indices


def apply_edits(
    trajectories: List[Trajectory], modified_by_tid: Dict[int, Trajectory],
) -> List[Trajectory]:
    """Swap edited trajectories in by trajectory_id, preserving length/order.

    Mirrors the editing runner's trajs_after reconstruction: an entry is
    replaced iff its trajectory_id appears in modified_by_tid.
    """
    return [modified_by_tid.get(t.trajectory_id, t) for t in trajectories]


def load_edited_trajectories(
    bundle: DataBundle, edit_dir: Union[str, Path],
) -> List[Trajectory]:
    """Build the FAMAIL edited corpus from a persisted editing run.

    Reads <edit_dir>/histories.pkl (each element exposes `.modified` carrying
    the relocated pickup and its trajectory_id) and swaps those into
    bundle.trajectories. Returns a list the same length/order as
    bundle.trajectories.
    """
    # histories.pkl is produced locally by FAMAIL's own editing runner
    # (see famail_temporal/algorithm/persistence.py); it is a trusted,
    # in-repo artifact, not external input — pickle.load is safe here.
    with open(Path(edit_dir) / "histories.pkl", "rb") as f:
        histories = pickle.load(f)
    modified_by_tid = {h.modified.trajectory_id: h.modified for h in histories}
    return apply_edits(bundle.trajectories, modified_by_tid)


def filtered_trajectories(bundle: DataBundle, n_remove: int) -> List[Trajectory]:
    """bundle.trajectories with the top-`n_remove` most-unfair removed."""
    if n_remove <= 0:
        return list(bundle.trajectories)
    removed = set(rank_unfair_trajectory_indices(bundle)[:n_remove])
    return [t for i, t in enumerate(bundle.trajectories) if i not in removed]
