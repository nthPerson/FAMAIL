"""Driver-index map and per-driver grouping for driver-conditioned generation.

`Trajectory.driver_id` is an int in [0, 49]. The generator's driver embedding
is sized by the number of DISTINCT drivers in the training corpus, so we map
each driver_id to a contiguous embedding index (sorted for determinism). The
map is persisted with a run so conditioned generation is reproducible.
"""
from __future__ import annotations
from typing import Dict, List


def build_driver_index(trajectories) -> Dict[int, int]:
    """{driver_id -> contiguous embedding idx}, ordered by sorted driver_id."""
    ids = sorted({int(t.driver_id) for t in trajectories})
    return {did: i for i, did in enumerate(ids)}


def invert_driver_index(driver_to_idx: Dict[int, int]) -> Dict[int, int]:
    """{embedding idx -> driver_id}."""
    return {idx: did for did, idx in driver_to_idx.items()}


def group_by_driver(trajectories) -> Dict[int, List]:
    """{driver_id -> [Trajectory, ...]} (insertion order preserved)."""
    groups: Dict[int, List] = {}
    for t in trajectories:
        groups.setdefault(int(t.driver_id), []).append(t)
    return groups


def driver_idxs_for(trajectories, driver_to_idx: Dict[int, int]) -> List[int]:
    """Index-aligned embedding indices for `trajectories`.

    Raises KeyError (clear message) if a trajectory's driver_id is absent from
    the map — that signals the map was built from a different corpus.
    """
    out: List[int] = []
    for t in trajectories:
        did = int(t.driver_id)
        if did not in driver_to_idx:
            raise KeyError(
                f"driver_id {did} not in driver_to_idx (built from a different "
                f"corpus?); known ids: {sorted(driver_to_idx)[:5]}..."
            )
        out.append(driver_to_idx[did])
    return out
