"""Core generation logic for trajectory pair datasets.

Implements loading, segment construction, sampling, alignment, and metadata
construction for the Streamlit UI defined in app.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import io
import json
import random

import numpy as np


DAY_BUCKETS = 288


@dataclass
class Segment:
    agent_id: str
    start_time: float
    end_time: float
    traj_indices: List[int]
    component_lengths: List[int]
    data: np.ndarray

    @property
    def length(self) -> int:
        return int(self.data.shape[0])


@dataclass
class GenerationConfig:
    data_path: Path
    positive_pairs: int
    negative_pairs: int
    days: int = 2
    feature_start: int = 4
    feature_end: int = 4  # default: no extra features beyond indices 0-3
    padding: str = "pad_to_longer"  # pad_to_longer | truncate_to_shorter | fixed_length
    fixed_length: Optional[int] = None
    positive_strategy: str = "random"  # random | sequential
    negative_strategy: str = "random"  # random | round_robin | temporal_hard | spatial_hard | mixed_hard
    agent_distribution: str = "proportional"  # proportional | uniform
    seed: Optional[int] = None
    ensure_agent_coverage: bool = True
    per_agent_counts: bool = False  # if True, positive_pairs/negative_pairs are per-agent counts
    identical_pair_ratio: float = 0.0  # ratio of positive pairs that should be identical (0.0-1.0)
    hard_negative_ratio: float = 0.0  # ratio of negative pairs that should be hard negatives (0.0-1.0)
    temporal_overlap_threshold: float = 0.5  # minimum temporal overlap for hard negatives (0.0-1.0)
    spatial_similarity_threshold: float = 0.3  # max spatial distance for hard negatives (normalized)

    def clamped_feature_bounds(self, state_dim: int) -> Tuple[int, int]:
        start = max(4, min(self.feature_start, state_dim))
        end = max(start, min(self.feature_end, state_dim))
        return start, end


def _compute_global_time(arr: np.ndarray) -> np.ndarray:
    days = arr[:, 3]
    buckets = arr[:, 2]
    return (days - 1) * DAY_BUCKETS + buckets


def load_dataset(path: Path) -> Dict[str, List[np.ndarray]]:
    import pickle

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Top-level dataset must be a dict of expert_id -> trajectories")
    expert_trajs: Dict[str, List[np.ndarray]] = {}
    for k, traj_list in raw.items():
        expert_id = str(k)
        if not isinstance(traj_list, (list, tuple)):
            raise ValueError(f"Expert {expert_id} payload must be list-like")
        converted: List[np.ndarray] = []
        for i, t in enumerate(traj_list):
            arr = np.asarray(t)
            if arr.ndim != 2:
                raise ValueError(f"Trajectory {i} for expert {expert_id} must be 2D")
            converted.append(arr)
        if converted:
            expert_trajs[agent_id_or_default(expert_id)] = converted
    if not expert_trajs:
        raise ValueError("No trajectories after conversion")
    dims = {traj.shape[1] for lst in expert_trajs.values() for traj in lst}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent state dims found: {dims}")
    return expert_trajs


def agent_id_or_default(eid: str) -> str:
    return eid if eid else "unknown"


def build_segments(expert_trajs: Dict[str, List[np.ndarray]], days: int) -> Dict[str, List[Segment]]:
    segments: Dict[str, List[Segment]] = {}
    for agent_id, trajs in expert_trajs.items():
        segs_for_agent: List[Segment] = []
        n = len(trajs)
        if n == 0:
            continue
        for start_idx in range(n):
            seg_arrays: List[np.ndarray] = []
            seg_traj_indices: List[int] = []
            comp_lengths: List[int] = []
            start_time: Optional[float] = None
            last_time: Optional[float] = None
            for cursor in range(start_idx, n):
                arr = trajs[cursor]
                times = _compute_global_time(arr)
                if start_time is None:
                    start_time = float(times[0])
                if last_time is not None and times[0] <= last_time:
                    break
                seg_arrays.append(arr)
                seg_traj_indices.append(cursor)
                comp_lengths.append(int(arr.shape[0]))
                last_time = float(times[-1])
                span_days = (last_time - start_time) / DAY_BUCKETS
                if span_days >= days:
                    data = np.concatenate(seg_arrays, axis=0)
                    segs_for_agent.append(
                        Segment(
                            agent_id=agent_id,
                            start_time=start_time,
                            end_time=last_time,
                            traj_indices=list(seg_traj_indices),
                            component_lengths=list(comp_lengths),
                            data=data,
                        )
                    )
                    break
                if cursor == n - 1:
                    data = np.concatenate(seg_arrays, axis=0)
                    segs_for_agent.append(
                        Segment(
                            agent_id=agent_id,
                            start_time=start_time,
                            end_time=last_time,
                            traj_indices=list(seg_traj_indices),
                            component_lengths=list(comp_lengths),
                            data=data,
                        )
                    )
        if segs_for_agent:
            segments[agent_id] = segs_for_agent
    return segments


def _non_overlapping(a: Segment, b: Segment) -> bool:
    return a.end_time < b.start_time or b.end_time < a.start_time


def _pick_positive_pair(segs: List[Segment], rng: random.Random, strategy: str) -> Optional[Tuple[Segment, Segment]]:
    if len(segs) < 2:
        return None
    attempts = 0
    segs_sorted = sorted(segs, key=lambda s: s.start_time)
    while attempts < 20:
        if strategy == "sequential":
            first = rng.choice(segs_sorted)
            candidates = [s for s in segs_sorted if _non_overlapping(first, s)]
            if not candidates:
                attempts += 1
                continue
            second = max(candidates, key=lambda s: abs(s.start_time - first.start_time))
        else:
            first, second = rng.sample(segs, 2)
            if not _non_overlapping(first, second):
                attempts += 1
                continue
        if _non_overlapping(first, second):
            return first, second
        attempts += 1
    return None


def _agent_weights(segments: Dict[str, List[Segment]], mode: str) -> List[float]:
    if mode == "uniform":
        return [1.0 for _ in segments]
    weights = []
    for segs in segments.values():
        weights.append(float(sum(s.length for s in segs)))
    return weights


def sample_positive_pairs(
    segments: Dict[str, List[Segment]],
    n_pairs: int,
    rng: random.Random,
    strategy: str,
    distribution: str,
    ensure_coverage: bool,
    per_agent_counts: bool = False,
) -> List[Tuple[Segment, Segment]]:
    """Sample positive (same-agent) pairs.
    
    If per_agent_counts=True, generates n_pairs for EACH agent (total = n_pairs * num_agents).
    Otherwise generates n_pairs total across all agents.
    """
    pairs: List[Tuple[Segment, Segment]] = []
    agents = [a for a, segs in segments.items() if len(segs) >= 2]
    if not agents:
        return pairs
    
    if per_agent_counts:
        # Generate n_pairs for each agent
        for agent_id in agents:
            agent_pairs = 0
            attempts = 0
            max_attempts = n_pairs * 20 + 100
            while agent_pairs < n_pairs and attempts < max_attempts:
                pair = _pick_positive_pair(segments[agent_id], rng, strategy)
                if pair:
                    pairs.append(pair)
                    agent_pairs += 1
                attempts += 1
        return pairs
    
    # Original behavior: total pairs across all agents
    weights = _agent_weights({a: segments[a] for a in agents}, distribution)
    if ensure_coverage:
        for agent_id in agents:
            if len(pairs) >= n_pairs:
                break
            pair = _pick_positive_pair(segments[agent_id], rng, strategy)
            if pair:
                pairs.append(pair)
    attempts = 0
    max_attempts = n_pairs * 10 + 200
    while len(pairs) < n_pairs and attempts < max_attempts:
        agent_id = rng.choices(agents, weights=weights, k=1)[0]
        pair = _pick_positive_pair(segments[agent_id], rng, strategy)
        if pair:
            pairs.append(pair)
        attempts += 1
    return pairs[:n_pairs]


def sample_identical_pairs(
    segments: Dict[str, List[Segment]],
    n_pairs: int,
    rng: random.Random,
    distribution: str,
) -> List[Tuple[Segment, Segment]]:
    """Sample identical (same segment vs itself) pairs.
    
    These pairs are crucial for training the discriminator to correctly
    output high similarity scores when comparing a trajectory to itself.
    
    Args:
        segments: Dictionary mapping agent_id to list of Segment objects
        n_pairs: Number of identical pairs to generate
        rng: Random number generator
        distribution: "proportional" or "uniform" agent selection
    
    Returns:
        List of (segment, segment) tuples where both elements are the same segment
    """
    pairs: List[Tuple[Segment, Segment]] = []
    agents = list(segments.keys())
    if not agents:
        return pairs
    
    weights = _agent_weights(segments, distribution)
    
    # Ensure at least one identical pair per agent for coverage
    for agent_id in agents:
        if len(pairs) >= n_pairs:
            break
        agent_segs = segments.get(agent_id, [])
        if agent_segs:
            seg = rng.choice(agent_segs)
            pairs.append((seg, seg))  # Same segment twice = identical pair
    
    # Fill remaining pairs
    attempts = 0
    max_attempts = n_pairs * 10 + 200
    while len(pairs) < n_pairs and attempts < max_attempts:
        agent_id = rng.choices(agents, weights=weights, k=1)[0]
        agent_segs = segments.get(agent_id, [])
        if agent_segs:
            seg = rng.choice(agent_segs)
            pairs.append((seg, seg))
        attempts += 1
    
    return pairs[:n_pairs]


def _compute_temporal_overlap(seg_a: Segment, seg_b: Segment) -> float:
    """Compute temporal overlap ratio between two segments.
    
    Returns a value between 0 and 1, where 1 means complete overlap
    and 0 means no overlap in time ranges.
    """
    overlap_start = max(seg_a.start_time, seg_b.start_time)
    overlap_end = min(seg_a.end_time, seg_b.end_time)
    
    if overlap_start >= overlap_end:
        return 0.0
    
    overlap_duration = overlap_end - overlap_start
    total_duration = max(seg_a.end_time, seg_b.end_time) - min(seg_a.start_time, seg_b.start_time)
    
    if total_duration <= 0:
        return 0.0
    
    return overlap_duration / total_duration


def _compute_spatial_similarity(seg_a: Segment, seg_b: Segment) -> float:
    """Compute spatial similarity between two segments based on grid overlap.
    
    Returns a value between 0 and 1, where 1 means high spatial similarity
    (visiting similar grid cells) and 0 means no spatial overlap.
    """
    # Extract x, y coordinates (indices 0 and 1)
    cells_a = set(zip(seg_a.data[:, 0].astype(int).tolist(), 
                      seg_a.data[:, 1].astype(int).tolist()))
    cells_b = set(zip(seg_b.data[:, 0].astype(int).tolist(), 
                      seg_b.data[:, 1].astype(int).tolist()))
    
    if not cells_a or not cells_b:
        return 0.0
    
    intersection = len(cells_a & cells_b)
    union = len(cells_a | cells_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union  # Jaccard similarity


def _compute_segment_centroid(seg: Segment) -> Tuple[float, float]:
    """Compute the spatial centroid of a segment."""
    x_mean = seg.data[:, 0].mean()
    y_mean = seg.data[:, 1].mean()
    return float(x_mean), float(y_mean)


def _pick_hard_negative_temporal(
    segments: Dict[str, List[Segment]],
    rng: random.Random,
    distribution: str,
    overlap_threshold: float = 0.5,
    anchor_agent: Optional[str] = None,
) -> Optional[Tuple[Segment, Segment]]:
    """Pick a negative pair where segments overlap temporally.
    
    This creates harder negative examples because the trajectories
    occurred at similar times, forcing the model to learn driver-specific
    patterns rather than relying on temporal differences.
    """
    agents = list(segments.keys())
    if len(agents) < 2:
        return None
    
    weights = _agent_weights(segments, distribution)
    if anchor_agent is None:
        anchor_agent = rng.choices(agents, weights=weights, k=1)[0]
    
    anchor_segs = segments.get(anchor_agent, [])
    if not anchor_segs:
        return None
    
    seg_a = rng.choice(anchor_segs)
    
    # Find segments from other agents that overlap temporally
    other_agents = [a for a in agents if a != anchor_agent and segments.get(a)]
    if not other_agents:
        return None
    
    # Collect all candidates with temporal overlap
    candidates = []
    for other_agent in other_agents:
        for seg_b in segments[other_agent]:
            overlap = _compute_temporal_overlap(seg_a, seg_b)
            if overlap >= overlap_threshold:
                candidates.append((seg_b, overlap))
    
    if not candidates:
        # Fallback: find best overlap even if below threshold
        for other_agent in other_agents:
            for seg_b in segments[other_agent]:
                overlap = _compute_temporal_overlap(seg_a, seg_b)
                if overlap > 0:
                    candidates.append((seg_b, overlap))
    
    if not candidates:
        return None
    
    # Weight by overlap (higher overlap = more likely to be selected)
    total_overlap = sum(o for _, o in candidates)
    if total_overlap > 0:
        weights_cand = [o / total_overlap for _, o in candidates]
        seg_b = rng.choices([c[0] for c in candidates], weights=weights_cand, k=1)[0]
    else:
        seg_b = rng.choice([c[0] for c in candidates])
    
    return seg_a, seg_b


def _pick_hard_negative_spatial(
    segments: Dict[str, List[Segment]],
    rng: random.Random,
    distribution: str,
    similarity_threshold: float = 0.3,
    anchor_agent: Optional[str] = None,
) -> Optional[Tuple[Segment, Segment]]:
    """Pick a negative pair where segments have high spatial similarity.
    
    This creates harder negative examples because the trajectories
    visit similar locations, forcing the model to learn driver-specific
    movement patterns rather than relying on different regions.
    """
    agents = list(segments.keys())
    if len(agents) < 2:
        return None
    
    weights = _agent_weights(segments, distribution)
    if anchor_agent is None:
        anchor_agent = rng.choices(agents, weights=weights, k=1)[0]
    
    anchor_segs = segments.get(anchor_agent, [])
    if not anchor_segs:
        return None
    
    seg_a = rng.choice(anchor_segs)
    centroid_a = _compute_segment_centroid(seg_a)
    
    # Find segments from other agents with similar spatial patterns
    other_agents = [a for a in agents if a != anchor_agent and segments.get(a)]
    if not other_agents:
        return None
    
    # Collect all candidates with spatial similarity
    candidates = []
    for other_agent in other_agents:
        for seg_b in segments[other_agent]:
            similarity = _compute_spatial_similarity(seg_a, seg_b)
            if similarity >= similarity_threshold:
                candidates.append((seg_b, similarity))
    
    if not candidates:
        # Fallback: find segments with similar centroids
        for other_agent in other_agents:
            for seg_b in segments[other_agent]:
                centroid_b = _compute_segment_centroid(seg_b)
                # Compute centroid distance (normalized by grid size ~50x90)
                dist = ((centroid_a[0] - centroid_b[0]) / 50) ** 2 + ((centroid_a[1] - centroid_b[1]) / 90) ** 2
                dist = dist ** 0.5
                if dist < 0.3:  # Within 30% of max grid distance
                    similarity = 1 - dist
                    candidates.append((seg_b, similarity))
    
    if not candidates:
        return None
    
    # Weight by similarity (higher similarity = more likely to be selected)
    total_sim = sum(s for _, s in candidates)
    if total_sim > 0:
        weights_cand = [s / total_sim for _, s in candidates]
        seg_b = rng.choices([c[0] for c in candidates], weights=weights_cand, k=1)[0]
    else:
        seg_b = rng.choice([c[0] for c in candidates])
    
    return seg_a, seg_b


def _pick_hard_negative_mixed(
    segments: Dict[str, List[Segment]],
    rng: random.Random,
    distribution: str,
    overlap_threshold: float = 0.3,
    similarity_threshold: float = 0.2,
    anchor_agent: Optional[str] = None,
) -> Optional[Tuple[Segment, Segment]]:
    """Pick a negative pair with both temporal overlap AND spatial similarity.
    
    This creates the hardest negative examples - different drivers
    operating in the same area at the same time.
    """
    agents = list(segments.keys())
    if len(agents) < 2:
        return None
    
    weights = _agent_weights(segments, distribution)
    if anchor_agent is None:
        anchor_agent = rng.choices(agents, weights=weights, k=1)[0]
    
    anchor_segs = segments.get(anchor_agent, [])
    if not anchor_segs:
        return None
    
    seg_a = rng.choice(anchor_segs)
    
    # Find segments from other agents with both temporal and spatial similarity
    other_agents = [a for a in agents if a != anchor_agent and segments.get(a)]
    if not other_agents:
        return None
    
    candidates = []
    for other_agent in other_agents:
        for seg_b in segments[other_agent]:
            temporal_overlap = _compute_temporal_overlap(seg_a, seg_b)
            spatial_sim = _compute_spatial_similarity(seg_a, seg_b)
            
            # Combined score: both must meet thresholds
            if temporal_overlap >= overlap_threshold and spatial_sim >= similarity_threshold:
                combined_score = (temporal_overlap + spatial_sim) / 2
                candidates.append((seg_b, combined_score))
    
    if not candidates:
        # Fallback: use just temporal or spatial
        if rng.random() < 0.5:
            return _pick_hard_negative_temporal(segments, rng, distribution, 
                                                overlap_threshold, anchor_agent)
        else:
            return _pick_hard_negative_spatial(segments, rng, distribution,
                                               similarity_threshold, anchor_agent)
    
    # Weight by combined score
    total_score = sum(s for _, s in candidates)
    if total_score > 0:
        weights_cand = [s / total_score for _, s in candidates]
        seg_b = rng.choices([c[0] for c in candidates], weights=weights_cand, k=1)[0]
    else:
        seg_b = rng.choice([c[0] for c in candidates])
    
    return seg_a, seg_b


def _pick_negative_pair(
    segments: Dict[str, List[Segment]],
    rng: random.Random,
    distribution: str,
    negative_strategy: str,
    anchor_agent: Optional[str] = None,
) -> Optional[Tuple[Segment, Segment]]:
    agents = list(segments.keys())
    if len(agents) < 2:
        return None
    weights = _agent_weights(segments, distribution)
    if anchor_agent is None:
        anchor_agent = rng.choices(agents, weights=weights, k=1)[0]
    anchor_segs = segments.get(anchor_agent, [])
    if not anchor_segs:
        return None
    other_agents = [a for a in agents if a != anchor_agent and segments.get(a)]
    if not other_agents:
        return None
    if negative_strategy == "round_robin":
        other_agents = sorted(other_agents)
    other_agent = rng.choice(other_agents)
    b_segs = segments[other_agent]
    seg_a = rng.choice(anchor_segs)
    seg_b = rng.choice(b_segs)
    return seg_a, seg_b


def sample_negative_pairs(
    segments: Dict[str, List[Segment]],
    n_pairs: int,
    rng: random.Random,
    strategy: str,
    distribution: str,
    ensure_coverage: bool,
    per_agent_counts: bool = False,
    hard_negative_ratio: float = 0.0,
    temporal_overlap_threshold: float = 0.5,
    spatial_similarity_threshold: float = 0.3,
) -> Tuple[List[Tuple[Segment, Segment]], Dict[str, int]]:
    """Sample negative (different-agent) pairs.
    
    If per_agent_counts=True, generates n_pairs for EACH ordered agent pair (agent_i, agent_j)
    where i != j. This ensures comprehensive cross-agent coverage for discriminator training.
    Total pairs = n_pairs * num_agents * (num_agents - 1).
    
    Otherwise generates n_pairs total across all agent combinations.
    
    Hard negative strategies:
    - temporal_hard: Pairs trajectories from different drivers at similar times
    - spatial_hard: Pairs trajectories from different drivers in similar locations  
    - mixed_hard: Combines temporal and spatial similarity for hardest negatives
    
    Args:
        segments: Dictionary mapping agent_id to list of Segment objects
        n_pairs: Number of pairs to generate (total or per agent depending on mode)
        rng: Random number generator
        strategy: "random", "round_robin", "temporal_hard", "spatial_hard", or "mixed_hard"
        distribution: "proportional" or "uniform" agent selection
        ensure_coverage: Whether to ensure all agents appear at least once
        per_agent_counts: If True, generate n_pairs per agent combination
        hard_negative_ratio: Fraction of pairs that should be hard negatives (0.0-1.0)
        temporal_overlap_threshold: Min temporal overlap for hard negatives
        spatial_similarity_threshold: Min spatial similarity for hard negatives
    
    Returns:
        Tuple of (list of pairs, stats dict with counts by type)
    """
    pairs: List[Tuple[Segment, Segment]] = []
    stats: Dict[str, int] = {
        "random": 0,
        "round_robin": 0,
        "temporal_hard": 0,
        "spatial_hard": 0,
        "mixed_hard": 0,
    }
    agents = list(segments.keys())
    if len(agents) < 2:
        return pairs, stats
    
    def _pick_pair_by_strategy(strat: str, anchor_agent: Optional[str] = None) -> Optional[Tuple[Segment, Segment]]:
        """Pick a negative pair using the specified strategy."""
        if strat == "temporal_hard":
            return _pick_hard_negative_temporal(segments, rng, distribution, 
                                                temporal_overlap_threshold, anchor_agent)
        elif strat == "spatial_hard":
            return _pick_hard_negative_spatial(segments, rng, distribution,
                                               spatial_similarity_threshold, anchor_agent)
        elif strat == "mixed_hard":
            return _pick_hard_negative_mixed(segments, rng, distribution,
                                            temporal_overlap_threshold * 0.6,  # Lower thresholds for combined
                                            spatial_similarity_threshold * 0.6,
                                            anchor_agent)
        else:  # random or round_robin
            return _pick_negative_pair(segments, rng, distribution, strat, anchor_agent)
    
    def _decide_strategy_for_pair(base_strategy: str, hard_ratio: float) -> str:
        """Decide which strategy to use for a single pair."""
        if hard_ratio <= 0 or base_strategy in ["temporal_hard", "spatial_hard", "mixed_hard"]:
            # If already a hard strategy or no hard negatives requested, use base
            return base_strategy
        
        if rng.random() < hard_ratio:
            # Use a hard negative strategy
            hard_strategies = ["temporal_hard", "spatial_hard", "mixed_hard"]
            return rng.choice(hard_strategies)
        else:
            return base_strategy
    
    if per_agent_counts:
        # Generate n_pairs for each (anchor, other) agent combination
        for anchor_agent in agents:
            anchor_segs = segments.get(anchor_agent, [])
            if not anchor_segs:
                continue
            for other_agent in agents:
                if other_agent == anchor_agent:
                    continue
                other_segs = segments.get(other_agent, [])
                if not other_segs:
                    continue
                combo_pairs = 0
                attempts = 0
                max_attempts = n_pairs * 20 + 100
                while combo_pairs < n_pairs and attempts < max_attempts:
                    strat = _decide_strategy_for_pair(strategy, hard_negative_ratio)
                    pair = _pick_pair_by_strategy(strat, anchor_agent)
                    if pair:
                        pairs.append(pair)
                        stats[strat if strat in stats else "random"] += 1
                        combo_pairs += 1
                    attempts += 1
        return pairs, stats
    
    # Original behavior: total pairs across all combinations
    if ensure_coverage:
        for agent_id in agents:
            if len(pairs) >= n_pairs:
                break
            strat = _decide_strategy_for_pair(strategy, hard_negative_ratio)
            pair = _pick_pair_by_strategy(strat, anchor_agent=agent_id)
            if pair:
                pairs.append(pair)
                stats[strat if strat in stats else "random"] += 1
    
    attempts = 0
    max_attempts = n_pairs * 10 + 200
    while len(pairs) < n_pairs and attempts < max_attempts:
        strat = _decide_strategy_for_pair(strategy, hard_negative_ratio)
        pair = _pick_pair_by_strategy(strat)
        if pair:
            pairs.append(pair)
            stats[strat if strat in stats else "random"] += 1
        attempts += 1
    
    return pairs[:n_pairs], stats


def _slice_features(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    base = arr[:, :4]
    sliced = arr[:, start:end]
    return np.concatenate([base, sliced], axis=1)


def align_pair(
    seq1: np.ndarray,
    seq2: np.ndarray,
    mode: str,
    fixed_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    len1, len2 = seq1.shape[0], seq2.shape[0]
    if mode == "truncate_to_shorter":
        target = min(len1, len2)
        seq1 = seq1[:target]
        seq2 = seq2[:target]
        mask1 = np.ones(target, dtype=np.int32)
        mask2 = np.ones(target, dtype=np.int32)
        return seq1, seq2, mask1, mask2, target
    if mode == "fixed_length" and fixed_length:
        target = fixed_length
    else:
        target = max(len1, len2)
    pad1 = target - len1
    pad2 = target - len2
    if pad1 > 0:
        seq1 = np.pad(seq1, ((0, pad1), (0, 0)), mode="constant", constant_values=0.0)
    else:
        seq1 = seq1[:target]
    if pad2 > 0:
        seq2 = np.pad(seq2, ((0, pad2), (0, 0)), mode="constant", constant_values=0.0)
    else:
        seq2 = seq2[:target]
    mask1 = np.zeros(target, dtype=np.int32)
    mask1[: min(len1, target)] = 1
    mask2 = np.zeros(target, dtype=np.int32)
    mask2[: min(len2, target)] = 1
    return seq1, seq2, mask1, mask2, target


def assemble_dataset(
    config: GenerationConfig,
    preview_only: bool = False,
    preview_cap: int = 12,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], List[Dict[str, Any]]]:
    rng = random.Random(config.seed)
    expert_trajs = load_dataset(config.data_path)
    state_dim = next(iter(expert_trajs.values()))[0].shape[1]
    feat_start, feat_end = config.clamped_feature_bounds(state_dim)
    segments = build_segments(expert_trajs, config.days)
    
    # Calculate how many identical pairs to include based on ratio
    total_positive = config.positive_pairs if not preview_only else min(config.positive_pairs, preview_cap)
    n_identical = int(total_positive * config.identical_pair_ratio)
    n_regular_positive = total_positive - n_identical
    
    # Sample identical pairs (same trajectory vs itself)
    identical_pairs: List[Tuple[Segment, Segment]] = []
    if n_identical > 0:
        identical_pairs = sample_identical_pairs(
            segments,
            n_pairs=n_identical,
            rng=rng,
            distribution=config.agent_distribution,
        )
    
    # Sample regular positive pairs (same agent, different trajectories)
    pos_pairs = sample_positive_pairs(
        segments,
        n_pairs=n_regular_positive,
        rng=rng,
        strategy=config.positive_strategy,
        distribution=config.agent_distribution,
        ensure_coverage=config.ensure_agent_coverage,
        per_agent_counts=config.per_agent_counts and not preview_only,
    )
    neg_pairs, neg_stats = sample_negative_pairs(
        segments,
        n_pairs=config.negative_pairs if not preview_only else min(config.negative_pairs, preview_cap),
        rng=rng,
        strategy=config.negative_strategy,
        distribution=config.agent_distribution,
        ensure_coverage=config.ensure_agent_coverage,
        per_agent_counts=config.per_agent_counts and not preview_only,
        hard_negative_ratio=config.hard_negative_ratio,
        temporal_overlap_threshold=config.temporal_overlap_threshold,
        spatial_similarity_threshold=config.spatial_similarity_threshold,
    )
    # Labels: 1 = same agent (positive), 0 = different agents (negative)
    # This aligns with optimization objectives that maximize same-agent probability
    # Identical pairs get label 1 (same agent) with special flag for tracking
    pairs = (
        [(p[0], p[1], 1, "identical") for p in identical_pairs] +
        [(p[0], p[1], 1, "same_agent") for p in pos_pairs] + 
        [(p[0], p[1], 0, "different_agent") for p in neg_pairs]
    )
    rng.shuffle(pairs)
    sequences1: List[np.ndarray] = []
    sequences2: List[np.ndarray] = []
    masks1: List[np.ndarray] = []
    masks2: List[np.ndarray] = []
    labels: List[int] = []
    lengths_raw: List[Tuple[int, int]] = []
    agent_usage: Dict[str, Dict[str, int]] = {}
    pair_info: List[Dict[str, Any]] = []
    identical_count = 0
    for seg_a, seg_b, label, pair_type in pairs:
        seq1 = _slice_features(seg_a.data, feat_start, feat_end).astype(np.float32)
        seq2 = _slice_features(seg_b.data, feat_start, feat_end).astype(np.float32)
        aligned1, aligned2, mask1, mask2, target_len = align_pair(seq1, seq2, config.padding, config.fixed_length)
        sequences1.append(aligned1)
        sequences2.append(aligned2)
        masks1.append(mask1)
        masks2.append(mask2)
        labels.append(label)
        lengths_raw.append((seq1.shape[0], seq2.shape[0]))
        agent_usage.setdefault(seg_a.agent_id, {"pos": 0, "neg": 0, "identical": 0})
        agent_usage.setdefault(seg_b.agent_id, {"pos": 0, "neg": 0, "identical": 0})
        if pair_type == "identical":
            agent_usage[seg_a.agent_id]["identical"] += 1
            identical_count += 1
        elif label == 1:  # Same agent (positive)
            agent_usage[seg_a.agent_id]["pos"] += 1
            agent_usage[seg_b.agent_id]["pos"] += 1
        else:  # Different agents (negative)
            agent_usage[seg_a.agent_id]["neg"] += 1
            agent_usage[seg_b.agent_id]["neg"] += 1
        pair_info.append(
            {
                "agent_a": seg_a.agent_id,
                "agent_b": seg_b.agent_id,
                "label": int(label),
                "pair_type": pair_type,  # "identical", "same_agent", or "different_agent"
                "len_raw_a": int(seq1.shape[0]),
                "len_raw_b": int(seq2.shape[0]),
                "align_len": int(target_len),
                "traj_indices_a": list(seg_a.traj_indices),
                "traj_indices_b": list(seg_b.traj_indices),
                "component_lengths_a": list(seg_a.component_lengths),
                "component_lengths_b": list(seg_b.component_lengths),
                "start_time_a": seg_a.start_time,
                "start_time_b": seg_b.start_time,
                "end_time_a": seg_a.end_time,
                "end_time_b": seg_b.end_time,
            }
        )
    if not sequences1:
        raise RuntimeError("No pairs generated. Check data availability and settings.")
    max_len = max(seq.shape[0] for seq in sequences1)
    # Ensure uniform length across dataset for saving convenience
    def pad_to_length(arr_list: List[np.ndarray]) -> np.ndarray:
        out = []
        for arr in arr_list:
            if arr.shape[0] == max_len:
                out.append(arr)
                continue
            pad = max_len - arr.shape[0]
            out.append(np.pad(arr, ((0, pad), (0, 0)), mode="constant", constant_values=0.0))
        return np.stack(out, axis=0)

    def pad_masks(mask_list: List[np.ndarray]) -> np.ndarray:
        out = []
        for mask in mask_list:
            if mask.shape[0] == max_len:
                out.append(mask)
                continue
            pad = max_len - mask.shape[0]
            out.append(np.pad(mask, (0, pad), mode="constant", constant_values=0))
        return np.stack(out, axis=0)

    x1 = pad_to_length(sequences1)
    x2 = pad_to_length(sequences2)
    mask1_arr = pad_masks(masks1)
    mask2_arr = pad_masks(masks2)
    label_arr = np.array(labels, dtype=np.int64)

    dataset = {
        "x1": x1,
        "x2": x2,
        "mask1": mask1_arr,
        "mask2": mask2_arr,
        "label": label_arr,
    }
    metadata = _build_metadata(
        config=config,
        feat_start=feat_start,
        feat_end=feat_end,
        max_len=max_len,
        lengths_raw=lengths_raw,
        agent_usage=agent_usage,
        total_pairs=len(pairs),
        pos_pairs=len(pos_pairs),
        neg_pairs=len(neg_pairs),
        identical_pairs=len(identical_pairs),
        negative_stats=neg_stats,
    )
    return dataset, metadata, pair_info


def _length_stats(lengths: List[int]) -> Dict[str, float]:
    arr = np.array(lengths, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _build_metadata(
    config: GenerationConfig,
    feat_start: int,
    feat_end: int,
    max_len: int,
    lengths_raw: List[Tuple[int, int]],
    agent_usage: Dict[str, Dict[str, int]],
    total_pairs: int,
    pos_pairs: int,
    neg_pairs: int,
    identical_pairs: int = 0,
    negative_stats: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    lens1 = [a for a, _ in lengths_raw]
    lens2 = [b for _, b in lengths_raw]
    combined = lens1 + lens2
    cfg_dict = asdict(config)
    cfg_dict["feature_start"] = feat_start
    cfg_dict["feature_end"] = feat_end
    cfg_dict["data_path"] = str(config.data_path)
    hash_payload = json.dumps(cfg_dict, sort_keys=True).encode()
    dataset_hash = hashlib.sha256(hash_payload).hexdigest()[:12]
    
    counts = {
        "total_pairs": total_pairs,
        "positive_pairs": pos_pairs,
        "negative_pairs": neg_pairs,
        "identical_pairs": identical_pairs,
    }
    
    # Add hard negative breakdown if available
    if negative_stats:
        counts["negative_breakdown"] = {
            "random": negative_stats.get("random", 0) + negative_stats.get("round_robin", 0),
            "temporal_hard": negative_stats.get("temporal_hard", 0),
            "spatial_hard": negative_stats.get("spatial_hard", 0),
            "mixed_hard": negative_stats.get("mixed_hard", 0),
        }
        hard_total = (negative_stats.get("temporal_hard", 0) + 
                      negative_stats.get("spatial_hard", 0) + 
                      negative_stats.get("mixed_hard", 0))
        counts["hard_negative_count"] = hard_total
        counts["hard_negative_actual_ratio"] = hard_total / neg_pairs if neg_pairs > 0 else 0.0
    
    return {
        "config": cfg_dict,
        "counts": counts,
        "length_stats": {
            "x1": _length_stats(lens1),
            "x2": _length_stats(lens2),
            "combined": _length_stats(combined),
            "padded_length": max_len,
        },
        "agent_usage": agent_usage,
        "dataset_hash": dataset_hash,
    }


def dataset_to_npz_bytes(dataset: Dict[str, np.ndarray]) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **dataset)
    return buf.getvalue()


def dataset_to_pt_bytes(dataset: Dict[str, np.ndarray]) -> bytes:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for .pt export") from exc
    buf = io.BytesIO()
    torch.save({k: torch.as_tensor(v) for k, v in dataset.items()}, buf)
    return buf.getvalue()


def sample_json(dataset: Dict[str, np.ndarray], metadata: Dict[str, any], k: int = 5) -> str:
    total = dataset["label"].shape[0]
    k = min(k, total)
    idx = list(range(total))
    random.shuffle(idx)
    idx = idx[:k]
    sample = []
    for i in idx:
        sample.append(
            {
                "label": int(dataset["label"][i]),
                "len_x1": int(dataset["mask1"][i].sum()),
                "len_x2": int(dataset["mask2"][i].sum()),
            }
        )
    return json.dumps({"sample_pairs": sample, "metadata": metadata}, indent=2)
