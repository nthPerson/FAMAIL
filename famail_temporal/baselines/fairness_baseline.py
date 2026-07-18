"""Fairness-intervention baseline arms (spec: docs/superpowers/specs/
2026-07-16-fairness-baseline-design.md). Pure functions only — wiring lives in
run_weighted_bc_smoke.py / gan/train_mle.py."""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import torch


def normalize_mean_one(w: List[float]) -> List[float]:
    arr = np.asarray(w, dtype=np.float64)
    m = float(arr.mean())
    if arr.size == 0 or not np.isfinite(m) or m <= 0:
        raise ValueError("weight mean must be positive")
    return list(arr / m)


def weights_from_groups(
    groups_of_trajs: List[int], sdr_by_group: Dict[int, float],
) -> List[float]:
    """Kamiran-Calders-style inverse-service weights: 1/SDR_g for group g,
    1.0 for excluded (-1), normalized to mean 1 (effective dataset size kept)."""
    raw = [
        1.0 / max(sdr_by_group[g], 1e-6) if g in sdr_by_group and g >= 0 else 1.0
        for g in groups_of_trajs
    ]
    return normalize_mean_one(raw)


def unit_groups_and_sdr(bundle) -> Tuple[Dict[Tuple[int, int], int], Dict[int, float]]:
    """Migrant axis, district-extremes grouping (disadvantaged_high=True) —
    same construction as run_external_fairness._run_one + _groups_for
    (famail_temporal/baselines/run_external_fairness.py:208-230, 17-23): same
    io.service_ratio_Y / io.per_unit_demographics calls, same
    ef.region_extremes call. _groups_for is module-private, so its 3-line
    body is replicated here rather than imported.

    Returns:
      cell_group: spatial (cx, cy) -> group label (0 adv, 1 disadv, -1
        excluded). Built by projecting the per-active-(cell, time-block)-unit
        group array down to each unit's origin cell. This is lossless: a
        cell's group is time-invariant, since it comes from the cell's own
        demographic value broadcast across every one of its active time
        blocks (external_fairness_io.per_unit_demographics), so collapsing
        repeated (cx, cy) entries across t always agrees. Keyed spatially
        (not by (cell, time-block)) because it must be looked up via a
        trajectory's `pickup_cell`, which carries no time component
        (utils/trajectory.py) — that is the whole point of this dict, since
        fairness_reweigh_weight_vector below indexes it with exactly that.
      sdr: group -> mean before-edit Y (supply/demand ratio), computed over
        every active (cell, time-block) unit in that group (i.e. NOT deduped
        by cell) — matching ef.supply_demand_ratio's / run_external_fairness's
        convention exactly, so busier cells contribute proportionally more.
    """
    from famail_temporal.baselines import external_fairness as ef
    from famail_temporal.baselines import external_fairness_io as io

    axis = "MigrantRatio"
    high = io.DISADVANTAGED_HIGH[axis]  # True
    demo = io.per_unit_demographics(bundle)
    # _groups_for's body (run_external_fairness.py:17-23) for grouping ==
    # "district_extremes", replicated since _groups_for is module-private:
    groups = ef.region_extremes(demo[axis], disadvantaged_high=high)

    Y_before = io.service_ratio_Y(bundle.pickup_3d, bundle)
    sdr_stats = ef.supply_demand_ratio(Y_before, groups)
    sdr: Dict[int, float] = {
        0: sdr_stats["mean_advantaged"],
        1: sdr_stats["mean_disadvantaged"],
    }

    # (N, 3) rows of (cx, cy, t) for each active unit, in the same row order
    # as the boolean-mask flattening used by per_unit_demographics /
    # service_ratio_Y (both index directly with bundle.mask_3d) — verified
    # against the enriched demographics grid during implementation.
    coords = np.argwhere(bundle.mask_3d)
    cell_group: Dict[Tuple[int, int], int] = {}
    for (cx, cy, _t), g in zip(coords, groups):
        cell_group[(int(cx), int(cy))] = int(g)
    return cell_group, sdr


def fairness_reweigh_weight_vector(trajs, bundle) -> List[float]:
    """Index-aligned per-trajectory weights: 1/SDR_g for the group g that a
    trajectory's origin (pickup) cell falls in, normalized to mean 1."""
    cell_group, sdr = unit_groups_and_sdr(bundle)
    groups_of_trajs = [cell_group.get(tuple(t.pickup_cell), -1) for t in trajs]
    return weights_from_groups(groups_of_trajs, sdr)


def dp_gap_penalty(logits, tgt, mask_disadv, mask_adv, pad_id: int):
    """Differentiable DP-gap analog over predicted next-cell distributions:
    (mean predicted mass per ADVANTAGED cell) - (per DISADVANTAGED cell),
    averaged over non-PAD positions. NOT F_causal (metric-firewall: the
    baseline optimizes an external-family quantity)."""
    probs = torch.softmax(logits, dim=-1)            # (B, L, V)
    valid = (tgt != pad_id).to(probs.dtype)          # (B, L)
    n_valid = valid.sum().clamp_min(1.0)
    mass_d = (probs[..., mask_disadv].sum(-1) * valid).sum() / (
        n_valid * int(mask_disadv.sum()))
    mass_a = (probs[..., mask_adv].sum(-1) * valid).sum() / (
        n_valid * int(mask_adv.sum()))
    return mass_a - mass_d


def dp_gap_penalty_abs(logits, tgt, mask_disadv, mask_adv, pad_id: int):
    """Absolute-value variant of :func:`dp_gap_penalty`: |mass_a - mass_d|.

    A thin composition over the proven signed penalty (which is NOT modified).
    Where the advantaged group is over-served (mass_a > mass_d) the two are
    identical in value and gradient; they diverge only if an overshoot crosses
    the gap through zero, where the signed penalty keeps pushing while this one
    pushes back toward equality. Same signature as dp_gap_penalty; the metric
    firewall is unchanged (still an external-family quantity, NOT F_causal).
    Spec: docs/superpowers/specs/2026-07-18-penalty-abs-probe-design.md."""
    return torch.abs(dp_gap_penalty(logits, tgt, mask_disadv, mask_adv, pad_id))


def cell_masks_for_vocab(
    cell_group: Dict[Tuple[int, int], int], vocab_size: int, token_of_cell,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Boolean (disadv, adv) masks over a generator vocabulary of size
    vocab_size, built by mapping each spatial cell in cell_group through
    token_of_cell. token_of_cell is caller-supplied (no production mapping
    is hardcoded here) — see fairness_baseline.py module docstring / the
    caller for the concrete production callable to pass."""
    mask_disadv = torch.zeros(vocab_size, dtype=torch.bool)
    mask_adv = torch.zeros(vocab_size, dtype=torch.bool)
    for cell, g in cell_group.items():
        if g not in (0, 1):
            continue
        tok = token_of_cell(cell)
        if tok is None or not (0 <= tok < vocab_size):
            continue
        if g == 1:
            mask_disadv[tok] = True
        else:
            mask_adv[tok] = True
    return mask_disadv, mask_adv
