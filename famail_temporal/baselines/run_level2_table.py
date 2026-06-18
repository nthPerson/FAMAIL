"""CLI: Level-2 usability table (fairness transfer).

Train a driver-conditioned BC policy on each of four matched, full-corpus data
sources -- raw, FAM-AIL edited, BC-generated, GAN-generated -- across paired
seeds, then evaluate each trained policy's generated demand on the Level-1 axes
(causal/spatial fairness, identity Fidelity-A with the real-anchored gate,
enriched Fidelity-B). Reports paired per-seed differences (edited vs raw
headline; edited vs BC-gen/GAN-gen secondary). HuMID is frozen, read-only.

See docs/superpowers/specs/2026-06-18-level2-usability-fairness-transfer-design.md
and docs/superpowers/plans/2026-06-18-level2-usability-fairness-transfer.md.
"""
from __future__ import annotations
import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.drivers import (
    build_driver_index, group_by_driver, driver_idxs_for,
)
from famail_temporal.baselines.gan.sequences import (
    trajectory_context, trajectory_to_tokens, flat_cell,
)
from famail_temporal.baselines.gan.rollout import (
    generate_trajectories, generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.metrics import data_level_fairness
from famail_temporal.fidelity.checkpoint import load_discriminator
from famail_temporal.baselines import fidelity_eval as fe
from famail_temporal.baselines.run_level1_table_v2 import (
    _select_eval_drivers, _real_context_tensors, _build_source_pairs,
    _train_and_generate_cond, _gen_cond_slot0, _gen_fidelity_full,
    _terminal_pickups_from_cells, _terminal_pickups_from_trajs,
    _edited_fairness_from_metrics, _curves_for_source,
)

_SOURCE_ORDER = ["raw", "edited", "bcgen", "gangen"]


def build_edited_corpus(raw_trajs, histories) -> list:
    """Full corpus with each modified trajectory swapped in by trajectory_id.

    Same length/order as raw_trajs; the 3,773 edited trajectories replace their
    originals, all others are kept (so D_edited is index-aligned to D_raw).
    """
    mod_by_id = {int(h.original.trajectory_id): h.modified for h in histories}
    return [mod_by_id.get(int(t.trajectory_id), t) for t in raw_trajs]


def traj_training_data(trajs, driver_to_idx) -> dict:
    """Token sequences + contexts + embedding indices for a list of Trajectories."""
    return {
        "sequences": [trajectory_to_tokens(t) for t in trajs],
        "contexts": [trajectory_context(t) for t in trajs],
        "driver_idxs": driver_idxs_for(trajs, driver_to_idx),
        "trajs": trajs,
    }


def gen_training_data(model, raw_trajs, driver_to_idx, *, max_len, device,
                      gen_batch_size) -> dict:
    """Driver-conditioned generated training set: one rollout per real seed.

    Each generated trajectory inherits its seed's driver + start-context.
    Empty rollouts fall back to [BOS, start_cell, EOS] (counted in n_empty) so
    the set stays index-aligned and full-corpus-sized.
    """
    contexts = [trajectory_context(t) for t in raw_trajs]
    driver_idxs = driver_idxs_for(raw_trajs, driver_to_idx)
    gen_cells = generate_trajectories(
        model, contexts, max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, driver_idxs=driver_idxs, progress=False,
    )
    sequences = []
    n_empty = 0
    for cells, (start_cell, _t) in zip(gen_cells, contexts):
        if cells:
            sequences.append([gc.BOS] + list(cells) + [gc.EOS])
        else:
            n_empty += 1
            sequences.append([gc.BOS, start_cell, gc.EOS])
    return {
        "sequences": sequences, "contexts": contexts,
        "driver_idxs": driver_idxs, "trajs": raw_trajs, "n_empty": n_empty,
    }


def _paired_diff_stats(per_seed: Dict[str, List[float]], *, baseline: str = "edited") -> dict:
    """Paired per-seed differences baseline - other, per other source.

    Returns {other: {diffs, mean, std, n, wilcoxon_p}}. wilcoxon_p is None when
    SciPy is unavailable, n < 1, or all differences are zero (no signed-rank
    test is defined).
    """
    try:
        from scipy.stats import wilcoxon  # optional dependency
    except Exception:
        wilcoxon = None
    base = per_seed[baseline]
    out: Dict[str, dict] = {}
    for other, vals in per_seed.items():
        if other == baseline:
            continue
        diffs = [float(b - o) for b, o in zip(base, vals)]
        n = len(diffs)
        mean = float(np.mean(diffs)) if n else float("nan")
        std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
        p = None
        if wilcoxon is not None and n >= 1 and any(d != 0.0 for d in diffs):
            try:
                p = float(wilcoxon(diffs).pvalue)
            except Exception:
                p = None
        out[other] = {"diffs": diffs, "mean": mean, "std": std, "n": n, "wilcoxon_p": p}
    return out
