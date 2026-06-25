"""Tests for the random-subset placebo selector in run_weighted_bc_smoke.

The placebo control upweights a RANDOM, matched-size subset of NON-edited
trajectories. If F_causal rises for the placebo the way it does for the edited
arm, the weighted-BC fairness gain would be an oversampling artifact rather than
something edit-specific. These tests pin the selector's correctness:

1. exactly k entries get weight w, every other entry is 1.0 (k defaults to the
   number of edited trajectories so the placebo subset is size-matched),
2. the upweighted indices are disjoint from the edited set (we upweight
   *unedited* data, so the only difference from the raw arm is the weighting),
3. the selection is reproducible given the seed, and
4. it is INDEPENDENT of the global torch/numpy/random RNG -- critical, because
   the paired-seed design relies on set_all_seeds() making model init + batch
   order byte-identical across arms; if subset selection consumed the global
   RNG, the raw/edited arms would stop reproducing the locked Level-2 baseline.
"""
from collections import namedtuple

from famail_temporal.baselines.run_weighted_bc_smoke import (
    random_subset_weight_vector,
)

_Traj = namedtuple("_Traj", ["trajectory_id"])


def _corpus(n):
    return [_Traj(trajectory_id=i) for i in range(n)]


def test_default_k_matches_edited_count_and_is_disjoint():
    trajs = _corpus(100)
    edited = {0, 1, 2, 3, 4}  # 5 edited ids
    w = random_subset_weight_vector(trajs, edited, 30.0)
    assert len(w) == len(trajs)
    upweighted = [i for i, v in enumerate(w) if v != 1.0]
    # exactly len(edited) entries upweighted, all == w, none in the edited set
    assert len(upweighted) == len(edited)
    assert all(w[i] == 30.0 for i in upweighted)
    assert all(int(trajs[i].trajectory_id) not in edited for i in upweighted)
    # every non-upweighted entry is exactly 1.0
    assert all(v == 1.0 for i, v in enumerate(w) if i not in set(upweighted))


def test_explicit_k_overrides_default():
    trajs = _corpus(100)
    edited = {0, 1, 2}
    w = random_subset_weight_vector(trajs, edited, 10.0, k=20)
    assert sum(1 for v in w if v == 10.0) == 20


def test_reproducible_given_seed():
    trajs = _corpus(200)
    edited = set(range(10))
    a = random_subset_weight_vector(trajs, edited, 30.0, seed=123)
    b = random_subset_weight_vector(trajs, edited, 30.0, seed=123)
    assert a == b


def test_different_seed_gives_different_subset():
    trajs = _corpus(200)
    edited = set(range(10))
    a = random_subset_weight_vector(trajs, edited, 30.0, seed=1)
    b = random_subset_weight_vector(trajs, edited, 30.0, seed=2)
    # 10-sample draw from a 190-element pool: collision is astronomically unlikely
    assert a != b


def test_does_not_consume_global_rng():
    """Selecting the subset must not advance torch/numpy/global-random state,
    or it would break the set_all_seeds paired-seed determinism."""
    import random as pyrandom
    import numpy as np
    import torch

    trajs = _corpus(200)
    edited = {0, 1, 2, 3, 4}

    pyrandom.seed(0); np.random.seed(0); torch.manual_seed(0)
    before_py = pyrandom.random()
    before_np = float(np.random.rand())
    before_th = float(torch.rand(1).item())

    pyrandom.seed(0); np.random.seed(0); torch.manual_seed(0)
    random_subset_weight_vector(trajs, edited, 30.0)
    after_py = pyrandom.random()
    after_np = float(np.random.rand())
    after_th = float(torch.rand(1).item())

    assert before_py == after_py
    assert before_np == after_np
    assert before_th == after_th
