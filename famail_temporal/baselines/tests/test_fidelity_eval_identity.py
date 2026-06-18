import random
import numpy as np
import torch

from famail_temporal.baselines import fidelity_eval as fe


def _traj_tensor(L, base):
    # [L,4] with distinguishable coords; +1 already applied by convention
    return torch.tensor(
        [[base + i + 1.0, base + i + 1.0, 10.0, 1.0] for i in range(L)],
        dtype=torch.float32,
    )


def _profile(v):
    return np.full(11, float(v), dtype=np.float32)


class _ProfileSameStub(torch.nn.Module):
    """Returns high prob iff the two branches' profiles are (near) equal.

    Stands in for an identity discriminator: same driver (same profile) -> ~1,
    different driver -> ~0. Ignores trajectories; exercises the plumbing + gate.
    """
    def forward(self, x1, x2, mask1=None, mask2=None, *, profile_1=None,
                profile_2=None, **kw):
        b = x1.shape[0]
        if profile_1 is None or profile_2 is None:
            return torch.full((b, 1), 0.5)
        same = (profile_1 - profile_2).abs().sum(dim=-1) < 1e-6
        return torch.where(same, torch.full((b,), 0.95),
                           torch.full((b,), 0.05)).unsqueeze(-1)


def _branch(slot0_base, ctx_bases, rng):
    slot0 = _traj_tensor(4, slot0_base)
    ctx = [_traj_tensor(3, b) for b in ctx_bases]
    return fe.build_identity_branch(slot0, ctx, rng=rng)


def test_build_identity_branch_shapes_and_slot0():
    rng = random.Random(0)
    s, m = _branch(0, [10, 20, 30, 40], rng)
    assert s.shape[0] == fe.N_TRAJS_PER_BRANCH
    assert m.shape[0] == fe.N_TRAJS_PER_BRANCH
    # slot 0's first real step keeps its identity coord (0+1)
    assert float(s[0, 0, 0]) == 1.0
    assert bool(m[0, 0])


def test_build_identity_branch_samples_with_replacement_when_short():
    rng = random.Random(0)
    s, m = _branch(0, [10], rng)   # only 1 context, needs n-1
    assert s.shape[0] == fe.N_TRAJS_PER_BRANCH


def test_identity_fidelity_high_for_same_profile():
    rng = random.Random(0)
    disc = _ProfileSameStub()
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p = _profile(1)
    pairs = [((sl, ml, p), (sr, mr, p))]   # same profile
    out = fe.humid_identity_fidelity(disc, pairs)
    assert out["mean"] > 0.9 and out["n"] == 1


def test_identity_gate_passes_when_matched_above_mismatched():
    rng = random.Random(0)
    disc = _ProfileSameStub()
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p_d, p_dp = _profile(1), _profile(2)
    matched = [((sl, ml, p_d), (sr, mr, p_d))]        # same driver
    mismatched = [((sl, ml, p_d), (sr, mr, p_dp))]    # different driver
    gate = fe.identity_validation_gate(
        disc, matched_pairs=matched, mismatched_pairs=mismatched,
    )
    assert gate["passed"] is True
    assert gate["high_matched"] > gate["low_mismatched"]


def test_identity_gate_fails_for_constant_discriminator():
    class _Const(torch.nn.Module):
        def forward(self, x1, x2, mask1=None, mask2=None, **kw):
            return torch.full((x1.shape[0], 1), 0.7)
    rng = random.Random(0)
    sl, ml = _branch(0, [10, 20, 30, 40], rng)
    sr, mr = _branch(5, [10, 20, 30, 40], rng)
    p = _profile(1)
    matched = [((sl, ml, p), (sr, mr, p))]
    mismatched = [((sl, ml, p), (sr, mr, p))]
    gate = fe.identity_validation_gate(
        _Const(), matched_pairs=matched, mismatched_pairs=mismatched,
    )
    assert gate["passed"] is False
