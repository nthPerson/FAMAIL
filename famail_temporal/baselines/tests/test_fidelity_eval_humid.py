"""fidelity_eval: HuMID paired fidelity + validation gate (stub discriminator)."""
import math

import torch
import torch.nn as nn

from famail_temporal.baselines import fidelity_eval as fe


class _ConstDiscriminator(nn.Module):
    """Returns a fixed same-agent probability for every pair."""
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x1, x2, **kwargs):
        b = x1.shape[0]
        return torch.full((b, 1), self.prob)


class _LengthSimDiscriminator(nn.Module):
    """High prob when the two trajectories have similar length, else low.

    Stands in for 'realistic' scoring: real-vs-real (similar lengths) -> high;
    real-vs-collapsed (very different lengths) -> low.
    """
    def forward(self, x1, x2, **kwargs):
        # x1, x2: [B, L, 4] padded; recover per-row nonzero length from coords.
        l1 = (x1[..., 0] > 0).sum(dim=1).float()
        l2 = (x2[..., 0] > 0).sum(dim=1).float()
        diff = (l1 - l2).abs()
        prob = torch.exp(-diff / 5.0).unsqueeze(1)   # close lengths -> ~1
        return prob


def _pair(len_a, len_b):
    # torch.ones (not zeros) is load-bearing: padding is 0.0, so a nonzero
    # x-coord marks a real step. _LengthSimDiscriminator counts (x > 0) to
    # recover per-row length, and _pad_pairs_to_batch zero-pads — keep real
    # steps nonzero or the length proxy (and any future mask-aware stub) breaks.
    a = torch.ones(len_a, 4)
    b = torch.ones(len_b, 4)
    return (a, b)


def test_humid_paired_fidelity_mean_over_pairs():
    disc = _ConstDiscriminator(0.8)
    pairs = [_pair(5, 5), _pair(6, 6), _pair(7, 7)]
    out = fe.humid_paired_fidelity(disc, pairs, batch_size=2)  # multi-batch
    assert out["n"] == 3
    assert math.isclose(out["mean"], 0.8, rel_tol=1e-6)
    assert math.isclose(out["std"], 0.0, abs_tol=1e-6)


def test_validation_gate_passes_with_clear_separation():
    disc = _LengthSimDiscriminator()
    real_pairs = [_pair(18, 18) for _ in range(8)]        # similar -> high
    collapsed_pairs = [_pair(18, 52) for _ in range(8)]   # mismatch -> low
    shuffled_pairs = [_pair(18, 50) for _ in range(8)]    # mismatch -> low
    out = fe.validation_gate(
        disc, real_pairs=real_pairs, collapsed_pairs=collapsed_pairs,
        shuffled_pairs=shuffled_pairs, batch_size=4,
    )
    assert out["high_real_real"] > out["low_collapsed"]
    assert out["high_real_real"] > out["low_shuffled"]
    assert out["passed"] is True


def test_validation_gate_fails_without_separation():
    disc = _ConstDiscriminator(0.5)   # cannot tell real from garbage
    pairs = [_pair(18, 18) for _ in range(4)]
    out = fe.validation_gate(
        disc, real_pairs=pairs, collapsed_pairs=pairs, shuffled_pairs=pairs,
        batch_size=4,
    )
    assert out["passed"] is False
