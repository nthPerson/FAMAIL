"""Unit tests for gan.critic.SequenceCritic."""
import torch
import torch.nn.functional as F

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.critic import SequenceCritic


def test_forward_ids_returns_per_sequence_logit():
    torch.manual_seed(0)
    critic = SequenceCritic()
    B, L = 5, 7
    ids = torch.randint(0, gc.N_CELLS, (B, L))
    lengths = torch.full((B,), L, dtype=torch.long)
    out = critic.forward_ids(ids, lengths)
    assert out.shape == (B,)


def test_soft_matches_hard_onehot():
    """forward_soft on a hard one-hot equals forward_ids on the same ids."""
    torch.manual_seed(0)
    critic = SequenceCritic()
    critic.train(False)
    B, L = 4, 6
    ids = torch.randint(0, gc.N_CELLS, (B, L))
    lengths = torch.full((B,), L, dtype=torch.long)
    onehot = F.one_hot(ids, num_classes=gc.VOCAB_SIZE).float()
    a = critic.forward_ids(ids, lengths)
    b = critic.forward_soft(onehot, lengths)
    assert torch.allclose(a, b, atol=1e-5)


def test_critic_can_separate_trivial_real_vs_fake():
    """A few D-steps should push real logits up and fake logits down on a
    trivially separable batch (real = low cell ids, fake = high cell ids)."""
    torch.manual_seed(0)
    critic = SequenceCritic()
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    L = 5
    real = torch.zeros(8, L, dtype=torch.long)               # all cell 0
    fake = torch.full((8, L), gc.N_CELLS - 1, dtype=torch.long)  # all last cell
    lengths = torch.full((8,), L, dtype=torch.long)
    for _ in range(50):
        d_real = critic.forward_ids(real, lengths)
        d_fake = critic.forward_ids(fake, lengths)
        loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        opt.zero_grad(); loss.backward(); opt.step()
    # Require a clear margin, not just ordering, so a broken optimizer loop or
    # misconfigured loss is caught (trivially separable + 50 Adam steps).
    real_mean = critic.forward_ids(real, lengths).mean()
    fake_mean = critic.forward_ids(fake, lengths).mean()
    assert real_mean - fake_mean > 1.0
