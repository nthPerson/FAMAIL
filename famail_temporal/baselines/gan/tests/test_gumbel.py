"""Unit tests for gan.gumbel.gumbel_rollout."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.gumbel import gumbel_rollout


def _ctx(B):
    cc = torch.randint(0, gc.N_CELLS, (B,))
    tb = torch.randint(0, gc.N_TBLOCKS, (B,))
    return cc, tb


def test_rollout_shapes_and_one_hot():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    B, max_len = 4, 10
    cc, tb = _ctx(B)
    soft, lengths = gumbel_rollout(
        model, cc, tb, max_len=max_len, tau=1.0,
        device=torch.device("cpu"), hard=True,
    )
    assert soft.shape == (B, max_len, gc.VOCAB_SIZE)
    # hard=True -> each step is a one-hot: sums to 1, max is 1.
    assert torch.allclose(soft.sum(dim=-1), torch.ones(B, max_len), atol=1e-5)
    assert torch.allclose(soft.max(dim=-1).values, torch.ones(B, max_len), atol=1e-5)
    assert lengths.shape == (B,)
    assert int(lengths.min()) >= 1 and int(lengths.max()) <= max_len


def test_rollout_gradient_flows_to_model():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    cc, tb = _ctx(2)
    soft, _ = gumbel_rollout(
        model, cc, tb, max_len=6, tau=1.0,
        device=torch.device("cpu"), hard=True,
    )
    soft.sum().backward()
    grad_total = sum(
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0


def test_soft_path_is_distribution_and_differentiable():
    """hard=False returns per-step softmax distributions (sum to 1, not one-hot)
    that still carry gradients to the model."""
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    B, max_len = 3, 6
    cc, tb = _ctx(B)
    soft, lengths = gumbel_rollout(
        model, cc, tb, max_len=max_len, tau=1.0,
        device=torch.device("cpu"), hard=False,
    )
    assert soft.shape == (B, max_len, gc.VOCAB_SIZE)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(B, max_len), atol=1e-5)
    assert not torch.isnan(soft).any()
    soft.sum().backward()
    grad_total = sum(
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    )
    assert grad_total > 0


def test_rollout_seed_deterministic():
    model = TrajectoryLSTM()
    cc, tb = _ctx(3)
    torch.manual_seed(7)
    a, la = gumbel_rollout(model, cc, tb, max_len=8, tau=1.0,
                           device=torch.device("cpu"), hard=True)
    torch.manual_seed(7)
    b, lb = gumbel_rollout(model, cc, tb, max_len=8, tau=1.0,
                           device=torch.device("cpu"), hard=True)
    assert torch.equal(a, b) and torch.equal(la, lb)
