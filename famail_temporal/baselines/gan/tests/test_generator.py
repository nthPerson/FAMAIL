"""Unit tests for gan.generator.TrajectoryLSTM."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def test_forward_returns_vocab_logits():
    model = TrajectoryLSTM()
    B, L = 4, 7
    tokens = torch.randint(0, gc.N_CELLS, (B, L))
    ctx_cell = torch.randint(0, gc.N_CELLS, (B,))
    ctx_tblock = torch.randint(0, gc.N_TBLOCKS, (B,))
    logits = model(tokens, ctx_cell, ctx_tblock)
    assert logits.shape == (B, L, gc.VOCAB_SIZE)


def test_context_changes_logits():
    """Different conditioning context must change the output distribution."""
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    tokens = torch.randint(0, gc.N_CELLS, (1, 5))
    c0 = torch.tensor([0]); c1 = torch.tensor([gc.N_CELLS - 1])
    tb = torch.tensor([0])
    out0 = model(tokens, c0, tb)
    out1 = model(tokens, c1, tb)
    assert not torch.allclose(out0, out1)


def test_step_matches_forward():
    """Stepwise decode (carried state) equals slicing forward() per position."""
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    model.train(False)
    tokens = torch.tensor([[gc.BOS, 10, 11, 12]])  # (1, 4)
    cc = torch.tensor([10]); tb = torch.tensor([0])
    full = model(tokens, cc, tb)                    # (1, 4, V)
    hidden = None
    for i in range(tokens.shape[1]):
        logits, hidden = model.step(tokens[:, i], cc, tb, hidden)
        assert torch.allclose(logits[0], full[0, i], atol=1e-5)
