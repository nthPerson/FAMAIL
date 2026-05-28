"""step_embed must reproduce step() when fed the hard-token embedding."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def test_step_embed_matches_step():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    model.train(False)
    B = 3
    token = torch.randint(0, gc.N_CELLS, (B,))
    cc = torch.randint(0, gc.N_CELLS, (B,))
    tb = torch.randint(0, gc.N_TBLOCKS, (B,))

    logits_step, h_step = model.step(token, cc, tb, None)
    embed = model.cell_embed(token)                      # (B, E)
    logits_embed, h_embed = model.step_embed(embed, cc, tb, None)

    assert logits_embed.shape == (B, gc.VOCAB_SIZE)
    assert torch.allclose(logits_step, logits_embed, atol=1e-6)
    assert torch.allclose(h_step[0], h_embed[0], atol=1e-6)
    assert torch.allclose(h_step[1], h_embed[1], atol=1e-6)


def test_step_embed_passes_gradient_to_input():
    model = TrajectoryLSTM()
    embed = torch.randn(2, gc.EMBED_DIM, requires_grad=True)
    cc = torch.zeros(2, dtype=torch.long)
    tb = torch.zeros(2, dtype=torch.long)
    logits, _ = model.step_embed(embed, cc, tb, None)
    logits.sum().backward()
    assert embed.grad is not None and embed.grad.abs().sum() > 0
