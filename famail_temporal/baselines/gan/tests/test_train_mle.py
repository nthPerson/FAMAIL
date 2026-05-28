"""Unit test for gan.train_mle: the model can overfit a tiny dataset."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_mle import train_mle


def test_overfits_tiny_dataset():
    torch.manual_seed(0)
    # Two short fixed sequences with fixed contexts.
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1)]
    model = TrajectoryLSTM()
    losses = train_mle(
        model, sequences, contexts,
        epochs=200, lr=1e-2, batch_size=2, device=torch.device("cpu"),
    )
    # Loss should fall substantially as the model memorizes the two sequences.
    assert losses[-1] < losses[0] * 0.3
