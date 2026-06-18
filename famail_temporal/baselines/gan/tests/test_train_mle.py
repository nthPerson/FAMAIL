"""Unit test for gan.train_mle: the model can overfit a tiny dataset."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_mle import train_mle


def test_overfits_tiny_dataset():
    torch.manual_seed(0)
    # Two short fixed sequences with fixed contexts.
    n = 2
    epochs = 200
    batch_size = 2
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1)]
    model = TrajectoryLSTM()
    out = train_mle(
        model, sequences, contexts,
        epochs=epochs, lr=1e-2, batch_size=batch_size, device=torch.device("cpu"),
    )
    # --- epoch_losses: same overfit check as before ---
    ep = out["epoch_losses"]
    assert ep[-1] < ep[0] * 0.3

    # --- batch_losses: per-batch curve assertions ---
    bl = out["batch_losses"]
    expected_batches_per_epoch = (n + batch_size - 1) // batch_size  # == 1
    assert isinstance(bl, list), "batch_losses must be a list"
    assert len(bl) > 0, "batch_losses must be non-empty"
    assert all(isinstance(v, float) and (v == v) and (v != float("inf")) for v in bl), \
        "all batch_losses must be finite floats"
    assert len(bl) == epochs * expected_batches_per_epoch, (
        f"expected {epochs * expected_batches_per_epoch} batch entries, got {len(bl)}"
    )
