"""Smoke test for gan.train_adversarial.adversarial_finetune."""
import copy
import math

import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_adversarial import adversarial_finetune


def test_finetune_runs_and_updates_generator():
    torch.manual_seed(0)
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
        [gc.BOS, 5, 6, 7, 8, gc.EOS],
        [gc.BOS, 20, 21, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1), (5, 0), (20, 2)]
    model = TrajectoryLSTM()
    before = copy.deepcopy(model.state_dict())

    history = adversarial_finetune(
        model, sequences, contexts,
        epochs=2, lr_g=1e-3, lr_d=1e-3, batch_size=2,
        max_len=8, tau_start=1.0, tau_end=0.5,
        device=torch.device("cpu"),
    )

    assert set(history) == {"g_losses", "d_losses", "g_batch_losses", "d_batch_losses"}
    assert len(history["g_losses"]) == 2 and len(history["d_losses"]) == 2
    assert all(math.isfinite(x) for x in history["g_losses"] + history["d_losses"])
    # Per-batch loss lists: non-empty, all finite, and d_batch_losses has >= #epochs entries.
    assert len(history["g_batch_losses"]) > 0
    assert len(history["d_batch_losses"]) >= len(history["d_losses"])
    assert all(math.isfinite(x) for x in history["g_batch_losses"] + history["d_batch_losses"])
    # The generator's parameters moved (fine-tune actually stepped G).
    after = model.state_dict()
    assert any(
        not torch.allclose(before[k], after[k]) for k in before
    )


def test_finetune_with_stabilization_knobs_runs():
    """Label smoothing + grad clip + a slowed critic (d_update_every>1) all
    run and keep losses finite; the generator still updates."""
    torch.manual_seed(0)
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
        [gc.BOS, 5, 6, 7, 8, gc.EOS],
        [gc.BOS, 20, 21, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1), (5, 0), (20, 2)]
    model = TrajectoryLSTM()
    before = copy.deepcopy(model.state_dict())

    history = adversarial_finetune(
        model, sequences, contexts,
        epochs=2, lr_g=1e-3, lr_d=1e-3, batch_size=2,
        max_len=8, tau_start=1.0, tau_end=0.5,
        real_label=0.9, grad_clip=1.0, d_update_every=2,
        device=torch.device("cpu"),
    )

    assert set(history) == {"g_losses", "d_losses", "g_batch_losses", "d_batch_losses"}
    assert all(math.isfinite(x) for x in history["g_losses"] + history["d_losses"])
    # d_update_every=2 with 2 batches/epoch -> the critic updates on batch 0
    # only, so d_losses is still populated (no divide-by-zero).
    assert len(history["d_losses"]) == 2
    after = model.state_dict()
    assert any(not torch.allclose(before[k], after[k]) for k in before)


def test_finetune_mle_lambda_disabled_runs():
    """mle_lambda=0 disables the MLE anchor; the loop still runs and reports the
    adversarial g_loss (finite)."""
    torch.manual_seed(0)
    sequences = [
        [gc.BOS, 10, 11, 12, gc.EOS],
        [gc.BOS, 40, 41, gc.EOS],
    ]
    contexts = [(10, 0), (40, 1)]
    model = TrajectoryLSTM()
    history = adversarial_finetune(
        model, sequences, contexts,
        epochs=1, lr_g=1e-3, lr_d=1e-3, batch_size=2,
        max_len=8, tau_start=1.0, tau_end=0.5, mle_lambda=0.0,
        device=torch.device("cpu"),
    )
    assert set(history) == {"g_losses", "d_losses", "g_batch_losses", "d_batch_losses"}
    assert all(math.isfinite(x) for x in history["g_losses"] + history["d_losses"])
