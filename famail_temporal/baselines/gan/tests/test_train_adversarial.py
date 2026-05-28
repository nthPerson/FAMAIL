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

    assert set(history) == {"g_losses", "d_losses"}
    assert len(history["g_losses"]) == 2 and len(history["d_losses"]) == 2
    assert all(math.isfinite(x) for x in history["g_losses"] + history["d_losses"])
    # The generator's parameters moved (fine-tune actually stepped G).
    after = model.state_dict()
    assert any(
        not torch.allclose(before[k], after[k]) for k in before
    )
