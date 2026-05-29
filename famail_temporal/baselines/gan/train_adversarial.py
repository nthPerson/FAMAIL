"""Gumbel-softmax adversarial fine-tune of an MLE-pretrained generator.

Non-saturating GAN: the discriminator maximizes log D(real) + log(1 - D(fake));
the generator maximizes log D(fake). The fake batch is a differentiable
straight-through Gumbel-softmax rollout, so generator gradients flow through
the discrete sequence. The Gumbel temperature is annealed across epochs. A
fresh SequenceCritic is created and trained alongside (the trained Siamese
discriminator is reserved for eval-time realism, not used here).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.critic import SequenceCritic
from famail_temporal.baselines.gan.gumbel import gumbel_rollout
from famail_temporal.baselines.gan.train_mle import _pad_batch
from famail_temporal.baselines.gan.progress import Progress


def _anneal(epoch: int, n_epochs: int, start: float, end: float) -> float:
    if n_epochs <= 1:
        return end
    return start + (end - start) * (epoch / (n_epochs - 1))


def adversarial_finetune(
    model: TrajectoryLSTM,
    sequences: List[List[int]],
    contexts: List[Tuple[int, int]],
    *,
    epochs: int,
    lr_g: float,
    lr_d: float,
    batch_size: int,
    max_len: int,
    tau_start: float,
    tau_end: float,
    device: torch.device,
    progress: bool = False,
) -> Dict[str, List[float]]:
    """Fine-tune `model` (in place) against a fresh critic. Returns per-epoch
    mean generator and discriminator losses. ``progress=True`` shows a per-epoch
    bar with live g_loss / d_loss / tau (the divergence readout)."""
    model.to(device).train()
    critic = SequenceCritic().to(device).train()
    opt_g = torch.optim.Adam(model.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    n = len(sequences)
    n_batches = (n + batch_size - 1) // batch_size
    g_losses: List[float] = []
    d_losses: List[float] = []

    for epoch in range(epochs):
        tau = _anneal(epoch, epochs, tau_start, tau_end)
        perm = torch.randperm(n)
        g_batch: List[float] = []
        d_batch: List[float] = []
        bar = Progress(
            n_batches, f"adv epoch {epoch + 1}/{epochs}", enabled=progress,
        )
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size].tolist()
            real = _pad_batch([sequences[i] for i in idx], device)      # (b, Lr)
            real_lengths = torch.tensor(
                [len(sequences[i]) for i in idx], dtype=torch.long, device=device,
            )
            cc = torch.tensor(
                [contexts[i][0] for i in idx], dtype=torch.long, device=device,
            )
            tb = torch.tensor(
                [contexts[i][1] for i in idx], dtype=torch.long, device=device,
            )

            # ----- Discriminator step (generator fixed) -----
            # no_grad detaches the fake, so the generator gets no gradient here;
            # the resulting hard one-hots feed forward_soft purely as data (the
            # critic still trains on them via its own params).
            with torch.no_grad():
                fake_soft, fake_len = gumbel_rollout(
                    model, cc, tb, max_len=max_len, tau=tau,
                    device=device, hard=True,
                )
            d_real = critic.forward_ids(real, real_lengths)
            d_fake = critic.forward_soft(fake_soft, fake_len)
            loss_d = (
                bce(d_real, torch.ones_like(d_real))
                + bce(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # ----- Generator step (non-saturating; gradients via Gumbel) -----
            fake_soft, fake_len = gumbel_rollout(
                model, cc, tb, max_len=max_len, tau=tau,
                device=device, hard=True,
            )
            d_fake_g = critic.forward_soft(fake_soft, fake_len)
            loss_g = bce(d_fake_g, torch.ones_like(d_fake_g))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            g_batch.append(float(loss_g.item()))
            d_batch.append(float(loss_d.item()))
            bar.update(
                1,
                g=f"{sum(g_batch) / len(g_batch):.3f}",
                d=f"{sum(d_batch) / len(d_batch):.3f}",
                tau=f"{tau:.2f}",
            )
        bar.close()
        g_losses.append(sum(g_batch) / len(g_batch))
        d_losses.append(sum(d_batch) / len(d_batch))

    return {"g_losses": g_losses, "d_losses": d_losses}
