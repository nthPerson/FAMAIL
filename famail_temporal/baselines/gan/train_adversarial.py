"""Gumbel-softmax adversarial fine-tune of an MLE-pretrained generator.

Non-saturating GAN: the discriminator maximizes log D(real) + log(1 - D(fake));
the generator maximizes log D(fake). The fake batch is a differentiable
straight-through Gumbel-softmax rollout, so generator gradients flow through
the discrete sequence. The Gumbel temperature is annealed across epochs. A
fresh SequenceCritic is created and trained alongside (the trained Siamese
discriminator is reserved for eval-time realism, not used here).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
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
    real_label: float = gc.D_REAL_LABEL,
    grad_clip: Optional[float] = gc.GRAD_CLIP,
    d_update_every: int = gc.D_UPDATE_EVERY,
    mle_lambda: float = gc.ADV_MLE_LAMBDA,
    progress: bool = False,
) -> Dict[str, List[float]]:
    """Fine-tune `model` (in place) against a fresh critic. Returns per-epoch
    mean generator (adversarial) and discriminator losses.

    Stabilizers against discriminator dominance / generator collapse:
      - ``mle_lambda`` adds a teacher-forced NLL term on the real batch to the
        generator loss, anchoring G to the data distribution so it can't drift
        (drift -> ever-longer fakes -> the critic separates on length ->
        collapse). This is the root-cause fix; 0 disables it.
      - one-sided label smoothing: the critic's real target is ``real_label``
        (< 1.0), capping its confidence so its gradient to G never vanishes;
      - gradient clipping to ``grad_clip`` (None disables) on both nets;
      - ``d_update_every`` updates the critic only every k-th batch, letting a
        lagging generator catch up.

    The reported g_loss is the *adversarial* component only (excludes the MLE
    term) so it stays comparable to d_loss. ``progress=True`` shows a per-epoch
    bar with live g / d / tau (and mle when active) and prints a per-epoch
    length diagnostic (mean real vs fake token length).
    """
    model.to(device).train()
    critic = SequenceCritic().to(device).train()
    opt_g = torch.optim.Adam(model.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=gc.PAD)
    n = len(sequences)
    n_batches = (n + batch_size - 1) // batch_size
    real_len_mean = sum(len(s) for s in sequences) / n
    g_losses: List[float] = []
    d_losses: List[float] = []

    for epoch in range(epochs):
        tau = _anneal(epoch, epochs, tau_start, tau_end)
        perm = torch.randperm(n)
        g_batch: List[float] = []
        d_batch: List[float] = []
        fake_len_sum = 0.0
        fake_len_cnt = 0
        bar = Progress(
            n_batches, f"adv epoch {epoch + 1}/{epochs}", enabled=progress,
        )
        for batch_i, start in enumerate(range(0, n, batch_size)):
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

            # ----- Discriminator step (generator fixed; every d_update_every) -
            # no_grad detaches the fake, so the generator gets no gradient here;
            # the resulting hard one-hots feed forward_soft purely as data (the
            # critic still trains on them via its own params).
            if batch_i % d_update_every == 0:
                with torch.no_grad():
                    fake_soft, fake_len = gumbel_rollout(
                        model, cc, tb, max_len=max_len, tau=tau,
                        device=device, hard=True,
                    )
                d_real = critic.forward_ids(real, real_lengths)
                d_fake = critic.forward_soft(fake_soft, fake_len)
                # One-sided label smoothing: real target < 1.0, fake stays 0.
                loss_d = (
                    bce(d_real, torch.full_like(d_real, real_label))
                    + bce(d_fake, torch.zeros_like(d_fake))
                )
                opt_d.zero_grad()
                loss_d.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                opt_d.step()
                d_batch.append(float(loss_d.item()))

            # ----- Generator step (non-saturating; gradients via Gumbel) -----
            fake_soft, fake_len = gumbel_rollout(
                model, cc, tb, max_len=max_len, tau=tau,
                device=device, hard=True,
            )
            d_fake_g = critic.forward_soft(fake_soft, fake_len)
            adv_g = bce(d_fake_g, torch.ones_like(d_fake_g))
            loss_g = adv_g
            mle_nll = None
            if mle_lambda > 0:
                # Teacher-forced NLL on the real batch anchors G to the data
                # distribution, so it can't drift toward unrealistic lengths.
                logits = model(real[:, :-1], cc, tb)            # (b, L-1, V)
                mle_nll = ce(
                    logits.reshape(-1, gc.VOCAB_SIZE), real[:, 1:].reshape(-1),
                )
                loss_g = adv_g + mle_lambda * mle_nll
            opt_g.zero_grad()
            loss_g.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt_g.step()

            g_batch.append(float(adv_g.item()))   # adversarial component only
            fake_len_sum += float(fake_len.float().sum().item())
            fake_len_cnt += int(fake_len.numel())
            postfix = {
                "g": f"{sum(g_batch) / len(g_batch):.3f}",
                "d": f"{sum(d_batch) / max(1, len(d_batch)):.3f}",
                "tau": f"{tau:.2f}",
            }
            if mle_nll is not None:
                postfix["mle"] = f"{float(mle_nll.item()):.3f}"
            bar.update(1, **postfix)
        bar.close()
        g_losses.append(sum(g_batch) / len(g_batch))
        d_losses.append(sum(d_batch) / max(1, len(d_batch)))
        if progress:
            fake_len_mean = fake_len_sum / max(1, fake_len_cnt)
            print(
                f"[adv epoch {epoch + 1}/{epochs}] mean length "
                f"real={real_len_mean:.1f} fake={fake_len_mean:.1f} "
                f"(if fake >> real, the critic may be cheating on length)",
                flush=True,
            )

    return {"g_losses": g_losses, "d_losses": d_losses}
