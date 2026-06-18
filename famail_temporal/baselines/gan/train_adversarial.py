"""Gumbel-softmax adversarial fine-tune of an MLE-pretrained generator.

Two loss modes (``gan_loss``):
  - "bce" (default): non-saturating GAN. The discriminator maximizes
    log D(real) + log(1 - D(fake)); the generator maximizes log D(fake).
  - "wgan-gp": Wasserstein GAN with gradient penalty (Gulrajani et al. 2017).
    The critic minimizes mean(D(fake)) - mean(D(real)) + gp_lambda * GP, where
    GP is computed on embedding-space interpolates (discrete tokens can't be
    interpolated); the generator minimizes -mean(D(fake)). The generator
    updates only every ``n_critic``-th batch (wgan convention).

In both modes the fake batch is a differentiable straight-through
Gumbel-softmax rollout, so generator gradients flow through the discrete
sequence. The Gumbel temperature is annealed across epochs. A fresh
SequenceCritic is created and trained alongside (the trained Siamese
discriminator is reserved for eval-time realism, not used here).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def _gradient_penalty(
    critic: SequenceCritic,
    real_ids: torch.Tensor,
    real_lengths: torch.Tensor,
    fake_soft: torch.Tensor,
    fake_lengths: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """WGAN-GP penalty on embedding-space interpolates (Gulrajani et al.).

    Token sequences are discrete, so the interpolation happens in the critic's
    embedding space: real ids are embedded, fakes are soft-mixed embeddings,
    both zero-padded to a common length, then x_hat = eps*real + (1-eps)*fake
    with per-sample eps ~ U[0,1]. The readout length for x_hat is the
    elementwise max of the pair so no valid timestep is cut off.
    """
    real_emb = critic.embed(real_ids)                       # (B, Lr, E)
    fake_emb = fake_soft @ critic.embed.weight              # (B, Lf, E)
    L = max(real_emb.size(1), fake_emb.size(1))
    real_emb = F.pad(real_emb, (0, 0, 0, L - real_emb.size(1)))
    fake_emb = F.pad(fake_emb, (0, 0, 0, L - fake_emb.size(1)))
    eps = torch.rand(real_emb.size(0), 1, 1, device=device)
    interp = (eps * real_emb + (1.0 - eps) * fake_emb).requires_grad_(True)
    lengths = torch.maximum(real_lengths, fake_lengths)
    # cuDNN's RNN kernels don't support double backward, which the GP needs
    # (loss_d.backward() differentiates through this grad graph). Disabling
    # cuDNN for just this forward records the native LSTM implementation,
    # which is twice-differentiable. CPU runs are unaffected.
    with torch.backends.cudnn.flags(enabled=False):
        scores = critic.forward_embed(interp, lengths)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=interp, create_graph=True,
    )[0]                                                    # (B, L, E)
    grad_norm = grads.flatten(1).norm(2, dim=1)             # (B,)
    return ((grad_norm - 1.0) ** 2).mean()


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
    gan_loss: str = gc.GAN_LOSS,
    gp_lambda: float = gc.WGAN_GP_LAMBDA,
    n_critic: int = 1,
    progress: bool = False,
    driver_idxs: List[int] | None = None,
) -> Dict[str, List[float]]:
    """Fine-tune `model` (in place) against a fresh critic. Returns per-epoch
    mean generator (adversarial) and discriminator losses, plus flat per-batch
    g/d loss lists in global-step order.

    Return keys:
      - ``g_losses``: per-epoch mean adversarial generator loss (length = epochs).
      - ``d_losses``: per-epoch mean discriminator/critic loss (length = epochs).
      - ``g_batch_losses``: flat list of adversarial g loss at every generator
        update step across all epochs (length = total g update count).
      - ``d_batch_losses``: flat list of discriminator/critic loss at every d
        update step across all epochs (length = total d update count).
    The per-batch g and d series have different lengths because the d-step runs
    every ``d_update_every`` batches while the g-step cadence differs (every
    batch for BCE, every ``n_critic``-th for WGAN-GP).

    ``gan_loss`` selects the adversarial objective:
      - "bce" (default): non-saturating BCE GAN, exactly the historical
        behavior (label smoothing active; G updates every batch).
      - "wgan-gp": Wasserstein critic loss + ``gp_lambda`` * gradient penalty
        on embedding-space interpolates; generator loss -mean(D(fake)). The
        generator updates only every ``n_critic``-th batch (wgan convention:
        n_critic critic updates per G update; composes with
        ``d_update_every``). ``real_label`` smoothing is inert in this mode,
        and the reported d_loss is the Wasserstein critic loss including the
        gradient-penalty term.

    Stabilizers against discriminator dominance / generator collapse:
      - ``mle_lambda`` adds a teacher-forced NLL term on the real batch to the
        generator loss, anchoring G to the data distribution so it can't drift
        (drift -> ever-longer fakes -> the critic separates on length ->
        collapse). This is the root-cause fix; 0 disables it. Available in
        both loss modes.
      - one-sided label smoothing: the critic's real target is ``real_label``
        (< 1.0), capping its confidence so its gradient to G never vanishes
        (bce mode only);
      - gradient clipping to ``grad_clip`` (None disables) on both nets;
      - ``d_update_every`` updates the critic only every k-th batch, letting a
        lagging generator catch up.

    The reported g_loss is the *adversarial* component only (excludes the MLE
    term) so it stays comparable to d_loss. ``progress=True`` shows a per-epoch
    bar with live g / d / tau (and mle when active) and prints a per-epoch
    length diagnostic (mean real vs fake token length).
    """
    if gan_loss not in ("bce", "wgan-gp"):
        raise ValueError(f"unknown gan_loss: {gan_loss!r} (use 'bce' or 'wgan-gp')")
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
    all_g_batch_losses: List[float] = []
    all_d_batch_losses: List[float] = []

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
            di = (
                torch.tensor(
                    [driver_idxs[i] for i in idx], dtype=torch.long, device=device,
                )
                if driver_idxs is not None else None
            )

            # ----- Discriminator step (generator fixed; every d_update_every) -
            # no_grad detaches the fake, so the generator gets no gradient here;
            # the resulting hard one-hots feed forward_soft purely as data (the
            # critic still trains on them via its own params).
            if batch_i % d_update_every == 0:
                with torch.no_grad():
                    fake_soft, fake_len = gumbel_rollout(
                        model, cc, tb, max_len=max_len, tau=tau,
                        device=device, hard=True, driver_idx=di,
                    )
                d_real = critic.forward_ids(real, real_lengths)
                d_fake = critic.forward_soft(fake_soft, fake_len)
                if gan_loss == "bce":
                    # One-sided label smoothing: real target < 1.0, fake stays 0.
                    loss_d = (
                        bce(d_real, torch.full_like(d_real, real_label))
                        + bce(d_fake, torch.zeros_like(d_fake))
                    )
                else:  # wgan-gp: critic maximizes real-fake gap, GP enforces 1-Lipschitz
                    loss_d = (
                        d_fake.mean() - d_real.mean()
                        + gp_lambda * _gradient_penalty(
                            critic, real, real_lengths, fake_soft, fake_len,
                            device=device,
                        )
                    )
                opt_d.zero_grad()
                loss_d.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                opt_d.step()
                d_batch.append(float(loss_d.item()))
                all_d_batch_losses.append(float(loss_d.item()))

            # ----- Generator step (gradients via Gumbel) -----
            g_step = gan_loss == "bce" or (batch_i % n_critic == n_critic - 1)
            mle_nll = None
            if g_step:
                fake_soft, fake_len = gumbel_rollout(
                    model, cc, tb, max_len=max_len, tau=tau,
                    device=device, hard=True, driver_idx=di,
                )
                d_fake_g = critic.forward_soft(fake_soft, fake_len)
                if gan_loss == "bce":
                    adv_g = bce(d_fake_g, torch.ones_like(d_fake_g))
                else:  # wgan: maximize critic score on fakes
                    adv_g = -d_fake_g.mean()
                loss_g = adv_g
                if mle_lambda > 0:
                    # Teacher-forced NLL on the real batch anchors G to the data
                    # distribution, so it can't drift toward unrealistic lengths.
                    logits = model(real[:, :-1], cc, tb, driver_idx=di)  # (b, L-1, V)
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
                all_g_batch_losses.append(float(adv_g.item()))
                fake_len_sum += float(fake_len.float().sum().item())
                fake_len_cnt += int(fake_len.numel())
            postfix = {
                "g": f"{sum(g_batch) / max(1, len(g_batch)):.3f}",
                "d": f"{sum(d_batch) / max(1, len(d_batch)):.3f}",
                "tau": f"{tau:.2f}",
            }
            if mle_nll is not None:
                postfix["mle"] = f"{float(mle_nll.item()):.3f}"
            bar.update(1, **postfix)
        bar.close()
        g_losses.append(sum(g_batch) / max(1, len(g_batch)))
        d_losses.append(sum(d_batch) / max(1, len(d_batch)))
        if progress:
            fake_len_mean = fake_len_sum / max(1, fake_len_cnt)
            print(
                f"[adv epoch {epoch + 1}/{epochs}] mean length "
                f"real={real_len_mean:.1f} fake={fake_len_mean:.1f} "
                f"(if fake >> real, the critic may be cheating on length)",
                flush=True,
            )

    return {
        "g_losses": g_losses,
        "d_losses": d_losses,
        "g_batch_losses": all_g_batch_losses,
        "d_batch_losses": all_d_batch_losses,
    }
