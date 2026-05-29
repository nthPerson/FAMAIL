"""Maximum-likelihood (next-token) training for the trajectory LSTM."""
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.progress import Progress


def _pad_batch(
    seqs: List[List[int]], device: torch.device,
) -> torch.Tensor:
    """Right-pad a list of token sequences to (B, Lmax) with PAD."""
    lmax = max(len(s) for s in seqs)
    out = torch.full((len(seqs), lmax), gc.PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
    return out


def train_mle(
    model: TrajectoryLSTM,
    sequences: List[List[int]],
    contexts: List[Tuple[int, int]],
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    progress: bool = False,
) -> List[float]:
    """Train `model` by next-token cross-entropy. Returns per-epoch mean loss.

    Teacher forcing: predict tokens[1:] from tokens[:-1]. PAD positions are
    ignored by the loss. ``progress=True`` shows a per-epoch loss bar.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=gc.PAD)
    n = len(sequences)
    n_batches = (n + batch_size - 1) // batch_size
    epoch_losses: List[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        batch_losses: List[float] = []
        with Progress(
            n_batches, f"MLE epoch {epoch + 1}/{epochs}", enabled=progress,
        ) as bar:
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size].tolist()
                batch = _pad_batch([sequences[i] for i in idx], device)
                ctx_cell = torch.tensor(
                    [contexts[i][0] for i in idx], dtype=torch.long, device=device,
                )
                ctx_tblock = torch.tensor(
                    [contexts[i][1] for i in idx], dtype=torch.long, device=device,
                )
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp, ctx_cell, ctx_tblock)         # (B, L-1, V)
                loss = loss_fn(
                    logits.reshape(-1, gc.VOCAB_SIZE), tgt.reshape(-1),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.item()))
                bar.update(1, loss=f"{sum(batch_losses) / len(batch_losses):.3f}")
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    return epoch_losses
