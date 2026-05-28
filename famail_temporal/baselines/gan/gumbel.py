"""Differentiable straight-through Gumbel-softmax rollout.

Decodes a fixed number of steps (max_len) so the batch keeps a static
(B, max_len, V) shape and gradients flow through every step; the first
sampled EOS per row is recorded in `lengths` for downstream masking. The
next-step input is a differentiable soft embedding (y @ cell_embed.weight),
so the recurrence is end-to-end differentiable wrt the generator's params.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def gumbel_rollout(
    model: TrajectoryLSTM,
    ctx_cell: torch.Tensor,     # (B,) long start-cell ids
    ctx_tblock: torch.Tensor,   # (B,) long start time-block ids
    *,
    max_len: int,
    tau: float,
    device: torch.device,
    hard: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (soft_onehots, lengths).

    soft_onehots: (B, max_len, VOCAB_SIZE) per-step straight-through one-hots,
        differentiable wrt model parameters.
    lengths: (B,) long — 1-based index of the first sampled EOS, or max_len.
    """
    cc = ctx_cell.to(device)
    tb = ctx_tblock.to(device)
    B = cc.shape[0]

    prev_embed = model.cell_embed(
        torch.full((B,), gc.BOS, dtype=torch.long, device=device)
    )                                                       # (B, E)
    hidden = None
    steps = []
    ended = torch.zeros(B, dtype=torch.bool, device=device)
    lengths = torch.full((B,), max_len, dtype=torch.long, device=device)

    for t in range(max_len):
        logits, hidden = model.step_embed(prev_embed, cc, tb, hidden)   # (B, V)
        y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)        # (B, V)
        steps.append(y)
        nxt = y.argmax(dim=-1)                                          # (B,)
        newly_ended = (~ended) & (nxt == gc.EOS)
        lengths = torch.where(
            newly_ended, torch.full_like(lengths, t + 1), lengths,
        )
        ended = ended | (nxt == gc.EOS)
        prev_embed = y @ model.cell_embed.weight                       # (B, E)

    soft_onehots = torch.stack(steps, dim=1)                           # (B, L, V)
    return soft_onehots, lengths
