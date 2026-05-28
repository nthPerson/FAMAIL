"""Real-vs-fake LSTM critic over the grid-cell vocabulary.

Mirrors the generator's representation (its own nn.Embedding over the same
VOCAB) so it can score BOTH hard real token sequences (forward_ids) and
differentiable Gumbel-softmax fake sequences (forward_soft, via
soft_onehot @ embed.weight). One realism logit per sequence, read off the
last valid timestep (BCEWithLogits convention: real = 1, fake = 0).
Unconditioned (Phase-3 simplification).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc


class SequenceCritic(nn.Module):
    def __init__(
        self,
        vocab_size: int = gc.VOCAB_SIZE,
        embed_dim: int = gc.EMBED_DIM,
        hidden_dim: int = gc.D_HIDDEN_DIM,
        n_layers: int = gc.N_LAYERS,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=gc.PAD)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def _forward_embed(
        self, embedded: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = self.lstm(embedded)                          # (B, L, H)
        idx = (lengths - 1).clamp(min=0)                      # last valid step
        last = out[torch.arange(out.size(0), device=out.device), idx]  # (B, H)
        return self.head(last).squeeze(-1)                    # (B,)

    def forward_ids(
        self, token_ids: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Score hard real sequences. token_ids: (B, L) long."""
        return self._forward_embed(self.embed(token_ids), lengths)

    def forward_soft(
        self, soft_onehots: torch.Tensor, lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Score soft generated sequences. soft_onehots: (B, L, VOCAB_SIZE)."""
        embedded = soft_onehots @ self.embed.weight           # (B, L, E)
        return self._forward_embed(embedded, lengths)
