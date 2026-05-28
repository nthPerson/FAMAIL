"""Conditional autoregressive LSTM over the grid-cell vocabulary.

forward() returns next-token logits for teacher-forced MLE training. The
conditioning context (start cell + start time-block) is injected by adding a
context embedding to every input-token embedding. A Gumbel-softmax sampling
path can be layered on later (Phase 3) without changing this interface.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc


class TrajectoryLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int = gc.VOCAB_SIZE,
        n_tblocks: int = gc.N_TBLOCKS,
        embed_dim: int = gc.EMBED_DIM,
        hidden_dim: int = gc.HIDDEN_DIM,
        n_layers: int = gc.N_LAYERS,
    ):
        super().__init__()
        self.cell_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=gc.PAD)
        self.tblock_embed = nn.Embedding(n_tblocks, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=n_layers, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, L) long input token ids
        ctx_cell: torch.Tensor,    # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,  # (B,) long start time-block ids
    ) -> torch.Tensor:
        x = self.cell_embed(tokens)                                   # (B, L, E)
        ctx = self.cell_embed(ctx_cell) + self.tblock_embed(ctx_tblock)  # (B, E)
        x = x + ctx.unsqueeze(1)                                      # broadcast
        out, _ = self.lstm(x)                                         # (B, L, H)
        return self.head(out)                                        # (B, L, V)

    def step_embed(
        self,
        input_embed: torch.Tensor,  # (B, E) precomputed input-token embedding
        ctx_cell: torch.Tensor,     # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,   # (B,) long start time-block ids
        hidden=None,                # (h, c) LSTM state from the previous step
    ):
        """Single-step decode from a precomputed input embedding.

        Used by the Gumbel-softmax rollout, where the next input is a
        differentiable soft embedding (soft_onehot @ cell_embed.weight) rather
        than a hard token id. Carries the recurrent state for O(L) decode.

        Note: cell_embed's padding_idx zeroing is NOT applied to input_embed
        (it only zeros a hard-index lookup); callers are responsible for
        masking PAD positions if they matter.
        """
        ctx = self.cell_embed(ctx_cell) + self.tblock_embed(ctx_tblock)  # (B, E)
        x = (input_embed + ctx).unsqueeze(1)                          # (B, 1, E)
        out, hidden = self.lstm(x, hidden)                            # (B, 1, H)
        return self.head(out[:, -1]), hidden                          # (B, V), state

    def step(
        self,
        token: torch.Tensor,       # (B,) long current token id
        ctx_cell: torch.Tensor,    # (B,) long start-cell ids
        ctx_tblock: torch.Tensor,  # (B,) long start time-block ids
        hidden=None,               # (h, c) LSTM state from the previous step
    ):
        """Single-step decode from a token id: next-token logits + LSTM state.

        Equivalent to slicing the last position of forward() over the full
        prefix (LSTM is a recurrence), with the same per-step additive
        conditioning. Delegates to step_embed after embedding the token.
        """
        return self.step_embed(self.cell_embed(token), ctx_cell, ctx_tblock, hidden)
