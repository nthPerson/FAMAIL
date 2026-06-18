"""Maximum-likelihood (next-token) training for the trajectory LSTM."""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.progress import Progress


def _token_budget_batches(perm, lengths, *, batch_size, max_batch_tokens):
    """Yield index batches from `perm` (a list of indices) so that, per batch,
    len(batch) * max(lengths in batch) <= max_batch_tokens, with at most
    batch_size indices per batch. A single trajectory longer than the budget
    forms its own batch (never dropped). `lengths[i]` is the token length of
    sequence i. Deterministic given `perm`.
    """
    i = 0
    n = len(perm)
    while i < n:
        batch = []
        cur_max = 0
        while i < n and len(batch) < batch_size:
            cand = perm[i]
            new_max = max(cur_max, lengths[cand])
            if batch and new_max * (len(batch) + 1) > max_batch_tokens:
                break
            batch.append(cand)
            cur_max = new_max
            i += 1
        yield batch


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
    driver_idxs: List[int] | None = None,
    max_batch_tokens: int | None = None,
) -> Dict[str, List[float]]:
    """Train `model` by next-token cross-entropy.

    Returns a dict with two keys:
    - ``"epoch_losses"``: per-epoch mean loss list (one value per epoch), same
      as the list this function previously returned directly.
    - ``"batch_losses"``: flat list of every batch's loss in global-step order
      across all epochs (length = epochs × batches_per_epoch). Useful for
      plotting fine-grained training curves.

    Teacher forcing: predict tokens[1:] from tokens[:-1]. PAD positions are
    ignored by the loss. ``progress=True`` shows a per-epoch loss bar.

    When ``max_batch_tokens`` is set, minibatches are formed greedily from the
    shuffled permutation so that ``len(batch) * max_len_in_batch <=
    max_batch_tokens`` (a single over-budget trajectory forms its own batch);
    ``batch_size`` remains an upper cap on count. When ``None`` (default), the
    original fixed-``batch_size`` slicing is used unchanged.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=gc.PAD)
    n = len(sequences)
    lengths = [len(s) for s in sequences]
    epoch_losses: List[float] = []
    all_batch_losses: List[float] = []

    for epoch in range(epochs):
        perm = torch.randperm(n)
        batch_losses: List[float] = []
        if max_batch_tokens is None:
            perm_list = perm.tolist()
            batches = [
                perm_list[start : start + batch_size]
                for start in range(0, n, batch_size)
            ]
        else:
            batches = list(_token_budget_batches(
                perm.tolist(), lengths,
                batch_size=batch_size, max_batch_tokens=max_batch_tokens,
            ))
        with Progress(
            len(batches), f"MLE epoch {epoch + 1}/{epochs}", enabled=progress,
        ) as bar:
            for idx in batches:
                batch = _pad_batch([sequences[i] for i in idx], device)
                ctx_cell = torch.tensor(
                    [contexts[i][0] for i in idx], dtype=torch.long, device=device,
                )
                ctx_tblock = torch.tensor(
                    [contexts[i][1] for i in idx], dtype=torch.long, device=device,
                )
                di = (
                    torch.tensor(
                        [driver_idxs[i] for i in idx], dtype=torch.long, device=device,
                    )
                    if driver_idxs is not None else None
                )
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp, ctx_cell, ctx_tblock, driver_idx=di)  # (B, L-1, V)
                loss = loss_fn(
                    logits.reshape(-1, gc.VOCAB_SIZE), tgt.reshape(-1),
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.item()))
                all_batch_losses.append(float(loss.item()))
                bar.update(1, loss=f"{sum(batch_losses) / len(batch_losses):.3f}")
        epoch_losses.append(sum(batch_losses) / len(batch_losses))
    return {"epoch_losses": epoch_losses, "batch_losses": all_batch_losses}
