"""Autoregressive sampling and demand-grid aggregation for generations."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.sequences import unflat_cell
from famail_temporal.baselines.gan.progress import Progress
from famail_temporal.baselines.datasets import pickup_mass
from famail_temporal.data.loader import DataBundle


@torch.no_grad()
def sample_trajectory_cells(
    model: TrajectoryLSTM, ctx_cell: int, ctx_tblock: int,
    *, max_len: int, device: torch.device, temperature: float = 1.0,
) -> List[int]:
    """Sample one trajectory's cell ids (BOS/EOS/specials stripped).

    Autoregressive multinomial decode from BOS; stops at EOS or max_len.
    Only in-vocabulary *cell* ids (< N_CELLS) are kept. Uses the generator's
    single-step decode (carried LSTM state) so cost is O(max_len), not
    O(max_len^2).
    """
    model.to(device).train(False)   # inference mode (no dropout/grad)
    cc = torch.tensor([ctx_cell], dtype=torch.long, device=device)
    tb = torch.tensor([ctx_tblock], dtype=torch.long, device=device)
    prev = gc.BOS
    hidden = None
    cells: List[int] = []
    for _ in range(max_len):
        tok = torch.tensor([prev], dtype=torch.long, device=device)
        logits, hidden = model.step(tok, cc, tb, hidden)  # (1, V), state
        probs = torch.softmax(logits[0] / temperature, dim=-1)
        nxt = int(torch.multinomial(probs, 1).item())
        if nxt == gc.EOS:
            break
        if nxt < gc.N_CELLS:                              # ignore stray specials
            cells.append(nxt)
        prev = nxt
    return cells


@torch.no_grad()
def sample_terminal_cells_batched(
    model: TrajectoryLSTM, ctx_cells: torch.Tensor, ctx_tblocks: torch.Tensor,
    *, max_len: int, device: torch.device, temperature: float = 1.0,
) -> torch.Tensor:
    """Batched autoregressive decode -> each row's terminal (last) cell id.

    Decodes ``ctx_cells.shape[0]`` rollouts in parallel from BOS, carrying the
    LSTM state. Each row records its most recent in-vocabulary cell while it is
    still active and freezes once it samples EOS; a row that never emits a cell
    falls back to its start cell. Returns a (B,) long tensor of terminal cells.
    Equivalent in contract to per-context ``sample_trajectory_cells`` + taking
    the last cell, but processes the whole batch per step (far faster on GPU).
    """
    model.to(device).train(False)
    cc = ctx_cells.to(device)
    tb = ctx_tblocks.to(device)
    B = cc.shape[0]
    prev = torch.full((B,), gc.BOS, dtype=torch.long, device=device)
    hidden = None
    terminal = cc.clone()                                  # fallback: start cell
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        logits, hidden = model.step(prev, cc, tb, hidden)  # (B, V), state
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(1)       # (B,)
        record = (~done) & (nxt < gc.N_CELLS)              # active + a real cell
        terminal = torch.where(record, nxt, terminal)
        done = done | (nxt == gc.EOS)
        prev = nxt
        if bool(done.all()):
            break
    return terminal


def generate_pickups(
    model: TrajectoryLSTM, contexts: List[Tuple[int, int]],
    *, max_len: int, device: torch.device,
    gen_batch_size: int = gc.GEN_BATCH_SIZE, progress: bool = False,
) -> List[Tuple[int, int, int]]:
    """One rollout per context; pickup = terminal cell, t_block = context block.

    Decodes ``gen_batch_size`` contexts in parallel per step (the old batch-1
    loop was the slowest phase). If a rollout produces no cells it falls back to
    the start cell, so every context yields a pickup (keeps the generated grid
    corpus-matched). ``progress=True`` shows a bar over contexts with an ETA.
    """
    out: List[Tuple[int, int, int]] = []
    bar = Progress(len(contexts), "generating rollouts", enabled=progress)
    for start in range(0, len(contexts), gen_batch_size):
        chunk = contexts[start : start + gen_batch_size]
        cc = torch.tensor([c for c, _ in chunk], dtype=torch.long, device=device)
        tb = torch.tensor([t for _, t in chunk], dtype=torch.long, device=device)
        terminals = sample_terminal_cells_batched(
            model, cc, tb, max_len=max_len, device=device,
        )
        for (_, ctx_tblock), term in zip(chunk, terminals.tolist()):
            x, y = unflat_cell(term)
            out.append((x, y, ctx_tblock))
        bar.update(len(chunk))
    bar.close()
    return out


def pickups_to_pickup_3d(
    bundle: DataBundle, pickups: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Aggregate generated pickups into a mean-hourly demand grid.

    Each pickup adds pickup_mass(t_block) at its (cell, t_block), mirroring the
    editing modifier's accounting so the generated grid is scale-comparable to
    bundle.pickup_3d.

    Pickups outside the bundle's grid are skipped. In production the generator's
    cell vocabulary is derived from config.GRID_DIMS, which equals the real
    bundle's grid, so the guard never fires. It only matters when the bundle
    grid is smaller than the vocabulary (e.g. the small synthetic test bundle),
    where it prevents an out-of-bounds index.
    """
    grid = np.zeros_like(bundle.pickup_3d)
    gx, gy, n_t = grid.shape
    for (x, y, t_block) in pickups:
        if 0 <= x < gx and 0 <= y < gy and 0 <= t_block < n_t:
            grid[x, y, t_block] += pickup_mass(bundle, t_block)
    return grid
