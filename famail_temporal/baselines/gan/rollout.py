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


def generate_pickups(
    model: TrajectoryLSTM, contexts: List[Tuple[int, int]],
    *, max_len: int, device: torch.device, progress: bool = False,
) -> List[Tuple[int, int, int]]:
    """One rollout per context; pickup = terminal cell, t_block = context block.

    If a rollout produces no cells, it falls back to the start cell so every
    context yields a pickup (keeps the generated grid corpus-matched).
    ``progress=True`` shows a bar over contexts (this batch-1 loop is the
    longest phase, so the bar's ETA is the main "not hung" signal).
    """
    out: List[Tuple[int, int, int]] = []
    bar = Progress(len(contexts), "generating rollouts", enabled=progress)
    for (ctx_cell, ctx_tblock) in contexts:
        cells = sample_trajectory_cells(
            model, ctx_cell, ctx_tblock, max_len=max_len, device=device,
        )
        terminal = cells[-1] if cells else ctx_cell
        x, y = unflat_cell(terminal)
        out.append((x, y, ctx_tblock))
        bar.update(1)
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
