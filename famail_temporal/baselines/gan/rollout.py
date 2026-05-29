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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched autoregressive decode -> (terminal cell, generated length) per row.

    Decodes ``ctx_cells.shape[0]`` rollouts in parallel from BOS, carrying the
    LSTM state. Each row records its most recent in-vocabulary cell while it is
    still active and freezes once it samples EOS; a row that never emits a cell
    falls back to its start cell. Returns ``(terminal, gen_len)``, both (B,)
    long: ``terminal`` is the pickup cell, ``gen_len`` is the 1-based step of
    the first EOS (or ``max_len`` if none) — the free-running rollout length,
    the real test of generation quality.
    """
    model.to(device).train(False)
    cc = ctx_cells.to(device)
    tb = ctx_tblocks.to(device)
    B = cc.shape[0]
    prev = torch.full((B,), gc.BOS, dtype=torch.long, device=device)
    hidden = None
    terminal = cc.clone()                                  # fallback: start cell
    gen_len = torch.full((B,), max_len, dtype=torch.long, device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    for t in range(max_len):
        logits, hidden = model.step(prev, cc, tb, hidden)  # (B, V), state
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1).squeeze(1)       # (B,)
        is_eos = nxt == gc.EOS
        record = (~done) & (nxt < gc.N_CELLS)              # active + a real cell
        terminal = torch.where(record, nxt, terminal)
        newly_eos = (~done) & is_eos
        gen_len = torch.where(newly_eos, torch.full_like(gen_len, t + 1), gen_len)
        done = done | is_eos
        prev = nxt
        if bool(done.all()):
            break
    return terminal, gen_len


def generate_pickups(
    model: TrajectoryLSTM, contexts: List[Tuple[int, int]],
    *, max_len: int, device: torch.device,
    gen_batch_size: int = gc.GEN_BATCH_SIZE, progress: bool = False,
) -> List[Tuple[int, int, int]]:
    """One rollout per context; pickup = terminal cell, t_block = context block.

    Decodes ``gen_batch_size`` contexts in parallel per step (the old batch-1
    loop was the slowest phase). If a rollout produces no cells it falls back to
    the start cell, so every context yields a pickup (keeps the generated grid
    corpus-matched). ``progress=True`` shows a bar over contexts with an ETA and
    prints the mean free-running rollout length (vs real ~18) at the end.
    """
    out: List[Tuple[int, int, int]] = []
    gen_len_sum = 0.0
    gen_len_cnt = 0
    bar = Progress(len(contexts), "generating rollouts", enabled=progress)
    for start in range(0, len(contexts), gen_batch_size):
        chunk = contexts[start : start + gen_batch_size]
        cc = torch.tensor([c for c, _ in chunk], dtype=torch.long, device=device)
        tb = torch.tensor([t for _, t in chunk], dtype=torch.long, device=device)
        terminals, gen_lens = sample_terminal_cells_batched(
            model, cc, tb, max_len=max_len, device=device,
        )
        for (_, ctx_tblock), term in zip(chunk, terminals.tolist()):
            x, y = unflat_cell(term)
            out.append((x, y, ctx_tblock))
        gen_len_sum += float(gen_lens.float().sum().item())
        gen_len_cnt += int(gen_lens.numel())
        bar.update(len(chunk))
    bar.close()
    if progress:
        print(
            f"[generation] mean generated length="
            f"{gen_len_sum / max(1, gen_len_cnt):.1f} (real mean ~18; "
            f">> that means the generator free-runs too long)",
            flush=True,
        )
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
