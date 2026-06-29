"""generate_trajectories: full cell-sequence capture, index-aligned with contexts."""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.rollout import generate_trajectories


def test_generate_trajectories_one_per_context_indexed_and_clean():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    contexts = [(0, 0), (5, 1), (10, 0), (20, 1), (30, 0)]
    out = generate_trajectories(
        model, contexts, max_len=8, device=torch.device("cpu"),
        gen_batch_size=2,  # exercises multi-batch path
    )
    assert isinstance(out, list) and len(out) == len(contexts)
    for seq in out:
        assert isinstance(seq, list)
        assert len(seq) <= 8
        # only in-vocabulary cell ids; no BOS/EOS/PAD
        assert all(0 <= c < gc.N_CELLS for c in seq)


def test_generate_trajectories_empty_contexts():
    model = TrajectoryLSTM()
    out = generate_trajectories(
        model, [], max_len=8, device=torch.device("cpu"), gen_batch_size=4,
    )
    assert out == []


class _ScriptedModel:
    """Deterministic generator: step 0 emits each row's start cell, then EOS.

    Carries the per-chunk step index through ``hidden`` (the decode loop resets
    ``hidden=None`` at the start of every chunk), so the script restarts for each
    batch. This pins the two correctness properties the random-model test cannot:
    index alignment (output[i] derives from contexts[i]) and EOS-stops-appending
    (each row stops the instant it samples EOS, well before max_len).
    """

    def to(self, device):
        return self

    def train(self, mode):
        return self

    def step(self, prev, cc, tb, hidden, driver_idx=None):
        # ``driver_idx`` accepts the optional driver-conditioning kwarg that
        # rollout.generate_trajectories threads through (added when driver_idxs
        # were wired into rollout); this deterministic stub ignores it.
        b = cc.shape[0]
        step_idx = 0 if hidden is None else hidden
        logits = torch.full((b, gc.VOCAB_SIZE), -1e9)
        if step_idx == 0:
            for i in range(b):
                logits[i, int(cc[i].item())] = 1e9   # emit this row's start cell
        else:
            logits[:, gc.EOS] = 1e9                   # then EOS for every row
        return logits, step_idx + 1


def test_generate_trajectories_index_alignment_and_eos_stops_appending():
    # Start cells are distinct, in-vocab, and span >1 chunk (gen_batch_size=2).
    contexts = [(3, 0), (7, 1), (11, 0)]
    out = generate_trajectories(
        _ScriptedModel(), contexts, max_len=8, device=torch.device("cpu"),
        gen_batch_size=2,
    )
    # Each row: exactly its own start cell (alignment), length 1 not 8 (EOS stop).
    assert out == [[3], [7], [11]]
