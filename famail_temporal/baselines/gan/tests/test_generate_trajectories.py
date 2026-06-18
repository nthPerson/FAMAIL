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
