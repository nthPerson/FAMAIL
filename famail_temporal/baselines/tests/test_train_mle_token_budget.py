import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.train_mle import train_mle, _token_budget_batches

DEV = torch.device("cpu")


def test_token_budget_batches_splits_long_trajectory():
    # lengths: three short (2) + one long (10); budget 8 tokens, cap 32 count.
    perm = [0, 1, 2, 3]
    lengths = [2, 2, 2, 10]
    batches = list(_token_budget_batches(perm, lengths, batch_size=32, max_batch_tokens=8))
    # short ones group while count*maxlen <= 8: [0,1,2] -> 3*2=6 ok, adding 3 (len10) -> 4*10>8 stop
    assert batches[0] == [0, 1, 2]
    # the long one forms its own batch (10 > 8 alone is allowed)
    assert batches[1] == [3]


def test_token_budget_batches_respects_count_cap():
    perm = list(range(10))
    lengths = [1] * 10
    batches = list(_token_budget_batches(perm, lengths, batch_size=4, max_batch_tokens=10_000))
    assert all(len(b) <= 4 for b in batches)
    assert sum(len(b) for b in batches) == 10


def test_none_path_unchanged():
    """max_batch_tokens=None trains via the original fixed-batch path."""
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    seqs = [[gc.BOS, 0, 1, gc.EOS], [gc.BOS, 2, 3, gc.EOS], [gc.BOS, 1, 0, gc.EOS]]
    ctx = [(0, 0), (2, 1), (1, 0)]
    out = train_mle(m, seqs, ctx, epochs=1, lr=1e-3, batch_size=2, device=DEV)
    assert len(out["epoch_losses"]) == 1


def test_budget_path_trains_full_corpus_shapes():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    seqs = [[gc.BOS, 0, 1, gc.EOS]] * 3 + [[gc.BOS] + list(range(8)) + [gc.EOS]]
    ctx = [(0, 0)] * 4
    out = train_mle(
        m, seqs, ctx, epochs=1, lr=1e-3, batch_size=32, device=DEV,
        driver_idxs=[0, 1, 0, 1], max_batch_tokens=8,
    )
    assert len(out["epoch_losses"]) == 1 and len(out["batch_losses"]) >= 1
