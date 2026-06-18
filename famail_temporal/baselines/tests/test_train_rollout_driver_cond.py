import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.gumbel import gumbel_rollout
from famail_temporal.baselines.gan.rollout import (
    generate_trajectories, generate_pickups,
)

DEV = torch.device("cpu")


def _seqs():
    # tiny in-vocab sequences: [BOS, cell, cell, EOS]
    return [[gc.BOS, 0, 1, gc.EOS], [gc.BOS, 2, 3, gc.EOS], [gc.BOS, 1, 0, gc.EOS]]


def _ctx():
    return [(0, 0), (2, 1), (1, 0)]


def test_train_mle_driver_idxs_runs_and_returns_curves():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    out = train_mle(
        m, _seqs(), _ctx(), epochs=1, lr=1e-3, batch_size=2, device=DEV,
        driver_idxs=[0, 1, 0],
    )
    assert "epoch_losses" in out and "batch_losses" in out
    assert len(out["epoch_losses"]) == 1


def test_train_mle_none_path_unchanged():
    """driver_idxs=None on an unconditioned model trains as before."""
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    out = train_mle(m, _seqs(), _ctx(), epochs=1, lr=1e-3, batch_size=2, device=DEV)
    assert len(out["epoch_losses"]) == 1


def test_gumbel_rollout_accepts_driver_idx():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    cc = torch.tensor([0, 2])
    tb = torch.tensor([0, 1])
    soft, lengths = gumbel_rollout(
        m, cc, tb, max_len=5, tau=1.0, device=DEV,
        driver_idx=torch.tensor([0, 1]),
    )
    assert soft.shape == (2, 5, gc.VOCAB_SIZE)
    assert lengths.shape == (2,)


def test_generate_trajectories_driver_idxs_aligned():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    ctxs = _ctx()
    out = generate_trajectories(
        m, ctxs, max_len=5, device=DEV, gen_batch_size=2, driver_idxs=[0, 1, 0],
    )
    assert len(out) == len(ctxs)


def test_generate_pickups_driver_idxs_aligned():
    torch.manual_seed(0)
    m = TrajectoryLSTM(n_drivers=2).to(DEV)
    ctxs = _ctx()
    out = generate_pickups(
        m, ctxs, max_len=5, device=DEV, gen_batch_size=2, driver_idxs=[0, 1, 0],
    )
    assert len(out) == len(ctxs)


def test_generate_trajectories_none_path_unchanged():
    torch.manual_seed(0)
    m = TrajectoryLSTM().to(DEV)
    ctxs = _ctx()
    out = generate_trajectories(m, ctxs, max_len=5, device=DEV, gen_batch_size=2)
    assert len(out) == len(ctxs)
