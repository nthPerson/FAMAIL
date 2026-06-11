"""WGAN-GP mode + generator gradient-direction verification.

The direction tests answer Dr. Zhang's first diagnostic (Meeting 37): "is the
generator loss / gradient update direction correct?" For both loss modes,
gradient DESCENT on the generator loss must push the critic's score on fakes
UPWARD (toward 'real'), i.e. d(loss)/d(score) < 0.
"""
import torch
import torch.nn as nn

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.critic import SequenceCritic
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_adversarial import (
    adversarial_finetune, _gradient_penalty,
)


def test_bce_generator_loss_gradient_direction_is_correct():
    # Non-saturating BCE: g_loss = BCE(score, 1). Descent must RAISE the score.
    scores = torch.zeros(4, requires_grad=True)
    loss = nn.BCEWithLogitsLoss()(scores, torch.ones_like(scores))
    loss.backward()
    assert (scores.grad < 0).all()  # -grad step increases the score


def test_wgan_generator_loss_gradient_direction_is_correct():
    # Wasserstein: g_loss = -mean(score). Descent must RAISE the score.
    scores = torch.zeros(4, requires_grad=True)
    loss = -scores.mean()
    loss.backward()
    assert (scores.grad < 0).all()


def test_gradient_penalty_finite_and_nonnegative():
    torch.manual_seed(0)
    critic = SequenceCritic()
    real = torch.randint(0, gc.N_CELLS, (2, 5))
    real_len = torch.tensor([5, 3])
    fake_soft = torch.softmax(torch.randn(2, 4, gc.VOCAB_SIZE), dim=-1)
    fake_len = torch.tensor([4, 2])
    gp = _gradient_penalty(
        critic, real, real_len, fake_soft, fake_len,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(gp)
    assert float(gp.item()) >= 0.0


def test_wgan_adversarial_finetune_smoke():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    sequences = [
        [gc.BOS, 10, 11, gc.EOS],
        [gc.BOS, 20, 21, 22, gc.EOS],
        [gc.BOS, 30, 31, gc.EOS],
        [gc.BOS, 40, 41, 42, 43, gc.EOS],
    ]
    contexts = [(10, 0), (20, 1), (30, 0), (40, 1)]
    out = adversarial_finetune(
        model, sequences, contexts,
        epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=2, max_len=8,
        tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
        gan_loss="wgan-gp", n_critic=2, mle_lambda=0.0,
    )
    assert set(out) == {"g_losses", "d_losses"}
    assert len(out["g_losses"]) == 1 and len(out["d_losses"]) == 1
    assert all(map(torch.isfinite, map(torch.tensor, out["g_losses"])))
    assert all(map(torch.isfinite, map(torch.tensor, out["d_losses"])))


def test_bce_mode_unchanged_smoke():
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    sequences = [[gc.BOS, 10, 11, gc.EOS], [gc.BOS, 20, 21, gc.EOS]]
    contexts = [(10, 0), (20, 1)]
    out = adversarial_finetune(
        model, sequences, contexts,
        epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=2, max_len=8,
        tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
    )
    assert len(out["g_losses"]) == 1 and len(out["d_losses"]) == 1


def test_unknown_gan_loss_raises():
    model = TrajectoryLSTM()
    try:
        adversarial_finetune(
            model, [[gc.BOS, 10, gc.EOS]], [(10, 0)],
            epochs=1, lr_g=1e-4, lr_d=1e-4, batch_size=1, max_len=4,
            tau_start=1.0, tau_end=0.5, device=torch.device("cpu"),
            gan_loss="nope",
        )
        assert False, "expected ValueError"
    except ValueError as e:
        assert "gan_loss" in str(e)
