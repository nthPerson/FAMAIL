"""Standard-adversarial training paradigm (spec B0): MLE pretrain -> Gumbel
adversarial fine-tune -> generate -> demand grid -> data-level fairness.

This completes the spec's "B0 end-to-end" by adding the adversarial stage that
the Phase-2 MLE keystone (b0.py) deferred. FAMAIL and B2 reuse this verbatim by
passing an edited / filtered bundle (Phase 4); only the training data changes.
"""
from __future__ import annotations
import time

import torch

from famail_temporal.data.loader import DataBundle
from famail_temporal.utils.seeding import set_all_seeds
from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.sequences import (
    trajectory_to_tokens, trajectory_context,
)
from famail_temporal.baselines.gan.train_mle import train_mle
from famail_temporal.baselines.gan.train_adversarial import adversarial_finetune
from famail_temporal.baselines.gan.rollout import (
    generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.gan.progress import log_phase
from famail_temporal.baselines.metrics import data_level_fairness


def fit_and_evaluate(
    bundle: DataBundle, *,
    train_trajectories: list | None = None,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    mle_batch_size: int = gc.MLE_BATCH_SIZE,
    adv_batch_size: int = gc.ADV_BATCH_SIZE,
    adv_lr_g: float = gc.ADV_LR_G,
    adv_lr_d: float = gc.ADV_LR_D,
    d_update_every: int = gc.D_UPDATE_EVERY,
    adv_mle_lambda: float = gc.ADV_MLE_LAMBDA,
    gan_loss: str = gc.GAN_LOSS,
    gp_lambda: float = gc.WGAN_GP_LAMBDA,
    n_critic: int = 1,
    adv_max_len: int | None = None,
    gen_batch_size: int = gc.GEN_BATCH_SIZE,
    max_tokens: int | None = gc.MAX_TRAIN_TOKENS,
    device: torch.device | None = None,
    seed: int = 0,
    progress: bool = False,
) -> dict:
    """Train (MLE + adversarial) on bundle.trajectories, generate one rollout
    per real context, and return generated-vs-corpus fairness + loss histories.

    Trajectories whose token sequence exceeds ``max_tokens`` are excluded from
    both training and generation (``max_tokens=None`` disables the filter). The
    corpus has a long length tail (max ~1654 tokens) and the MLE logits tensor
    is (batch, seq_len, VOCAB=4323), so the cap bounds peak memory on small
    GPUs; it drops only ~1% of the corpus (p99 length is 213). ``mle_batch_size``
    / ``adv_batch_size`` are exposed for the same reason. ``progress=True``
    prints phase markers and per-phase bars (training metrics + a generation
    ETA) to stderr.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(seed)

    t0 = time.monotonic()

    def _phase(msg: str) -> None:
        if progress:
            log_phase(t0, msg)

    _phase(f"device={device}")

    train_trajectories = (
        bundle.trajectories if train_trajectories is None else train_trajectories
    )
    if not train_trajectories:
        raise ValueError("fit_and_evaluate requires a non-empty training corpus")
    pairs = [
        (trajectory_to_tokens(t), trajectory_context(t))
        for t in train_trajectories
    ]
    n_all = len(pairs)
    if max_tokens is not None:
        pairs = [(s, c) for (s, c) in pairs if len(s) <= max_tokens]
    if not pairs:
        raise ValueError(
            f"no training trajectories remain after the max_tokens={max_tokens} filter"
        )
    sequences = [s for s, _ in pairs]
    contexts = [c for _, c in pairs]
    _phase(
        f"corpus: {n_all} trajectories; training on {len(sequences)} "
        f"(max_tokens={max_tokens} dropped {n_all - len(sequences)})"
    )

    model = TrajectoryLSTM().to(device)
    _phase(f"MLE pretrain: {mle_epochs} epochs, batch {mle_batch_size}")
    mle_losses = train_mle(
        model, sequences, contexts,
        epochs=mle_epochs, lr=gc.MLE_LR, batch_size=mle_batch_size,
        device=device, progress=progress,
    )
    adv_len = adv_max_len if adv_max_len is not None else max_len
    _phase(
        f"adversarial fine-tune: {adv_epochs} epochs, batch {adv_batch_size}, "
        f"gan_loss={gan_loss}, mle_lambda={adv_mle_lambda}, "
        f"rollout max_len={adv_len}"
    )
    adv_losses = adversarial_finetune(
        model, sequences, contexts,
        epochs=adv_epochs, lr_g=adv_lr_g, lr_d=adv_lr_d,
        batch_size=adv_batch_size, max_len=adv_len,
        tau_start=gc.GUMBEL_TAU_START, tau_end=gc.GUMBEL_TAU_END,
        d_update_every=d_update_every, mle_lambda=adv_mle_lambda,
        gan_loss=gan_loss, gp_lambda=gp_lambda, n_critic=n_critic,
        device=device, progress=progress,
    )

    _phase(f"generating {len(contexts)} rollouts (max_len {max_len})")
    pickups = generate_pickups(
        model, contexts, max_len=max_len, device=device,
        gen_batch_size=gen_batch_size, progress=progress,
    )
    gen_grid = pickups_to_pickup_3d(bundle, pickups)

    _phase("scoring fairness")
    result = {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
        "pickups": pickups,
        "mle_losses": mle_losses,
        "adv_losses": adv_losses,
    }
    _phase(
        f"done: generated f_causal={result['generated']['f_causal']:.4f} "
        f"vs corpus {result['corpus']['f_causal']:.4f}"
    )
    return result
