"""Standard-adversarial training paradigm (spec B0): MLE pretrain -> Gumbel
adversarial fine-tune -> generate -> demand grid -> data-level fairness.

This is the spec's "B0 end-to-end" with the adversarial stage Phase 2 deferred.
FAMAIL and B2 reuse this verbatim by passing an edited / filtered bundle
(Phase 4); only the training data changes.
"""
from __future__ import annotations
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
from famail_temporal.baselines.metrics import data_level_fairness


def fit_and_evaluate(
    bundle: DataBundle, *,
    mle_epochs: int = gc.MLE_EPOCHS,
    adv_epochs: int = gc.ADV_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train (MLE + adversarial) on bundle.trajectories, generate one rollout
    per real context, and return generated-vs-corpus fairness + loss histories.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not bundle.trajectories:
        raise ValueError(
            "fit_and_evaluate requires a non-empty corpus (bundle.trajectories)"
        )
    set_all_seeds(seed)

    sequences = [trajectory_to_tokens(t) for t in bundle.trajectories]
    contexts = [trajectory_context(t) for t in bundle.trajectories]

    model = TrajectoryLSTM().to(device)
    mle_losses = train_mle(
        model, sequences, contexts,
        epochs=mle_epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE,
        device=device,
    )
    adv_losses = adversarial_finetune(
        model, sequences, contexts,
        epochs=adv_epochs, lr_g=gc.ADV_LR_G, lr_d=gc.ADV_LR_D,
        batch_size=gc.ADV_BATCH_SIZE, max_len=max_len,
        tau_start=gc.GUMBEL_TAU_START, tau_end=gc.GUMBEL_TAU_END,
        device=device,
    )

    pickups = generate_pickups(model, contexts, max_len=max_len, device=device)
    gen_grid = pickups_to_pickup_3d(bundle, pickups)

    return {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
        "mle_losses": mle_losses,
        "adv_losses": adv_losses,
    }
