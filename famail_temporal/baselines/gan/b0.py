"""B0 baseline: train an MLE trajectory generator on a dataset, generate
rollouts, and measure the generations' data-level fairness against the corpus.

The B0 claim: a generative model trained on biased data reproduces the bias
in its generations (generated fairness ~ corpus fairness, possibly worse via
mode collapse). This module also IS the reusable train->generate->grid->
fairness pipeline that the filtered/edited variants and the adversarial layer
build on.
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
from famail_temporal.baselines.gan.rollout import (
    generate_pickups, pickups_to_pickup_3d,
)
from famail_temporal.baselines.metrics import data_level_fairness


def run_b0(
    bundle: DataBundle, *,
    epochs: int = gc.MLE_EPOCHS,
    max_len: int = gc.MAX_GEN_LEN,
    device: torch.device | None = None,
    seed: int = 0,
) -> dict:
    """Train on bundle.trajectories, generate one rollout per trajectory's
    context, and return generated vs corpus fairness."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(seed)

    sequences = [trajectory_to_tokens(t) for t in bundle.trajectories]
    contexts = [trajectory_context(t) for t in bundle.trajectories]

    model = TrajectoryLSTM().to(device)
    train_mle(
        model, sequences, contexts,
        epochs=epochs, lr=gc.MLE_LR, batch_size=gc.MLE_BATCH_SIZE, device=device,
    )

    pickups = generate_pickups(model, contexts, max_len=max_len, device=device)
    gen_grid = pickups_to_pickup_3d(bundle, pickups)

    return {
        "generated": data_level_fairness(bundle, pickup_3d=gen_grid),
        "corpus": data_level_fairness(bundle),
        "n_generated": len(pickups),
    }
