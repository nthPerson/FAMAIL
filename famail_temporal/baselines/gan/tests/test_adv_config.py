"""Sanity checks for the Phase-3 adversarial hyperparameters."""
from famail_temporal.baselines.gan import config as gc


def test_adversarial_constants_present_and_sane():
    assert gc.ADV_EPOCHS >= 1
    assert gc.ADV_LR_G > 0 and gc.ADV_LR_D > 0
    assert gc.ADV_BATCH_SIZE >= 1
    # Temperature is annealed downward toward (but never to) zero.
    assert gc.GUMBEL_TAU_START >= gc.GUMBEL_TAU_END > 0
    assert gc.D_HIDDEN_DIM >= 1
