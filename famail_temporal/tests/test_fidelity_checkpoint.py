"""Tests for fidelity.checkpoint."""
import pytest

from famail_temporal import config
from famail_temporal.fidelity.checkpoint import load_discriminator


@pytest.mark.slow
def test_load_discriminator_inference_mode():
    ckpt_path = (
        config.DISCRIMINATOR_CHECKPOINT_DIR
        / config.DISCRIMINATOR_CHECKPOINT_FILENAME
    )
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not present at {ckpt_path}")
    model = load_discriminator(ckpt_path)
    assert not model.training
    for p in model.parameters():
        assert not p.requires_grad
