"""Load a pre-trained MultiStreamSiameseDiscriminator checkpoint.

The checkpoint is treated as an opaque artifact. Canonical location:
    discriminator_checkpoints/default/best.pt

To substitute, edit config.DISCRIMINATOR_CHECKPOINT_FILENAME.
"""

from __future__ import annotations
from pathlib import Path

import torch

from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


class MissingArchitectureConfig(RuntimeError):
    pass


def load_discriminator(checkpoint_path: Path) -> MultiStreamSiameseDiscriminator:
    """Load weights, switch to inference mode, freeze parameters.

    The checkpoint should be a dict with:
      - 'model_state_dict': PyTorch state dict
      - 'architecture_config': kwargs for MultiStreamSiameseDiscriminator.__init__

    If 'architecture_config' is missing, the default constructor is used; if
    shapes mismatch, MissingArchitectureConfig is raised with remediation text.
    """
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False,
    )

    arch_config = checkpoint.get("architecture_config", None)
    if arch_config is None:
        model = MultiStreamSiameseDiscriminator()
    else:
        model = MultiStreamSiameseDiscriminator(**arch_config)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        if arch_config is None:
            raise MissingArchitectureConfig(
                "Checkpoint state dict does not match default architecture. "
                "The checkpoint is missing 'architecture_config'. Add it "
                "via a one-time preprocessing step."
            ) from e
        raise

    model.train(False)  # inference mode
    for p in model.parameters():
        p.requires_grad = False
    return model
