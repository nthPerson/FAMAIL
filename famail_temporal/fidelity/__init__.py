"""Fidelity term: discriminator-based realism check."""

from famail_temporal.fidelity.context import (
    MultiStreamData,
    MultiStreamContextBuilder,
)
from famail_temporal.fidelity.model import (
    FeatureNormalizer,
    SiameseLSTMEncoder,
    ProfileEncoder,
    MultiStreamSiameseDiscriminator,
)
from famail_temporal.fidelity.checkpoint import (
    load_discriminator,
    MissingArchitectureConfig,
)
from famail_temporal.fidelity.compute import compute_ffidelity

__all__ = [
    "MultiStreamData", "MultiStreamContextBuilder",
    "FeatureNormalizer", "SiameseLSTMEncoder", "ProfileEncoder",
    "MultiStreamSiameseDiscriminator",
    "load_discriminator", "MissingArchitectureConfig",
    "compute_ffidelity",
]
