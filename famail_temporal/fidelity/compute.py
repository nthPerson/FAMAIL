"""
Compute F_fidelity = Discriminator(tau, tau_prime).

cuDNN workaround: cuDNN's RNN backward requires training mode, but we
need inference-mode behavior while allowing gradient flow through the
LSTM for ST-iFGSM. Disabling cuDNN for the forward pass uses the
pure-PyTorch LSTM implementation, which supports backward in inference
mode.
"""

from __future__ import annotations
from typing import Dict, Tuple

import torch


def compute_ffidelity(
    discriminator: torch.nn.Module,
    tau_features: torch.Tensor,
    tau_prime_features: torch.Tensor,
    multi_stream_kwargs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, dict]:
    """Forward the discriminator; return F_fidelity in [0, 1] + debug dict.

    When multi_stream_kwargs contains 'x1' and 'x2', those 4D seeking
    tensors replace tau_features/tau_prime_features. Other kwargs pass through.
    """
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        if "x1" in multi_stream_kwargs and "x2" in multi_stream_kwargs:
            x1 = multi_stream_kwargs["x1"]
            x2 = multi_stream_kwargs["x2"]
            extra = {
                k: v for k, v in multi_stream_kwargs.items()
                if k not in {"x1", "x2"}
            }
            similarity = discriminator(x1, x2, **extra)
        else:
            similarity = discriminator(tau_features, tau_prime_features)

    f_fidelity = similarity.mean() if similarity.dim() > 0 else similarity
    f_fidelity = torch.clamp(f_fidelity, 0.0, 1.0)
    return f_fidelity, {"similarity_raw": float(similarity.mean().detach())}
