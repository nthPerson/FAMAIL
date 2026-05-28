"""
FAMAILObjective — orchestrates F_spatial + F_causal + F_fidelity.

L = alpha_spatial * F_spatial + alpha_causal * F_causal + alpha_fidelity * F_fidelity

Input: soft_pickup_3d (grid_x, grid_y, T) with gradient through one (cell, t) slice.
Output: scalar total objective + per-term dict.

The single grid-to-unit conversion (grid_x, grid_y, T) -> (N,) happens via
bundle.mask_3d. Every fairness module consumes only (N,) vectors. This is the
sole conversion point — no other module in the system touches both grid geometry
and the flat active-unit vector.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from famail_temporal import config
from famail_temporal.data.loader import DataBundle
from famail_temporal.fairness.causal import compute_fcausal_from_compact
from famail_temporal.fairness.spatial import compute_fspatial
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fidelity.compute import compute_ffidelity


class FAMAILObjective(nn.Module):
    """Orchestrator for the three-term FAMAIL objective.

    L = alpha_spatial * F_spatial + alpha_causal * F_causal + alpha_fidelity * F_fidelity

    The forward pass:
    1. Gathers soft_pickup_3d[mask_3d] -> pickup_N (N-vector, carries gradient)
    2. Gathers dropoff_3d[mask_3d] and active_taxis_3d[mask_3d] -> (N-vectors, no grad)
    3. Computes F_spatial from (pickup_N, dropoff_N, active_taxis_N)
    4. Computes g0(D_clamped) with no_grad (frozen function)
    5. Computes F_causal from (pickup_N, active_taxis_N, g0_D_N)
    6. Optionally computes F_fidelity from discriminator(tau, tau')
    7. Returns weighted sum

    Gradient flows: soft_pickup_3d -> pickup_N -> [F_spatial, F_causal] -> total
    """

    def __init__(
        self,
        bundle: DataBundle,
        alpha_spatial: float | None = None,
        alpha_causal: float | None = None,
        alpha_fidelity: float | None = None,
    ):
        super().__init__()
        # Resolve alpha defaults at __init__ time (NOT at function-definition
        # time) so runtime config overrides (e.g. --override ALPHA_FIDELITY=0.1)
        # are respected. Default-arg evaluation happens once at module import,
        # which would silently freeze the values from that moment.
        self.alpha_spatial = config.ALPHA_SPATIAL if alpha_spatial is None else alpha_spatial
        self.alpha_causal = config.ALPHA_CAUSAL if alpha_causal is None else alpha_causal
        self.alpha_fidelity = config.ALPHA_FIDELITY if alpha_fidelity is None else alpha_fidelity

        # Pre-materialize constant tensors as registered buffers
        self.register_buffer("mask_3d", torch.from_numpy(bundle.mask_3d))
        self.register_buffer("dropoff_3d", torch.from_numpy(bundle.dropoff_3d).float())
        self.register_buffer("active_taxis_3d", torch.from_numpy(bundle.active_taxis_3d).float())
        # Pre-flattened active-unit vectors. These are constant for the entire
        # run (bundle data never changes inside an optimizer pass), so gathering
        # them once at __init__ rather than every forward saves an N-element
        # indexed-gather × 2 per iter at production scale (N=34,524).
        # ``pickup_N`` still has to be re-gathered per forward because it carries
        # the gradient through ``soft_pickup_3d``.
        self.register_buffer(
            "dropoff_N",
            torch.from_numpy(bundle.dropoff_3d[bundle.mask_3d].copy()).float(),
        )
        self.register_buffer(
            "active_taxis_N",
            torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d].copy()).float(),
        )

        # Hat-matrix building blocks for F_causal (compact FWL form — O(Np)
        # memory instead of O(N²)). At N=34,524 (T=24) the dense form is
        # ~19 GB; compact form is ~1 MB. See fairness/hat_matrices.py for
        # the algebraic identity.
        tensors = hat_matrices_to_torch(bundle.hat_matrices)
        self.register_buffer("X_demo", tensors['X_demo'])
        self.register_buffer("XtX_inv", tensors['XtX_inv'])

        self.g0_func = bundle.g0_func
        self.discriminator = bundle.discriminator

    def forward(
        self,
        soft_pickup_3d: torch.Tensor,
        tau_features: Optional[torch.Tensor] = None,
        tau_prime_features: Optional[torch.Tensor] = None,
        multi_stream_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute L = weighted sum of F_spatial, F_causal, F_fidelity.

        Args:
            soft_pickup_3d: (grid_x, grid_y, T) tensor — the soft pickup counts
                with gradient from soft cell assignment. THIS is the input the
                ST-iFGSM loop optimizes.
            tau_features: Optional trajectory features for discriminator branch 1.
            tau_prime_features: Optional trajectory features for discriminator branch 2.
            multi_stream_kwargs: Optional dict of V3 multi-stream inputs for
                discriminator (when present, replaces tau_features/tau_prime_features).

        Returns:
            (total, terms) where total is a scalar tensor (differentiable) and
            terms is a dict with per-term values and debug info.
        """
        device = soft_pickup_3d.device
        mask = self.mask_3d

        # ── THE single grid -> unit conversion point ────────────────────
        pickup_N = soft_pickup_3d[mask]
        # dropoff_N / active_taxis_N are constant across iters — read the
        # pre-flattened buffers instead of re-gathering every forward.
        dropoff_N = self.dropoff_N
        active_taxis_N = self.active_taxis_N

        # ── F_spatial ───────────────────────────────────────────────────
        f_spatial, sp_debug = compute_fspatial(pickup_N, dropoff_N, active_taxis_N)

        # ── g0(D) computed without grad (frozen function, torch-native) ──
        # Stays on ``device`` end-to-end: no .cpu().numpy() round-trip per iter.
        with torch.no_grad():
            D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
            g0_D_N = self.g0_func.eval_torch(D_clamped).to(dtype=torch.float32)

        # ── F_causal ───────────────────────────────────────────────────
        f_causal, cs_debug = compute_fcausal_from_compact(
            demand_N=pickup_N,
            supply_N=active_taxis_N,
            g0_D_N=g0_D_N,
            X_demo=self.X_demo,
            XtX_inv=self.XtX_inv,
        )

        # ── F_fidelity ─────────────────────────────────────────────────
        if self.alpha_fidelity > 0 and tau_features is not None:
            f_fidelity, fd_debug = compute_ffidelity(
                self.discriminator,
                tau_features,
                tau_prime_features,
                multi_stream_kwargs or {},
            )
        else:
            f_fidelity = torch.tensor(0.0, device=device)
            fd_debug = {}

        # ── Weighted combination ────────────────────────────────────────
        total = (
            self.alpha_spatial * f_spatial
            + self.alpha_causal * f_causal
            + self.alpha_fidelity * f_fidelity
        )

        terms = {
            "f_spatial": f_spatial,
            "f_causal": f_causal,
            "f_fidelity": f_fidelity,
            "total": total,
        }
        terms.update({f"debug_spatial_{k}": v for k, v in sp_debug.items()})
        terms.update({f"debug_causal_{k}": v for k, v in cs_debug.items()})
        terms.update({f"debug_fidelity_{k}": v for k, v in fd_debug.items()})

        return total, terms
