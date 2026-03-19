"""
Multi-stream context builder for V3 discriminator integration.

Prepares driving trajectory, seeking trajectory, and profile feature tensors
for the V3 MultiStreamSiameseDiscriminator during trajectory modification.

Architecture Design Decisions
=============================

Decision 1: Same-Driver Branch Construction
    Both branches of the Siamese comparison represent the SAME driver.
    driving_1 == driving_2 and profile_1 == profile_2. Only the seeking stream
    differs (original vs modified). The fidelity term asks "does the modified
    trajectory still look like it came from the same driver as the original?"

Decision 2: Seeking Fill Strategy = "sample"
    The V3 model expects N=5 seeking trajectories per branch. We fill as:
    - Slot 0: the target trajectory (original in branch 1, modified in branch 2)
    - Slots 1-4: 4 additional seeking trajectories from the SAME driver,
      sampled from the multi-stream seeking_trajs data (preferably same calendar day)
    - Slots 1-4 are IDENTICAL in both branches — only slot 0 changes.
    This keeps the model in-distribution (trained on batches of 5 per driver-day).

Decision 3: Coordinate Conversion
    The modifier operates in 0-indexed coords [0-47, 0-89]. The V3 discriminator
    was trained on 1-indexed data [1-48, 1-90]. We add +1 to x,y when converting
    modifier trajectories for V3 input. Context trajectories from multi-stream
    extracted data are already 1-indexed and need no conversion.

Decision 4: Gradient Flow Through Slot 0 Only
    Only the modified trajectory tensor (slot 0 of x2) maintains requires_grad=True.
    Context trajectories (slots 1-4), driving, and profile are detached constants.
    The ST-iFGSM gradient flows only through the trajectory being optimized.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import at module level to avoid circular imports; used only for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .trajectory import Trajectory


class MultiStreamContextBuilder:
    """Builds driving/profile/seeking kwargs for V3 discriminator.

    For fidelity computation, both branches compare the SAME driver:
    - Branch 1 (x1): original seeking trajectory + driver's context
    - Branch 2 (x2): modified seeking trajectory + SAME driver's context

    All driving and profile tensors are identical across branches.
    Only the seeking stream slot 0 differs (original vs modified).
    """

    def __init__(
        self,
        driving_trajs: Dict[int, List],
        seeking_trajs: Dict[int, List],
        profile_features: Dict[int, np.ndarray],
        seeking_days: Optional[Dict[int, List[int]]] = None,
        driving_days: Optional[Dict[int, List[int]]] = None,
        n_trajs: int = 5,
        fill_strategy: str = 'sample',
        device: str = 'cpu',
        seed: int = 42,
    ):
        """
        Args:
            driving_trajs: {driver_idx: [trajectories]}, 1-indexed coords.
                Each trajectory is a list of [x, y, t, d] states.
            seeking_trajs: {driver_idx: [trajectories]}, 1-indexed coords.
            profile_features: {driver_idx: ndarray(11,)}, z-score normalized.
            seeking_days: {driver_idx: [cal_day_index_per_traj]}.
            driving_days: {driver_idx: [cal_day_index_per_traj]}.
            n_trajs: Number of trajectories per stream per branch (default 5).
            fill_strategy: How to fill seeking stream slots:
                'sample' - 1 target + (N-1) context from same driver (recommended)
                'replicate' - copy target trajectory N times
                'single' - 1 target, V3 zero-pads remaining slots
            device: PyTorch device string.
            seed: Random seed for reproducible context sampling.
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MultiStreamContextBuilder")

        self.driving_trajs = driving_trajs
        self.seeking_trajs = seeking_trajs
        self.profile_features = profile_features
        self.seeking_days = seeking_days
        self.driving_days = driving_days
        self.n_trajs = n_trajs
        self.fill_strategy = fill_strategy
        self.device = device
        self._rng = random.Random(seed)

    def build_fidelity_kwargs(
        self,
        trajectory: 'Trajectory',
        modified: 'Trajectory',
    ) -> Dict[str, 'torch.Tensor']:
        """Build all V3 discriminator inputs for fidelity computation.

        The returned dict contains BOTH the seeking streams (x1, x2) and
        the driving/profile streams. When present, these replace the
        tau_features / tau_prime_features that the modifier normally builds.

        Returns dict with keys:
            x1:        [1, N, L, 4]   original seeking (1-indexed)
            x2:        [1, N, L, 4]   modified seeking (1-indexed), slot 0 = modified traj
            mask1:     [1, N, L]
            mask2:     [1, N, L]
            driving_1: [1, N, L_d, 4] driving context (1-indexed)
            driving_2: [1, N, L_d, 4] same as driving_1
            mask_d1:   [1, N, L_d]
            mask_d2:   [1, N, L_d]
            profile_1: [1, 11]
            profile_2: [1, 11]        same as profile_1
        """
        driver_idx = int(trajectory.driver_id)

        # --- Profile features (identical both branches) ---
        profile = self._get_profile_tensor(driver_idx)  # [1, 11]

        # --- Driving trajectories (identical both branches) ---
        driving, mask_d = self._select_driving_trajs(driver_idx)  # [1, N, L_d, 4], [1, N, L_d]

        # --- Seeking trajectories ---
        # Convert modifier trajectories (0-indexed) to 1-indexed for V3
        orig_traj_1idx = self._trajectory_to_1indexed(trajectory)   # [L, 4]
        mod_traj_1idx = self._trajectory_to_1indexed(modified)      # [L, 4]

        if self.fill_strategy == 'sample':
            x1, mask1 = self._build_seeking_with_context(
                driver_idx, orig_traj_1idx, target_requires_grad=False
            )
            x2, mask2 = self._build_seeking_with_context(
                driver_idx, mod_traj_1idx, target_requires_grad=True
            )
        elif self.fill_strategy == 'replicate':
            x1, mask1 = self._build_seeking_replicated(orig_traj_1idx, requires_grad=False)
            x2, mask2 = self._build_seeking_replicated(mod_traj_1idx, requires_grad=True)
        else:  # 'single'
            x1, mask1 = self._build_seeking_single(orig_traj_1idx, requires_grad=False)
            x2, mask2 = self._build_seeking_single(mod_traj_1idx, requires_grad=True)

        return {
            'x1': x1, 'x2': x2,
            'mask1': mask1, 'mask2': mask2,
            'driving_1': driving, 'driving_2': driving.clone(),
            'mask_d1': mask_d, 'mask_d2': mask_d.clone(),
            'profile_1': profile, 'profile_2': profile.clone(),
        }

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def _get_profile_tensor(self, driver_idx: int) -> 'torch.Tensor':
        """Get z-score normalized profile features for driver. Returns [1, 11]."""
        if driver_idx in self.profile_features:
            feat = self.profile_features[driver_idx]
        else:
            # Fallback: zero vector (V3 model uses default embedding for missing)
            feat = np.zeros(11, dtype=np.float32)
        return torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Driving trajectories
    # ------------------------------------------------------------------

    def _select_driving_trajs(
        self, driver_idx: int,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Sample N driving trajectories for driver. Returns ([1, N, L_d, 4], [1, N, L_d])."""
        trajs = self.driving_trajs.get(driver_idx, [])

        if len(trajs) == 0:
            # No driving data — return zeros (V3 uses driving_default_embedding)
            dummy = torch.zeros(1, self.n_trajs, 1, 4, dtype=torch.float32, device=self.device)
            mask = torch.zeros(1, self.n_trajs, 1, dtype=torch.bool, device=self.device)
            return dummy, mask

        # Sample N trajectories (with replacement if fewer than N available)
        if len(trajs) >= self.n_trajs:
            selected = self._rng.sample(trajs, self.n_trajs)
        else:
            selected = self._rng.choices(trajs, k=self.n_trajs)

        tensor, mask = self._pad_trajs_to_tensor(selected)  # [N, L_d, 4], [N, L_d]
        return tensor.unsqueeze(0), mask.unsqueeze(0)  # [1, N, L_d, 4], [1, N, L_d]

    # ------------------------------------------------------------------
    # Seeking trajectories — fill strategies
    # ------------------------------------------------------------------

    def _build_seeking_with_context(
        self,
        driver_idx: int,
        target_traj: 'torch.Tensor',
        target_requires_grad: bool,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Build N-trajectory seeking input with 1 target + (N-1) context.

        Args:
            driver_idx: Driver index for context sampling.
            target_traj: [L, 4] tensor (1-indexed coords).
            target_requires_grad: Whether slot 0 needs gradient flow.

        Returns:
            (tensor [1, N, L, 4], mask [1, N, L])
        """
        context_trajs_raw = self.seeking_trajs.get(driver_idx, [])

        # Sample N-1 context trajectories
        n_context = self.n_trajs - 1
        if len(context_trajs_raw) >= n_context:
            context_selected = self._rng.sample(context_trajs_raw, n_context)
        elif len(context_trajs_raw) > 0:
            context_selected = self._rng.choices(context_trajs_raw, k=n_context)
        else:
            # No context available — use zeros
            context_selected = [[[0, 0, 0, 1]]] * n_context

        # Pad context trajectories to tensors
        context_tensor, context_mask = self._pad_trajs_to_tensor(context_selected)
        # context_tensor: [N-1, L_ctx, 4], context_mask: [N-1, L_ctx]

        # Find max length across target and context
        target_len = target_traj.shape[0]
        context_len = context_tensor.shape[1]
        max_len = max(target_len, context_len)

        # Pad target to max_len
        if target_len < max_len:
            pad = torch.zeros(max_len - target_len, 4, dtype=torch.float32, device=self.device)
            target_padded = torch.cat([target_traj, pad], dim=0)
        else:
            target_padded = target_traj

        target_mask = torch.zeros(max_len, dtype=torch.bool, device=self.device)
        target_mask[:target_len] = True

        # Pad context to max_len if needed
        if context_len < max_len:
            pad = torch.zeros(
                n_context, max_len - context_len, 4,
                dtype=torch.float32, device=self.device
            )
            context_tensor = torch.cat([context_tensor, pad], dim=1)
            pad_mask = torch.zeros(
                n_context, max_len - context_len,
                dtype=torch.bool, device=self.device
            )
            context_mask = torch.cat([context_mask, pad_mask], dim=1)

        # Stack: slot 0 = target, slots 1..N-1 = context
        # Context is detached (no gradients)
        all_trajs = torch.cat(
            [target_padded.unsqueeze(0), context_tensor.detach()], dim=0
        )  # [N, max_len, 4]
        all_masks = torch.cat(
            [target_mask.unsqueeze(0), context_mask.detach()], dim=0
        )  # [N, max_len]

        if target_requires_grad:
            all_trajs[0].requires_grad_(True)

        return all_trajs.unsqueeze(0), all_masks.unsqueeze(0)  # [1, N, L, 4], [1, N, L]

    def _build_seeking_replicated(
        self,
        target_traj: 'torch.Tensor',
        requires_grad: bool,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Replicate target trajectory N times. Returns ([1, N, L, 4], [1, N, L])."""
        L = target_traj.shape[0]
        # Repeat target N times
        replicated = target_traj.unsqueeze(0).expand(self.n_trajs, -1, -1).clone()  # [N, L, 4]
        mask = torch.ones(self.n_trajs, L, dtype=torch.bool, device=self.device)

        if requires_grad:
            replicated[0].requires_grad_(True)

        return replicated.unsqueeze(0), mask.unsqueeze(0)

    def _build_seeking_single(
        self,
        target_traj: 'torch.Tensor',
        requires_grad: bool,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Pass single trajectory as [1, 1, L, 4]. V3 auto-pads to N."""
        traj = target_traj.unsqueeze(0).unsqueeze(0)  # [1, 1, L, 4]
        mask = torch.ones(1, 1, target_traj.shape[0], dtype=torch.bool, device=self.device)
        if requires_grad:
            traj = traj.clone()
            traj[0, 0].requires_grad_(True)
        return traj, mask

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _trajectory_to_1indexed(self, trajectory: 'Trajectory') -> 'torch.Tensor':
        """Convert a modifier Trajectory (0-indexed) to 1-indexed tensor [L, 4].

        The modifier uses 0-indexed grid coords [0-47, 0-89].
        V3 was trained on 1-indexed [1-48, 1-90].
        We add +1 to x and y coordinates.
        """
        tensor = trajectory.to_tensor().to(self.device)  # [L, 4]
        # Add +1 to x (col 0) and y (col 1) — time_bucket and day_index unchanged
        tensor = tensor.clone()
        tensor[:, 0] += 1
        tensor[:, 1] += 1
        return tensor

    # ------------------------------------------------------------------
    # Utility: pad variable-length trajectories
    # ------------------------------------------------------------------

    def _pad_trajs_to_tensor(
        self,
        trajs: List[List[List[int]]],
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Pad variable-length trajectory lists to uniform tensor.

        Args:
            trajs: List of trajectories, each a list of [x, y, t, d] states.

        Returns:
            (tensor [N, max_len, 4], mask [N, max_len])
            where mask[i, j] = True if state j of traj i is real (not padding).
        """
        n = len(trajs)
        max_len = max(len(t) for t in trajs) if trajs else 1

        tensor = torch.zeros(n, max_len, 4, dtype=torch.float32, device=self.device)
        mask = torch.zeros(n, max_len, dtype=torch.bool, device=self.device)

        for i, traj in enumerate(trajs):
            length = len(traj)
            for j, state in enumerate(traj):
                if len(state) >= 4:
                    tensor[i, j, 0] = float(state[0])
                    tensor[i, j, 1] = float(state[1])
                    tensor[i, j, 2] = float(state[2])
                    tensor[i, j, 3] = float(state[3])
            mask[i, :length] = True

        return tensor, mask
