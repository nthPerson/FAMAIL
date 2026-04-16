# FAMAIL-Temporal Implementation Plan — Phases 7–8

> **MODEL REQUIREMENT — OPUS ONLY:** Same as the main plan file.
>
> **Prerequisite:** Phases 1–6 complete.

**Scope:** Phase 7 (Algorithm — soft cell assignment, objective, attribution, modifier) and Phase 8 (Integration tests).

---

## Phase 7: Algorithm (Tasks 26–30)

### Task 26: algorithm/soft_cell_assignment.py — SoftCellAssignment module

**Files:**
- Create: famail_temporal/algorithm/soft_cell_assignment.py
- Create: famail_temporal/tests/test_soft_cell_assignment.py
- Source reference: objective_function/soft_cell_assignment/module.py (existing, already clean)

- [ ] **Step 1: Write failing tests**

    """Tests for algorithm.soft_cell_assignment."""
    import torch

    from famail_temporal.algorithm.soft_cell_assignment import SoftCellAssignment


    def test_soft_assignment_probs_shape():
        s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                               initial_temperature=1.0)
        loc = torch.tensor([[10.3, 20.7]])
        cell = torch.tensor([[10, 20]]).float()
        probs = s(loc, cell)
        # [batch, neighborhood_size, neighborhood_size]
        assert probs.shape == (1, 5, 5)


    def test_soft_assignment_probs_sum_to_one():
        s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                               initial_temperature=1.0)
        loc = torch.tensor([[10.3, 20.7]])
        cell = torch.tensor([[10, 20]]).float()
        probs = s(loc, cell)
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


    def test_soft_assignment_set_temperature():
        s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                               initial_temperature=1.0)
        s.set_temperature(0.2)
        assert float(s.temperature) == 0.2


    def test_gradient_flows_to_loc():
        """Gradient should flow from probs back to loc (requires_grad)."""
        s = SoftCellAssignment(grid_dims=(48, 90), neighborhood_size=5,
                               initial_temperature=1.0)
        loc = torch.tensor([[10.3, 20.7]], requires_grad=True)
        cell = torch.tensor([[10, 20]]).float()
        probs = s(loc, cell)
        probs.sum().backward()
        assert loc.grad is not None
        assert not torch.isnan(loc.grad).any()

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_soft_cell_assignment.py -v

- [ ] **Step 3: Port and write famail_temporal/algorithm/soft_cell_assignment.py**

Open objective_function/soft_cell_assignment/module.py for reference. Copy the SoftCellAssignment class into the new file, replacing imports that reference the parent project. Full module:

    """
    Differentiable soft cell assignment.

    For a continuous pickup location (x, y) in R^2, produces a probability
    distribution over a (2k+1) x (2k+1) neighborhood centered at floor((x, y)).
    Temperature controls sharpness (tau=1 soft, tau=0.1 near-hard).

    Ported from objective_function/soft_cell_assignment/module.py.
    """

    from __future__ import annotations
    from typing import Tuple

    import torch
    import torch.nn as nn

    from famail_temporal import config


    class SoftCellAssignment(nn.Module):
        def __init__(
            self,
            grid_dims: Tuple[int, int] = config.GRID_DIMS,
            neighborhood_size: int = config.SOFT_NEIGHBORHOOD_SIZE,
            initial_temperature: float = config.TAU_MAX,
        ):
            super().__init__()
            assert neighborhood_size % 2 == 1, "neighborhood_size must be odd"
            self.grid_dims = grid_dims
            self.k = neighborhood_size // 2
            self.register_buffer(
                "temperature",
                torch.tensor(float(initial_temperature)),
            )

        def forward(self, loc: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
            """Compute soft probability over the (2k+1) x (2k+1) neighborhood.

            loc: (batch, 2) continuous coordinates, may require_grad.
            cell: (batch, 2) integer cell coordinates (float tensor).
            returns: (batch, 2k+1, 2k+1) probability distribution.
            """
            batch_size = loc.shape[0]
            k = self.k
            ns = 2 * k + 1

            # Build neighborhood cell coordinates relative to cell
            offsets = torch.arange(-k, k + 1, device=loc.device, dtype=loc.dtype)
            dx, dy = torch.meshgrid(offsets, offsets, indexing="ij")
            # (ns, ns, 2) — relative offsets
            rel = torch.stack([dx, dy], dim=-1)

            # Absolute neighborhood cells for each batch entry
            # cell is (batch, 2); rel is (ns, ns, 2)
            # abs_cells: (batch, ns, ns, 2)
            abs_cells = cell.unsqueeze(1).unsqueeze(2) + rel.unsqueeze(0)

            # Squared distance from loc (center of each cell) to loc
            # loc is (batch, 2); abs_cells + 0.5 is the cell center
            loc_exp = loc.unsqueeze(1).unsqueeze(2)
            # Use cell center = abs_cells + 0.5 for distance calc
            dist_sq = ((abs_cells + 0.5) - loc_exp).pow(2).sum(dim=-1)

            # Gaussian kernel with temperature as the softmax temperature
            logits = -dist_sq / (self.temperature + config.EPS)
            # Flatten neighborhood for softmax
            logits_flat = logits.view(batch_size, -1)
            probs_flat = torch.softmax(logits_flat, dim=-1)
            probs = probs_flat.view(batch_size, ns, ns)
            return probs

        def set_temperature(self, tau: float) -> None:
            if tau <= 0:
                raise ValueError(f"Temperature must be > 0, got {tau}")
            self.temperature = torch.tensor(float(tau), device=self.temperature.device)

        def get_annealed_temperature(
            self, iteration: int, total_iterations: int,
            tau_max: float = config.TAU_MAX, tau_min: float = config.TAU_MIN,
        ) -> float:
            if total_iterations <= 1:
                return tau_min
            progress = iteration / (total_iterations - 1)
            return tau_max * (tau_min / tau_max) ** progress

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_soft_cell_assignment.py -v

Expected: 4 passed.

- [ ] **Step 5: Commit**

    git add famail_temporal/algorithm/soft_cell_assignment.py \
            famail_temporal/tests/test_soft_cell_assignment.py
    git commit -m "feat(algorithm): port SoftCellAssignment module"

---

### Task 27: inject_soft_counts_into_3d helper

**Files:**
- Modify: famail_temporal/algorithm/soft_cell_assignment.py
- Modify: famail_temporal/tests/test_soft_cell_assignment.py

- [ ] **Step 1: Append failing tests**

    from famail_temporal.algorithm.soft_cell_assignment import (
        inject_soft_counts_into_3d,
    )


    def test_inject_only_modifies_t_block_slice():
        base = torch.zeros(48, 90, 4)
        base[:, :, 1] = 5.0  # constant in block 1; we modify block 0
        probs = torch.ones(5, 5) / 25.0  # uniform, sums to 1
        out = inject_soft_counts_into_3d(
            base_counts_3d=base,
            probs_2d=probs,
            cell_xy=(10, 20),
            t_block=0,
            k=2,
            pickup_mass=1.0,
        )
        # Block 1 (and others) unchanged
        assert torch.equal(out[:, :, 1], base[:, :, 1])
        # Block 0 has changes in the 5x5 neighborhood only
        changed = (out[:, :, 0] != base[:, :, 0]).sum()
        assert changed == 25


    def test_inject_mass_balance():
        """Total injected mass equals pickup_mass (sum of probs * pickup_mass)."""
        base = torch.zeros(48, 90, 4)
        probs = torch.rand(5, 5)
        probs = probs / probs.sum()  # normalize to sum to 1
        pickup_mass = 0.01  # example mass
        out = inject_soft_counts_into_3d(
            base, probs, cell_xy=(10, 20), t_block=0, k=2, pickup_mass=pickup_mass,
        )
        total_injected = (out[:, :, 0] - base[:, :, 0]).sum()
        assert torch.isclose(total_injected, torch.tensor(pickup_mass), atol=1e-5)


    def test_inject_preserves_gradient():
        """Gradient flows from the injected tensor back to probs."""
        base = torch.zeros(48, 90, 4)
        probs = torch.rand(5, 5, requires_grad=True)
        out = inject_soft_counts_into_3d(
            base, probs, cell_xy=(10, 20), t_block=0, k=2, pickup_mass=1.0,
        )
        out.sum().backward()
        assert probs.grad is not None
        assert not torch.isnan(probs.grad).any()

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_soft_cell_assignment.py -v

- [ ] **Step 3: Append to algorithm/soft_cell_assignment.py**

    def inject_soft_counts_into_3d(
        base_counts_3d: torch.Tensor,
        probs_2d: torch.Tensor,
        cell_xy: Tuple[int, int],
        t_block: int,
        k: int,
        pickup_mass: float,
    ) -> torch.Tensor:
        """Inject probs_2d * pickup_mass into base_counts_3d at slice t_block.

        Uses the delta-tensor pattern so autograd flows cleanly through probs_2d:
            delta = zeros_like(base)
            delta[:, :, t_block] = scatter(probs * pickup_mass)
            return base + delta

        Only cells in the (2k+1, 2k+1) neighborhood of cell_xy in slice t_block
        are modified. Cells outside the grid bounds are silently skipped.
        """
        gx, gy, t_total = base_counts_3d.shape
        assert probs_2d.shape == (2 * k + 1, 2 * k + 1)
        assert 0 <= t_block < t_total

        delta = torch.zeros_like(base_counts_3d)
        cx, cy = cell_xy
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                ni, nj = cx + di, cy + dj
                if 0 <= ni < gx and 0 <= nj < gy:
                    delta[ni, nj, t_block] = (
                        delta[ni, nj, t_block]
                        + probs_2d[di + k, dj + k] * pickup_mass
                    )
        return base_counts_3d + delta

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_soft_cell_assignment.py -v

Expected: 7 passed.

- [ ] **Step 5: Commit**

    git add famail_temporal/algorithm/soft_cell_assignment.py \
            famail_temporal/tests/test_soft_cell_assignment.py
    git commit -m "feat(algorithm): inject_soft_counts_into_3d with delta-tensor pattern"

---

### Task 28: algorithm/objective.py — FAMAILObjective

**Files:**
- Create: famail_temporal/algorithm/objective.py
- Create: famail_temporal/tests/test_objective.py

- [ ] **Step 1: Write failing tests**

    """Tests for algorithm.objective FAMAILObjective."""
    import numpy as np
    import pytest
    import torch
    import torch.nn as nn

    from famail_temporal import config
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.data.active_mask import UnitIndexMap
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.g0_power_basis import G0Function
    from famail_temporal.fairness.hat_matrices import precompute_hat_matrices
    from famail_temporal.fidelity.context import MultiStreamData


    def _make_synthetic_bundle(N_cells_per_block=20, seed=0):
        """Build a minimal DataBundle for testing — small grid, synthetic data."""
        rng = np.random.RandomState(seed)
        gx, gy, t = 8, 8, 4
        # Make N active cells per block — spread evenly in the grid
        mask = np.zeros((gx, gy, t), dtype=bool)
        n_active = N_cells_per_block * t
        positions = rng.choice(gx * gy * t, n_active, replace=False)
        for p in positions:
            cell = p // t
            tb = p % t
            x, y = cell // gy, cell % gy
            mask[x, y, tb] = True

        unit_map = UnitIndexMap.from_mask(mask, grid_shape=(gx, gy))
        N = unit_map.n_units

        pickup_3d = np.zeros((gx, gy, t), dtype=np.float32)
        dropoff_3d = np.zeros((gx, gy, t), dtype=np.float32)
        active_3d = np.zeros((gx, gy, t), dtype=np.float32)

        for i in range(N):
            fc = unit_map.to_flat_cell(i)
            tb = unit_map.to_time_block(i)
            x, y = fc // gy, fc % gy
            pickup_3d[x, y, tb] = rng.uniform(1.0, 5.0)
            dropoff_3d[x, y, tb] = rng.uniform(1.0, 5.0)
            active_3d[x, y, tb] = rng.uniform(1.0, 10.0)

        demographics = rng.randn(N, 3)
        hat = precompute_hat_matrices(
            demands=np.maximum(pickup_3d[mask], config.DEMAND_FLOOR),
            demographic_features=demographics,
            feature_names=["a", "b", "c"],
        )
        g0 = G0Function(
            coefficients=np.array([0.5, 0.1, 0.1, 0.01]),
            d_min=0.01, d_max=10.0,
        )

        bundle = DataBundle(
            pickup_3d=pickup_3d, dropoff_3d=dropoff_3d, active_taxis_3d=active_3d,
            mask_3d=mask, unit_map=unit_map,
            n_hours_per_block=np.array([3, 6, 4, 11], dtype=np.int32),
            n_days=65,
            g0_func=g0, hat_matrices=hat,
            trajectories=[],
            multi_stream=MultiStreamData(
                driving_trajs={}, seeking_trajs={},
                profile_features={}, seeking_days={}, driving_days={},
            ),
            discriminator=nn.Identity(),
        )
        return bundle


    def test_famailobjective_forward_returns_scalar_total():
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)  # skip fidelity
        soft_3d = torch.from_numpy(bundle.pickup_3d).float()
        total, terms = obj(
            soft_pickup_3d=soft_3d,
            tau_features=None,
            tau_prime_features=None,
            multi_stream_kwargs=None,
        )
        assert total.dim() == 0
        assert "f_spatial" in terms
        assert "f_causal" in terms
        assert "f_fidelity" in terms


    def test_famailobjective_metrics_in_unit_interval():
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        soft_3d = torch.from_numpy(bundle.pickup_3d).float()
        total, terms = obj(soft_pickup_3d=soft_3d)
        assert 0.0 <= float(terms["f_spatial"]) <= 1.0
        assert 0.0 <= float(terms["f_causal"]) <= 1.0


    def test_gradient_flows_through_soft_pickup():
        """Gradient should flow from total back to soft_pickup_3d."""
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        soft_3d = torch.from_numpy(bundle.pickup_3d).float().requires_grad_(True)
        total, _ = obj(soft_pickup_3d=soft_3d)
        total.backward()
        assert soft_3d.grad is not None
        assert not torch.isnan(soft_3d.grad).any()
        # Gradient should be non-zero at active-mask locations
        mask_t = torch.from_numpy(bundle.mask_3d)
        assert (soft_3d.grad[mask_t].abs() > 0).any()

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_objective.py -v

- [ ] **Step 3: Write famail_temporal/algorithm/objective.py**

    """
    FAMAILObjective — orchestrates F_spatial + F_causal + F_fidelity.

    Input: soft_pickup_3d (48, 90, T) with gradient through one (cell, t) slice;
           dropoff_3d, active_taxis_3d (constants); optional trajectory tensors
           for fidelity.
    Output: scalar total objective + per-term dict.

    The one-way (48, 90, T) -> (N,) conversion happens via bundle.mask_3d.
    Every fairness module consumes (N,) vectors only.
    """

    from __future__ import annotations
    from typing import Dict, Optional, Tuple

    import numpy as np
    import torch
    import torch.nn as nn

    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.causal import compute_fcausal
    from famail_temporal.fairness.spatial import compute_fspatial
    from famail_temporal.fidelity.compute import compute_ffidelity


    class FAMAILObjective(nn.Module):
        def __init__(
            self,
            bundle: DataBundle,
            alpha_spatial: float = config.ALPHA_SPATIAL,
            alpha_causal: float = config.ALPHA_CAUSAL,
            alpha_fidelity: float = config.ALPHA_FIDELITY,
        ):
            super().__init__()
            self.bundle = bundle
            self.alpha_spatial = alpha_spatial
            self.alpha_causal = alpha_causal
            self.alpha_fidelity = alpha_fidelity

            self.register_buffer(
                "mask_3d",
                torch.from_numpy(bundle.mask_3d),
            )
            self.register_buffer(
                "dropoff_3d",
                torch.from_numpy(bundle.dropoff_3d).float(),
            )
            self.register_buffer(
                "active_taxis_3d",
                torch.from_numpy(bundle.active_taxis_3d).float(),
            )
            self.register_buffer(
                "I_minus_H_demo",
                torch.from_numpy(bundle.hat_matrices["I_minus_H_demo"]).float(),
            )
            self.register_buffer(
                "M",
                torch.from_numpy(bundle.hat_matrices["M"]).float(),
            )
            self.g0_func = bundle.g0_func
            self.discriminator = bundle.discriminator

        def forward(
            self,
            soft_pickup_3d: torch.Tensor,
            tau_features: Optional[torch.Tensor] = None,
            tau_prime_features: Optional[torch.Tensor] = None,
            multi_stream_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, dict]:
            device = soft_pickup_3d.device
            mask = self.mask_3d

            pickup_N = soft_pickup_3d[mask]
            dropoff_N = self.dropoff_3d[mask]
            active_taxis_N = self.active_taxis_3d[mask]

            f_spatial, sp_debug = compute_fspatial(pickup_N, dropoff_N, active_taxis_N)

            # g0(D) computed without grad (frozen function)
            with torch.no_grad():
                D_clamped = torch.clamp(pickup_N, min=config.DEMAND_FLOOR)
                g0_D_np = self.g0_func(D_clamped.detach().cpu().numpy())
                g0_D_N = torch.from_numpy(g0_D_np).float().to(device)

            f_causal, cs_debug = compute_fcausal(
                demand_N=pickup_N,
                supply_N=active_taxis_N,
                g0_D_N=g0_D_N,
                I_minus_H_demo=self.I_minus_H_demo,
                M=self.M,
            )

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
                **{f"debug_spatial_{k}": v for k, v in sp_debug.items()},
                **{f"debug_causal_{k}": v for k, v in cs_debug.items()},
                **{f"debug_fidelity_{k}": v for k, v in fd_debug.items()},
            }
            return total, terms

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_objective.py -v

Expected: 3 passed.

- [ ] **Step 5: Commit**

    git add famail_temporal/algorithm/objective.py famail_temporal/tests/test_objective.py
    git commit -m "feat(algorithm): FAMAILObjective orchestrator with gradient flow"

---

### Task 29: algorithm/attribution.py — per-unit + per-trajectory

**Files:**
- Create: famail_temporal/algorithm/attribution.py
- Create: famail_temporal/tests/test_attribution.py

- [ ] **Step 1: Write failing tests**

    """Tests for algorithm.attribution."""
    import numpy as np
    import torch

    from famail_temporal import config
    from famail_temporal.algorithm.attribution import (
        compute_per_unit_attribution,
        rank_trajectories,
        select_top_k,
    )
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState

    # Reuse the synthetic bundle builder from test_objective
    from famail_temporal.tests.test_objective import _make_synthetic_bundle


    def test_per_unit_attribution_returns_N_vector():
        bundle = _make_synthetic_bundle()
        attribution, signed = compute_per_unit_attribution(bundle)
        assert attribution.shape == (bundle.unit_map.n_units,)
        assert signed.shape == (bundle.unit_map.n_units,)


    def test_rank_trajectories_orders_by_attribution():
        bundle = _make_synthetic_bundle()
        attribution, _ = compute_per_unit_attribution(bundle)

        # Build two synthetic trajectories: one pointing at a high-attribution
        # unit, one at a low-attribution unit.
        high_idx = int(np.argmax(attribution))
        low_idx = int(np.argmin(attribution))
        high_cell = bundle.unit_map.to_flat_cell(high_idx)
        high_t = bundle.unit_map.to_time_block(high_idx)
        low_cell = bundle.unit_map.to_flat_cell(low_idx)
        low_t = bundle.unit_map.to_time_block(low_idx)

        gy = bundle.pickup_3d.shape[1]
        def _make_traj(cell, t_block, traj_id):
            x, y = cell // gy, cell % gy
            # time_bucket must map to t_block; pick a bucket in the block's range
            _, start_hour, _ = config.TIME_BLOCKS[t_block]
            tb = 1 + (start_hour * 12)  # first 5-min bucket of the block's first hour
            states = [
                TrajectoryState(x_grid=0.0, y_grid=0.0, time_bucket=tb, day_index=1),
                TrajectoryState(x_grid=float(x), y_grid=float(y),
                                time_bucket=tb, day_index=1),
            ]
            return Trajectory(trajectory_id=traj_id, driver_id=0, states=states)

        trajs = [_make_traj(high_cell, high_t, 0), _make_traj(low_cell, low_t, 1)]
        ranked = rank_trajectories(trajs, attribution, bundle.unit_map)
        assert ranked[0][0] == 0  # high-attribution traj first


    def test_select_top_k_drops_zero_attribution():
        scored = [(0, 0.5), (1, 0.3), (2, 0.0), (3, -0.1)]
        picks = select_top_k(scored, k=4)
        # Zeros and negatives should be excluded
        assert picks == [0, 1]

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_attribution.py -v

- [ ] **Step 3: Write famail_temporal/algorithm/attribution.py**

    """
    Attribution pipeline: per-unit fairness contribution -> per-trajectory ranking.

    Per-unit attribution comes from fairness.causal.per_unit_attribution, which
    decomposes 1 - F_causal into per-unit contributions summing to r^2_demo.

    Each trajectory inherits the attribution of its pickup's (cell, t) unit.
    """

    from __future__ import annotations
    from typing import List, Tuple

    import numpy as np
    import torch

    from famail_temporal import config
    from famail_temporal.data.active_mask import UnitIndexMap
    from famail_temporal.data.aggregation import (
        hour_to_block_index, time_bucket_to_hour,
    )
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.causal import (
        per_unit_attribution, per_unit_attribution_signed,
    )
    from famail_temporal.utils.trajectory import Trajectory


    def compute_per_unit_attribution(
        bundle: DataBundle,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute unsigned and signed attribution over active units."""
        D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
        S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
        D = torch.clamp(D, min=config.DEMAND_FLOOR)
        Y = S / (D + config.EPS)
        g0_D_np = bundle.g0_func(D.numpy())
        g0_D = torch.from_numpy(g0_D_np).float()
        R = Y - g0_D

        IH = torch.from_numpy(bundle.hat_matrices["I_minus_H_demo"]).float()
        M = torch.from_numpy(bundle.hat_matrices["M"]).float()
        unsigned = per_unit_attribution(R, IH, M).numpy()
        signed = per_unit_attribution_signed(R, IH, M).numpy()
        return unsigned, signed


    def rank_trajectories(
        trajectories: List[Trajectory],
        unit_attribution: np.ndarray,
        unit_map: UnitIndexMap,
    ) -> List[Tuple[int, float]]:
        """Map each trajectory's pickup (cell, t) -> attribution score.

        Returns [(trajectory_idx, score), ...] sorted descending.
        Trajectories in inactive units get score 0 and are placed at the end.
        """
        gy = config.GRID_DIMS[1]
        scored = []
        for i, traj in enumerate(trajectories):
            cx, cy = traj.pickup_cell
            time_bucket = traj.pickup_state.time_bucket
            hour = time_bucket_to_hour(time_bucket)
            t_block = hour_to_block_index(hour)

            flat_cell = cx * gy + cy
            unit_idx = unit_map.from_cell_time(flat_cell, t_block)
            score = float(unit_attribution[unit_idx]) if unit_idx >= 0 else 0.0
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


    def select_top_k(
        scored: List[Tuple[int, float]], k: int,
    ) -> List[int]:
        """Return indices of the top-k trajectories with strictly positive scores."""
        return [idx for idx, score in scored[:k] if score > 0]

- [ ] **Step 4: Run tests (expect pass)**

    pytest famail_temporal/tests/test_attribution.py -v

Expected: 3 passed.

- [ ] **Step 5: Commit**

    git add famail_temporal/algorithm/attribution.py \
            famail_temporal/tests/test_attribution.py
    git commit -m "feat(algorithm): per-unit + per-trajectory attribution"

---

### Task 30: algorithm/modifier.py — TrajectoryModifier

**Files:**
- Create: famail_temporal/algorithm/modifier.py
- Create: famail_temporal/tests/test_modifier.py
- Modify: famail_temporal/algorithm/__init__.py

- [ ] **Step 1: Write failing tests**

    """Tests for algorithm.modifier TrajectoryModifier."""
    from dataclasses import dataclass

    import numpy as np
    import pytest
    import torch

    from famail_temporal import config
    from famail_temporal.algorithm.modifier import TrajectoryModifier, ModificationHistory
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    from famail_temporal.tests.test_objective import _make_synthetic_bundle


    def _make_test_trajectory(driver_id=0, pickup_xy=(3, 4), time_bucket=90):
        """Make a trajectory with pickup at given coords, time_bucket in morning_peak."""
        states = [
            TrajectoryState(x_grid=0.0, y_grid=0.0,
                            time_bucket=time_bucket - 1, day_index=1),
            TrajectoryState(x_grid=float(pickup_xy[0]), y_grid=float(pickup_xy[1]),
                            time_bucket=time_bucket, day_index=1),
        ]
        return Trajectory(trajectory_id=0, driver_id=driver_id, states=states)


    def test_modify_single_returns_history():
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        modifier = TrajectoryModifier(
            objective=obj, bundle=bundle, multi_stream_builder=None,
            max_iterations=3,
        )
        # Pick an active unit
        any_active_idx = 0
        cell = bundle.unit_map.to_flat_cell(any_active_idx)
        t_block = bundle.unit_map.to_time_block(any_active_idx)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy
        _, start_hour, _ = config.TIME_BLOCKS[t_block]
        tb = 1 + (start_hour * 12)
        traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

        history = modifier.modify_single(traj)
        assert isinstance(history, ModificationHistory)
        assert history.total_iterations <= 3
        assert len(history.iterations) == history.total_iterations


    def test_modify_single_respects_epsilon_ball():
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        modifier = TrajectoryModifier(
            objective=obj, bundle=bundle,
            max_iterations=50,
        )
        any_active_idx = 0
        cell = bundle.unit_map.to_flat_cell(any_active_idx)
        t_block = bundle.unit_map.to_time_block(any_active_idx)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy
        _, start_hour, _ = config.TIME_BLOCKS[t_block]
        tb = 1 + (start_hour * 12)
        traj = _make_test_trajectory(pickup_xy=(x, y), time_bucket=tb)

        history = modifier.modify_single(traj)
        orig = np.array([float(x), float(y)])
        final = np.array([
            history.modified.pickup_state.x_grid,
            history.modified.pickup_state.y_grid,
        ])
        diff = np.abs(final - orig)
        assert (diff <= config.EPSILON_BALL + 1e-5).all(), (
            f"Final pickup {final} strayed {diff} from original {orig}, "
            f"exceeding epsilon={config.EPSILON_BALL}"
        )

- [ ] **Step 2: Run tests (expect failure)**

    pytest famail_temporal/tests/test_modifier.py -v

- [ ] **Step 3: Write famail_temporal/algorithm/modifier.py**

    """
    TrajectoryModifier — ST-iFGSM with soft cell assignment and pooled (cell, t)
    fairness.

    Algorithm per trajectory:
      1. Determine the trajectory's time block t* from its pickup's time_bucket
      2. Compute pickup_mass = 1 / (n_hours_per_block[t*] * n_days)
      3. Subtract trajectory's contribution from _base_pickup_3d[orig_cell, t*]
      4. For each iteration t:
         a. Anneal temperature
         b. Build pickup_tensor with requires_grad=True
         c. Compute soft probs; inject * pickup_mass into t* slice
         d. Forward + backward through FAMAILObjective
         e. Apply delta = clip(alpha * sign(grad), -epsilon, epsilon)
         f. Update cumulative delta, clip pickup to grid bounds
      5. Persist changes to shared _base_pickup_3d
    """

    from __future__ import annotations
    from dataclasses import dataclass, field
    from typing import List, Optional

    import numpy as np
    import torch

    from famail_temporal import config
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.algorithm.soft_cell_assignment import (
        SoftCellAssignment, inject_soft_counts_into_3d,
    )
    from famail_temporal.data.aggregation import (
        hour_to_block_index, time_bucket_to_hour,
    )
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.utils.trajectory import Trajectory


    @dataclass
    class ModificationResult:
        iteration: int
        objective_value: float
        f_spatial: float
        f_causal: float
        f_fidelity: float
        gradient_norm: float
        cumulative_delta: np.ndarray


    @dataclass
    class ModificationHistory:
        original: Trajectory
        modified: Trajectory
        iterations: List[ModificationResult] = field(default_factory=list)
        converged: bool = False
        total_iterations: int = 0
        final_objective: float = 0.0


    class TrajectoryModifier:
        def __init__(
            self,
            objective: FAMAILObjective,
            bundle: DataBundle,
            multi_stream_builder=None,
            alpha: float = config.STEP_SIZE_ALPHA,
            epsilon: float = config.EPSILON_BALL,
            max_iterations: int = config.MAX_ITERATIONS,
            convergence_tol: float = config.CONVERGENCE_TOL,
        ):
            self.objective = objective
            self.bundle = bundle
            self.multi_stream_builder = multi_stream_builder
            self.alpha = alpha
            self.epsilon = epsilon
            self.max_iterations = max_iterations
            self.convergence_tol = convergence_tol

            self.soft_assign = SoftCellAssignment()
            self._base_pickup_3d = torch.from_numpy(bundle.pickup_3d).float().clone()

        def _get_annealed_temperature(self, iteration: int) -> float:
            if not config.ANNEAL_TEMPERATURE or self.max_iterations <= 1:
                return config.TAU_MIN
            progress = iteration / (self.max_iterations - 1)
            return config.TAU_MAX * (config.TAU_MIN / config.TAU_MAX) ** progress

        def _neighborhood_has_active_units(
            self, cell_xy, t_block: int,
        ) -> bool:
            k = self.soft_assign.k
            cx, cy = cell_xy
            gx, gy = config.GRID_DIMS
            for di in range(-k, k + 1):
                for dj in range(-k, k + 1):
                    ni, nj = cx + di, cy + dj
                    if 0 <= ni < gx and 0 <= nj < gy:
                        if self.bundle.mask_3d[ni, nj, t_block]:
                            return True
            return False

        def modify_single(self, trajectory: Trajectory) -> ModificationHistory:
            pickup_state = trajectory.states[-1]
            orig_cx = int(pickup_state.x_grid)
            orig_cy = int(pickup_state.y_grid)
            hour = time_bucket_to_hour(pickup_state.time_bucket)
            t_block = hour_to_block_index(hour)

            if not self._neighborhood_has_active_units((orig_cx, orig_cy), t_block):
                import warnings
                warnings.warn(
                    f"Trajectory {trajectory.trajectory_id} pickup at "
                    f"({orig_cx}, {orig_cy}, t={t_block}) has no active "
                    f"neighbors — skipping.",
                )
                return ModificationHistory(
                    original=trajectory, modified=trajectory.clone(),
                    iterations=[], converged=False, total_iterations=0,
                )

            n_hours = int(self.bundle.n_hours_per_block[t_block])
            pickup_mass = 1.0 / (n_hours * self.bundle.n_days)

            # Base_3d = shared state minus this trajectory's own contribution
            base_3d = self._base_pickup_3d.clone()
            base_3d[orig_cx, orig_cy, t_block] -= pickup_mass

            original_pickup = np.array([float(orig_cx), float(orig_cy)], dtype=np.float32)
            cumulative_delta = np.zeros(2, dtype=np.float32)

            iterations: List[ModificationResult] = []
            prev_objective = float("-inf")
            converged = False

            for it in range(self.max_iterations):
                if config.ANNEAL_TEMPERATURE:
                    self.soft_assign.set_temperature(
                        self._get_annealed_temperature(it)
                    )

                current_pickup = original_pickup + cumulative_delta
                pickup_tensor = torch.tensor(
                    current_pickup, dtype=torch.float32, requires_grad=True,
                )
                cell_tensor = torch.tensor(
                    [orig_cx, orig_cy], dtype=torch.float32,
                ).unsqueeze(0)

                probs = self.soft_assign(
                    pickup_tensor.unsqueeze(0), cell_tensor,
                )[0]  # (ns, ns)

                soft_3d = inject_soft_counts_into_3d(
                    base_3d, probs, (orig_cx, orig_cy), t_block,
                    k=self.soft_assign.k, pickup_mass=pickup_mass,
                )

                # Build tau_prime with pickup_tensor gradient
                tau_features = None
                tau_prime_features = None
                ms_kwargs = None
                if self.objective.alpha_fidelity > 0:
                    tau_features = trajectory.to_tensor().unsqueeze(0)
                    tau_prime_features = tau_features.clone()
                    tau_prime_features[0, -1, 0] = pickup_tensor[0]
                    tau_prime_features[0, -1, 1] = pickup_tensor[1]
                    if self.multi_stream_builder is not None:
                        modified = trajectory.apply_perturbation(cumulative_delta)
                        ms_kwargs = self.multi_stream_builder.build_fidelity_kwargs(
                            trajectory, modified,
                        )
                        # Inject gradient through slot 0 of x2 (+1 for 1-indexed coords)
                        x2 = ms_kwargs["x2"]
                        x2_new = x2.clone()
                        x2_new[0, 0, -1, 0] = pickup_tensor[0] + 1
                        x2_new[0, 0, -1, 1] = pickup_tensor[1] + 1
                        ms_kwargs["x2"] = x2_new

                total, terms = self.objective(
                    soft_pickup_3d=soft_3d,
                    tau_features=tau_features,
                    tau_prime_features=tau_prime_features,
                    multi_stream_kwargs=ms_kwargs,
                )
                self.objective.zero_grad()
                total.backward(retain_graph=True)

                if pickup_tensor.grad is None:
                    grad = np.zeros(2)
                else:
                    grad = pickup_tensor.grad.detach().cpu().numpy()
                grad_norm = float(np.linalg.norm(grad))

                if grad_norm > 1e-8:
                    delta = self.alpha * np.sign(grad)
                    cumulative_delta = np.clip(
                        cumulative_delta + delta, -self.epsilon, self.epsilon,
                    ).astype(np.float32)

                new_pickup = np.clip(
                    original_pickup + cumulative_delta,
                    [0.0, 0.0],
                    [config.GRID_DIMS[0] - 1, config.GRID_DIMS[1] - 1],
                ).astype(np.float32)
                # Re-sync cumulative_delta after grid-clip
                cumulative_delta = new_pickup - original_pickup

                result = ModificationResult(
                    iteration=it,
                    objective_value=float(total),
                    f_spatial=float(terms["f_spatial"]),
                    f_causal=float(terms["f_causal"]),
                    f_fidelity=float(terms["f_fidelity"]),
                    gradient_norm=grad_norm,
                    cumulative_delta=cumulative_delta.copy(),
                )
                iterations.append(result)

                if abs(float(total) - prev_objective) < self.convergence_tol:
                    converged = True
                    break
                prev_objective = float(total)

            modified = trajectory.apply_perturbation(cumulative_delta)

            # Persist change to shared base state
            new_cx = int(modified.pickup_state.x_grid)
            new_cy = int(modified.pickup_state.y_grid)
            if (new_cx, new_cy) != (orig_cx, orig_cy):
                self._base_pickup_3d[orig_cx, orig_cy, t_block] -= pickup_mass
                self._base_pickup_3d[new_cx, new_cy, t_block] += pickup_mass

            return ModificationHistory(
                original=trajectory,
                modified=modified,
                iterations=iterations,
                converged=converged,
                total_iterations=len(iterations),
                final_objective=iterations[-1].objective_value if iterations else 0.0,
            )

        def modify_batch(
            self, trajectories: List[Trajectory],
        ) -> List[ModificationHistory]:
            return [self.modify_single(t) for t in trajectories]

- [ ] **Step 4: Update famail_temporal/algorithm/__init__.py**

    """Algorithm orchestration — objective, modifier, attribution, soft assignment."""

    from famail_temporal.algorithm.attribution import (
        compute_per_unit_attribution,
        rank_trajectories,
        select_top_k,
    )
    from famail_temporal.algorithm.modifier import (
        TrajectoryModifier,
        ModificationResult,
        ModificationHistory,
    )
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.algorithm.soft_cell_assignment import (
        SoftCellAssignment,
        inject_soft_counts_into_3d,
    )

    __all__ = [
        "FAMAILObjective",
        "TrajectoryModifier", "ModificationResult", "ModificationHistory",
        "SoftCellAssignment", "inject_soft_counts_into_3d",
        "compute_per_unit_attribution", "rank_trajectories", "select_top_k",
    ]

- [ ] **Step 5: Run tests (expect pass)**

    pytest famail_temporal/tests/test_modifier.py -v

Expected: 2 passed.

- [ ] **Step 6: Commit**

    git add famail_temporal/algorithm/modifier.py \
            famail_temporal/algorithm/__init__.py \
            famail_temporal/tests/test_modifier.py
    git commit -m "feat(algorithm): TrajectoryModifier with ST-iFGSM loop"

---

## Phase 8: Integration tests (Tasks 31–32)

### Task 31: End-to-end gradient flow test

**Files:**
- Create: famail_temporal/tests/test_gradient_flow.py

- [ ] **Step 1: Write the tests**

    """End-to-end gradient flow tests for the full objective."""
    import numpy as np
    import torch

    from famail_temporal import config
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.algorithm.soft_cell_assignment import (
        SoftCellAssignment, inject_soft_counts_into_3d,
    )
    from famail_temporal.tests.test_objective import _make_synthetic_bundle


    def test_gradient_flows_through_pooled_objective():
        """Gradient from total objective flows to a pickup_tensor (x, y)."""
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

        # Pick any active unit and use its cell as the pickup location
        cell = bundle.unit_map.to_flat_cell(0)
        t_block = bundle.unit_map.to_time_block(0)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy

        pickup_tensor = torch.tensor([float(x), float(y)], requires_grad=True)
        soft = SoftCellAssignment()
        cell_t = torch.tensor([x, y]).float().unsqueeze(0)
        probs = soft(pickup_tensor.unsqueeze(0), cell_t)[0]

        base_3d = torch.from_numpy(bundle.pickup_3d).float()
        pickup_mass = 1.0 / (int(bundle.n_hours_per_block[t_block]) * bundle.n_days)
        soft_3d = inject_soft_counts_into_3d(
            base_3d, probs, (x, y), t_block, k=soft.k, pickup_mass=pickup_mass,
        )

        total, _ = obj(soft_pickup_3d=soft_3d)
        total.backward()

        assert pickup_tensor.grad is not None
        assert not torch.isnan(pickup_tensor.grad).any()
        assert not torch.isinf(pickup_tensor.grad).any()


    def test_gradient_only_flows_through_correct_t_block():
        """The gradient should affect only the target time block's slice."""
        bundle = _make_synthetic_bundle()
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

        cell = bundle.unit_map.to_flat_cell(0)
        t_block = bundle.unit_map.to_time_block(0)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy

        pickup_tensor = torch.tensor([float(x), float(y)], requires_grad=True)
        soft = SoftCellAssignment()
        cell_t = torch.tensor([x, y]).float().unsqueeze(0)
        probs = soft(pickup_tensor.unsqueeze(0), cell_t)[0]

        base_3d = torch.from_numpy(bundle.pickup_3d).float().requires_grad_(False)
        # Make base_3d a parameter-like tensor so we can inspect downstream grads
        soft_3d = inject_soft_counts_into_3d(
            base_3d, probs, (x, y), t_block, k=soft.k, pickup_mass=1.0,
        )

        # Slices other than t_block are bit-identical to base_3d (no grad entered)
        for t in range(config.T):
            if t == t_block:
                continue
            assert torch.equal(soft_3d[:, :, t], base_3d[:, :, t])

- [ ] **Step 2: Run tests (expect pass)**

    pytest famail_temporal/tests/test_gradient_flow.py -v

Expected: 2 passed.

- [ ] **Step 3: Commit**

    git add famail_temporal/tests/test_gradient_flow.py
    git commit -m "test(integration): gradient flow through pooled objective"

---

### Task 32: Modifier integration — convergence + mass balance

**Files:**
- Create: famail_temporal/tests/test_modifier_integration.py

- [ ] **Step 1: Write the tests**

    """Integration tests for the full modifier loop."""
    import numpy as np
    import torch

    from famail_temporal import config
    from famail_temporal.algorithm.modifier import TrajectoryModifier
    from famail_temporal.algorithm.objective import FAMAILObjective
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
    from famail_temporal.tests.test_objective import _make_synthetic_bundle


    def _make_test_trajectory(pickup_xy, time_bucket):
        states = [
            TrajectoryState(x_grid=0.0, y_grid=0.0,
                            time_bucket=time_bucket - 1, day_index=1),
            TrajectoryState(x_grid=float(pickup_xy[0]), y_grid=float(pickup_xy[1]),
                            time_bucket=time_bucket, day_index=1),
        ]
        return Trajectory(trajectory_id=0, driver_id=0, states=states)


    def test_five_iteration_objective_improves_or_plateaus():
        bundle = _make_synthetic_bundle(seed=0)
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        modifier = TrajectoryModifier(
            objective=obj, bundle=bundle, max_iterations=5,
        )

        cell = bundle.unit_map.to_flat_cell(0)
        t_block = bundle.unit_map.to_time_block(0)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy
        _, start_hour, _ = config.TIME_BLOCKS[t_block]
        tb = 1 + (start_hour * 12)

        traj = _make_test_trajectory((x, y), tb)
        history = modifier.modify_single(traj)

        # Objective should improve or stay flat (modifier maximizes L)
        values = [r.objective_value for r in history.iterations]
        assert len(values) > 0
        # Allow some wiggle — overall trajectory should not drop substantially
        first = values[0]
        last = values[-1]
        assert last >= first - 1e-3, (
            f"Objective decreased from {first} to {last}"
        )


    def test_mass_balance_after_single_modification():
        """pickup_3d total mass should be preserved after one modification."""
        bundle = _make_synthetic_bundle(seed=1)
        obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
        modifier = TrajectoryModifier(
            objective=obj, bundle=bundle, max_iterations=3,
        )

        mass_before = modifier._base_pickup_3d.sum().item()

        cell = bundle.unit_map.to_flat_cell(0)
        t_block = bundle.unit_map.to_time_block(0)
        gy = bundle.pickup_3d.shape[1]
        x, y = cell // gy, cell % gy
        _, start_hour, _ = config.TIME_BLOCKS[t_block]
        tb = 1 + (start_hour * 12)

        traj = _make_test_trajectory((x, y), tb)
        modifier.modify_single(traj)

        mass_after = modifier._base_pickup_3d.sum().item()
        # Mass preserved within EPS
        assert abs(mass_after - mass_before) < 1e-5, (
            f"Mass imbalance: {mass_before} -> {mass_after}"
        )

- [ ] **Step 2: Run tests (expect pass)**

    pytest famail_temporal/tests/test_modifier_integration.py -v

Expected: 2 passed.

- [ ] **Step 3: Commit**

    git add famail_temporal/tests/test_modifier_integration.py
    git commit -m "test(integration): modifier convergence + mass balance"

---

**End of Phase 7–8 file.** At this checkpoint the full algorithm is functional end-to-end. Continue with 2026-04-16-famail-temporal-phase9.md for documentation (sub-READMEs and top-level README).
