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

The modifier is the only module that MUTATES _base_pickup_3d. This creates
ordering dependencies across trajectories when modify_batch is called:
trajectory B's optimization sees the updated baseline after trajectory A's
modification. This is intentional.
"""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, List, Optional
import warnings

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
from famail_temporal.utils.trajectory import Trajectory, taper_weights
from famail_temporal.algorithm.supply import (
    soft_delta_supply, hard_delta_supply, state_presence_mass,
)


@dataclass
class ModificationResult:
    """Single iteration record from the ST-iFGSM loop."""
    iteration: int
    objective_value: float
    f_spatial: float
    f_causal: float
    f_fidelity: float
    gradient_norm: float
    cumulative_delta: np.ndarray
    # Tier A diagnostics - None when diagnostics_enabled=False.
    grad_spatial_norm: float | None = None
    grad_causal_norm: float | None = None
    grad_fidelity_norm: float | None = None
    grad_cosine_spatial_causal: float | None = None
    grad_cosine_fairness_fidelity: float | None = None
    sign_flipped: bool | None = None
    dominant_term: str | None = None


@dataclass
class ModificationHistory:
    """Full history of modifying one trajectory.

    ``modified`` is the trajectory built from the *best-iterate* perturbation
    (the iter with the highest objective value seen during optimization),
    not the last iterate. ``best_iteration`` records which iter that was;
    ``best_objective`` is the corresponding objective value. ``iterations``
    still contains every iter that ran, so downstream consumers can inspect
    the full optimization trajectory.

    ``converged=True`` means the patience-based early-stop fired (no
    improvement above CONVERGENCE_TOL for PATIENCE consecutive iters);
    ``converged=False`` means the run hit MAX_ITERATIONS.
    """
    original: Trajectory
    modified: Trajectory
    iterations: List[ModificationResult] = field(default_factory=list)
    converged: bool = False
    total_iterations: int = 0
    final_objective: float = 0.0
    best_iteration: int = -1
    best_objective: float = float("-inf")


class TrajectoryModifier:
    """ST-iFGSM trajectory modifier with soft cell assignment.

    Perturbation rule:  delta = clip(alpha * sign(grad), -epsilon, epsilon)
    Cumulative delta is clipped to the epsilon-ball: ||delta||_inf <= epsilon.
    Mass balance is maintained: pickup_3d.sum() is preserved after each
    modification (subtract at old cell, add at new cell).
    """

    def __init__(
        self,
        objective: FAMAILObjective,
        bundle: DataBundle,
        multi_stream_builder=None,
        alpha: float | None = None,
        epsilon: float | None = None,
        max_iterations: int | None = None,
        convergence_tol: float | None = None,
        diagnostics_enabled: bool | None = None,
        device: torch.device | str | None = None,
        patience: int | None = None,
        accept_rule: str | None = None,
        epsilon_cap: float | None = None,
        use_ste: bool | None = None,
    ):
        # Resolve each config-backed default at __init__ time (NOT at
        # function-definition time) so config-override mutations applied
        # after module import — e.g. by run_experiment's _apply_config_overrides —
        # are picked up correctly. Default-arg evaluation happens once at
        # import, which would silently ignore runtime overrides.
        self.objective = objective
        self.bundle = bundle
        self.multi_stream_builder = multi_stream_builder
        self.alpha = config.STEP_SIZE_ALPHA if alpha is None else alpha
        self.epsilon = config.EPSILON_BALL if epsilon is None else epsilon
        self.max_iterations = (
            config.MAX_ITERATIONS if max_iterations is None else max_iterations
        )
        self.convergence_tol = (
            config.CONVERGENCE_TOL if convergence_tol is None else convergence_tol
        )
        self.diagnostics_enabled = (
            config.DIAGNOSTICS_ENABLED if diagnostics_enabled is None
            else diagnostics_enabled
        )
        # Patience-based early stop. ``None`` disables early stopping (always
        # runs max_iterations). Any non-negative integer N triggers
        # convergence when the best objective hasn't improved by more than
        # convergence_tol for N consecutive iters.
        self.patience = (
            config.PATIENCE if patience is None else patience
        )
        # Inner-loop acceptance gate (see config.ACCEPT_RULE).
        self.accept_rule = (
            config.ACCEPT_RULE if accept_rule is None else accept_rule
        )
        # Cumulative L-inf cap from the true original cell, across rounds (see
        # config.EPSILON_CAP). Equals EPSILON_BALL by default ⇒ no-op for a
        # single edit anchored at its own cell.
        self.epsilon_cap = (
            config.EPSILON_CAP if epsilon_cap is None else epsilon_cap
        )
        # Straight-through (hard-metric) editing toggle (see config.STE_ENABLED).
        self.use_ste = config.STE_ENABLED if use_ste is None else use_ste

        # Resolve device. If unspecified, inherit from the objective's first
        # buffer (it's an nn.Module with registered buffers already on the
        # correct device). Default to CPU when no inference source exists.
        if device is None:
            buf = next(objective.buffers(), None)
            device = buf.device if buf is not None else torch.device("cpu")
        self.device = torch.device(device)

        # Resolve neighborhood_size from config at construction time (NOT via
        # SoftCellAssignment's import-time default arg) so a runtime
        # `--override SOFT_NEIGHBORHOOD_SIZE` actually takes effect — mirrors how
        # the other config-backed values are resolved here.
        self.soft_assign = SoftCellAssignment(
            neighborhood_size=config.SOFT_NEIGHBORHOOD_SIZE,
        ).to(self.device)
        # Clone so we don't mutate the original bundle array; place on device.
        self._base_pickup_3d = (
            torch.from_numpy(bundle.pickup_3d).float().to(self.device).clone()
        )
        self._prev_grad_sign = None

        # ── Supply-lift state (Task 7) ───────────────────────────────────
        # Shared DELTA-supply accumulator (evaluation-honest tier-1 ΔS). Shape
        # is derived from the bundle tensors (NOT config.GRID_DIMS) so it always
        # matches _base_pickup_3d and the objective's grid buffers. Zero at init;
        # holds deltas, never totals (baseline presence lives in S_base).
        self._delta_supply_3d = torch.zeros_like(self._base_pickup_3d)
        self._grid_shape = tuple(self._base_pickup_3d.shape)  # (gx, gy, T)
        # Boolean grid→unit mask. Boolean-indexing (gx,gy,T) with this in C order
        # yields the (N,) vector in the SAME unit order the objective uses for
        # its active_taxis_N / mask_3d buffers (both built from bundle.mask_3d).
        self._mask_3d_t = torch.from_numpy(bundle.mask_3d).to(self.device)
        # Tapered-tail infeasibility counters (Task 8 persists these).
        self.n_taper_infeasible_trim = 0
        self.n_taper_infeasible_lift = 0

    def current_pickup_3d(self) -> np.ndarray:
        """Return the post-modification pickup tensor as a numpy ndarray.

        Shape (grid_x, grid_y, T), float32. Returns a copy so callers
        cannot mutate modifier state.
        """
        return self._base_pickup_3d.detach().cpu().numpy().copy().astype(np.float32)

    def current_delta_supply_3d(self) -> np.ndarray:
        """Return the accumulated tier-1 ΔS grid as a float64 numpy copy.

        Shape (grid_x, grid_y, T). This is the evaluation-honest supply delta
        contributed by every discretized edit (trim tails and lift tails alike);
        the downstream evaluator adds it to the baseline supply. Returns a copy
        so callers cannot mutate modifier state.
        """
        return (
            self._delta_supply_3d.detach().cpu().numpy().astype(np.float64).copy()
        )

    def _hard_tail_delta_supply(
        self, original: Trajectory, modified: Trajectory,
    ) -> Optional[torch.Tensor]:
        """Hard tier-1 ΔS of a discretized edit: −presence box at each ORIGINAL
        state cell, +presence box at each MODIFIED state cell, over exactly the
        rows that changed cell (unmoved rows cancel, so iterating all states is
        safe and captures any tail-deepening apply_tail_perturbation performed).

        Returns a torch tensor (matching the accumulator's dtype/device) or
        ``None`` when nothing moved.
        """
        olds, news, tbs, masses = [], [], [], []
        for s_old, s_new in zip(original.states, modified.states):
            oc = (int(s_old.x_grid), int(s_old.y_grid))
            nc = (int(s_new.x_grid), int(s_new.y_grid))
            if oc == nc:
                continue
            tb = hour_to_block_index(time_bucket_to_hour(s_old.time_bucket))
            mass = state_presence_mass(
                self.bundle.n_hours_per_block, self.bundle.n_days, tb,
            )
            olds.append(oc)
            news.append(nc)
            tbs.append(tb)
            masses.append(mass)
        if not olds:
            return None
        ds_np = hard_delta_supply(olds, news, tbs, masses, self._grid_shape)
        return torch.from_numpy(ds_np).to(self._delta_supply_3d)

    def _discretize_trim(
        self, trajectory: Trajectory, best_cumulative_delta: np.ndarray,
    ) -> Trajectory:
        """Trim-mode discretization of the best iterate's perturbation.

        TAIL_LEN == 0 → legacy ``apply_perturbation`` (bit-for-bit, G1).
        TAIL_LEN > 0 → tail translation; on infeasible repair, fall back to
        the legacy pickup-only move (counted) so the pickup NEVER differs
        from legacy (G3).
        """
        # G3: must match legacy persist cell arithmetic exactly. Legacy applies
        # the FRACTIONAL delta (apply_perturbation: clip(coord + delta, 0,
        # dim-1), default grid_dims) and the persist step int()-TRUNCATES the
        # result. apply_tail_perturbation instead round()s the offset, which
        # diverges by one cell at fractional deltas (e.g. int(10-0.4)=9 vs
        # 10+round(-0.4)=10). So compute the legacy deployed cell FIRST, then
        # hand the repair the INTEGER offset vector — round() of an
        # integer-valued float is a no-op, so the repair deploys exactly that
        # cell. Feasibility/repair/taper logic unchanged.
        legacy_frac = trajectory.apply_perturbation(best_cumulative_delta)
        if config.TAIL_LEN == 0:
            # TAIL_LEN == 0 → bit-for-bit legacy (G1).
            return legacy_frac
        pickup = trajectory.pickup_state
        legacy_cx = int(legacy_frac.pickup_state.x_grid)
        legacy_cy = int(legacy_frac.pickup_state.y_grid)
        delta_int = np.array(
            [float(legacy_cx - int(pickup.x_grid)),
             float(legacy_cy - int(pickup.y_grid))],
            dtype=np.float32,
        )
        modified = trajectory.apply_tail_perturbation(
            delta_int, config.TAIL_LEN, config.GRID_DIMS,
        )
        if modified is None:
            self.n_taper_infeasible_trim += 1
            modified = legacy_frac
        return modified

    def _get_annealed_temperature(self, iteration: int) -> float:
        """Exponential temperature annealing: tau_max * (tau_min/tau_max)^(t/T)."""
        if not config.ANNEAL_TEMPERATURE or self.max_iterations <= 1:
            return config.TAU_MIN
        progress = iteration / (self.max_iterations - 1)
        return config.TAU_MAX * (config.TAU_MIN / config.TAU_MAX) ** progress

    def _neighborhood_has_active_units(
        self, cell_xy, t_block: int,
    ) -> bool:
        """Check if any cell in the soft-assignment neighborhood is active."""
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

    def _compute_decomposed_gradient(
        self,
        f_spatial: torch.Tensor,
        f_causal: torch.Tensor,
        f_fidelity: torch.Tensor,
        pickup_tensor: torch.Tensor,
    ):
        """Return (grad_combined_ndarray, diagnostics_dict)."""
        alpha_sp = self.objective.alpha_spatial
        alpha_ca = self.objective.alpha_causal
        alpha_fi = self.objective.alpha_fidelity
        zero_grad = np.zeros(2, dtype=np.float32)

        # Skip backwards when a term's alpha is zero — its contribution to the
        # combined gradient is identically zero, and running the backward would
        # be wasted compute (or fail, in the fidelity case where f_fidelity
        # is an unconnected constant when alpha_fidelity == 0).
        if alpha_sp > 0:
            grad_spatial = torch.autograd.grad(
                f_spatial, pickup_tensor, retain_graph=True,
            )[0].detach().cpu().numpy()
        else:
            grad_spatial = zero_grad

        if alpha_ca > 0:
            grad_causal = torch.autograd.grad(
                f_causal, pickup_tensor, retain_graph=True,
            )[0].detach().cpu().numpy()
        else:
            grad_causal = zero_grad

        if alpha_fi > 0:
            grad_fidelity = torch.autograd.grad(
                f_fidelity, pickup_tensor, retain_graph=True,
            )[0].detach().cpu().numpy()
        else:
            grad_fidelity = zero_grad

        grad_combined = (
            alpha_sp * grad_spatial
            + alpha_ca * grad_causal
            + alpha_fi * grad_fidelity
        )

        def _cos(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

        norms = {
            "spatial":  float(np.linalg.norm(grad_spatial)),
            "causal":   float(np.linalg.norm(grad_causal)),
            "fidelity": float(np.linalg.norm(grad_fidelity)),
        }
        weighted = {
            "spatial":  alpha_sp * norms["spatial"],
            "causal":   alpha_ca * norms["causal"],
            "fidelity": alpha_fi * norms["fidelity"],
        }
        # When all weighted norms are ~zero (e.g., at convergence or when
        # all alphas happen to give zero-norm gradients simultaneously),
        # there is no meaningful dominant term. Return None rather than
        # silently picking one via dict-insertion-order tiebreak — otherwise
        # aggregate metrics like frac_iters_spatial_dominant get biased.
        max_weighted = max(weighted.values())
        if max_weighted < 1e-8:
            dominant = None
        else:
            dominant = max(weighted, key=weighted.get)

        fairness_grad = alpha_sp * grad_spatial + alpha_ca * grad_causal
        diagnostics = {
            "grad_spatial_norm":              norms["spatial"],
            "grad_causal_norm":               norms["causal"],
            "grad_fidelity_norm":             norms["fidelity"],
            "grad_cosine_spatial_causal":     _cos(grad_spatial, grad_causal),
            "grad_cosine_fairness_fidelity":  _cos(fairness_grad, grad_fidelity),
            "dominant_term":                  dominant,
        }
        return grad_combined, diagnostics

    def modify_single(
        self,
        trajectory: Trajectory,
        on_iteration: Optional[Callable[[int, "ModificationResult"], None]] = None,
        *,
        original_cell: Optional[tuple] = None,
        mode: str = "trim",
    ) -> ModificationHistory:
        """Run the ST-iFGSM loop on a single trajectory.

        Steps:
        1. Determine time block and pickup mass
        2. Subtract this trajectory's contribution from the shared base
        3. Iteratively perturb the pickup location using signed gradients
        4. Persist the final change to the shared _base_pickup_3d

        Args:
            trajectory: The trajectory whose pickup location to perturb.
            on_iteration: Optional callback invoked after each ST-iFGSM step
                with ``(iteration_index, ModificationResult)``. Pure
                instrumentation — does not affect any algorithmic state.
                Default ``None`` (no callback). Use for progress bars,
                live diagnostics, etc.
        """
        self._prev_grad_sign = None
        pickup_state = trajectory.states[-1]
        orig_cx = int(pickup_state.x_grid)
        orig_cy = int(pickup_state.y_grid)
        hour = time_bucket_to_hour(pickup_state.time_bucket)
        t_block = hour_to_block_index(hour)

        # Safeguard: skip if no active units in the neighborhood
        if not self._neighborhood_has_active_units((orig_cx, orig_cy), t_block):
            warnings.warn(
                f"Trajectory {trajectory.trajectory_id} pickup at "
                f"({orig_cx}, {orig_cy}, t={t_block}) has no active "
                f"neighbors — skipping.",
            )
            return ModificationHistory(
                original=trajectory, modified=trajectory.clone(),
                iterations=[], converged=False, total_iterations=0,
            )

        # pickup_mass = 1 / (n_hours_per_block[t*] * n_days)
        n_hours = int(self.bundle.n_hours_per_block[t_block])
        pickup_mass = 1.0 / (n_hours * self.bundle.n_days)

        # Subtract this trajectory's contribution BEFORE the loop
        base_3d = self._base_pickup_3d.clone()
        base_3d[orig_cx, orig_cy, t_block] -= pickup_mass

        original_pickup = np.array([float(orig_cx), float(orig_cy)], dtype=np.float32)
        true_original = (
            np.array([float(original_cell[0]), float(original_cell[1])],
                     dtype=np.float32)
            if original_cell is not None
            else original_pickup
        )
        cumulative_delta = np.zeros(2, dtype=np.float32)

        # Cache the dense-fidelity feature tensor once per trajectory. The
        # underlying ``trajectory.states`` never change inside this loop —
        # only the cumulative_delta does, which we splice into a clone every
        # iter. Skipping the rebuild saves one Python list-comp + numpy alloc
        # + torch tensor construction per iter. Lands on self.device so the
        # downstream discriminator forward stays on-device.
        tau_features_cached = (
            trajectory.to_tensor().unsqueeze(0).to(self.device)
            if self.objective.alpha_fidelity > 0
            else None
        )

        # Cache the multi-stream context kwargs once per trajectory. Of the
        # kwargs dict that ``build_fidelity_kwargs`` returns, only
        # ``x2[0, 0, -1, 0:2]`` (the perturbed pickup coords on slot 0 of the
        # modified branch) actually changes across iters. driving_1/2,
        # profile_1/2, x1/mask1, slots 1..N-1 of x2, and mask2 all depend
        # only on ``trajectory.driver_id`` and ``trajectory.states`` — both
        # constant within this loop. Building it once saves the heaviest
        # per-iter allocation in the fidelity path: ~5 seeking-context pads,
        # driving sample + pad, profile fetch, two coordinate conversions.
        ms_kwargs_cached = (
            self.multi_stream_builder.build_fidelity_kwargs(trajectory, trajectory)
            if (self.objective.alpha_fidelity > 0
                and self.multi_stream_builder is not None)
            else None
        )

        # ── Lift-mode per-trajectory constants ────────────────────────────
        # Precomputed ONCE (independent of the optimizer's delta): the moving
        # tail rows (last L_eff+1 states, matching apply_tail_perturbation /
        # lift_candidates), their presence masses, taper weights, and the
        # CONSTANT hard removal at the ORIGINAL positions. Guarded by ``mode`` so
        # the trim optimization path pays for none of this (G1).
        if mode == "lift":
            # Guard float32 demand epsilon (Task-10 production incident): after
            # thousands of trim persists (chains of -= mass / += mass float32
            # ops on the shared grid) a drained cell can sit a few ULP below
            # zero (verified: 67 aggregated pickups − 67 masses = −1.86e-9),
            # and compute_fspatial's strict negativity check raises on the
            # first lift objective call. Sanitize the LOCAL clone only — one
            # data-cleaning op per lift trajectory on a constant (no-autograd)
            # tensor. The trim/legacy path is untouched: its tensor ops and
            # objective inputs stay byte-identical (G1/G3).
            base_3d = torch.clamp(base_3d, min=0.0)
            n_states = trajectory.n_states
            l_eff = max(0, min(config.TAIL_LEN, n_states - 2))
            M = l_eff + 1
            moving_idx = list(range(n_states - M, n_states))
            tail_states = [trajectory.states[i] for i in moving_idx]
            tail_cells = [(int(s.x_grid), int(s.y_grid)) for s in tail_states]
            tail_tblocks = [
                hour_to_block_index(time_bucket_to_hour(s.time_bucket))
                for s in tail_states
            ]
            tail_masses = [
                state_presence_mass(
                    self.bundle.n_hours_per_block, self.bundle.n_days, tb,
                )
                for tb in tail_tblocks
            ]
            tail_signs_pos = [1] * M
            # Taper: the L_eff tail rows get taper_weights(l_eff); the pickup
            # (last moving row) gets the full 1.0 delta.
            taper_vec = list(taper_weights(l_eff)) + [1.0]
            tail_orig_tensor = torch.tensor(
                [[float(cx), float(cy)] for (cx, cy) in tail_cells],
                dtype=torch.float32, device=self.device,
            )  # (M, 2) — original integer positions of the moving rows
            tail_cell_tensor = tail_orig_tensor.clone()  # neighborhood centers
            taper_col = torch.tensor(
                taper_vec, dtype=torch.float32, device=self.device,
            ).unsqueeze(1)  # (M, 1)
            # Constant hard removal at the ORIGINAL tail positions (sign already
            # −mass; not differentiable — only the +soft-add term carries grad).
            removal_const = torch.from_numpy(
                hard_delta_supply(
                    tail_cells, [], tail_tblocks, tail_masses, self._grid_shape,
                )
            ).to(self._delta_supply_3d)

        iterations: List[ModificationResult] = []
        # Best-iterate tracking. We persist the perturbation associated with
        # the best objective value seen across all iters (not the last) —
        # standard methodology in PGD/MI-FGSM literature.
        best_objective = float("-inf")
        best_cumulative_delta = np.zeros(2, dtype=np.float32)
        best_iteration = -1
        iters_since_improvement = 0
        converged = False
        f_causal_0 = None
        f_spatial_0 = None

        # Cache the discriminator's constant (iteration-invariant) stream
        # encodings for the whole loop: within one trajectory's ST-iFGSM loop
        # only x2 slot 0 changes, so ~15 of the ~20 LSTM row-encodes per
        # fidelity forward are otherwise recomputed identically. The cache is
        # bitwise-transparent (eval mode => deterministic encoders); trip_s2 and
        # the gradient path stay live. nullcontext when fidelity is off or the
        # discriminator is a stub => those paths are byte-unchanged (G1).
        _disc = getattr(self.objective, "discriminator", None)
        _cache_cm = (
            _disc.cache_constant_streams()
            if (self.objective.alpha_fidelity > 0
                and hasattr(_disc, "cache_constant_streams"))
            else nullcontext()
        )
        with _cache_cm:
            for it in range(self.max_iterations):
                # (a) Anneal temperature
                if config.ANNEAL_TEMPERATURE:
                    self.soft_assign.set_temperature(
                        self._get_annealed_temperature(it)
                    )

                # (b) Build the differentiable leaf + the soft pickup-demand probs.
                current_pickup = original_pickup + cumulative_delta
                delta_supply_N = None
                if mode == "lift":
                    # Leaf = the tail-translation delta. Every moving row's soft
                    # position (supply) AND the pickup-demand position derive from
                    # it, so gradients from the demand grid, the endogenous supply
                    # channel, and fidelity all land on this single (2,) tensor —
                    # exactly as cumulative_delta plays the pickup-delta role in trim.
                    delta_tensor = torch.tensor(
                        cumulative_delta, dtype=torch.float32,
                        device=self.device, requires_grad=True,
                    )
                    grad_leaf = delta_tensor
                    # pos_j = orig_j + taper_j * delta  (differentiable in delta)
                    pos_stack = tail_orig_tensor + taper_col * delta_tensor  # (M, 2)
                    probs_new = self.soft_assign(pos_stack, tail_cell_tensor)  # (M,ns,ns)
                    probs = probs_new[-1]  # the pickup row's soft assignment (ns, ns)
                else:
                    pickup_tensor = torch.tensor(
                        current_pickup, dtype=torch.float32,
                        device=self.device, requires_grad=True,
                    )
                    grad_leaf = pickup_tensor
                    cell_tensor = torch.tensor(
                        [orig_cx, orig_cy], dtype=torch.float32, device=self.device,
                    ).unsqueeze(0)

                    # (c) Compute soft probs -> inject into t_block slice
                    probs = self.soft_assign(
                        pickup_tensor.unsqueeze(0), cell_tensor,
                    )[0]  # (ns, ns)

                # Pickup demand injection (UNCHANGED; lift moves demand too, by
                # design — the pickup row's soft position feeds the same t_block slice).
                soft_3d = inject_soft_counts_into_3d(
                    base_3d, probs, (orig_cx, orig_cy), t_block,
                    k=self.soft_assign.k, pickup_mass=pickup_mass,
                )

                # (c2) Straight-through hard-metric grid (opt-in). Forward value =
                # the HARD (realizable) grid: full pickup mass at int(current_pickup),
                # the exact cell the persist step writes. Gradient flows via the soft
                # assignment (soft_3d - soft_3d.detach()). This makes best-iterate +
                # the acceptance gate select on the metric actually deployed (§8.8).
                # int(current_pickup), NOT argmax(probs): soft uses cell centers, so
                # argmax can tie-break wrong at integer coords.
                if self.use_ste:
                    k_half = self.soft_assign.k
                    snap_x, snap_y = int(current_pickup[0]), int(current_pickup[1])
                    ox, oy = snap_x - orig_cx + k_half, snap_y - orig_cy + k_half
                    hard_probs = torch.zeros_like(probs)
                    if 0 <= ox < probs.shape[0] and 0 <= oy < probs.shape[1]:
                        hard_probs[ox, oy] = 1.0
                    hard_3d = inject_soft_counts_into_3d(
                        base_3d, hard_probs, (orig_cx, orig_cy), t_block,
                        k=k_half, pickup_mass=pickup_mass,
                    )
                    objective_grid = hard_3d + (soft_3d - soft_3d.detach())
                else:
                    objective_grid = soft_3d

                # (c3) Endogenous supply channel (lift only). Build this trajectory's
                # soft ΔS: +1 soft-add at the moving rows' NEW soft positions, plus
                # the constant −1 hard removal at the ORIGINAL positions; add it to
                # the shared accumulator and gather to (N,) in the objective's unit
                # order. Only the +soft-add term carries gradient w.r.t. delta_tensor.
                if mode == "lift":
                    soft_add = soft_delta_supply(
                        probs_new, tail_cells, tail_tblocks, tail_masses,
                        tail_signs_pos, self._grid_shape,
                    )
                    traj_soft_ds = soft_add + removal_const
                    delta_supply_N = (
                        self._delta_supply_3d + traj_soft_ds
                    )[self._mask_3d_t]

                # Build fidelity features if needed
                tau_features = None
                tau_prime_features = None
                ms_kwargs = None
                if self.objective.alpha_fidelity > 0:
                    tau_features = tau_features_cached
                    tau_prime_features = tau_features.clone()
                    if mode == "lift":
                        # Splice ALL moving rows (not just the pickup): moving row j
                        # is feature row -(M)+j == -(L_eff+1)+j.
                        for j in range(M):
                            row = -M + j
                            tau_prime_features[0, row, 0] = pos_stack[j, 0]
                            tau_prime_features[0, row, 1] = pos_stack[j, 1]
                    else:
                        tau_prime_features[0, -1, 0] = pickup_tensor[0]
                        tau_prime_features[0, -1, 1] = pickup_tensor[1]
                    if self.multi_stream_builder is not None:
                        # Reuse the per-trajectory cache and splice only the
                        # iter-dependent slot(s) into a fresh clone. The cache itself
                        # is never mutated, so subsequent iters start from clean state.
                        x2_new = ms_kwargs_cached["x2"].clone()
                        if mode == "lift":
                            for j in range(M):
                                row = -M + j
                                x2_new[0, 0, row, 0] = pos_stack[j, 0] + 1
                                x2_new[0, 0, row, 1] = pos_stack[j, 1] + 1
                        else:
                            x2_new[0, 0, -1, 0] = pickup_tensor[0] + 1
                            x2_new[0, 0, -1, 1] = pickup_tensor[1] + 1
                        ms_kwargs = {**ms_kwargs_cached, "x2": x2_new}

                # (d) Forward through FAMAILObjective. Trim passes NOTHING for
                # delta_supply_N (G1: the None path is byte-identical to legacy).
                if mode == "lift":
                    total, terms = self.objective(
                        soft_pickup_3d=objective_grid,
                        tau_features=tau_features,
                        tau_prime_features=tau_prime_features,
                        multi_stream_kwargs=ms_kwargs,
                        delta_supply_N=delta_supply_N,
                    )
                else:
                    total, terms = self.objective(
                        soft_pickup_3d=objective_grid,
                        tau_features=tau_features,
                        tau_prime_features=tau_prime_features,
                        multi_stream_kwargs=ms_kwargs,
                    )

                # (e) Backward - decomposed if diagnostics_enabled, else single-backward
                self.objective.zero_grad()
                tier_a_metrics = None
                if self.diagnostics_enabled:
                    grad, tier_a_metrics = self._compute_decomposed_gradient(
                        terms["f_spatial"], terms["f_causal"], terms["f_fidelity"],
                        grad_leaf,
                    )
                else:
                    total.backward(retain_graph=True)
                    if grad_leaf.grad is None:
                        grad = np.zeros(2)
                    else:
                        grad = grad_leaf.grad.detach().cpu().numpy()
                grad_norm = float(np.linalg.norm(grad))

                # (f) ST-iFGSM: delta = clip(alpha * sign(grad), -epsilon, epsilon)
                if grad_norm > 1e-8:
                    delta = self.alpha * np.sign(grad)
                    cumulative_delta = np.clip(
                        cumulative_delta + delta, -self.epsilon, self.epsilon,
                    ).astype(np.float32)

                # Clip pickup to grid bounds
                new_pickup = np.clip(
                    original_pickup + cumulative_delta,
                    [0.0, 0.0],
                    [config.GRID_DIMS[0] - 1, config.GRID_DIMS[1] - 1],
                ).astype(np.float32)
                # Cumulative-epsilon cap: keep within self.epsilon_cap (L-inf) of the
                # TRUE original cell, across rounds. With epsilon_cap == EPSILON_BALL
                # and original_cell == this call's start cell, this is a no-op
                # (new_pickup is already within EPSILON_BALL of original_pickup).
                if self.epsilon_cap is not None and np.isfinite(self.epsilon_cap):
                    new_pickup = np.clip(
                        new_pickup,
                        true_original - self.epsilon_cap,
                        true_original + self.epsilon_cap,
                    ).astype(np.float32)
                # Re-sync cumulative_delta after grid + cumulative-cap clips
                cumulative_delta = new_pickup - original_pickup

                prev_sign = self._prev_grad_sign
                cur_sign = np.sign(grad)
                sign_flipped = (
                    bool(np.any(prev_sign != cur_sign))
                    if (self.diagnostics_enabled and prev_sign is not None)
                    else (False if self.diagnostics_enabled else None)
                )
                self._prev_grad_sign = cur_sign

                result = ModificationResult(
                    iteration=it,
                    objective_value=float(total.detach()),
                    f_spatial=float(terms["f_spatial"].detach()),
                    f_causal=float(terms["f_causal"].detach()),
                    f_fidelity=float(terms["f_fidelity"].detach()),
                    gradient_norm=grad_norm,
                    cumulative_delta=cumulative_delta.copy(),
                    grad_spatial_norm=(tier_a_metrics or {}).get("grad_spatial_norm"),
                    grad_causal_norm=(tier_a_metrics or {}).get("grad_causal_norm"),
                    grad_fidelity_norm=(tier_a_metrics or {}).get("grad_fidelity_norm"),
                    grad_cosine_spatial_causal=(tier_a_metrics or {}).get("grad_cosine_spatial_causal"),
                    grad_cosine_fairness_fidelity=(tier_a_metrics or {}).get("grad_cosine_fairness_fidelity"),
                    sign_flipped=sign_flipped,
                    dominant_term=(tier_a_metrics or {}).get("dominant_term"),
                )
                iterations.append(result)
                if on_iteration is not None:
                    on_iteration(it, result)

                # (g) Best-iterate tracking + patience-based convergence.
                # iter-0 sits at the pre-edit pickup ⇒ captures the baseline F the
                # non-regression gate compares against.
                if it == 0:
                    f_causal_0 = result.f_causal
                    f_spatial_0 = result.f_spatial
                current_objective = float(total.detach())
                if self.accept_rule == "non-regression":
                    qualifies = (
                        result.f_causal >= f_causal_0 + self.convergence_tol
                        and result.f_spatial >= f_spatial_0 - self.convergence_tol
                    )
                else:
                    qualifies = True
                if qualifies and current_objective > best_objective + self.convergence_tol:
                    best_objective = current_objective
                    best_cumulative_delta = cumulative_delta.copy()
                    best_iteration = it
                    iters_since_improvement = 0
                else:
                    iters_since_improvement += 1
                    if (self.patience is not None
                            and iters_since_improvement >= self.patience):
                        converged = True
                        break

        # ── Discretize the BEST iterate's perturbation ────────────────────
        # If no iter exceeded the initial -inf by more than tol (rare),
        # best_cumulative_delta stays at its zero default — no perturbation,
        # the correct fallback.
        if mode == "lift":
            # Move the whole seeking tail toward under-served cells. Infeasible
            # tail repair → skip the edit entirely: the shared demand grid was
            # never touched (the demand subtraction lived only in the local
            # base_3d clone) and the ΔS accumulator stays untouched.
            modified = trajectory.apply_tail_perturbation(
                best_cumulative_delta, config.TAIL_LEN, config.GRID_DIMS,
            )
            if modified is None:
                self.n_taper_infeasible_lift += 1
                return ModificationHistory(
                    original=trajectory,
                    modified=trajectory.clone(),
                    iterations=iterations,
                    converged=converged,
                    total_iterations=len(iterations),
                    final_objective=(
                        iterations[-1].objective_value if iterations else 0.0
                    ),
                    best_iteration=best_iteration,
                    best_objective=best_objective if best_iteration >= 0 else 0.0,
                )
        else:
            modified = self._discretize_trim(trajectory, best_cumulative_delta)

        # Persist change to shared _base_pickup_3d (sub at old, add at new).
        # Uses the BEST iterate's pickup cell, matching the returned trajectory.
        new_cx = int(modified.pickup_state.x_grid)
        new_cy = int(modified.pickup_state.y_grid)
        if (new_cx, new_cy) != (orig_cx, orig_cy):
            self._base_pickup_3d[orig_cx, orig_cy, t_block] -= pickup_mass
            self._base_pickup_3d[new_cx, new_cy, t_block] += pickup_mass

        # Accumulate the FINAL hard tier-1 ΔS of the (tail) move — evaluation
        # honesty. Lift always contributes; trim contributes only when a real
        # tail exists (TAIL_LEN > 0), so a TAIL_LEN == 0 run is bit-for-bit
        # legacy (the accumulator is never read on the trim optimization path).
        if mode == "lift" or config.TAIL_LEN > 0:
            ds = self._hard_tail_delta_supply(trajectory, modified)
            if ds is not None:
                self._delta_supply_3d = self._delta_supply_3d + ds

        return ModificationHistory(
            original=trajectory,
            modified=modified,
            iterations=iterations,
            converged=converged,
            total_iterations=len(iterations),
            final_objective=iterations[-1].objective_value if iterations else 0.0,
            best_iteration=best_iteration,
            best_objective=best_objective if best_iteration >= 0 else 0.0,
        )

    def modify_batch(
        self, trajectories: List[Trajectory],
    ) -> List[ModificationHistory]:
        """Sequentially modify trajectories. Each sees the updated baseline
        from all prior modifications."""
        return [self.modify_single(t) for t in trajectories]
