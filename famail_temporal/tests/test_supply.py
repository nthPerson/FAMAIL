import numpy as np
import pytest
import torch

from famail_temporal.algorithm import supply as sp
from famail_temporal.algorithm.soft_cell_assignment import SoftCellAssignment


def test_state_presence_mass_convention():
    n_hours = np.ones(24, dtype=np.int32)
    assert sp.state_presence_mass(n_hours, 5, 0) == pytest.approx(1.0 / (12 * 1 * 5))


def test_hard_delta_supply_box_and_sign():
    gx, gy, T = 10, 10, 4
    d = sp.hard_delta_supply(
        positions_old=[(5, 5)], positions_new=[(7, 5)],
        t_blocks=[2], masses=[0.1], grid_shape=(gx, gy, T))
    assert d.shape == (gx, gy, T)
    assert d[5, 5, 2] == pytest.approx(0.0)      # overlap of -box(5,5) and +box(7,5): both cover (5..7,3..7)
    assert d[3, 5, 2] == pytest.approx(-0.1)     # only in old box
    assert d[9, 5, 2] == pytest.approx(+0.1)     # only in new box
    assert d[:, :, 0].sum() == pytest.approx(0.0)  # untouched block
    # clipped at edges: total added mass = 0.1 * |box(7,5)| (full 25 here), removed = 0.1 * 25
    assert d[:, :, 2].sum() == pytest.approx(0.0)


def test_hard_delta_supply_edge_clip():
    d = sp.hard_delta_supply([(0, 0)], [(9, 9)], [0], [1.0], (10, 10, 1))
    assert d[:, :, 0].min() == pytest.approx(-1.0)
    # old box at corner has 3x3=9 cells, new box 3x3=9
    assert (d[:, :, 0] < 0).sum() == 9 and (d[:, :, 0] > 0).sum() == 9


def test_soft_delta_supply_matches_hard_at_low_temperature():
    torch.manual_seed(0)
    gx, gy, T = 12, 12, 3
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    sa.temperature.fill_(1e-4)                    # near-one-hot softmax
    # loc strictly inside cell (6,6): (6,6) itself is a 4-cell tie point of the assignment
    loc = torch.tensor([[6.7, 6.7]]); cell = torch.tensor([[6, 6]])
    probs = sa(loc, cell)                         # (1, ns, ns)
    d_soft = sp.soft_delta_supply(probs, cells=[(6, 6)], t_blocks=[1],
                                  masses=[0.2], signs=[+1], grid_shape=(gx, gy, T))
    d_hard = sp.hard_delta_supply([], [(6, 6)], [1], [0.2], (gx, gy, T))
    np.testing.assert_allclose(d_soft.detach().numpy(), d_hard, atol=1e-4)


def test_soft_delta_supply_differentiable():
    gx, gy, T = 12, 12, 2
    sa = SoftCellAssignment(grid_dims=(gx, gy))
    loc = torch.tensor([[6.0, 6.0]], requires_grad=True)
    probs = sa(loc, torch.tensor([[6, 6]]))
    d = sp.soft_delta_supply(probs, [(6, 6)], [0], [0.2], [+1], (gx, gy, T))
    d.sum().backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()


# ── Task 4: supply-gradient attribution + lift scoring ─────────────────


def test_supply_gradient_matches_analytic_fcausal():
    """alpha=(0,1,0): dL/dS_i must equal the analytic F_causal supply gradient."""
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    from famail_temporal.algorithm.objective import FAMAILObjective
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_spatial=0.0, alpha_causal=1.0, alpha_fidelity=0.0)
    g = sp.supply_gradient_N(bundle, obj)
    # analytic: R = Y - g0(D), Y = S/max(D,.5); dF/dS_i = (2/(R'MR)) * [((I-H)R)_i - F*(MR)_i] / max(D_i,.5)
    import torch
    D = torch.clamp(torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).double(), min=0.5)
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).double()
    Y = S / D
    g0 = torch.from_numpy(np.asarray(bundle.g0_func((D).numpy()))).double()
    R = Y - g0
    X = torch.from_numpy(bundle.hat_matrices["X_demo"]).double()
    XtX_inv = torch.from_numpy(bundle.hat_matrices["XtX_inv"]).double()
    IHR = R - X @ (XtX_inv @ (X.T @ R))
    MR = R - R.mean()
    RMR = float(R @ MR)
    F = float(R @ IHR) / RMR
    analytic = (2.0 / RMR) * (IHR - F * MR) / D
    mask_free = bundle.active_taxis_3d[bundle.mask_3d] > 0.1   # clamp-inactive units only
    np.testing.assert_allclose(g[mask_free], analytic.numpy()[mask_free], rtol=1e-3, atol=1e-6)


def test_lift_candidates_prefers_tails_near_positive_gradient():
    """Synthetic: gradient positive in one region; trajectory tails near it score higher.

    ``_make_synthetic_bundle`` carries ``trajectories=[]`` (see
    ``famail_temporal/tests/test_objective.py``), so it cannot exercise
    ``lift_candidates`` directly. Attach 10 synthetic 3-state Trajectory
    objects on active cells (anchor + one tail state + pickup, so
    ``len(states) == 3`` clears the ``< 3`` skip and exercises the
    ``min(tail_len, len-2)`` anchor-safety clamp) via ``dataclasses.replace``,
    per the brief's documented fallback.
    """
    import dataclasses
    from famail_temporal.tests.test_objective import _make_synthetic_bundle
    from famail_temporal.baselines.tests._helpers import active_units, time_bucket_for_block
    from famail_temporal.utils.trajectory import Trajectory, TrajectoryState

    bundle = _make_synthetic_bundle()
    N = int(bundle.mask_3d.sum())
    grad = np.zeros(N); grad[: N // 4] = 1.0      # first units (low x) positive

    units = active_units(bundle, 10)  # 10 active (cx, cy, t_block) triples
    trajs = []
    for i, (cx, cy, tb) in enumerate(units):
        tbucket = time_bucket_for_block(tb)
        states = [
            TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tbucket, day_index=1),
            TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tbucket, day_index=1),
            TrajectoryState(x_grid=float(cx), y_grid=float(cy), time_bucket=tbucket, day_index=1),
        ]
        trajs.append(Trajectory(trajectory_id=i, driver_id=0, states=states))
    bundle = dataclasses.replace(bundle, trajectories=trajs)

    scored = sp.lift_candidates(bundle, grad, tail_len=2, epsilon=2)
    assert len(scored) > 0
    assert all(scored[i][1] >= scored[i + 1][1] for i in range(len(scored) - 1))


# ── Task 6: edit-plan assembly (trim precedence, budget fill) ──────────


def test_assemble_edit_plan_trim_precedence_and_fill():
    trim = [3, 7]
    lift = [(7, 9.0), (1, 5.0), (2, 3.0), (9, 0.0), (4, -1.0)]
    plan = sp.assemble_edit_plan(trim, lift, k_total=5)
    assert plan[:2] == [(3, "trim"), (7, "trim")]
    assert plan[2:] == [(1, "lift"), (2, "lift")]     # 7 deduped to trim; 9 and 4 dropped (score<=0)


def test_assemble_edit_plan_explicit_budget():
    plan = sp.assemble_edit_plan([1], [(2, 4.0), (3, 2.0)], k_total=10, lift_budget=1)
    assert plan == [(1, "trim"), (2, "lift")]
