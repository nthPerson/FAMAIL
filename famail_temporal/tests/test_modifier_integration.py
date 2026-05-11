"""Integration tests for the full modifier loop.

These tests exercise the complete pipeline: attribution -> ranking -> ST-iFGSM
modification. Both synthetic and real-data tests are included.

Trajectory length considerations:
- Real Shenzhen trajectories range from ~2 to >500 states.
- The modifier only perturbs the terminal state (pickup), but the full
  trajectory is passed to the fidelity discriminator for realism assessment.
- Tests here use varied lengths (2, 10, 15, 50) to verify the modifier
  handles short and medium-length sequences correctly.
"""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.modifier import TrajectoryModifier
from famail_temporal.algorithm.objective import FAMAILObjective
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def _make_test_trajectory(pickup_xy, time_bucket, n_states=10):
    """Build a test trajectory with a specified number of states.

    States interpolate from (0,0) to the pickup location to give the
    discriminator a realistic-length sequence to process.
    """
    px, py = pickup_xy
    states = []
    for i in range(n_states):
        frac = i / max(n_states - 1, 1)
        states.append(TrajectoryState(
            x_grid=float(px * frac),
            y_grid=float(py * frac),
            time_bucket=time_bucket,
            day_index=1,
        ))
    return Trajectory(trajectory_id=0, driver_id=0, states=states)


def _pick_active_unit_coords(bundle):
    """Return (x, y, t_block, time_bucket) for unit index 0 in the bundle."""
    cell = bundle.unit_map.to_flat_cell(0)
    t_block = bundle.unit_map.to_time_block(0)
    gy = bundle.pickup_3d.shape[1]
    x, y = cell // gy, cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[t_block]
    tb = 1 + (start_hour * 12)  # 1-indexed time_bucket in that block
    return x, y, t_block, tb


def test_five_iteration_objective_improves_or_plateaus():
    """Over 5 iterations the objective should improve or plateau, not regress."""
    bundle = _make_synthetic_bundle(seed=0)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5,
    )

    x, y, _, tb = _pick_active_unit_coords(bundle)
    traj = _make_test_trajectory((x, y), tb, n_states=15)
    history = modifier.modify_single(traj)

    values = [r.objective_value for r in history.iterations]
    assert len(values) > 0
    first = values[0]
    last = values[-1]
    # Allow a small tolerance for numerical noise
    assert last >= first - 1e-3, (
        f"Objective decreased from {first} to {last}"
    )


def test_mass_balance_after_single_modification():
    """pickup_3d.sum() must be preserved after modification (mass balance)."""
    bundle = _make_synthetic_bundle(seed=1)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=3,
    )

    mass_before = modifier._base_pickup_3d.sum().item()

    x, y, _, tb = _pick_active_unit_coords(bundle)
    traj = _make_test_trajectory((x, y), tb, n_states=10)
    modifier.modify_single(traj)

    mass_after = modifier._base_pickup_3d.sum().item()
    assert abs(mass_after - mass_before) < 1e-5, (
        f"Mass imbalance: {mass_before} -> {mass_after}"
    )


def test_modifier_handles_varied_trajectory_lengths():
    """The modifier should work correctly on trajectories of varying lengths.

    Real Shenzhen trajectories range from ~2 to >500 states. The modifier
    only perturbs the terminal state (pickup), but the full trajectory is
    passed to the fidelity discriminator for realism assessment.
    """
    bundle = _make_synthetic_bundle(seed=2)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)

    x, y, _, tb = _pick_active_unit_coords(bundle)

    for n_states in [2, 10, 50]:
        modifier = TrajectoryModifier(
            objective=obj, bundle=bundle, max_iterations=3,
        )
        traj = _make_test_trajectory((x, y), tb, n_states=n_states)
        history = modifier.modify_single(traj)
        assert history.total_iterations <= 3
        assert history.modified.n_states == n_states
        # The modification only touches the last state
        assert history.modified.states[0].x_grid == traj.states[0].x_grid


def test_modify_batch_sequential_baseline_update():
    """modify_batch processes trajectories sequentially — each sees the
    updated baseline from previous modifications."""
    bundle = _make_synthetic_bundle(seed=3)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=3,
    )

    x, y, _, tb = _pick_active_unit_coords(bundle)
    trajs = [_make_test_trajectory((x, y), tb, n_states=10) for _ in range(3)]
    for i, t in enumerate(trajs):
        t.trajectory_id = i

    histories = modifier.modify_batch(trajs)
    assert len(histories) == 3
    for h in histories:
        assert isinstance(h.total_iterations, int)
        assert h.total_iterations >= 0


def test_convergence_stops_early():
    """With convergence_tol large enough, the loop should stop before max_iterations."""
    bundle = _make_synthetic_bundle(seed=4)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle,
        max_iterations=50, convergence_tol=1.0,  # very loose tolerance
    )

    x, y, _, tb = _pick_active_unit_coords(bundle)
    traj = _make_test_trajectory((x, y), tb, n_states=15)
    history = modifier.modify_single(traj)

    # With such a loose tolerance, should converge in 2 iterations (first
    # iteration sets prev_objective, second compares and triggers convergence)
    assert history.converged, "Should have converged with convergence_tol=1.0"
    assert history.total_iterations < 50


@pytest.mark.slow
def test_modifier_on_real_data():
    """Run modify_single on real Shenzhen trajectories.

    Filters trajectories to 5 <= n_states <= 300 for meaningful testing.
    Verifies: metrics in range, epsilon-ball respected, mass-balance preserved.
    """
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.algorithm.attribution import (
        compute_per_unit_attribution, rank_trajectories, select_top_k,
    )

    required = [
        config.SOURCE_DATA_DIR / "pickup_dropoff_counts.pkl",
        config.SOURCE_DATA_DIR / "cell_demographics.pkl",
    ]
    for path in required:
        if not path.exists():
            pytest.skip(f"Raw data missing: {path}")
    cache_files = list(config.CACHE_DIR.glob("*.pkl"))
    if not cache_files:
        pytest.skip("Cache empty -- run preprocess first")

    bundle = DataBundle.load(max_trajectories=200, max_drivers=10)

    # Filter to effective trajectory lengths
    valid_trajs = [t for t in bundle.trajectories if 5 <= t.n_states <= 300]
    if len(valid_trajs) < 5:
        pytest.skip(f"Too few valid-length trajectories: {len(valid_trajs)}")

    # Print length statistics for researcher inspection
    lengths = [t.n_states for t in valid_trajs]
    print(f"\n  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"median={sorted(lengths)[len(lengths)//2]}, n={len(lengths)}")

    # Attribution + ranking
    attribution = compute_per_unit_attribution(bundle)
    ranked = rank_trajectories(valid_trajs, attribution, bundle.unit_map)
    top_indices = select_top_k(ranked, k=3)

    if len(top_indices) == 0:
        pytest.skip("No trajectories with strictly negative attribution found")

    # Modify top-ranked trajectory
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    modifier = TrajectoryModifier(
        objective=obj, bundle=bundle, max_iterations=5,
    )
    mass_before = modifier._base_pickup_3d.sum().item()

    traj = valid_trajs[top_indices[0]]
    print(f"  Modifying trajectory {traj.trajectory_id} "
          f"(driver={traj.driver_id}, n_states={traj.n_states})")

    history = modifier.modify_single(traj)

    # Assertions
    assert history.total_iterations > 0
    for r in history.iterations:
        assert 0.0 <= r.f_spatial <= 1.0, f"f_spatial={r.f_spatial}"
        assert 0.0 <= r.f_causal <= 1.0, f"f_causal={r.f_causal}"

    # Epsilon-ball
    orig = np.array([traj.pickup_state.x_grid, traj.pickup_state.y_grid])
    final = np.array([history.modified.pickup_state.x_grid,
                       history.modified.pickup_state.y_grid])
    displacement = np.abs(final - orig)
    assert (displacement <= config.EPSILON_BALL + 1e-5).all(), (
        f"Displacement {displacement} exceeds epsilon={config.EPSILON_BALL}"
    )

    # Mass balance — use relative tolerance because float32 precision at
    # production-scale sums (~10^4) limits absolute accuracy to ~10^-3.
    mass_after = modifier._base_pickup_3d.sum().item()
    rel_diff = abs(mass_after - mass_before) / (abs(mass_before) + 1e-10)
    assert rel_diff < 1e-6, (
        f"Mass imbalance on real data: {mass_before:.6f} -> {mass_after:.6f} "
        f"(relative diff={rel_diff:.2e})"
    )

    print(f"  Final objective: {history.final_objective:.4f}")
    print(f"  Displacement: dx={displacement[0]:.2f}, dy={displacement[1]:.2f}")
    print(f"  Converged: {history.converged}")
