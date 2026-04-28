"""Tests for algorithm.attribution.

The attribution pipeline uses the canonical 1/N-shifted decomposition
(``per_cell_fairness_attribution_causal``):

    Σᵢ αᵢ == F_causal
    αᵢ > 0 → cell contributes more than 1/N baseline to fairness
    αᵢ < 0 → cell drags fairness below baseline (priority for modification)

``rank_trajectories`` sorts ASCENDING (most-negative first); inactive
cells get a sentinel score of ``+inf`` and sort to the end.
``select_top_k`` returns trajectories with strictly negative attribution.
"""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.algorithm.attribution import (
    compute_per_unit_attribution,
    rank_trajectories,
    select_top_k,
)
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_compute_per_unit_attribution_returns_single_N_vector():
    """Returns a 1-D numpy array of length N (no tuple, no signed variant)."""
    bundle = _make_synthetic_bundle()
    attribution = compute_per_unit_attribution(bundle)
    assert isinstance(attribution, np.ndarray)
    assert attribution.shape == (bundle.unit_map.n_units,)


def test_attribution_sum_matches_fcausal():
    """Σᵢ attributionᵢ == F_causal (1/N-shifted decomposition invariant)."""
    bundle = _make_synthetic_bundle()
    attribution = compute_per_unit_attribution(bundle)

    from famail_temporal.fairness.hat_matrices import (
        hat_matrices_to_torch, compute_fcausal_torch,
    )

    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D_np = bundle.g0_func(D.numpy())
    g0_D = torch.from_numpy(np.asarray(g0_D_np, dtype=np.float32))
    R = Y - g0_D

    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])

    expected = float(f_causal)
    actual = float(attribution.sum())
    np.testing.assert_allclose(actual, expected, atol=1e-5,
                               err_msg="Attribution sum != F_causal")


def test_rank_trajectories_orders_ascending_most_negative_first():
    """rank_trajectories sorts ascending: most-negative αᵢ (drag cells) first."""
    bundle = _make_synthetic_bundle()
    attribution = compute_per_unit_attribution(bundle)

    high_idx = int(np.argmax(attribution))   # above baseline
    low_idx = int(np.argmin(attribution))    # below baseline (priority)
    high_cell = bundle.unit_map.to_flat_cell(high_idx)
    high_t = bundle.unit_map.to_time_block(high_idx)
    low_cell = bundle.unit_map.to_flat_cell(low_idx)
    low_t = bundle.unit_map.to_time_block(low_idx)

    gy = bundle.unit_map.grid_shape[1]

    def _make_traj(cell, t_block, traj_id):
        x, y = cell // gy, cell % gy
        _, start_hour, _ = config.TIME_BLOCKS[t_block]
        tb = 1 + (start_hour * 12)
        states = [
            TrajectoryState(x_grid=0.0, y_grid=0.0, time_bucket=tb, day_index=1),
            TrajectoryState(x_grid=float(x), y_grid=float(y),
                            time_bucket=tb, day_index=1),
        ]
        return Trajectory(trajectory_id=traj_id, driver_id=0, states=states)

    trajs = [_make_traj(high_cell, high_t, 0), _make_traj(low_cell, low_t, 1)]
    ranked = rank_trajectories(trajs, attribution, bundle.unit_map)
    # Ascending: most-negative first → low-attribution traj should be first.
    assert ranked[0][0] == 1


def test_select_top_k_keeps_only_strictly_negative():
    """select_top_k filters by αᵢ < 0 (cells dragging fairness below baseline)."""
    scored = [(0, -0.5), (1, -0.3), (2, 0.0), (3, 0.1)]
    picks = select_top_k(scored, k=4)
    assert picks == [0, 1]


def test_select_top_k_excludes_inactive_inf_sentinel():
    """Inactive cells (+inf score) must never enter the top-k under any k."""
    scored = [(0, -0.2), (1, float("inf")), (2, float("inf"))]
    assert select_top_k(scored, k=3) == [0]


def test_inactive_pickup_gets_inf_score():
    """A trajectory whose pickup maps to an inactive unit gets the +inf sentinel."""
    bundle = _make_synthetic_bundle()
    attribution = compute_per_unit_attribution(bundle)
    gx, gy = bundle.unit_map.grid_shape

    # Find an inactive (cell, t_block) pair.
    inactive_cell = None
    inactive_t = None
    for x in range(gx):
        for y in range(gy):
            for t in range(bundle.mask_3d.shape[2]):
                if not bundle.mask_3d[x, y, t]:
                    inactive_cell = x * gy + y
                    inactive_t = t
                    break
            if inactive_cell is not None:
                break
        if inactive_cell is not None:
            break

    assert inactive_cell is not None, "No inactive cell found in synthetic bundle"

    x, y = inactive_cell // gy, inactive_cell % gy
    _, start_hour, _ = config.TIME_BLOCKS[inactive_t]
    tb = 1 + (start_hour * 12)
    traj = Trajectory(
        trajectory_id=99, driver_id=0,
        states=[
            TrajectoryState(x_grid=0.0, y_grid=0.0, time_bucket=tb, day_index=1),
            TrajectoryState(x_grid=float(x), y_grid=float(y),
                            time_bucket=tb, day_index=1),
        ],
    )
    ranked = rank_trajectories([traj], attribution, bundle.unit_map)
    assert ranked[0][1] == float("inf"), (
        "Inactive-unit trajectory must get the +inf sentinel"
    )


# ── Slow real-data tests ───────────────────────────────────────────────


@pytest.mark.slow
def test_attribution_on_real_data():
    """Compute per-cell attribution on real Shenzhen data — verify finite,
    in-range, and the sum invariant holds at production scale."""
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.hat_matrices import (
        compute_fcausal_compact, hat_matrices_to_torch,
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
        pytest.skip("Cache empty — run preprocess first")

    bundle = DataBundle.load(max_trajectories=50, max_drivers=5)
    attribution = compute_per_unit_attribution(bundle)

    N = bundle.unit_map.n_units
    assert attribution.shape == (N,)
    assert np.isfinite(attribution).all(), "attribution contains non-finite values"

    # Sum invariant at production scale (compact FWL form).
    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D = torch.from_numpy(np.asarray(bundle.g0_func(D.numpy()), dtype=np.float32))
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal = compute_fcausal_compact(R, tensors['X_demo'], tensors['XtX_inv'])
    attr_sum = float(attribution.sum())
    expected = float(f_causal)
    diff = abs(attr_sum - expected)
    assert diff < 0.01, (
        f"Attribution sum invariant broken at production scale: "
        f"sum={attr_sum:.6f}, F_causal={expected:.6f}, diff={diff:.2e}"
    )

    if len(bundle.trajectories) > 0:
        ranked = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
        assert len(ranked) == len(bundle.trajectories)
        # Drag cells: trajectories with strictly negative attribution.
        drag = [s for _, s in ranked if s < 0]
        print(f"\n  Real data: {len(drag)}/{len(ranked)} trajectories have negative αᵢ")
        print(f"  Top-5 (most-negative) scores: {[f'{s:.4f}' for _, s in ranked[:5]]}")
        print(f"  Attribution sum invariant diff: {diff:.2e}")
