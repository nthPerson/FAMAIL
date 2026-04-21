"""Tests for algorithm.attribution."""
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


def test_per_unit_attribution_returns_N_vector():
    bundle = _make_synthetic_bundle()
    attribution, signed = compute_per_unit_attribution(bundle)
    assert attribution.shape == (bundle.unit_map.n_units,)
    assert signed.shape == (bundle.unit_map.n_units,)


def test_rank_trajectories_orders_by_attribution():
    bundle = _make_synthetic_bundle()
    attribution, _ = compute_per_unit_attribution(bundle)

    high_idx = int(np.argmax(attribution))
    low_idx = int(np.argmin(attribution))
    high_cell = bundle.unit_map.to_flat_cell(high_idx)
    high_t = bundle.unit_map.to_time_block(high_idx)
    low_cell = bundle.unit_map.to_flat_cell(low_idx)
    low_t = bundle.unit_map.to_time_block(low_idx)

    gy = bundle.pickup_3d.shape[1]

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
    assert ranked[0][0] == 0  # high-attribution traj first


def test_select_top_k_drops_zero_attribution():
    scored = [(0, 0.5), (1, 0.3), (2, 0.0), (3, -0.1)]
    picks = select_top_k(scored, k=4)
    assert picks == [0, 1]


# ── Hardening tests ─────────────────────────────────────────────────────


def test_inactive_pickup_gets_score_zero():
    """A trajectory whose pickup maps to an inactive unit gets score 0."""
    bundle = _make_synthetic_bundle()
    attribution, _ = compute_per_unit_attribution(bundle)
    gy = bundle.pickup_3d.shape[1]
    gx = bundle.pickup_3d.shape[0]

    # Find an inactive (cell, t_block) pair
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
    assert ranked[0][1] == 0.0, "Inactive-unit trajectory should have score 0"


def test_attribution_sum_matches_one_minus_fcausal():
    """unsigned.sum() should equal 1 - F_causal (the load-bearing invariant)."""
    bundle = _make_synthetic_bundle()
    attribution, _ = compute_per_unit_attribution(bundle)

    # Recompute F_causal independently
    from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch, compute_fcausal_torch

    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D_np = bundle.g0_func(D.numpy())
    g0_D = torch.from_numpy(np.asarray(g0_D_np, dtype=np.float32))
    R = Y - g0_D

    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])

    expected = 1.0 - float(f_causal)
    actual = float(attribution.sum())
    np.testing.assert_allclose(actual, expected, atol=1e-5,
                               err_msg="Attribution sum != 1 - F_causal")


def test_signed_magnitude_equals_unsigned():
    """|signed| should equal unsigned per element."""
    bundle = _make_synthetic_bundle()
    unsigned, signed = compute_per_unit_attribution(bundle)
    np.testing.assert_allclose(np.abs(signed), np.abs(unsigned), atol=1e-7,
                               err_msg="|signed| != |unsigned| per element")


# ── Slow real-data tests ───────────────────────────────────────────────


@pytest.mark.slow
def test_attribution_on_real_data():
    """Compute per-unit attribution on real Shenzhen data — verify finite,
    in-range, and the sum invariant holds at production scale."""
    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle
    from famail_temporal.fairness.hat_matrices import compute_fcausal_torch, hat_matrices_to_torch

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
    attribution, signed = compute_per_unit_attribution(bundle)

    N = bundle.unit_map.n_units
    assert attribution.shape == (N,)
    assert signed.shape == (N,)

    # All values finite
    assert np.isfinite(attribution).all(), "attribution contains non-finite values"
    assert np.isfinite(signed).all(), "signed attribution contains non-finite values"

    # Sum invariant at production scale
    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D
    g0_D = torch.from_numpy(np.asarray(bundle.g0_func(D.numpy()), dtype=np.float32))
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    f_causal = compute_fcausal_torch(R, tensors['I_minus_H_demo'], tensors['M'])
    attr_sum = float(attribution.sum())
    expected = 1.0 - float(f_causal)
    diff = abs(attr_sum - expected)
    assert diff < 0.01, (
        f"Attribution sum invariant broken at production scale: "
        f"sum={attr_sum:.6f}, 1-F_causal={expected:.6f}, diff={diff:.2e}"
    )

    # Rank real trajectories
    if len(bundle.trajectories) > 0:
        ranked = rank_trajectories(bundle.trajectories, attribution, bundle.unit_map)
        assert len(ranked) == len(bundle.trajectories)
        # At least some trajectories should have non-zero scores
        non_zero = [s for _, s in ranked if s > 0]
        print(f"\n  Real data: {len(non_zero)}/{len(ranked)} trajectories have positive attribution")
        print(f"  Top-5 scores: {[f'{s:.4f}' for _, s in ranked[:5]]}")
        print(f"  Attribution sum invariant diff: {diff:.2e}")
