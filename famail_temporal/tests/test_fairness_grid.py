"""Tests for evaluation.grid.build_fairness_grid."""
import numpy as np
import pytest
import torch

from famail_temporal import config
from famail_temporal.evaluation.grid import build_fairness_grid
from famail_temporal.fairness.spatial import compute_fspatial
from famail_temporal.fairness.hat_matrices import hat_matrices_to_torch
from famail_temporal.fairness.causal import per_unit_attribution
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_returns_correct_shape():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    gx, gy = bundle.pickup_3d.shape[:2]
    assert grid.shape == (gx, gy, config.T, 4)
    assert grid.dtype == np.float32


def test_inactive_cells_are_nan():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    inactive = ~bundle.mask_3d
    assert np.isnan(grid[inactive]).all()


def test_active_cells_are_finite():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    grid = build_fairness_grid(bundle)
    active = bundle.mask_3d
    for c in range(4):
        assert np.isfinite(grid[active, c]).all(), f"channel {c} has NaN on active cells"


def test_spatial_attr_channel_sums_to_one_minus_fspatial():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    grid = build_fairness_grid(bundle)
    pickup_N = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    f_spatial, _ = compute_fspatial(pickup_N, dropoff_N, active_N)
    assert np.isclose(np.nansum(grid[..., 0]), 1.0 - float(f_spatial), atol=1e-5)


def test_causal_attr_channel_sums_to_one_minus_fcausal():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    grid = build_fairness_grid(bundle)
    D = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    S = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    D_clamped = torch.clamp(D, min=config.DEMAND_FLOOR)
    Y = S / D_clamped
    g0_D = torch.from_numpy(np.asarray(bundle.g0_func(D_clamped.numpy()), dtype=np.float32))
    R = Y - g0_D
    tensors = hat_matrices_to_torch(bundle.hat_matrices)
    expected = float(per_unit_attribution(R, tensors["I_minus_H_demo"], tensors["M"]).sum())
    assert np.isclose(np.nansum(grid[..., 1]), expected, atol=1e-5)


def test_gini_dsr_channel_sums_to_dsr_gini():
    from famail_temporal.fairness.spatial import pairwise_gini
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=4)
    grid = build_fairness_grid(bundle)
    pickup_N = torch.from_numpy(bundle.pickup_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    dsr = pickup_N / (active_N + config.EPS)
    assert np.isclose(np.nansum(grid[..., 2]), float(pairwise_gini(dsr)), atol=1e-6)


def test_gini_asr_channel_sums_to_asr_gini():
    from famail_temporal.fairness.spatial import pairwise_gini
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=6)
    grid = build_fairness_grid(bundle)
    dropoff_N = torch.from_numpy(bundle.dropoff_3d[bundle.mask_3d]).float()
    active_N = torch.from_numpy(bundle.active_taxis_3d[bundle.mask_3d]).float()
    asr = dropoff_N / (active_N + config.EPS)
    assert np.isclose(np.nansum(grid[..., 3]), float(pairwise_gini(asr)), atol=1e-6)


def test_channel_0_equals_half_sum_of_channels_2_3_on_active():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=8)
    grid = build_fairness_grid(bundle)
    active = bundle.mask_3d
    lhs = grid[..., 0][active]
    rhs = 0.5 * (grid[..., 2][active] + grid[..., 3][active])
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_pickup_override_changes_grid():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=9)
    grid_default = build_fairness_grid(bundle)
    pickup_mod = bundle.pickup_3d.copy()
    active_ix = np.argwhere(bundle.mask_3d)
    x0, y0, t0 = active_ix[0]
    x1, y1, t1 = active_ix[1]
    pickup_mod[x0, y0, t0] = max(0.0, pickup_mod[x0, y0, t0] - 0.5)
    pickup_mod[x1, y1, t1] += 0.5
    grid_mod = build_fairness_grid(bundle, pickup_3d=pickup_mod)
    assert not np.allclose(
        grid_default[..., 0][bundle.mask_3d],
        grid_mod[..., 0][bundle.mask_3d],
    ), "Channel 0 should change when pickup_3d changes"
