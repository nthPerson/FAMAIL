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
    """Build a minimal DataBundle for testing — small grid, synthetic data.

    Uses config.T for the time-block axis so the bundle matches whatever
    T the global config is set to (4 during framework validation, 24 in
    production). Hardcoded previously to T=4.
    """
    rng = np.random.RandomState(seed)
    gx, gy, t = 8, 8, config.T

    # Random active mask with approximately N_cells_per_block per block
    mask = np.zeros((gx, gy, t), dtype=bool)
    n_target = N_cells_per_block * t
    positions = rng.choice(gx * gy * t, min(n_target, gx * gy * t), replace=False)
    for p in positions:
        cell = p // t
        tb = p % t
        x, y = cell // gy, cell % gy
        mask[x, y, tb] = True

    unit_map = UnitIndexMap.from_mask(mask, grid_shape=(gx, gy))
    N = unit_map.n_units

    # Build tensors with some variation for meaningful Gini/R^2
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

    # Demographics and hat matrices at the active-unit level
    demographics = rng.randn(N, 3).astype(np.float32)
    D_vec = np.maximum(pickup_3d[mask], config.DEMAND_FLOOR)
    hat = precompute_hat_matrices(D_vec, demographics, ["a", "b", "c"])

    # Fit g0 on the active-unit scale
    S_vec = active_3d[mask]
    Y_vec = S_vec / D_vec
    g0 = G0Function(
        coefficients=np.array([0.5, 0.1, 0.1, 0.01]),
        d_min=float(D_vec.min()),
        d_max=float(D_vec.max()),
    )

    bundle = DataBundle(
        pickup_3d=pickup_3d,
        dropoff_3d=dropoff_3d,
        active_taxis_3d=active_3d,
        mask_3d=mask,
        n_hours_per_block=np.array(
            [config.TIME_BLOCKS[i][2] - config.TIME_BLOCKS[i][1]
             for i in range(t)],
            dtype=np.int32,
        ),
        n_days=65,
        unit_map=unit_map,
        g0_func=g0,
        hat_matrices=hat,
        trajectories=[],
        multi_stream=MultiStreamData(
            driving_trajs={}, seeking_trajs={},
            profile_features={}, seeking_days={}, driving_days={},
        ),
        discriminator=nn.Identity(),
    )
    return bundle


# ── Core required tests ─────────────────────────────────────────────────


def test_famailobjective_forward_returns_scalar_total():
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    total, terms = obj(soft_pickup_3d=soft_3d)
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
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float().requires_grad_(True)
    total, _ = obj(soft_pickup_3d=soft_3d)
    total.backward()
    assert soft_3d.grad is not None
    assert not torch.isnan(soft_3d.grad).any()
    mask_t = torch.from_numpy(bundle.mask_3d)
    assert (soft_3d.grad[mask_t].abs() > 0).any()


# ── Hardening tests ─────────────────────────────────────────────────────


def test_alpha_spatial_zero_does_not_contribute():
    """Setting alpha_spatial=0 means F_spatial has zero weight in total."""
    bundle = _make_synthetic_bundle()
    obj_no_sp = FAMAILObjective(bundle, alpha_spatial=0.0, alpha_causal=1.0, alpha_fidelity=0.0)
    obj_with_sp = FAMAILObjective(bundle, alpha_spatial=1.0, alpha_causal=1.0, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    total_no_sp, terms_no_sp = obj_no_sp(soft_pickup_3d=soft_3d)
    total_with_sp, terms_with_sp = obj_with_sp(soft_pickup_3d=soft_3d)

    # Total should equal just f_causal when alpha_spatial=0
    assert torch.isclose(total_no_sp, terms_no_sp["f_causal"], atol=1e-6)
    # With spatial on, total should be different (unless f_spatial happens to be 0)
    f_sp = float(terms_with_sp["f_spatial"])
    if f_sp > 1e-6:
        assert not torch.isclose(total_with_sp, terms_with_sp["f_causal"], atol=1e-6)


def test_alpha_fidelity_zero_skips_discriminator():
    """alpha_fidelity=0 should skip fidelity computation without error,
    even when no tau_features are provided."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    total, terms = obj(soft_pickup_3d=soft_3d)
    assert float(terms["f_fidelity"]) == 0.0


def test_debug_dict_completeness():
    """Verify expected debug keys are present in the terms dict."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    _, terms = obj(soft_pickup_3d=soft_3d)

    # Core terms
    assert "f_spatial" in terms
    assert "f_causal" in terms
    assert "f_fidelity" in terms
    assert "total" in terms

    # Debug keys from spatial
    assert "debug_spatial_gini_dsr" in terms
    assert "debug_spatial_gini_asr" in terms

    # Debug keys from causal
    assert "debug_causal_Y_min" in terms
    assert "debug_causal_Y_max" in terms
    assert "debug_causal_R_min" in terms
    assert "debug_causal_R_max" in terms
    assert "debug_causal_f_causal" in terms


def test_device_consistency_of_buffers():
    """All registered buffers should be on the same device."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    devices = set()
    for name, buf in obj.named_buffers():
        devices.add(buf.device)
    assert len(devices) == 1, f"Buffers on multiple devices: {devices}"


def test_total_equals_weighted_sum():
    """Verify total = alpha_spatial * f_spatial + alpha_causal * f_causal + alpha_fidelity * f_fidelity."""
    bundle = _make_synthetic_bundle()
    a_sp, a_cs, a_fi = 0.4, 0.5, 0.0
    obj = FAMAILObjective(bundle, alpha_spatial=a_sp, alpha_causal=a_cs, alpha_fidelity=a_fi)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    total, terms = obj(soft_pickup_3d=soft_3d)
    expected = a_sp * terms["f_spatial"] + a_cs * terms["f_causal"] + a_fi * terms["f_fidelity"]
    assert torch.isclose(total, expected, atol=1e-6), (
        f"total={float(total)}, expected={float(expected)}"
    )


def test_gradient_is_zero_outside_mask():
    """Gradient should be zero at grid positions outside the active mask."""
    bundle = _make_synthetic_bundle()
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float().requires_grad_(True)
    total, _ = obj(soft_pickup_3d=soft_3d)
    total.backward()
    mask_t = torch.from_numpy(bundle.mask_3d)
    outside_grad = soft_3d.grad[~mask_t]
    assert (outside_grad == 0.0).all(), "Gradient should be zero outside mask"


def test_different_seeds_produce_different_bundles():
    """Sanity check: different seeds produce different data."""
    b1 = _make_synthetic_bundle(seed=0)
    b2 = _make_synthetic_bundle(seed=42)
    assert not np.array_equal(b1.pickup_3d, b2.pickup_3d)


# ── Slow real-data tests ───────────────────────────────────────────────


@pytest.mark.slow
def test_famailobjective_on_real_data():
    """Run FAMAILObjective on real Shenzhen data — verify metrics are
    finite and in [0, 1] at production scale."""
    from famail_temporal import config
    from famail_temporal.data.loader import DataBundle

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

    bundle = DataBundle.load(max_trajectories=10, max_drivers=2)
    obj = FAMAILObjective(bundle, alpha_fidelity=0.0)
    soft_3d = torch.from_numpy(bundle.pickup_3d).float()
    total, terms = obj(soft_pickup_3d=soft_3d)

    # All metrics finite and in [0, 1]
    assert torch.isfinite(total), f"total is not finite: {float(total)}"
    assert 0.0 <= float(terms['f_spatial']) <= 1.0, (
        f"f_spatial out of range: {float(terms['f_spatial'])}"
    )
    assert 0.0 <= float(terms['f_causal']) <= 1.0, (
        f"f_causal out of range: {float(terms['f_causal'])}"
    )

    # Print values for researcher inspection
    print(f"\n  Real data F_spatial = {float(terms['f_spatial']):.4f}")
    print(f"  Real data F_causal  = {float(terms['f_causal']):.4f}")
    print(f"  Real data total     = {float(total):.4f}")
    print(f"  N active units      = {bundle.unit_map.n_units}")
