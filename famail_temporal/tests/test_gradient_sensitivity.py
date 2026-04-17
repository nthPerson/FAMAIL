"""Tests for evaluation.diagnostics.compute_gradient_sensitivity."""
import numpy as np
import pytest

from famail_temporal import config
from famail_temporal.evaluation.diagnostics import compute_gradient_sensitivity
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_returns_correct_shape():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=0)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    gx, gy = bundle.pickup_3d.shape[:2]
    assert sens.shape == (gx, gy, config.T, 2)
    assert sens.dtype == np.float32


def test_inactive_cells_are_nan():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=1)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    inactive = ~bundle.mask_3d
    assert np.isnan(sens[inactive]).all()


def test_active_cells_are_finite():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=2)
    sens = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    active = bundle.mask_3d
    for c in range(2):
        assert np.isfinite(sens[active, c]).all()


def test_sensitivity_changes_under_pickup_modification():
    bundle = _make_synthetic_bundle(N_cells_per_block=10, seed=3)
    sens_a = compute_gradient_sensitivity(bundle, bundle.pickup_3d)
    pickup_mod = bundle.pickup_3d.copy()
    active_ix = np.argwhere(bundle.mask_3d)
    x0, y0, t0 = active_ix[0]
    pickup_mod[x0, y0, t0] += 1.0
    sens_b = compute_gradient_sensitivity(bundle, pickup_mod)
    assert not np.allclose(
        sens_a[bundle.mask_3d], sens_b[bundle.mask_3d],
    )
