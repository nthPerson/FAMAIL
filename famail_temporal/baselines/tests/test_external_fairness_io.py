import numpy as np
import pytest

from famail_temporal.baselines import external_fairness_io as io
from famail_temporal.tests.test_objective import _make_synthetic_bundle


def test_service_ratio_matches_manual():
    bundle = _make_synthetic_bundle()
    Y = io.service_ratio_Y(bundle.pickup_3d, bundle)
    mask = bundle.mask_3d
    # Cast to float64 before dividing: bundle tensors are float32, and the
    # implementation intentionally computes in float64 (matching
    # per_unit_demographics' convention). Comparing against a float32
    # computation here would fail rtol=1e-9 on pure float32 rounding noise
    # rather than an actual formula mismatch.
    demand = bundle.pickup_3d[mask].astype(np.float64)
    supply = bundle.active_taxis_3d[mask].astype(np.float64)
    expected = supply / np.maximum(demand, 0.5)
    assert Y.shape == (int(mask.sum()),)
    np.testing.assert_allclose(Y, expected, rtol=1e-9)


def test_per_unit_demographics_injected_grid_shapes_and_values():
    bundle = _make_synthetic_bundle()
    gx, gy, _ = bundle.mask_3d.shape
    # synthetic (gx, gy, 3) grid: axis j = constant j+1 everywhere
    sel = np.zeros((gx, gy, 3), dtype=np.float64)
    for j in range(3):
        sel[..., j] = j + 1
    demo = io.per_unit_demographics(bundle, selected_grid=sel)
    n = int(bundle.mask_3d.sum())
    for j, axis in enumerate(io.EQUITY_AXES):
        assert demo[axis].shape == (n,)
        np.testing.assert_allclose(demo[axis], j + 1)


def test_equity_axes_and_pole_constants():
    assert io.EQUITY_AXES == ["AvgHousingPricePerSqM", "CompPerCapita", "MigrantRatio"]
    assert io.DISADVANTAGED_HIGH["MigrantRatio"] is True
    assert io.DISADVANTAGED_HIGH["AvgHousingPricePerSqM"] is False
