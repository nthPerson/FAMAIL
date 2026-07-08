import numpy as np
import pytest

from famail_temporal import config
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


def test_service_ratio_supply_override_used():
    bundle = _make_synthetic_bundle()
    mask = bundle.mask_3d
    # distinct from bundle.active_taxis_3d so the override is detectable
    custom_supply = bundle.active_taxis_3d + 5.0
    Y_default = io.service_ratio_Y(bundle.pickup_3d, bundle)
    Y_override = io.service_ratio_Y(bundle.pickup_3d, bundle, supply_3d=custom_supply)
    demand = bundle.pickup_3d[mask].astype(np.float64)
    expected_override = (custom_supply[mask].astype(np.float64)
                          / np.maximum(demand, config.DEMAND_FLOOR))
    np.testing.assert_allclose(Y_override, expected_override, rtol=1e-9)
    assert not np.allclose(Y_override, Y_default)


def test_service_ratio_default_path_unchanged_when_supply_omitted():
    bundle = _make_synthetic_bundle()
    mask = bundle.mask_3d
    # pre-change reference: the exact formula before the override was added
    demand = bundle.pickup_3d[mask].astype(np.float64)
    supply = bundle.active_taxis_3d[mask].astype(np.float64)
    expected = supply / np.maximum(demand, config.DEMAND_FLOOR)

    Y_omitted = io.service_ratio_Y(bundle.pickup_3d, bundle)
    Y_explicit_none = io.service_ratio_Y(bundle.pickup_3d, bundle, supply_3d=None)

    np.testing.assert_array_equal(Y_omitted, expected)
    np.testing.assert_array_equal(Y_omitted, Y_explicit_none)


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


import pickle as _pickle
from types import SimpleNamespace

from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj_at(x, y, time_bucket=0):
    return Trajectory(
        trajectory_id=0, driver_id=0,
        states=[TrajectoryState(
            x_grid=int(x), y_grid=int(y), time_bucket=int(time_bucket), day_index=0)],
    )


def test_build_edited_pickup_relocates_mass(tmp_path):
    bundle = _make_synthetic_bundle()
    mask = bundle.mask_3d
    xs, ys, ts = np.where(mask)
    # pick an active origin unit with a high pickup, and a distinct active dest
    demand_vals = bundle.pickup_3d[mask]
    o = int(np.argmax(demand_vals))
    ox, oy, ot = int(xs[o]), int(ys[o]), int(ts[o])
    # require ts[i] != ot (not just tuple inequality) so dt != ot: this makes
    # mass_o and mass_d distinct below, which is what lets this test catch a
    # per-block mass mix-up (e.g. accidentally using mass_o at the dest cell).
    d = next(i for i in range(len(xs)) if ts[i] != ot)
    dx, dy, dt = int(xs[d]), int(ys[d]), int(ts[d])
    # build trajectories whose terminal state maps to (ox,oy,ot)/(dx,dy,dt).
    # pickup_unit_of computes t_block = hour_to_block_index(time_bucket_to_hour(tb)).
    # config.TIME_BLOCKS is hourly (block i == hour i), so hour_to_block_index
    # is the identity on 0..23. time_bucket_to_hour(tb) = max(0, (tb-1)//12)
    # (1-indexed, 12 five-minute buckets per hour), so tb = 12*t_block + 1 is
    # the value whose hour is exactly t_block for any t_block in 0..23.
    orig = _traj_at(ox, oy, time_bucket=12 * ot + 1)
    modif = _traj_at(dx, dy, time_bucket=12 * dt + 1)
    histories = [SimpleNamespace(original=orig, modified=modif)]
    with open(tmp_path / "histories.pkl", "wb") as f:
        _pickle.dump(histories, f)

    # Give the origin and destination blocks distinct hour-counts so
    # mass_o != mass_d: under the default uniform-hourly config all blocks
    # have n_hours_per_block == 1, which would make a per-block mass
    # mix-up (e.g. using mass_o at the destination) undetectable below.
    # n_hours_per_block is a numpy array on this frozen dataclass, so
    # in-place mutation of this freshly-built synthetic bundle is safe.
    bundle.n_hours_per_block[ot] = 2
    bundle.n_hours_per_block[dt] = 3

    before = bundle.pickup_3d.copy()
    after = io.build_edited_pickup_3d(bundle, tmp_path)
    mass_o = 1.0 / (int(bundle.n_hours_per_block[ot]) * bundle.n_days)
    mass_d = 1.0 / (int(bundle.n_hours_per_block[dt]) * bundle.n_days)
    assert before[ox, oy, ot] - after[ox, oy, ot] == pytest.approx(mass_o)
    assert after[dx, dy, dt] - before[dx, dy, dt] == pytest.approx(mass_d)
