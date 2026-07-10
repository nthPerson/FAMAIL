"""Engine tests for the Demographic Oversampling baseline (Mission-3 4th arm)."""
from types import SimpleNamespace

import numpy as np
import pytest

from famail_temporal.baselines import demographic_oversampling as dov
from famail_temporal.baselines.external_fairness_io import EQUITY_AXES
from famail_temporal.utils.trajectory import Trajectory, TrajectoryState


def _traj(cells, traj_id, driver="d0", time_bucket=13, day=0):
    """Trajectory through integer `cells` [(x, y), ...]; last cell is the pickup."""
    states = [TrajectoryState(x_grid=float(x), y_grid=float(y),
                              time_bucket=time_bucket, day_index=day)
              for x, y in cells]
    return Trajectory(trajectory_id=traj_id, driver_id=driver, states=states)


def _selected_grid():
    """(6, 4, 3) cell values: 6 distinct 'district' values along x, same per axis.

    region_extremes(frac=1/3) over 6 distinct values -> k=2 extreme regions per
    pole. For housing/comp (disadvantaged LOW) D = rows {0, 1}; for migrant
    (disadvantaged HIGH) D = rows {4, 5}.
    """
    vals = np.arange(6, dtype=np.float64)          # 0..5, one value per x-row
    grid = np.zeros((6, 4, 3))
    for j in range(3):
        grid[:, :, j] = vals[:, None]
    return grid


def test_disadvantaged_cell_masks_follow_evaluation_convention():
    masks = dov.disadvantaged_cell_masks(_selected_grid())
    assert set(masks) == set(EQUITY_AXES)
    housing = masks["AvgHousingPricePerSqM"]      # disadvantaged LOW -> rows 0, 1
    migrant = masks["MigrantRatio"]               # disadvantaged HIGH -> rows 4, 5
    assert housing.shape == (6, 4) and housing.dtype == bool
    assert housing[0].all() and housing[1].all() and not housing[2:].any()
    assert migrant[4].all() and migrant[5].all() and not migrant[:4].any()


def test_disadvantaged_cell_masks_nan_cells_excluded():
    grid = _selected_grid()
    grid[0, 0, :] = np.nan
    masks = dov.disadvantaged_cell_masks(grid)
    assert not masks["AvgHousingPricePerSqM"][0, 0]


def test_eligible_pools_by_origin_cell():
    masks = dov.disadvantaged_cell_masks(_selected_grid())
    trajs = [
        _traj([(0, 0), (2, 2)], "a"),   # origin row 0 -> housing+comp D
        _traj([(5, 1), (3, 3)], "b"),   # origin row 5 -> migrant D
        _traj([(3, 0), (0, 0)], "c"),   # origin row 3 -> no pool (pickup row ignored)
    ]
    pools = dov.eligible_pools(trajs, masks)
    assert pools["AvgHousingPricePerSqM"].tolist() == [0]
    assert pools["CompPerCapita"].tolist() == [0]
    assert pools["MigrantRatio"].tolist() == [1]


def test_sample_duplicates_quotas_and_dedupe():
    pools = {
        "AvgHousingPricePerSqM": np.array([0, 1, 2, 3]),
        "CompPerCapita": np.array([0, 1, 2, 3]),      # fully overlaps housing
        "MigrantRatio": np.array([10, 11, 12, 13]),
    }
    specs = dov.sample_duplicates(pools, n_corpus=20, dose=7, seed=0)
    assert len(specs) == 7
    # quotas in EQUITY_AXES order: 7 = 3 + 2 + 2
    per = {a: sum(1 for s in specs if s.stratum == a) for a in EQUITY_AXES}
    assert per == {"AvgHousingPricePerSqM": 3, "CompPerCapita": 2, "MigrantRatio": 2}
    # cross-stratum dedupe: housing draws 3 of the shared {0,1,2,3} pool,
    # leaving 1 for comp's quota of 2 -> exactly one flagged fallback draw;
    # distinct sources = 4 (shared pool) + 2 (migrant) = 6 of 7 draws.
    srcs = [s.source_index for s in specs]
    assert sum(1 for s in specs if s.with_replacement) == 1
    assert len(set(srcs)) == 6
    # offsets are rigid, radius-1, never zero
    assert all(max(abs(s.offset[0]), abs(s.offset[1])) == 1 for s in specs)
    # eligible_axes recorded for overlap sources
    housing_specs = [s for s in specs if s.stratum == "AvgHousingPricePerSqM"]
    assert all(set(s.eligible_axes) == {"AvgHousingPricePerSqM", "CompPerCapita"}
               for s in housing_specs)


def test_sample_duplicates_deterministic_and_seed_sensitive():
    pools = {a: np.arange(50) for a in EQUITY_AXES}
    a1 = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=3)
    a2 = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=3)
    b = dov.sample_duplicates(pools, n_corpus=100, dose=12, seed=4)
    assert a1 == a2
    assert [s.source_index for s in a1] != [s.source_index for s in b]


def test_sample_duplicates_placebo_uniform_over_corpus():
    pools = {a: np.array([0]) for a in EQUITY_AXES}   # pools must be IGNORED
    specs = dov.sample_duplicates(pools, n_corpus=30, dose=10, seed=0,
                                  variant=dov.PLACEBO)
    assert len(specs) == 10
    assert all(s.stratum == dov.PLACEBO for s in specs)
    assert len({s.source_index for s in specs}) == 10          # without replacement
    assert max(s.source_index for s in specs) < 30


def test_sample_duplicates_empty_pool_hard_error():
    pools = {a: (np.array([], dtype=np.int64) if a == "MigrantRatio"
                 else np.arange(9)) for a in EQUITY_AXES}
    with pytest.raises(ValueError, match="empty pool"):
        dov.sample_duplicates(pools, n_corpus=9, dose=9, seed=0)


def test_sample_duplicates_dose_zero_is_empty():
    pools = {a: np.arange(5) for a in EQUITY_AXES}
    assert dov.sample_duplicates(pools, n_corpus=5, dose=0, seed=0) == []


def test_make_phantom_rigid_shift_and_identity():
    src = _traj([(3, 3), (4, 3), (4, 4)], "t9", driver="real_driver")
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(1, -1),
                             phantom_id="phantom_targeted_s0_000000",
                             with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 0
    assert ph.driver_id == "phantom_targeted_s0_000000"
    assert ph.driver_id != src.driver_id
    assert str(src.trajectory_id) in str(ph.trajectory_id)
    # rigid: every state shifted by exactly (1, -1); times/days unchanged
    for s_src, s_ph in zip(src.states, ph.states):
        assert (s_ph.x_grid, s_ph.y_grid) == (s_src.x_grid + 1, s_src.y_grid - 1)
        assert s_ph.time_bucket == s_src.time_bucket
        assert s_ph.day_index == s_src.day_index
    # source untouched (deep copy)
    assert (src.states[0].x_grid, src.states[0].y_grid) == (3.0, 3.0)


def test_make_phantom_clips_at_boundary_and_counts():
    src = _traj([(0, 0), (1, 0)], "t10")
    spec = dov.DuplicateSpec(source_index=0, stratum="CompPerCapita",
                             eligible_axes=("CompPerCapita",), offset=(-1, -1),
                             phantom_id="p", with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 2                        # both states clipped in x and/or y
    assert (ph.states[0].x_grid, ph.states[0].y_grid) == (0.0, 0.0)
    assert (ph.states[1].x_grid, ph.states[1].y_grid) == (0.0, 0.0)


def test_adjacency_preserved_without_clipping():
    from famail_temporal.baselines.stifgsm_baseline import adjacency_violation_rate
    src = _traj([(5, 5), (6, 5), (6, 6), (7, 6)], "t11")
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(1, 1),
                             phantom_id="p", with_replacement=False)
    ph, n_clipped = dov.make_phantom(src, spec, grid_dims=(48, 90))
    assert n_clipped == 0
    assert adjacency_violation_rate([ph]) == adjacency_violation_rate([src])


def test_escape_fractions():
    masks = dov.disadvantaged_cell_masks(_selected_grid())      # migrant D rows 4-5
    src = _traj([(5, 1), (4, 1), (3, 1)], "t12")                # origin row 5; pickup row 3
    spec = dov.DuplicateSpec(source_index=0, stratum="MigrantRatio",
                             eligible_axes=("MigrantRatio",), offset=(-1, 0),
                             phantom_id="p", with_replacement=False)
    ph, _ = dov.make_phantom(src, spec, grid_dims=(6, 4))
    fr = dov.escape_fractions([spec], [ph], masks)
    # shifted origin = row 4 (still D) -> no escape; shifted pickup = row 2 (outside D)
    assert fr == {"origin_escape_frac": 0.0, "pickup_outside_frac": 1.0}


def test_escape_fractions_placebo_none():
    spec = dov.DuplicateSpec(source_index=0, stratum=dov.PLACEBO,
                             eligible_axes=(), offset=(1, 0),
                             phantom_id="p", with_replacement=False)
    ph, _ = dov.make_phantom(_traj([(2, 2), (3, 2)], "t13"), spec, grid_dims=(6, 4))
    fr = dov.escape_fractions([spec], [ph],
                              dov.disadvantaged_cell_masks(_selected_grid()))
    assert fr == {"origin_escape_frac": None, "pickup_outside_frac": None}


def _stub_bundle(gx=48, gy=90, n_days=2):
    from famail_temporal import config as cfg
    from famail_temporal.data.aggregation import block_n_hours
    T = cfg.T
    return SimpleNamespace(
        pickup_3d=np.zeros((gx, gy, T), dtype=np.float32),
        active_taxis_3d=np.zeros((gx, gy, T), dtype=np.float32),
        n_hours_per_block=np.array([block_n_hours(t) for t in range(T)],
                                   dtype=np.int32),
        n_days=n_days,
    )


def test_additive_demand_mass_conservation_and_placement():
    from famail_temporal.baselines.datasets import pickup_mass, pickup_unit_of
    b = _stub_bundle()
    phantoms = [_traj([(3, 3), (4, 4)], "p1"), _traj([(7, 8), (9, 9)], "p2")]
    D = dov.additive_demand(b, phantoms)
    assert D.dtype == np.float64
    expected = sum(pickup_mass(b, pickup_unit_of(ph)[2]) for ph in phantoms)
    assert np.isclose(D.sum() - np.float64(b.pickup_3d).sum(), expected)
    cx, cy, t = pickup_unit_of(phantoms[0])
    assert D[cx, cy, t] == pytest.approx(pickup_mass(b, t))


def test_additive_supply_distinct_count_semantics():
    b = _stub_bundle()
    # two states of ONE phantom in the same cell/hour -> counted ONCE
    ph = _traj([(10, 10), (10, 10), (11, 10)], "p3", time_bucket=13)
    S = dov.additive_supply(b, [ph])
    from famail_temporal.data.aggregation import (
        hour_to_block_index, time_bucket_to_hour,
    )
    t = hour_to_block_index(time_bucket_to_hour(13))
    unit = 1.0 / (float(b.n_hours_per_block[t]) * b.n_days)
    # states[-1] (the pickup) is EXCLUDED from supply; the two remaining
    # states are both at (10, 10) -> the 5x5 neighborhood around each is
    # identical -> every covered cell gets exactly ONE unit, not two.
    assert S[10, 10, t] == pytest.approx(unit)
    assert S[8, 8, t] == pytest.approx(unit)          # 5x5 reach (K=2)
    assert S[13, 10, t] == 0.0                        # outside the neighborhood
    # two DISTINCT phantoms in the same cell/hour -> counted TWICE
    ph2 = _traj([(10, 10), (11, 10)], "p4", time_bucket=13)
    S2 = dov.additive_supply(b, [ph, ph2])
    assert S2[10, 10, t] == pytest.approx(2 * unit)


def test_additive_supply_matches_production_counter_on_fixture():
    """Pin additive_supply to the PRODUCTION tier-2 convention: the same pings
    pushed through build_active_taxis_counts + aggregate_active_taxis must
    reproduce additive_supply's delta on every cell the phantom covers."""
    import pandas as pd
    from famail_temporal.data.aggregation import aggregate_active_taxis
    from famail_temporal.data.source_generation.views.active_taxis import (
        build_active_taxis_counts,
    )
    from famail_temporal import config as cfg

    b = _stub_bundle(n_days=1)
    ph = _traj([(10, 10), (12, 11), (12, 12)], "pfix", time_bucket=13, day=0)
    S = dov.additive_supply(b, [ph])

    # The production path: raw pings are 1-indexed; the terminal state is the
    # pickup-transition -> give it passenger_indicator=1 so the production
    # counter drops it too (mirrors the supply-only convention).
    hour = 1  # time_bucket 13 -> hour 1 (1-indexed 5-min buckets, 12/hour)
    rows = [
        {"plate_id": "pfix", "x_grid": 10 + 1, "y_grid": 10 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 0},
        {"plate_id": "pfix", "x_grid": 12 + 1, "y_grid": 11 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 0},
        {"plate_id": "pfix", "x_grid": 12 + 1, "y_grid": 12 + 1,
         "hour": hour, "day_index": 0, "passenger_indicator": 1},   # pickup
    ]
    counts = build_active_taxis_counts(pd.DataFrame(rows))
    agg = aggregate_active_taxis(counts, n_days=1)

    covered = S > 0
    assert covered.any()
    # Where the phantom contributed, production and additive agree exactly.
    assert np.allclose(S[covered], agg[covered])
    # Where it didn't, production shows only its SUPPLY_FLOOR.
    assert np.all(agg[~covered] <= cfg.SUPPLY_FLOOR + 1e-12)


def test_additive_grids_dose_zero_identity():
    b = _stub_bundle()
    b.pickup_3d = np.random.default_rng(0).random(b.pickup_3d.shape).astype(np.float32)
    b.active_taxis_3d = np.random.default_rng(1).random(b.active_taxis_3d.shape).astype(np.float32)
    assert np.array_equal(dov.additive_demand(b, []), np.float64(b.pickup_3d))
    assert np.array_equal(dov.additive_supply(b, []), np.float64(b.active_taxis_3d))
