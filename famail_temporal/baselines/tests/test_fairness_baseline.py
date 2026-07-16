import numpy as np
import pytest

from famail_temporal.baselines.fairness_baseline import (
    normalize_mean_one, weights_from_groups,
)


def test_normalize_mean_one():
    w = normalize_mean_one([2.0, 4.0, 6.0])
    assert np.isclose(np.mean(w), 1.0)
    assert np.isclose(w[1] / w[0], 2.0)  # ratios preserved


def test_weights_from_groups_inverse_sdr():
    # group 1 (disadv) has SDR 2.0, group 0 (adv) has SDR 8.0 -> disadv gets 4x
    groups_of_trajs = [1, 0, -1, 1]
    sdr_by_group = {0: 8.0, 1: 2.0}
    w = weights_from_groups(groups_of_trajs, sdr_by_group)
    assert np.isclose(np.mean(w), 1.0)
    assert np.isclose(w[0] / w[1], 4.0)       # inverse-SDR ratio
    assert np.isfinite(w[2])                   # excluded stays finite
    assert w[0] == w[3]                        # same group, same weight


def test_normalize_mean_one_empty():
    with pytest.raises(ValueError):
        normalize_mean_one([])


def test_unit_groups_real_bundle():
    pytest.importorskip("torch")
    from famail_temporal.data.loader import DataBundle
    try:
        bundle = DataBundle.load()
    except Exception:
        pytest.skip("bundle data not available")
    from famail_temporal.baselines.fairness_baseline import unit_groups_and_sdr
    cell_group, sdr = unit_groups_and_sdr(bundle)
    n_d = sum(1 for g in cell_group.values() if g == 1)
    # NOTE on the check value: the task brief/plan cited N_D = 6,950,
    # copied from run_external_fairness's group_sizes.n_disadvantaged (see
    # famail_temporal/baselines/external_fairness/results/*/external_fairness.json,
    # metrics.MigrantRatio.district_extremes.group_sizes). That figure counts
    # active (cell, time-block) UNITS — the same spatial cell is counted once
    # per active hour it has, e.g. mask_3d.sum() == 34524 total active units
    # vs GRID_DIMS (48, 90) == 4320 total *cells* — so 6,950 cannot be the
    # size of any dict keyed purely by spatial cell (max possible is 4320).
    # cell_group here IS keyed purely by (cx, cy) — required so it can be
    # looked up via a trajectory's time-free `pickup_cell` (see
    # fairness_reweigh_weight_vector) — so it is deduped to unique
    # disadvantaged CELLS, not unit-hours. Verified empirically against this
    # bundle: 1,879 unique active cells total (462 disadvantaged / 406
    # advantaged / 1,011 excluded), and real trajectory pickup_cells match
    # cell_group at a 92.8% rate (88,431 / 95,297), confirming the lookup is
    # meaningful. Flagged in the implementer's report for the plan owner.
    assert n_d == 462
    assert sdr[1] < sdr[0]        # disadvantaged group is under-served


def test_fairness_reweigh_weight_vector_real_bundle():
    pytest.importorskip("torch")
    from famail_temporal.data.loader import DataBundle
    try:
        bundle = DataBundle.load()
    except Exception:
        pytest.skip("bundle data not available")
    from famail_temporal.baselines.fairness_baseline import (
        fairness_reweigh_weight_vector, unit_groups_and_sdr,
    )
    trajs = bundle.trajectories[:2000]
    w = np.asarray(fairness_reweigh_weight_vector(trajs, bundle))
    assert len(w) == len(trajs)                 # index-aligned
    assert np.isclose(w.mean(), 1.0)            # normalized to mean 1
    assert (w > 0).all() and np.isfinite(w).all()
    cell_group, sdr = unit_groups_and_sdr(bundle)
    gs = [cell_group.get(tuple(t.pickup_cell), -1) for t in trajs]
    i_d = next(i for i, g in enumerate(gs) if g == 1)
    i_a = next(i for i, g in enumerate(gs) if g == 0)
    assert w[i_d] > w[i_a]                      # disadvantaged-origin upweighted
