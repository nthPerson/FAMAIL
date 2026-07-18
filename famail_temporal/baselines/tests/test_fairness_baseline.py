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


import torch
from famail_temporal.baselines.fairness_baseline import dp_gap_penalty


def test_dp_gap_zero_for_uniform_logits():
    B, L, V = 2, 3, 10
    logits = torch.zeros(B, L, V)                    # uniform after softmax
    tgt = torch.ones(B, L, dtype=torch.long)         # no PAD
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True   # 3 disadv cells
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True  # 3 adv cells
    g = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    assert torch.isclose(g, torch.tensor(0.0), atol=1e-6)


def test_dp_gap_positive_when_adv_favored_and_differentiable():
    B, L, V = 1, 2, 6
    logits = torch.full((B, L, V), -10.0, requires_grad=True)
    with torch.no_grad():
        logits[..., 3:6] = 10.0                      # all mass on adv cells
    tgt = torch.ones(B, L, dtype=torch.long)
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True
    g = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    assert g.item() > 0.2
    g.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_dp_gap_abs_equals_abs_of_signed_on_fixtures():
    # The abs variant is |signed| on the module's existing fixtures: the
    # uniform-logits fixture (gap == 0) and the adv-favored fixture (gap > 0),
    # where mass_a >= mass_d so the two formulations already agree in value.
    from famail_temporal.baselines.fairness_baseline import dp_gap_penalty_abs
    # Fixture A -- uniform logits (same construction as test_dp_gap_zero_*).
    B, L, V = 2, 3, 10
    logits = torch.zeros(B, L, V)
    tgt = torch.ones(B, L, dtype=torch.long)
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True
    s = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    a = dp_gap_penalty_abs(logits, tgt, m_d, m_a, pad_id=0)
    assert torch.isclose(a, s.abs())
    # Fixture B -- adv-favored (same construction as test_dp_gap_positive_*).
    B, L, V = 1, 2, 6
    logits = torch.full((B, L, V), -10.0)
    with torch.no_grad():
        logits[..., 3:6] = 10.0                      # all mass on adv cells
    tgt = torch.ones(B, L, dtype=torch.long)
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True
    s = dp_gap_penalty(logits, tgt, m_d, m_a, pad_id=0)
    a = dp_gap_penalty_abs(logits, tgt, m_d, m_a, pad_id=0)
    assert torch.isclose(a, s.abs())
    assert a.item() > 0.2                            # signed gap is positive here


def test_dp_gap_abs_opposite_gradient_on_negative_gap():
    # DISCRIMINATING test -- the entire behavioral difference between the two
    # formulations lives in the mass_d > mass_a region (signed gap < 0). There
    # |x| = -x, so the absolute penalty's gradient is the NEGATION of the signed
    # penalty's: they push the logits in OPPOSITE directions. A test that only
    # exercised the positive-gap region (where the two agree) would prove
    # nothing, so this fixture puts all mass on the DISADVANTAGED cells.
    from famail_temporal.baselines.fairness_baseline import dp_gap_penalty_abs
    B, L, V = 1, 2, 6
    m_d = torch.zeros(V, dtype=torch.bool); m_d[:3] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[3:6] = True
    tgt = torch.ones(B, L, dtype=torch.long)

    def _neg_gap_logits():
        # More mass on DISADVANTAGED cells -> mass_d > mass_a -> signed gap < 0.
        # Moderate (unsaturated) logits so softmax gradients stay non-trivial;
        # ~one-hot logits would give a correct-but-vanishing gradient (~1e-9).
        lg = torch.zeros((B, L, V), requires_grad=True)
        with torch.no_grad():
            lg[..., :3] = 2.0
        return lg

    lg_s = _neg_gap_logits()
    g_signed = dp_gap_penalty(lg_s, tgt, m_d, m_a, pad_id=0)
    assert g_signed.item() < -0.2                    # genuinely in the negative-gap region
    g_signed.backward()

    lg_a = _neg_gap_logits()
    g_abs = dp_gap_penalty_abs(lg_a, tgt, m_d, m_a, pad_id=0)
    assert torch.isclose(g_abs, g_signed.detach().abs())   # magnitude preserved
    g_abs.backward()

    # opposite directions everywhere, and non-trivially so (not 0 == -0)
    assert lg_s.grad.abs().max() > 1e-4
    assert torch.allclose(lg_a.grad, -lg_s.grad, atol=1e-6)


def test_cell_masks_for_vocab_disjoint_and_correct():
    cell_group = {(0, 0): 1, (1, 1): 0}
    token_of_cell = lambda c: {(0, 0): 4, (1, 1): 7}[c]
    from famail_temporal.baselines.fairness_baseline import cell_masks_for_vocab
    m_d, m_a = cell_masks_for_vocab(cell_group, vocab_size=10, token_of_cell=token_of_cell)
    assert m_d.dtype == torch.bool and m_a.dtype == torch.bool
    assert m_d.shape == (10,) and m_a.shape == (10,)
    assert m_d.sum().item() == 1 and m_a.sum().item() == 1
    assert m_d[4].item() is True
    assert m_a[7].item() is True
    assert not (m_d & m_a).any()


def _gc():
    from famail_temporal.baselines.gan import config as gc
    return gc


def _tiny_training(penalty_fn=None, penalty_lambda=0.0):
    import torch
    from famail_temporal.utils.seeding import set_all_seeds
    from famail_temporal.baselines.gan.generator import TrajectoryLSTM
    from famail_temporal.baselines.gan.train_mle import train_mle
    set_all_seeds(0)
    model = TrajectoryLSTM(n_drivers=2)
    seqs = [[1, 2, 3], [2, 3, 4, 5], [3, 4]]
    ctxs = [(1, 0), (2, 1), (3, 0)]
    kwargs = dict(epochs=2, lr=1e-3, batch_size=2,
                  device=torch.device("cpu"), driver_idxs=[0, 1, 0])
    if penalty_fn is not None:
        kwargs.update(penalty_fn=penalty_fn, penalty_lambda=penalty_lambda)
    return train_mle(model, seqs, ctxs, **kwargs)["epoch_losses"]


def test_penalty_default_off_is_identical():
    # Two implicit-default runs agree (determinism sanity)...
    base = _tiny_training()
    assert base == _tiny_training()
    # ...and an EXPLICIT penalty_fn=None, penalty_lambda=0.0 call to train_mle
    # itself (not via the helper, whose guard would swallow the kwargs) is
    # bit-identical to the implicit-defaults path.
    import torch
    from famail_temporal.utils.seeding import set_all_seeds
    from famail_temporal.baselines.gan.generator import TrajectoryLSTM
    from famail_temporal.baselines.gan.train_mle import train_mle
    set_all_seeds(0)
    model = TrajectoryLSTM(n_drivers=2)
    seqs = [[1, 2, 3], [2, 3, 4, 5], [3, 4]]
    ctxs = [(1, 0), (2, 1), (3, 0)]
    explicit = train_mle(
        model, seqs, ctxs, epochs=2, lr=1e-3, batch_size=2,
        device=torch.device("cpu"), driver_idxs=[0, 1, 0],
        penalty_fn=None, penalty_lambda=0.0,
    )
    assert explicit["epoch_losses"] == base
    assert "penalty_values" not in explicit   # key absent when penalty_fn is None


def test_penalty_changes_loss_when_active():
    import torch
    from famail_temporal.baselines.fairness_baseline import dp_gap_penalty
    gc = _gc()
    V = gc.VOCAB_SIZE
    m_d = torch.zeros(V, dtype=torch.bool); m_d[1] = True
    m_a = torch.zeros(V, dtype=torch.bool); m_a[2] = True
    fn = lambda lg, tg: dp_gap_penalty(lg, tg, m_d, m_a, pad_id=gc.PAD)
    base = _tiny_training()
    pen = _tiny_training(penalty_fn=fn, penalty_lambda=100.0)
    assert base != pen
