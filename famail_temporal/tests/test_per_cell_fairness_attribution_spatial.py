"""Tests for fairness.spatial.per_cell_fairness_attribution_spatial.

The 1/N-shifted spatial attribution: αᵢ = 1/N − 0.5·(gini_decomp_DSR_i +
gini_decomp_ASR_i). Sums to F_spatial (not 1 - F_spatial). Sign convention:

    positive  → cell contributes more than 1/N baseline to fairness
    negative  → cell drags fairness below baseline (priority for modification)

See ``famail_temporal/docs/FAIRNESS_DECOMPOSITION_FORMULATION.md``.
"""
import torch

from famail_temporal import config
from famail_temporal.fairness.spatial import (
    per_cell_fairness_attribution_spatial,
    per_unit_gini_decomposition,
    compute_fspatial,
    pairwise_gini,
)


def _synth(N, seed=0):
    torch.manual_seed(seed)
    pickup = torch.rand(N) * 5.0 + 0.1
    dropoff = torch.rand(N) * 5.0 + 0.1
    active = torch.rand(N) * 3.0 + 1.0
    return pickup, dropoff, active


def test_returns_1d_tensor_of_length_N():
    N = 40
    pickup, dropoff, active = _synth(N)
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    assert attr.dim() == 1
    assert attr.shape == (N,)


def test_sums_to_fspatial():
    """Σᵢ αᵢ_spatial == F_spatial — the load-bearing decomposition identity."""
    N = 60
    pickup, dropoff, active = _synth(N, seed=3)
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    f_spatial, _ = compute_fspatial(pickup, dropoff, active)
    assert torch.isclose(attr.sum(), f_spatial, atol=1e-5)


def test_matches_manual_one_over_n_shift():
    """αᵢ = 1/N − 0.5·(gini_decomp_DSR_i + gini_decomp_ASR_i)."""
    N = 30
    pickup, dropoff, active = _synth(N, seed=11)
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    dsr = pickup / (active + config.EPS)
    asr = dropoff / (active + config.EPS)
    expected = (1.0 / N) - 0.5 * (
        per_unit_gini_decomposition(dsr) + per_unit_gini_decomposition(asr)
    )
    assert torch.allclose(attr, expected, atol=1e-7)


def test_constant_inputs_give_uniform_one_over_n():
    """When DSR and ASR are constant, every cell contributes 1/N to F_spatial."""
    N = 50
    pickup = torch.full((N,), 3.0)
    dropoff = torch.full((N,), 3.0)
    active = torch.full((N,), 5.0)
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    expected = torch.full((N,), 1.0 / N)
    assert torch.allclose(attr, expected, atol=1e-5)
    # And sum should be ~1.0 (perfect fairness => F_spatial == 1).
    assert torch.isclose(attr.sum(), torch.tensor(1.0), atol=1e-5)


def test_outlier_cell_drags_below_baseline():
    """A cell that contributes disproportionately to inequality has αᵢ < 1/N."""
    N = 30
    pickup = torch.full((N,), 1.0)
    dropoff = torch.full((N,), 1.0)
    active = torch.full((N,), 1.0)
    # Spike one cell's pickup ratio.
    pickup[0] = 50.0
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    # The outlier cell should be the most negative (or smallest) attribution.
    assert torch.argmin(attr).item() == 0
    # And its attribution should be strictly below the 1/N baseline.
    assert float(attr[0]) < 1.0 / N


def test_negative_values_correspond_to_drag_cells():
    """Σ negative αᵢ ≤ 0; cells with αᵢ < 0 are below the 1/N baseline."""
    N = 60
    pickup, dropoff, active = _synth(N, seed=17)
    # Inject heavy inequality so some cells go negative.
    pickup[0] = 200.0
    pickup[1] = 200.0
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    assert (attr < 0).any(), "Expected at least one negative attribution under heavy inequality"


def test_sign_convention_pairwise_gini_aligns():
    """Sanity: gini sum equals 1 - F_spatial (the unfairness side)."""
    N = 50
    pickup, dropoff, active = _synth(N, seed=23)
    attr = per_cell_fairness_attribution_spatial(pickup, dropoff, active)
    dsr = pickup / (active + config.EPS)
    asr = dropoff / (active + config.EPS)
    half_gini_sum = 0.5 * (pairwise_gini(dsr) + pairwise_gini(asr))
    f_spatial, _ = compute_fspatial(pickup, dropoff, active)
    # 1 - F_spatial = half-Gini sum (the unfairness).
    assert torch.isclose(1.0 - f_spatial, half_gini_sum, atol=1e-5)
    # Attribution sums to F_spatial = 1 - half-Gini sum.
    assert torch.isclose(attr.sum(), 1.0 - half_gini_sum, atol=1e-5)
