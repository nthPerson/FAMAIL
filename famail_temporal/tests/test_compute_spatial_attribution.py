"""Tests for fairness.spatial.compute_spatial_attribution."""
import torch
import pytest

from famail_temporal.fairness.spatial import (
    compute_spatial_attribution, compute_fspatial, pairwise_gini,
)


def _synth(N, seed=0):
    torch.manual_seed(seed)
    pickup = torch.rand(N) * 5.0 + 0.1
    dropoff = torch.rand(N) * 5.0 + 0.1
    active = torch.rand(N) * 3.0 + 1.0
    return pickup, dropoff, active


def test_returns_three_channels_of_length_N():
    N = 40
    pickup, dropoff, active = _synth(N)
    result = compute_spatial_attribution(pickup, dropoff, active)
    assert set(result.keys()) == {"gini_decomp_dsr", "gini_decomp_asr", "spatial_attr"}
    for key, vec in result.items():
        assert vec.shape == (N,), f"{key} shape {vec.shape} != ({N},)"


def test_spatial_attr_sums_to_one_minus_fspatial():
    N = 60
    pickup, dropoff, active = _synth(N, seed=3)
    result = compute_spatial_attribution(pickup, dropoff, active)
    f_spatial, _ = compute_fspatial(pickup, dropoff, active)
    assert torch.isclose(
        result["spatial_attr"].sum(),
        1.0 - f_spatial,
        atol=1e-5,
    )


def test_dsr_decomp_sums_to_dsr_gini():
    N = 50
    pickup, dropoff, active = _synth(N, seed=5)
    result = compute_spatial_attribution(pickup, dropoff, active)
    from famail_temporal import config
    dsr = pickup / (active + config.EPS)
    assert torch.isclose(result["gini_decomp_dsr"].sum(), pairwise_gini(dsr), atol=1e-6)


def test_asr_decomp_sums_to_asr_gini():
    N = 50
    pickup, dropoff, active = _synth(N, seed=7)
    result = compute_spatial_attribution(pickup, dropoff, active)
    from famail_temporal import config
    asr = dropoff / (active + config.EPS)
    assert torch.isclose(result["gini_decomp_asr"].sum(), pairwise_gini(asr), atol=1e-6)


def test_spatial_attr_equals_half_sum_of_components():
    N = 30
    pickup, dropoff, active = _synth(N, seed=11)
    result = compute_spatial_attribution(pickup, dropoff, active)
    expected = 0.5 * (result["gini_decomp_dsr"] + result["gini_decomp_asr"])
    assert torch.allclose(result["spatial_attr"], expected, atol=1e-7)
