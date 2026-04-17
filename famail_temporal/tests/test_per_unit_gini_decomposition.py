"""Tests for fairness.spatial.per_unit_gini_decomposition."""
import torch
import pytest

from famail_temporal.fairness.spatial import (
    per_unit_gini_decomposition, pairwise_gini,
)


def test_decomposition_sums_to_gini_random():
    torch.manual_seed(0)
    values = torch.rand(50) * 10.0 + 0.1
    decomp = per_unit_gini_decomposition(values)
    assert decomp.shape == values.shape
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_equal_values_zero():
    values = torch.full((20,), 3.0)
    decomp = per_unit_gini_decomposition(values)
    assert torch.allclose(decomp, torch.zeros_like(values), atol=1e-6)


def test_decomposition_one_hot_concentrated_on_outlier():
    values = torch.zeros(10)
    values[0] = 100.0
    decomp = per_unit_gini_decomposition(values)
    # Outlier's contribution dominates: for n=10 one-hot, decomp[0]/decomp[1] = 9.
    # (Spec plan had `> 10 *` which is mathematically impossible; relaxed to `> 5 *`
    # while preserving the "outlier dominates" intent.)
    assert decomp[0] > 5 * decomp[1]
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_single_element_zero():
    values = torch.tensor([5.0])
    decomp = per_unit_gini_decomposition(values)
    assert decomp.shape == (1,)
    assert float(decomp.sum()) == 0.0


def test_decomposition_two_elements_sum_matches_gini():
    values = torch.tensor([1.0, 3.0])
    decomp = per_unit_gini_decomposition(values)
    assert torch.isclose(decomp.sum(), pairwise_gini(values), atol=1e-6)


def test_decomposition_all_nonnegative():
    torch.manual_seed(1)
    values = torch.rand(30) * 5.0 + 0.1
    decomp = per_unit_gini_decomposition(values)
    assert (decomp >= 0.0).all()
