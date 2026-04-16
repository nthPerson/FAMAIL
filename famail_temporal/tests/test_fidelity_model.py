"""Tests for fidelity.model components."""
import torch

from famail_temporal.fidelity.model import FeatureNormalizer


def test_feature_normalizer_output_shape():
    norm = FeatureNormalizer()
    x = torch.randn(2, 20, 4)
    out = norm(x)
    # FeatureNormalizer expands 4 raw features to 6 normalized features:
    # [x_norm, y_norm, sin_time, cos_time, sin_day, cos_day]
    assert out.shape == (2, 20, 6)


def test_feature_normalizer_has_params():
    norm = FeatureNormalizer()
    # Constructor stores normalization constants as plain attributes
    assert hasattr(norm, "x_max")
    assert hasattr(norm, "y_max")
    assert hasattr(norm, "time_buckets")
    assert hasattr(norm, "days_in_week")
    # Default values match the Shenzhen 48x90 / 288-bucket grid
    assert norm.x_max == 49.0
    assert norm.y_max == 89.0
    assert norm.time_buckets == 288
    assert norm.days_in_week == 5
