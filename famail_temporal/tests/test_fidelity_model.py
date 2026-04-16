"""Tests for fidelity.model components."""
import torch

from famail_temporal.fidelity.model import FeatureNormalizer, SiameseLSTMEncoder, ProfileEncoder


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


def test_siamese_lstm_encoder_forward_shape():
    # Constructor: input_dim, lstm_hidden_dims=(200,100), dropout=0.2, bidirectional=True
    enc = SiameseLSTMEncoder(input_dim=6, lstm_hidden_dims=(64,), dropout=0.0, bidirectional=False)
    x = torch.randn(3, 15, 6)
    out = enc(x)
    assert out.shape[0] == 3
    assert out.shape[-1] in (64, 128)  # uni- or bidirectional


def test_profile_encoder_forward_shape():
    # Constructor: input_dim=11, hidden_dims=(64,32), output_dim=8, dropout=0.2
    enc = ProfileEncoder(input_dim=11, hidden_dims=(64, 32), output_dim=32, dropout=0.0)
    x = torch.randn(3, 11)
    out = enc(x)
    assert out.shape == (3, 32)


from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


def test_multistream_discriminator_shape():
    model = MultiStreamSiameseDiscriminator()
    model.train(False)   # inference mode
    batch_size, n_trajs, seq_len = 2, 5, 20
    x1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    x2 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    driving_1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    driving_2 = driving_1.clone()
    profile_1 = torch.randn(batch_size, 11)
    profile_2 = profile_1.clone()

    with torch.no_grad():
        out = model(x1, x2,
                    driving_1=driving_1, driving_2=driving_2,
                    profile_1=profile_1, profile_2=profile_2)
    assert out.shape[0] == batch_size
