"""Tests for fidelity.compute."""
import torch

from famail_temporal.fidelity.compute import compute_ffidelity
from famail_temporal.fidelity.model import MultiStreamSiameseDiscriminator


def test_compute_ffidelity_in_unit_interval():
    torch.manual_seed(0)
    model = MultiStreamSiameseDiscriminator()
    model.train(False)
    for p in model.parameters():
        p.requires_grad = False

    batch_size, n_trajs, seq_len = 1, 5, 15
    x1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    x2 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    driving_1 = torch.rand(batch_size, n_trajs, seq_len, 4) * 10.0
    driving_2 = driving_1.clone()
    profile_1 = torch.randn(batch_size, 11)
    profile_2 = profile_1.clone()

    ms_kwargs = {
        "x1": x1, "x2": x2,
        "driving_1": driving_1, "driving_2": driving_2,
        "profile_1": profile_1, "profile_2": profile_2,
    }
    tau = torch.rand(1, seq_len, 4)
    tau_prime = tau.clone()

    f, _ = compute_ffidelity(model, tau, tau_prime, ms_kwargs)
    assert 0.0 <= float(f) <= 1.0
