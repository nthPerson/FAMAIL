import torch

from famail_temporal.baselines.gan.generator import TrajectoryLSTM


def _tiny(n_drivers=None):
    return TrajectoryLSTM(
        vocab_size=20, n_tblocks=4, embed_dim=8, hidden_dim=8, n_layers=1,
    ) if n_drivers is None else TrajectoryLSTM(
        vocab_size=20, n_tblocks=4, embed_dim=8, hidden_dim=8, n_layers=1,
        n_drivers=n_drivers,
    )


def test_driver_idx_none_matches_unconditioned():
    """driver_idx=None on a driver-aware model == an unconditioned model with
    the same shared weights (the driver embedding contributes nothing)."""
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    plain = _tiny()
    # Copy the shared (non-driver) weights so the two models are identical
    # except for the unused driver embedding.
    plain.load_state_dict(
        {k: v for k, v in cond.state_dict().items() if k != "driver_embed.weight"}
    )
    tokens = torch.randint(0, 20, (2, 5))
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    out_cond = cond(tokens, cc, tb, driver_idx=None)
    out_plain = plain(tokens, cc, tb)
    assert torch.allclose(out_cond, out_plain, atol=1e-6)


def test_driver_idx_changes_logits():
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    tokens = torch.randint(0, 20, (2, 5))
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    a = cond(tokens, cc, tb, driver_idx=torch.tensor([0, 0]))
    b = cond(tokens, cc, tb, driver_idx=torch.tensor([1, 2]))
    assert a.shape == (2, 5, 20)
    assert not torch.allclose(a, b)


def test_step_and_step_embed_accept_driver_idx():
    torch.manual_seed(0)
    cond = _tiny(n_drivers=3)
    tok = torch.tensor([1, 2])
    cc = torch.tensor([1, 2])
    tb = torch.tensor([0, 3])
    di = torch.tensor([0, 1])
    logits_step, h = cond.step(tok, cc, tb, None, driver_idx=di)
    assert logits_step.shape == (2, 20)
    emb = cond.cell_embed(tok)
    logits_se, _ = cond.step_embed(emb, cc, tb, None, driver_idx=di)
    assert logits_se.shape == (2, 20)
    # step delegates to step_embed: same driver_idx -> same logits at step 0
    assert torch.allclose(logits_step, logits_se, atol=1e-6)


def test_unconditioned_model_unchanged_regression():
    """A model built without n_drivers has no driver_embed and behaves exactly
    as before (positional call still works)."""
    torch.manual_seed(0)
    m = _tiny()
    assert not hasattr(m, "driver_embed")
    tokens = torch.randint(0, 20, (2, 5))
    out = m(tokens, torch.tensor([1, 2]), torch.tensor([0, 3]))
    assert out.shape == (2, 5, 20)
