"""Tests for optional per-sequence loss weighting in gan.train_mle.

The weighted-BC smoke test upweights the edited trajectories so they are not
averaged away by the flat per-token MLE mean. These tests pin the two
properties that make that experiment trustworthy:

1. uniform weights (or weights=None) reproduce the unweighted training exactly
   (so the locked Level-2 baseline is untouched), and
2. upweighting a sequence biases the trained model toward fitting it.
"""
import torch

from famail_temporal.baselines.gan import config as gc
from famail_temporal.baselines.gan.generator import TrajectoryLSTM
from famail_temporal.baselines.gan.train_mle import train_mle


_SEQS = [
    [gc.BOS, 10, 11, 12, gc.EOS],
    [gc.BOS, 40, 41, gc.EOS],
]
_CTX = [(10, 0), (40, 1)]


def _seq_ce(model: TrajectoryLSTM, seq, ctx) -> float:
    """Mean next-token cross-entropy of `model` on a single sequence."""
    model.eval()
    dev = torch.device("cpu")
    batch = torch.tensor([seq], dtype=torch.long, device=dev)
    cc = torch.tensor([ctx[0]], dtype=torch.long, device=dev)
    ct = torch.tensor([ctx[1]], dtype=torch.long, device=dev)
    with torch.no_grad():
        logits = model(batch[:, :-1], cc, ct)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, gc.VOCAB_SIZE), batch[:, 1:].reshape(-1),
            ignore_index=gc.PAD,
        )
    return float(loss.item())


def _train(weights):
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    out = train_mle(
        model, _SEQS, _CTX,
        epochs=5, lr=1e-2, batch_size=2, device=torch.device("cpu"),
        sample_weights=weights,
    )
    return model, out


def test_uniform_weights_match_unweighted():
    """weights=[1,1] must reproduce weights=None epoch-loss-for-epoch-loss."""
    torch.manual_seed(0)
    base_model = TrajectoryLSTM()
    base = train_mle(
        base_model, _SEQS, _CTX,
        epochs=5, lr=1e-2, batch_size=2, device=torch.device("cpu"),
        sample_weights=None,
    )
    _, weighted = _train([1.0, 1.0])
    for a, b in zip(base["epoch_losses"], weighted["epoch_losses"]):
        assert abs(a - b) < 1e-4, f"uniform-weight loss {b} != unweighted {a}"


def test_upweighting_biases_model_toward_that_sequence():
    """Heavily weighting seq0 fits seq0 better than heavily weighting seq1."""
    fav0, _ = _train([10.0, 1.0])
    fav1, _ = _train([1.0, 10.0])
    ce0_under_fav0 = _seq_ce(fav0, _SEQS[0], _CTX[0])
    ce0_under_fav1 = _seq_ce(fav1, _SEQS[0], _CTX[0])
    assert ce0_under_fav0 < ce0_under_fav1, (
        f"upweighting seq0 should lower its CE: fav0={ce0_under_fav0} "
        f"!< fav1={ce0_under_fav1}"
    )


def test_wrong_length_weights_rejected():
    """sample_weights must be one weight per sequence."""
    torch.manual_seed(0)
    model = TrajectoryLSTM()
    try:
        train_mle(
            model, _SEQS, _CTX,
            epochs=1, lr=1e-2, batch_size=2, device=torch.device("cpu"),
            sample_weights=[1.0],  # len 1 != 2 sequences
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched sample_weights length")
