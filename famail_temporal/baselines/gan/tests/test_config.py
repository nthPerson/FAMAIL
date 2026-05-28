"""Vocabulary/config sanity for the GAN baselines."""
from famail_temporal import config as root_config
from famail_temporal.baselines.gan import config as gc


def test_vocab_layout():
    gx, gy = root_config.GRID_DIMS
    assert gc.N_CELLS == gx * gy
    # Three special tokens above the cell ids, all distinct and contiguous.
    assert gc.BOS == gc.N_CELLS
    assert gc.EOS == gc.N_CELLS + 1
    assert gc.PAD == gc.N_CELLS + 2
    assert gc.VOCAB_SIZE == gc.N_CELLS + 3
