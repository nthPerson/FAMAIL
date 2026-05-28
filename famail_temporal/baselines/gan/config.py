"""Vocabulary constants and hyperparameters for the trajectory generator."""
from famail_temporal import config as _root

GX, GY = _root.GRID_DIMS          # (48, 90)
N_CELLS = GX * GY                 # 4320 flat cell ids: 0 .. N_CELLS-1
BOS = N_CELLS                     # begin-of-sequence
EOS = N_CELLS + 1                 # end-of-sequence
PAD = N_CELLS + 2                 # padding (ignored by the loss)
VOCAB_SIZE = N_CELLS + 3
N_TBLOCKS = _root.T               # conditioning time-block cardinality

# Generator
EMBED_DIM = 64
HIDDEN_DIM = 128
N_LAYERS = 1

# Training
MLE_EPOCHS = 5
MLE_LR = 1e-3
MLE_BATCH_SIZE = 256

# Generation
MAX_GEN_LEN = 64                  # hard cap on rollout length (cells)
