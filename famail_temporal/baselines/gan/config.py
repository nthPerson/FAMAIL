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
MLE_BATCH_SIZE = 32               # small: MLE logits are (B, seq_len, VOCAB=4323);
                                  # the corpus has a long length tail (max ~1654
                                  # tokens), so a large batch OOMs an 8 GB GPU
MAX_TRAIN_TOKENS = 256            # exclude trajectories longer than this from
                                  # training (p99 length is 213; drops ~1% tail)

# Generation
MAX_GEN_LEN = 64                  # hard cap on rollout length (cells)

# Adversarial fine-tune (Phase 3)
ADV_EPOCHS = 3
ADV_LR_G = 1e-4                   # generator LR during fine-tune (small: don't undo MLE)
ADV_LR_D = 1e-4                   # critic LR
ADV_BATCH_SIZE = 64              # smaller than MLE: the G-step backprops through
                                 # a 64-step Gumbel rollout (memory-heavy)
GUMBEL_TAU_START = 1.0            # Gumbel-softmax temperature, annealed start
GUMBEL_TAU_END = 0.5              #   -> end (sharper, closer to discrete)
D_HIDDEN_DIM = 128                # critic LSTM hidden size
