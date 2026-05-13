# Discriminator Checkpoints

The canonical checkpoint used by `famail_temporal/` lives at:

    discriminator_checkpoints/default/best.pt

## Provisioning

`python -m famail_temporal.fetch_data` downloads this file as part of the
public HuggingFace dataset bundle — no manual step needed.

Dataset: <https://huggingface.co/datasets/nthPerson/famail-temporal-data>

To skip the discriminator entirely (fairness-only experiments), set
`ALPHA_FIDELITY = 0` in `config.py`; the loader will not look for the file.

## Checkpoint format

The checkpoint should contain:

  - `model_state_dict` — PyTorch state dict
  - `architecture_config` — dict of constructor kwargs

If the checkpoint lacks `architecture_config`, the loader raises
`MissingArchitectureConfig` with a remediation message.

## Substituting a different checkpoint

Edit `famail_temporal/config.py`:

    DISCRIMINATOR_CHECKPOINT_FILENAME = "other_dir/best.pt"

Place the `.pt` file at the matching path under `discriminator_checkpoints/`.
