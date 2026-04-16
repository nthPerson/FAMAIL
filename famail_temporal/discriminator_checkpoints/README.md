# Discriminator Checkpoints

The canonical checkpoint used by `famail_temporal/` lives at:

    discriminator_checkpoints/default/best.pt

## Canonical checkpoint provenance

Copy from the parent project:

    cp discriminator/model/checkpoints/20260316_223817/best.pt \
       famail_temporal/discriminator_checkpoints/default/best.pt

The checkpoint should contain:

  - `model_state_dict` — PyTorch state dict
  - `architecture_config` — dict of constructor kwargs

If the checkpoint lacks `architecture_config`, the loader raises
`MissingArchitectureConfig` with a remediation message.

## Substituting a different checkpoint

Edit `famail_temporal/config.py`:

    DISCRIMINATOR_CHECKPOINT_FILENAME = "other_dir/best.pt"

Place the `.pt` file at the matching path under `discriminator_checkpoints/`.
