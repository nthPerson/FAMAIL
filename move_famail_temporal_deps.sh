#!/usr/bin/bash
# Provision famail_temporal/ with non-source-generation dependencies.
#
# Most source datasets are now produced by the unified source-generation tool:
#   python -m famail_temporal.data.source_generation \
#       --input-dir raw_data/ --output-dir famail_temporal/source_data/
#
# This script only copies the files that that tool does NOT produce:
# 1. The 2 external inputs (census/district data — not GPS-derived).
# 2. The discriminator checkpoint.
#
# See famail_temporal/source_data/README.md for the full file inventory.

set -e

# External inputs consumed by preprocess.py (see source_data/README.md, Group B).
cp source_data/cell_demographics.pkl        famail_temporal/source_data/
cp source_data/grid_to_district_mapping.pkl famail_temporal/source_data/

# Discriminator checkpoint for F_fidelity (retrain after regenerating
# ms_* source datasets; see CHANGELOG 2026-04-20 entry for the scheduled-next
# retraining task).
cp discriminator/model/checkpoints/20260316_223817/best.pt \
   famail_temporal/discriminator_checkpoints/default/best.pt
