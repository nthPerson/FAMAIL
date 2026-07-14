#!/usr/bin/env bash
# R5b: rollout-allocation eval of the ALPHA* TRIM-ONLY corpus — the one demand-only
# baseline never re-run at the adopted weights (2026-07-13 era audit; Robert approved
# the run). Restores a like-for-like attenuation ratio for §4.4.
# Config-independent (uses config.GRID_DIMS only; verified 2026-07-13), so it may run
# in any DEMOGRAPHIC_FEATURES window; launched as a guarded companion beside q6a
# (measured: editor 471 MiB + rollout ~2.7 GB ≈ 3.2/8.2 GB).
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash guarded_companion.sh r5b "bash /home/robert/FAMAIL/famail_temporal/results/experiments_campaign/r5b.sh" >> guarded_r5b.log 2>&1 &
set -u
cd /home/robert/FAMAIL

EDIT_DIR=famail_temporal/results/2026-07-11T12-11-31_trimonly_a10_shz
OUT_DIR=famail_temporal/baselines/external_fairness/results/option_a_rollout_trimonly_a10
CMD="python PAPER/external-metrics/scripts/option_a_rollout_eval.py --edit-dir $EDIT_DIR --out-dir $OUT_DIR"

echo "[r5b $(date +%H:%M:%S)] start: $CMD"
mkdir -p "$OUT_DIR"
python -m famail_temporal.analysis.run_ledger start --queue-id R5b-trimonly-rollout \
  --cmd "$CMD" --artifact-dir "$OUT_DIR" \
  --config-note "shenzhen; alpha* TRIM-ONLY corpus rollout (demand-only comparator for §4.4 attenuation ratio); config-independent tool — runs during the HGC window beside q6a"
if $CMD; then
  python -m famail_temporal.analysis.run_ledger finish --queue-id R5b-trimonly-rollout --artifact-dir "$OUT_DIR"
  echo "[r5b $(date +%H:%M:%S)] R5b DONE: $OUT_DIR/summary.json"
else
  rc=$?  # capture BEFORE the $(date) substitution resets $? (masked the 07-13 SIGTERM as rc=0)
  echo "[r5b $(date +%H:%M:%S)] R5b FAILED rc=$rc"
  exit 1
fi
