#!/usr/bin/env bash
# R5: rollout-allocation eval at alpha* on the s10 headline corpus.
# Same tool + protocol as the 2026-07-09 supply-lift run (PAPER/external-metrics/
# scripts/option_a_rollout_eval.py, 6 seeds x {raw,edited,edited_w10,edited_w30},
# 20 epochs) so the "attenuated, not reversed" boundary is re-measured 1:1.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash r5.sh >> r5.log 2>&1 &
set -u
cd /home/robert/FAMAIL

EDIT_DIR=famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered
OUT_DIR=famail_temporal/baselines/external_fairness/results/option_a_rollout_a10
CMD="python PAPER/external-metrics/scripts/option_a_rollout_eval.py --edit-dir $EDIT_DIR --out-dir $OUT_DIR"

echo "[r5 $(date +%H:%M:%S)] start: $CMD"
mkdir -p "$OUT_DIR"
python -m famail_temporal.analysis.run_ledger start --queue-id R5-rollout-a10 \
  --cmd "$CMD" --artifact-dir "$OUT_DIR" \
  --config-note "PRIMARY / shenzhen; policy-rollout allocation shares at alpha* (protocol = 2026-07-09 run)"
if $CMD; then
  python -m famail_temporal.analysis.run_ledger finish --queue-id R5-rollout-a10 --artifact-dir "$OUT_DIR"
  echo "[r5 $(date +%H:%M:%S)] R5 DONE: $OUT_DIR/summary.json"
else
  echo "[r5 $(date +%H:%M:%S)] R5 FAILED rc=$?"
  exit 1
fi
