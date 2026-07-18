#!/usr/bin/env bash
# Absolute-value penalty probe (Robert-approved 2026-07-18; spec
# docs/superpowers/specs/2026-07-18-penalty-abs-probe-design.md).
# Stage 1: seed-0 pilot on the signed suite's exact lambda grid (comparability).
# Stage 2: n=6 suite at the inert- and catastrophic-representative doses.
# Idempotent via markers; relaunch verbatim to resume.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash fb_abs_chain.sh >> fb_abs_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
M=famail_temporal/results/experiments_campaign/markers
mkdir -p "$M"
S10=famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered
PILOT_OUT=famail_temporal/results/weighted_bc_sweep/fairness_penalty_abs_pilot
SUITE_OUT=famail_temporal/results/weighted_bc_sweep/fairness_penalty_abs_6seed

stage() { # $1 marker $2 qid $3 note $4 outdir $5... cmd
  local marker=$1 qid=$2 note=$3 outdir=$4; shift 4
  if [ -f "$M/$marker" ]; then echo "[fbabs $(date +%H:%M:%S)] $qid SKIP"; return 0; fi
  echo "[fbabs $(date +%H:%M:%S)] $qid START: $*"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$qid" --cmd "$*" \
    --artifact-dir "$outdir" --config-note "$note"
  if "$@"; then
    python -m famail_temporal.analysis.run_ledger finish --queue-id "$qid" --artifact-dir "$outdir"
    touch "$M/$marker"
    echo "[fbabs $(date +%H:%M:%S)] $qid DONE: $outdir"
  else
    echo "[fbabs $(date +%H:%M:%S)] $qid FAILED rc=$? — HALT"
    return 1
  fi
}

stage fb_abs_pilot.done FB-PENALTY-ABS-PILOT \
  "PRIMARY / shenzhen; abs-penalty pilot seed 0, lambda grid identical to the signed suite {1,3.16,10,100,1000} for point-by-point comparability (spec 2026-07-18; decision rule pre-committed)" \
  "$PILOT_OUT" \
  python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$S10" \
    --seeds 0 --weights "" --fairness-penalty-abs "1,3.16,10,100,1000" \
    --device auto --out-dir "$PILOT_OUT" || exit 1

stage fb_abs_suite.done FB-PENALTY-ABS \
  "PRIMARY / shenzhen; abs-penalty n=6 suite at lambda {10,1000} (inert- and catastrophic-representative doses; escalate only if pilot disagrees with signed)" \
  "$SUITE_OUT" \
  python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$S10" \
    --seeds 0,1,2,3,4,5 --weights "" --fairness-penalty-abs "10,1000" \
    --device auto --out-dir "$SUITE_OUT" || exit 1

echo "[fbabs $(date +%H:%M:%S)] FB-ABS CHAIN COMPLETE"
