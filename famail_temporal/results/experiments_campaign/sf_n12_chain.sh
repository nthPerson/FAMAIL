#!/usr/bin/env bash
# SF WBC n=12 extension (Robert-approved 2026-07-18): seeds 6-11 on the SF
# supply-lift corpus -> pooled n=12 for §4.7's recovery (mirror of WBC-N12).
# Waits on the h-chain's n12.done marker so the GPU is free before starting.
# Idempotent via markers; relaunch verbatim to resume.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash sf_n12_chain.sh >> sf_n12_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
M=famail_temporal/results/experiments_campaign/markers
mkdir -p "$M"
SF_TL=famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered
OUT=famail_temporal/results/weighted_bc_sweep/supply_lift_a10_sf12_seeds6to11

stage() { # $1 marker $2 qid $3 note $4 outdir $5... cmd
  local marker=$1 qid=$2 note=$3 outdir=$4; shift 4
  if [ -f "$M/$marker" ]; then echo "[sfn12 $(date +%H:%M:%S)] $qid SKIP"; return 0; fi
  echo "[sfn12 $(date +%H:%M:%S)] $qid START: $*"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$qid" --cmd "$*" \
    --artifact-dir "$outdir" --config-note "$note"
  if "$@"; then
    python -m famail_temporal.analysis.run_ledger finish --queue-id "$qid" --artifact-dir "$outdir"
    touch "$M/$marker"
    echo "[sfn12 $(date +%H:%M:%S)] $qid DONE: $outdir"
  else
    echo "[sfn12 $(date +%H:%M:%S)] $qid FAILED rc=$?"
    return 1
  fi
}

# Gate: wait for the h-chain's final stage (WBC-N12) to finish so the GPU is idle.
echo "[sfn12 $(date +%H:%M:%S)] waiting on $M/n12.done (h-chain drain) ..."
while [ ! -f "$M/n12.done" ]; do sleep 120; done
echo "[sfn12 $(date +%H:%M:%S)] gate open — GPU free"

stage sf_n12.done SF-WBC-N12 \
  "PRIMARY(ACS) / sf12; SF WBC seeds 6-11 (raw/edited/edited_w30 + w30 controls) -> pooled n=12 so the SF recovery in §4.7 clears the p=.031 floor with correction headroom (mirror of WBC-N12; Robert-approved 2026-07-18)" \
  "$OUT" \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_weighted_bc_smoke \
    --edit-dir "$SF_TL" --seeds 6,7,8,9,10,11 --weights 30 --placebo 30 \
    --most-fair 30 --device auto --out-dir "$OUT"

echo "[sfn12 $(date +%H:%M:%S)] SF-N12 CHAIN COMPLETE"
