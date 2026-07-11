#!/usr/bin/env bash
# Re-anchor r-chain (alpha* = (0.1, 0.8, 0.1)) — serial GPU work, skip-if-done per step.
# r1: SF edit at alpha* + filter + external metrics + channel decomposition
# r2a: SZ trim-only baseline at alpha* (TAIL_LEN=0 legacy) + external metrics
# r2b: SF trim-only baseline at alpha* + external metrics
# r4:  SZ weighted-BC sweep on the s10 corpus (10 arms x 6 seeds)
# Launch: nohup setsid bash r_chain.sh >> r_chain.log 2>&1 &   (survives session restarts)
set -uo pipefail
cd /home/robert/FAMAIL
export PYTHONUNBUFFERED=1
R=famail_temporal/results
S10=$R/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered
AOV="--override ALPHA_SPATIAL=0.1 --override ALPHA_CAUSAL=0.8 --override ALPHA_FIDELITY=0.1"
RL="python -m famail_temporal.analysis.run_ledger"
step() { echo "[r_chain $(date -u +%H:%M:%S)] $*"; }

# ---------- r1: SF edit at alpha* ----------
if ls $R/*_supply_lift_a10_sf12_filtered/metrics.json >/dev/null 2>&1; then
  step "r1 SKIP (filtered SF a10 corpus exists)"
else
  step "r1 START: SF edit at (0.1,0.8,0.1)"
  $RL start --queue-id R1-sf-a10 --cmd "FAMAIL_CITY=sf12 runner -k 2000 --name supply_lift_a10_sf12 $AOV; filter; externals; channel" --artifact-dir $R/experiments_campaign/ledger/R1-a10 --config-note "sf12; alpha*=(0.1,0.8,0.1); PRIMARY features"
  FAMAIL_CITY=sf12 python -m famail_temporal.evaluation.runner -k 2000 --name supply_lift_a10_sf12 --device auto $AOV || { step "r1 FAILED (runner)"; exit 1; }
  EDIT=$(ls -dt $R/*_supply_lift_a10_sf12 | head -1)
  FAMAIL_CITY=sf12 python -m famail_temporal.analysis.filter_infeasible_trims --edit-dir "$EDIT" || { step "r1 FAILED (filter)"; exit 1; }
  F="${EDIT}_filtered"
  $RL finish --queue-id R1-sf-a10 --artifact-dir "$F"
  FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_external_fairness --edit-dir "$F" --dataset sf12-supplylift-a10 --delta-supply "$F/delta_supply_3d.npz" --bootstrap 1000 --seed 0 || step "r1 WARN: externals failed"
  FAMAIL_CITY=sf12 python -m famail_temporal.analysis.channel_decomposition --edit-dir "$F" --bootstrap 2000 --seed 0 || step "r1 WARN: channel failed"
  step "r1 DONE: $F"
fi

# ---------- r2a: SZ trim-only at alpha* ----------
if ls $R/*_trimonly_a10_shz/metrics.json >/dev/null 2>&1; then
  step "r2a SKIP"
else
  step "r2a START: SZ trim-only (legacy TAIL_LEN=0) at (0.1,0.8,0.1)"
  $RL start --queue-id R2a-trimonly-shz --cmd "runner -k 10000 --name trimonly_a10_shz $AOV --override TAIL_LEN=0; externals" --artifact-dir $R/experiments_campaign/ledger/R2a --config-note "shenzhen; trim-only ablation baseline at alpha*"
  python -m famail_temporal.evaluation.runner -k 10000 --name trimonly_a10_shz --device auto $AOV --override TAIL_LEN=0 || { step "r2a FAILED"; exit 1; }
  T=$(ls -dt $R/*_trimonly_a10_shz | head -1)
  $RL finish --queue-id R2a-trimonly-shz --artifact-dir "$T"
  python -m famail_temporal.baselines.run_external_fairness --edit-dir "$T" --dataset shenzhen-trimonly-a10 --bootstrap 1000 --seed 0 || step "r2a WARN: externals failed"
  step "r2a DONE: $T"
fi

# ---------- r2b: SF trim-only at alpha* ----------
if ls $R/*_trimonly_a10_sf12/metrics.json >/dev/null 2>&1; then
  step "r2b SKIP"
else
  step "r2b START: SF trim-only at (0.1,0.8,0.1)"
  $RL start --queue-id R2b-trimonly-sf12 --cmd "FAMAIL_CITY=sf12 runner -k 2000 --name trimonly_a10_sf12 $AOV --override TAIL_LEN=0; externals" --artifact-dir $R/experiments_campaign/ledger/R2b --config-note "sf12; trim-only ablation baseline at alpha*"
  FAMAIL_CITY=sf12 python -m famail_temporal.evaluation.runner -k 2000 --name trimonly_a10_sf12 --device auto $AOV --override TAIL_LEN=0 || { step "r2b FAILED"; exit 1; }
  T=$(ls -dt $R/*_trimonly_a10_sf12 | head -1)
  $RL finish --queue-id R2b-trimonly-sf12 --artifact-dir "$T"
  FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_external_fairness --edit-dir "$T" --dataset sf12-trimonly-a10 --bootstrap 1000 --seed 0 || step "r2b WARN: externals failed"
  step "r2b DONE: $T"
fi

# ---------- r4: SZ weighted-BC sweep on the s10 corpus ----------
WBC_OUT=$R/weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed
if [ -f "$WBC_OUT/paired_stats.json" ]; then
  step "r4 SKIP"
else
  step "r4 START: SZ weighted-BC sweep on s10 corpus (10 arms x 6 seeds, ~10h)"
  $RL start --queue-id R4-wbc-shz --cmd "run_weighted_bc_smoke --edit-dir $S10 --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir $WBC_OUT" --artifact-dir "$WBC_OUT" --config-note "shenzhen; s10 headline corpus"
  python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$S10" --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir "$WBC_OUT" || { step "r4 FAILED"; exit 1; }
  $RL finish --queue-id R4-wbc-shz --artifact-dir "$WBC_OUT"
  step "r4 DONE: $WBC_OUT"
fi

step "R-CHAIN COMPLETE"
