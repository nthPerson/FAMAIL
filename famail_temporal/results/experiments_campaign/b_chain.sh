#!/usr/bin/env bash
# Post-campaign hardening chain (runs-menu 2026-07-15, Robert-approved 2026-07-16):
#   B1  — perturbation baselines (ifgsm/fgsm/random) on the SF alpha* corpus (~0.3 GPU-h)
#   B2  — variance suites at n=10 seeds, SF then SZ (~0.9 GPU-h)
#   C1  — SZ weighted-BC dose extension w40/w50 + both controls (~5-7 GPU-h)
# Cheapest-first so early results land early. Idempotent: each stage skips if its
# marker/artifact exists — relaunch this script verbatim to resume.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash b_chain.sh >> b_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
MARKERS=famail_temporal/results/experiments_campaign/markers
mkdir -p "$MARKERS"

SF_CORPUS=famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered
S10_CORPUS=famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered

stage() { # $1 marker  $2 queue-id  $3 config-note  $4 start-dir  $5 finish-glob  $6... command
  # start-dir must EXIST (run_ledger start mkdirs it and drops environment.json there).
  # finish-glob may be a pattern (B1 arms write timestamped dirs); the newest match at
  # finish time becomes the ledger row's artifact dir.
  local marker=$1 qid=$2 note=$3 sdir=$4 fglob=$5; shift 5
  if [ -f "$MARKERS/$marker" ]; then echo "[chain $(date +%H:%M:%S)] $qid SKIP (marker)"; return 0; fi
  echo "[chain $(date +%H:%M:%S)] $qid START: $*"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$qid" --cmd "$*" \
    --artifact-dir "$sdir" --config-note "$note"
  if "$@"; then
    local final
    final=$(ls -dt $fglob 2>/dev/null | head -1)
    [ -z "$final" ] && final=$sdir
    python -m famail_temporal.analysis.run_ledger finish --queue-id "$qid" --artifact-dir "$final"
    touch "$MARKERS/$marker"
    echo "[chain $(date +%H:%M:%S)] $qid DONE: $final"
  else
    local rc=$?
    echo "[chain $(date +%H:%M:%S)] $qid FAILED rc=$rc — chain continues to next stage"
    return 0
  fi
}

# ---- B1: SF perturbation arms (SF discriminator; k=2000 corpus) ----
stage b1_ifgsm.done  B1-IFGSM  "PRIMARY(ACS) / sf12; runs-menu B1: iFGSM arm on the SF alpha* corpus"  "$SF_CORPUS" 'famail_temporal/results/*_baseline_ifgsm_sf12' \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir "$SF_CORPUS" --mode ifgsm --seed 0 --device auto --score-fidelity
stage b1_fgsm.done   B1-FGSM   "PRIMARY(ACS) / sf12; runs-menu B1: FGSM arm on the SF alpha* corpus"   "$SF_CORPUS" 'famail_temporal/results/*_baseline_fgsm_sf12' \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir "$SF_CORPUS" --mode fgsm --seed 0 --device auto --score-fidelity
stage b1_random.done B1-RANDOM "PRIMARY(ACS) / sf12; runs-menu B1: random-jitter arm on the SF alpha* corpus" "$SF_CORPUS" 'famail_temporal/results/*_baseline_random_sf12' \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir "$SF_CORPUS" --mode random --seed 0 --device auto --score-fidelity

# ---- B2: variance suites at n=10 (SF first: ~9 min; then SZ: ~45 min) ----
stage b2_var_sf.done B2-VAR-SF "PRIMARY(ACS) / sf12; runs-menu B2: variance suite n=10 (seeds 0-9)" \
  famail_temporal/results/variance_suite/supply_lift_sf12_10seed \
  famail_temporal/results/variance_suite/supply_lift_sf12_10seed \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_variance_suite --edit-dir "$SF_CORPUS" --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir famail_temporal/results/variance_suite/supply_lift_sf12_10seed
stage b2_var_sz.done B2-VAR-SZ "PRIMARY / shenzhen; runs-menu B2: variance suite n=10 (seeds 0-9)" \
  famail_temporal/results/variance_suite/supply_lift_shz_10seed \
  famail_temporal/results/variance_suite/supply_lift_shz_10seed \
  python -m famail_temporal.baselines.run_variance_suite --edit-dir "$S10_CORPUS" --seeds 0,1,2,3,4,5,6,7,8,9 --out-dir famail_temporal/results/variance_suite/supply_lift_shz_10seed

# ---- C1: SZ weighted-BC dose extension w40/w50, edited + both controls ----
stage c1_wbc.done C1-WBC-DOSEEXT "PRIMARY / shenzhen; runs-menu C1: WBC dose extension w40/w50 (saturation), edited + placebo + most-fair, 6 seeds" \
  famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_dose_ext_6seed \
  famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_dose_ext_6seed \
  python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$S10_CORPUS" --seeds 0,1,2,3,4,5 --weights 40,50 --placebo 40,50 --most-fair 40,50 --out-dir famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_dose_ext_6seed

echo "[chain $(date +%H:%M:%S)] B-CHAIN COMPLETE"
