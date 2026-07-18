#!/usr/bin/env bash
# Hardening chain 2 (Robert-approved 2026-07-17): A1' tier-2 featsets splits ->
# SF WBC dose extension w40/w50 -> C3 SF rollout comparators (trim-only + trim+lift)
# -> WBC headline seeds 6-11 (n=12 pooled). C2 HELD (Robert: keep the disclosure).
# Idempotent via markers; relaunch verbatim to resume.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash h_chain.sh >> h_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
M=famail_temporal/results/experiments_campaign/markers
mkdir -p "$M"
S10=famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered
HGC=famail_temporal/results/2026-07-13T04-41-12_supply_lift_v1_shz_hgc_filtered
FEAT4=famail_temporal/results/2026-07-13T17-04-22_supply_lift_v1_shz_4feat_filtered
SF_TL=famail_temporal/results/2026-07-11T11-31-55_supply_lift_a10_sf12_filtered
SF_TO=famail_temporal/results/2026-07-11T13-43-37_trimonly_a10_sf12

stage() { # $1 marker $2 qid $3 note $4 outdir $5... cmd
  local marker=$1 qid=$2 note=$3 outdir=$4; shift 4
  if [ -f "$M/$marker" ]; then echo "[h $(date +%H:%M:%S)] $qid SKIP"; return 0; fi
  echo "[h $(date +%H:%M:%S)] $qid START: $*"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$qid" --cmd "$*" \
    --artifact-dir "$outdir" --config-note "$note"
  if "$@"; then
    python -m famail_temporal.analysis.run_ledger finish --queue-id "$qid" --artifact-dir "$outdir"
    touch "$M/$marker"
    echo "[h $(date +%H:%M:%S)] $qid DONE: $outdir"
  else
    echo "[h $(date +%H:%M:%S)] $qid FAILED rc=$? — continuing to next stage"
    return 0
  fi
}

# ---- A1': tier-2 recount + tier-2 channel decomposition on the alternate feature sets ----
stage a1p_hgc_recount.done A1P-HGC-RECOUNT "HGC / shenzhen; A1' tier-2 recount with persisted grids" "$HGC" \
  python -m famail_temporal.analysis.supply_recount --edit-dir "$HGC" --persist-grids
stage a1p_hgc_chan.done A1P-HGC-CHAN "HGC / shenzhen; A1' channel decomposition WITH tier-2 grid" "$HGC" \
  python -m famail_temporal.analysis.channel_decomposition --edit-dir "$HGC" --bootstrap 2000 --seed 0 --tier2-grid "$HGC/S_tier2_after.npz"
stage a1p_4feat_recount.done A1P-4FEAT-RECOUNT "4FEAT / shenzhen; A1' tier-2 recount with persisted grids" "$FEAT4" \
  python -m famail_temporal.analysis.supply_recount --edit-dir "$FEAT4" --persist-grids
stage a1p_4feat_chan.done A1P-4FEAT-CHAN "4FEAT / shenzhen; A1' channel decomposition WITH tier-2 grid" "$FEAT4" \
  python -m famail_temporal.analysis.channel_decomposition --edit-dir "$FEAT4" --bootstrap 2000 --seed 0 --tier2-grid "$FEAT4/S_tier2_after.npz"

# ---- SF WBC dose extension w40/w50 (saturation counterpart) ----
SFDOSE=famail_temporal/results/weighted_bc_sweep/supply_lift_a10_sf12_dose_ext_6seed
stage sf_dose.done SF-WBC-DOSEEXT "PRIMARY(ACS) / sf12; SF dose extension w40/w50, edited + both controls, 6 seeds" "$SFDOSE" \
  env FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$SF_TL" --seeds 0,1,2,3,4,5 --weights 40,50 --placebo 40,50 --most-fair 40,50 --device auto --out-dir "$SFDOSE"

# ---- C3: SF rollout comparators (trim+lift AND trim-only) ----
SFR_TL=famail_temporal/baselines/external_fairness/results/option_a_rollout_sf12_tl
stage c3_tl.done C3-SF-ROLLOUT-TL "PRIMARY(ACS) / sf12; SF trim+lift rollout (allocation boundary, city 2)" "$SFR_TL" \
  env FAMAIL_CITY=sf12 python PAPER/external-metrics/scripts/option_a_rollout_eval.py --edit-dir "$SF_TL" --seeds 0,1,2,3,4,5 --arms raw,edited,edited_w10,edited_w30 --device auto --out-dir "$SFR_TL"
SFR_TO=famail_temporal/baselines/external_fairness/results/option_a_rollout_sf12_trimonly
stage c3_to.done C3-SF-ROLLOUT-TO "PRIMARY(ACS) / sf12; SF trim-only rollout comparator (attenuation denominator, city 2)" "$SFR_TO" \
  env FAMAIL_CITY=sf12 python PAPER/external-metrics/scripts/option_a_rollout_eval.py --edit-dir "$SF_TO" --seeds 0,1,2,3,4,5 --arms raw,edited,edited_w10,edited_w30 --device auto --out-dir "$SFR_TO"

# ---- WBC headline seeds 6-11 (pool to n=12 analysis-side) ----
N12=famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_seeds6to11
stage n12.done WBC-N12 "PRIMARY / shenzhen; headline WBC seeds 6-11 (raw/edited/edited_w30 + w30 controls) -> pooled n=12 so the flagship recovery clears p<.05 with correction headroom" "$N12" \
  python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir "$S10" --seeds 6,7,8,9,10,11 --weights 30 --placebo 30 --most-fair 30 --device auto --out-dir "$N12"

echo "[h $(date +%H:%M:%S)] H-CHAIN COMPLETE"
