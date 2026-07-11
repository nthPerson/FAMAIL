#!/usr/bin/env bash
# Experiments-section campaign driver — trim+lift re-run queue Q1–Q8b.
# Plan: docs/superpowers/plans/2026-07-11-experiments-section.md (Task 2).
#
# Stages (fixed campaign order; one GPU job at a time on the RTX 3070 eGPU):
#   q1   perturbation arms (ifgsm/fgsm/random + 2 --no-random-start ablations)
#        + per-arm external fairness + tier-2 supply recount + 6-row table
#   q2   SF weighted-BC sweep (trim+lift sf12 filtered corpus)
#   q3   SZ L1v2 four-source table (fresh generators)
#   q4   SF L1v2 four-source table (fresh generators)
#   q5   variance suites, SZ then SF
#   q6a  HGC alternate-set trim+lift edit run + infeasible-trim filter
#   q6b  4FEAT alternate-set trim+lift edit run + infeasible-trim filter
#   q7   per-set external fairness (hgc/4feat, each under its matching config)
#        + PRIMARY filter@K Pareto
#   q8a  HGC downstream block: L1v2 + weighted-BC + variance (serial)
#   q8b  4FEAT downstream block: L1v2 + weighted-BC + variance (serial)
#
# Usage:   nohup setsid bash famail_temporal/results/experiments_campaign/driver.sh q1 \
#            >> famail_temporal/results/experiments_campaign/q1.driver.log 2>&1 &
# Status:  bash famail_temporal/results/experiments_campaign/driver.sh --status
#
# RESUMABLE: every sub-step is skip-if-done via its terminal artifact; a stage
# whose DONE marker exists is skipped outright. Interrupting loses only the
# in-flight sub-step; relaunch the same stage to resume.
#
# LEDGER DISCIPLINE: nothing runs without a ledger row — every command is
# wrapped by `python -m famail_temporal.analysis.run_ledger start/finish`
# (Task 1). For commands whose results dir is timestamped (arm runs, editor
# runs) the start row's artifact dir is a placeholder under
# experiments_campaign/ledger/<QID>/ (environment.json lands there) and
# `finish` is pointed at the discovered results dir (checksums land with the
# real artifacts).
#
# CONFIG-FLIP GUARDS: q1–q5 and q7's Pareto part require the PRIMARY feature
# set; q6a/q8a (+ q7's hgc externals) require HGC; q6b/q8b (+ q7's 4feat
# externals) require 4FEAT. Detection greps the DEMOGRAPHIC_FEATURES list
# BLOCK of famail_temporal/config.py (not the whole file — 'LogPopDensity'
# appears in a comment at config.py:63, so a whole-file grep would
# misclassify PRIMARY). The flips themselves are Task-15/16 commits
# ('paper-campaign: config -> <set>'), never driver actions.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /home/robert/FAMAIL

RESULTS=famail_temporal/results
STATE_DIR=$RESULTS/experiments_campaign
mkdir -p "$STATE_DIR/ledger"

# ---- fixed inputs (headline trim+lift filtered corpora + oversampling s0 arms)
# RE-ANCHOR (Robert, 2026-07-11; PAPER/objective-motivation/weight-sensitivity/DECISION.md):
# headline corpora are the alpha*=(0.55,0.35,0.1) runs. SZ = the promoted s55 sweep
# corpus (headline-grade, no re-run); SF = the R1 re-edit at alpha*.
SHZ_FILTERED=$RESULTS/2026-07-10T17-45-40_alpha_sweep_s55_c35_f10_filtered
SF_FILTERED=__R1_PENDING__  # set to the <ts>_supply_lift_a55_sf12_filtered dir when R1 lands; q2/q4/q5-sf refuse while unset
OVR_D2500=$RESULTS/2026-07-10T00-39-51_baseline_demo_oversample_targeted_d2500_s0_shenzhen
OVR_D5000=$RESULTS/2026-07-10T00-47-07_baseline_demo_oversample_targeted_d5000_s0_shenzhen
OVR_D10000=$RESULTS/2026-07-10T00-47-33_baseline_demo_oversample_targeted_d10000_s0_shenzhen
EXT_RESULTS=famail_temporal/baselines/external_fairness/results
STUB_FAMAIL=famail_temporal/baselines/famail_headline_stub.json
STUB_RAW=famail_temporal/baselines/raw_stub.json

# ---- stage DONE markers
Q1_MARKER=famail_temporal/baselines/baseline_table/baseline_table.md
Q2_MARKER=$RESULTS/weighted_bc_sweep/supply_lift_v1_sf12_filtered_6seed/paired_stats.json
Q3_MARKER=$RESULTS/level1_table_v2/supply_lift_shz_5seed/level1_v2_multiseed.json
Q4_MARKER=$RESULTS/level1_table_v2/supply_lift_sf12_5seed/level1_v2_multiseed.json
Q5_MARKER_SHZ=$RESULTS/variance_suite/supply_lift_shz_5seed/aggregate.json
Q5_MARKER_SF=$RESULTS/variance_suite/supply_lift_sf12_5seed/aggregate.json
Q7_MARKER_HGC=$EXT_RESULTS/shenzhen-hgc-supplylift/external_fairness.json
Q7_MARKER_4FEAT=$EXT_RESULTS/shenzhen-4feat-supplylift/external_fairness.json
Q7_MARKER_PARETO=$RESULTS/analysis/pareto_supplylift/pareto_points.json
Q8A_MARKER_L1=$RESULTS/level1_table_v2/supply_lift_shz_hgc_5seed/level1_v2_multiseed.json
Q8A_MARKER_WBC=$RESULTS/weighted_bc_sweep/supply_lift_v1_shz_hgc_filtered_6seed/paired_stats.json
Q8A_MARKER_VAR=$RESULTS/variance_suite/supply_lift_shz_hgc_5seed/aggregate.json
Q8B_MARKER_L1=$RESULTS/level1_table_v2/supply_lift_shz_4feat_5seed/level1_v2_multiseed.json
Q8B_MARKER_WBC=$RESULTS/weighted_bc_sweep/supply_lift_v1_shz_4feat_filtered_6seed/paired_stats.json
Q8B_MARKER_VAR=$RESULTS/variance_suite/supply_lift_shz_4feat_5seed/aggregate.json

STAGE=""
LOG=""

# ---------------------------------------------------------------- helpers ----

demo_block() {  # the DEMOGRAPHIC_FEATURES list block of config.py (comments excluded)
  sed -n '/^DEMOGRAPHIC_FEATURES/,/^\]/p' famail_temporal/config.py
}
demo_has() { demo_block | grep -q "\"$1\""; }
config_is_primary() { demo_has MigrantRatio && ! demo_has LogPopDensity && ! demo_has GDPperCapita; }

config_label() {
  if config_is_primary; then echo PRIMARY
  elif demo_has GDPperCapita; then echo HGC
  elif demo_has LogPopDensity; then echo 4FEAT
  else echo UNKNOWN
  fi
}

require_primary() {
  config_is_primary && return 0
  echo "ERROR: stage $STAGE requires the PRIMARY feature set; config.py is currently $(config_label)." >&2
  echo "  Edit famail_temporal/config.py DEMOGRAPHIC_FEATURES to" >&2
  echo "  [\"AvgHousingPricePerSqM\", \"CompPerCapita\", \"MigrantRatio\"]," >&2
  echo "  commit 'paper-campaign: config -> PRIMARY (restore)' (plan Tasks 15/16), then re-run $STAGE." >&2
  exit 1
}
require_hgc() {
  demo_has GDPperCapita && return 0
  echo "ERROR: stage $STAGE requires the HGC feature set; config.py is currently $(config_label)." >&2
  echo "  Edit famail_temporal/config.py DEMOGRAPHIC_FEATURES to" >&2
  echo "  [\"AvgHousingPricePerSqM\", \"GDPperCapita\", \"CompPerCapita\"]," >&2
  echo "  commit 'paper-campaign: config -> housing-gdp-comp' (plan Task 15 Step 1 / Task 16 Step 1), then re-run $STAGE." >&2
  exit 1
}
require_4feat() {
  demo_has LogPopDensity && return 0
  echo "ERROR: stage $STAGE requires the 4FEAT feature set; config.py is currently $(config_label)." >&2
  echo "  Edit famail_temporal/config.py DEMOGRAPHIC_FEATURES to" >&2
  echo "  [\"AvgHousingPricePerSqM\", \"CompPerCapita\", \"MigrantRatio\", \"LogPopDensity\"]," >&2
  echo "  commit 'paper-campaign: config -> 4feat' (plan Task 15 Step 3 / Task 16 Step 2), then re-run $STAGE." >&2
  exit 1
}

require_ledger() {
  [ -f famail_temporal/analysis/run_ledger.py ] && return 0
  echo "ERROR: famail_temporal/analysis/run_ledger.py is missing — implement plan Task 1 first (ledger discipline: nothing runs without a ledger row)." >&2
  exit 1
}

find_editor_run() {  # $1 = runner --name suffix -> newest completed run dir (has metrics.json); _filtered excluded by ends-with glob
  local name="$1" d
  for d in $(ls -dt "$RESULTS"/*"$name" 2>/dev/null || true); do
    [ -f "$d/metrics.json" ] && { echo "$d"; return 0; }
  done
  return 0
}

find_arm_dir() {  # $1 mode, $2 random_start (true|false) -> newest COMPLETE arm dir (fidelity scored)
  local mode="$1" rs="$2" d
  for d in $(ls -dt "$RESULTS"/*"_baseline_${mode}_shenzhen" 2>/dev/null || true); do
    [ -f "$d/metrics.json" ] || continue
    grep -q "\"random_start\": ${rs}" "$d/metrics.json" || continue
    grep -q '"fidelity"' "$d/metrics.json" || continue
    echo "$d"
    return 0
  done
  return 0
}

log_echo() { echo "--- [$(date -Is)] $*" | tee -a "$LOG"; }

ledger_start() {  # qid, artifact_dir, config_note, cmd
  mkdir -p "$2"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$1" \
    --cmd "$4" --artifact-dir "$2" --config-note "$3" >> "$LOG" 2>&1
}
ledger_finish() {  # qid, artifact_dir
  python -m famail_temporal.analysis.run_ledger finish --queue-id "$1" \
    --artifact-dir "$2" >> "$LOG" 2>&1
}
run_logged() {  # cmd (a single string; FAMAIL_CITY=sf12 prefixes stay scoped to it)
  log_echo "RUN: $1"
  bash -c "$1" >> "$LOG" 2>&1
}
ledger_run() {  # qid, artifact_dir (static, known up-front), config_note, cmd
  ledger_start "$1" "$2" "$3" "$4"
  run_logged "$4"
  ledger_finish "$1" "$2"
}

# ----------------------------------------------------------------- status ----

flag() { if [ -f "$1" ]; then echo OK; else echo --; fi; }

q6_status_line() {  # $1 stage, $2 runner name suffix
  local run
  run=$(find_editor_run "$2")
  if [ -n "$run" ] && [ -f "${run}_filtered/metrics.json" ]; then
    echo "$1   DONE     ${run}_filtered/metrics.json"
  elif [ -n "$run" ]; then
    echo "$1   PENDING  ${run}_filtered/metrics.json (edit run done, filter pending)"
  else
    echo "$1   PENDING  $RESULTS/<ts>_$2_filtered/metrics.json (edit run not started)"
  fi
}

print_status() {
  echo "config: $(config_label)   (guards: q1-q5+q7-pareto=PRIMARY, q6a/q8a=HGC, q6b/q8b=4FEAT)"
  if [ -f "$Q1_MARKER" ]; then echo "q1    DONE     $Q1_MARKER"; else echo "q1    PENDING  $Q1_MARKER"; fi
  if [ -f "$Q2_MARKER" ]; then echo "q2    DONE     $Q2_MARKER"; else echo "q2    PENDING  $Q2_MARKER"; fi
  if [ -f "$Q3_MARKER" ]; then echo "q3    DONE     $Q3_MARKER"; else echo "q3    PENDING  $Q3_MARKER"; fi
  if [ -f "$Q4_MARKER" ]; then echo "q4    DONE     $Q4_MARKER"; else echo "q4    PENDING  $Q4_MARKER"; fi
  if [ -f "$Q5_MARKER_SHZ" ] && [ -f "$Q5_MARKER_SF" ]; then
    echo "q5    DONE     $Q5_MARKER_SHZ + $Q5_MARKER_SF"
  else
    echo "q5    PENDING  $Q5_MARKER_SHZ [$(flag "$Q5_MARKER_SHZ")] + $Q5_MARKER_SF [$(flag "$Q5_MARKER_SF")]"
  fi
  q6_status_line q6a supply_lift_v1_shz_hgc
  q6_status_line q6b supply_lift_v1_shz_4feat
  if [ -f "$Q7_MARKER_HGC" ] && [ -f "$Q7_MARKER_4FEAT" ] && [ -f "$Q7_MARKER_PARETO" ]; then
    echo "q7    DONE     $Q7_MARKER_PARETO [hgc-ext:OK 4feat-ext:OK pareto:OK]"
  else
    echo "q7    PENDING  $Q7_MARKER_PARETO [hgc-ext:$(flag "$Q7_MARKER_HGC") 4feat-ext:$(flag "$Q7_MARKER_4FEAT") pareto:$(flag "$Q7_MARKER_PARETO")]"
  fi
  if [ -f "$Q8A_MARKER_L1" ] && [ -f "$Q8A_MARKER_WBC" ] && [ -f "$Q8A_MARKER_VAR" ]; then
    echo "q8a   DONE     $Q8A_MARKER_VAR [l1v2:OK wbc:OK var:OK]"
  else
    echo "q8a   PENDING  $Q8A_MARKER_VAR [l1v2:$(flag "$Q8A_MARKER_L1") wbc:$(flag "$Q8A_MARKER_WBC") var:$(flag "$Q8A_MARKER_VAR")]"
  fi
  if [ -f "$Q8B_MARKER_L1" ] && [ -f "$Q8B_MARKER_WBC" ] && [ -f "$Q8B_MARKER_VAR" ]; then
    echo "q8b   DONE     $Q8B_MARKER_VAR [l1v2:OK wbc:OK var:OK]"
  else
    echo "q8b   PENDING  $Q8B_MARKER_VAR [l1v2:$(flag "$Q8B_MARKER_L1") wbc:$(flag "$Q8B_MARKER_WBC") var:$(flag "$Q8B_MARKER_VAR")]"
  fi
}

# ----------------------------------------------------------------- stages ----

stage_q1() {
  require_primary
  if [ -f "$Q1_MARKER" ]; then log_echo "q1 already DONE ($Q1_MARKER) — skip"; return 0; fi
  for stub in "$STUB_FAMAIL" "$STUB_RAW"; do
    if [ ! -f "$stub" ]; then
      echo "ERROR: missing $stub — hand-author the 6-row-table stubs first (plan Task 11 Step 1)." >&2
      exit 1
    fi
  done

  local mode d cmd
  # 3 perturbation arms (PGD-style random start — the paper-facing arms).
  for mode in ifgsm fgsm random; do
    d=$(find_arm_dir "$mode" true)
    if [ -n "$d" ]; then
      log_echo "q1 arm $mode (random-start) already complete: $d (skip)"
    else
      cmd="python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir $SHZ_FILTERED --mode $mode --seed 0 --device auto --score-fidelity"
      ledger_start "Q1-arm-$mode" "$STATE_DIR/ledger/Q1-arm-$mode" "PRIMARY / shenzhen / $mode random-start" "$cmd"
      run_logged "$cmd"
      d=$(find_arm_dir "$mode" true)
      if [ -z "$d" ]; then log_echo "!!! q1 arm $mode produced no complete arm dir — see $LOG"; exit 1; fi
      ledger_finish "Q1-arm-$mode" "$d"
      log_echo "q1 arm $mode complete: $d"
    fi
  done
  # 2 vanilla-no-op ablation arms (--no-random-start; ignored by mode=random).
  for mode in ifgsm fgsm; do
    d=$(find_arm_dir "$mode" false)
    if [ -n "$d" ]; then
      log_echo "q1 arm $mode (--no-random-start) already complete: $d (skip)"
    else
      cmd="python -m famail_temporal.baselines.run_stifgsm_baseline --edit-dir $SHZ_FILTERED --mode $mode --seed 0 --device auto --score-fidelity --no-random-start"
      ledger_start "Q1-arm-$mode-nors" "$STATE_DIR/ledger/Q1-arm-$mode-nors" "PRIMARY / shenzhen / $mode --no-random-start ablation" "$cmd"
      run_logged "$cmd"
      d=$(find_arm_dir "$mode" false)
      if [ -z "$d" ]; then log_echo "!!! q1 arm $mode --no-random-start produced no complete arm dir — see $LOG"; exit 1; fi
      ledger_finish "Q1-arm-$mode-nors" "$d"
      log_echo "q1 arm $mode --no-random-start complete: $d"
    fi
  done

  local arm_ifgsm arm_fgsm arm_random arm_ifgsm_nors arm_fgsm_nors
  arm_ifgsm=$(find_arm_dir ifgsm true)
  arm_fgsm=$(find_arm_dir fgsm true)
  arm_random=$(find_arm_dir random true)
  arm_ifgsm_nors=$(find_arm_dir ifgsm false)
  arm_fgsm_nors=$(find_arm_dir fgsm false)

  # Per-arm external fairness + tier-2 supply recount (run-book step 2; the
  # run-book glob covers ablation arms too, so all 5 arms get both).
  local arm base ext_out
  for arm in "$arm_ifgsm" "$arm_fgsm" "$arm_random" "$arm_ifgsm_nors" "$arm_fgsm_nors"; do
    base=$(basename "$arm")
    ext_out=$EXT_RESULTS/baseline-$base
    if [ -f "$ext_out/external_fairness.json" ]; then
      log_echo "q1 external fairness already done for $base (skip)"
    else
      cmd="python -m famail_temporal.baselines.run_external_fairness --edit-dir $arm --dataset baseline-$base"
      ledger_run "Q1-ext-$base" "$ext_out" "PRIMARY / shenzhen" "$cmd"
    fi
    if [ -f "$arm/supply_recount.json" ]; then
      log_echo "q1 supply recount already done for $base (skip)"
    else
      cmd="python -m famail_temporal.analysis.supply_recount --edit-dir $arm --city shenzhen --persist-grids"
      ledger_run "Q1-recount-$base" "$arm" "PRIMARY / shenzhen" "$cmd"
    fi
  done

  # 6-row comparison table: 3 random-start arms + the 3 targeted oversampling
  # d-dose s0 arms + the two hand-authored stubs (plan Task 11 Step 2).
  cmd="python -m famail_temporal.baselines.assemble_baseline_table --arm-dirs $arm_ifgsm $arm_fgsm $arm_random $OVR_D2500 $OVR_D5000 $OVR_D10000 --famail-json $STUB_FAMAIL --raw-json $STUB_RAW --out famail_temporal/baselines/baseline_table"
  ledger_run "Q1-table" "famail_temporal/baselines/baseline_table" "PRIMARY / shenzhen" "$cmd"
  [ -f "$Q1_MARKER" ] || { log_echo "!!! q1 table missing after assemble — see $LOG"; exit 1; }
  log_echo "q1 COMPLETE: $Q1_MARKER"
}

stage_q2() {
  require_primary
  if [ -f "$Q2_MARKER" ]; then log_echo "q2 already DONE ($Q2_MARKER) — skip"; return 0; fi
  local cmd="FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir $SF_FILTERED --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir $RESULTS/weighted_bc_sweep/supply_lift_v1_sf12_filtered_6seed"
  ledger_run "Q2" "$RESULTS/weighted_bc_sweep/supply_lift_v1_sf12_filtered_6seed" "PRIMARY / sf12" "$cmd"
  [ -f "$Q2_MARKER" ] || { log_echo "!!! q2 marker missing after run — see $LOG"; exit 1; }
  log_echo "q2 COMPLETE: $Q2_MARKER"
}

stage_q3() {
  require_primary
  if [ -f "$Q3_MARKER" ]; then log_echo "q3 already DONE ($Q3_MARKER) — skip"; return 0; fi
  local cmd="python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir $SHZ_FILTERED --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir $RESULTS/level1_table_v2/supply_lift_shz_5seed"
  ledger_run "Q3" "$RESULTS/level1_table_v2/supply_lift_shz_5seed" "PRIMARY / shenzhen" "$cmd"
  [ -f "$Q3_MARKER" ] || { log_echo "!!! q3 marker missing after run — see $LOG"; exit 1; }
  log_echo "q3 COMPLETE: $Q3_MARKER"
}

stage_q4() {
  require_primary
  if [ -f "$Q4_MARKER" ]; then log_echo "q4 already DONE ($Q4_MARKER) — skip"; return 0; fi
  local cmd="FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir $SF_FILTERED --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir $RESULTS/level1_table_v2/supply_lift_sf12_5seed"
  ledger_run "Q4" "$RESULTS/level1_table_v2/supply_lift_sf12_5seed" "PRIMARY / sf12" "$cmd"
  [ -f "$Q4_MARKER" ] || { log_echo "!!! q4 marker missing after run — see $LOG"; exit 1; }
  log_echo "q4 COMPLETE: $Q4_MARKER"
}

stage_q5() {
  require_primary
  local cmd
  if [ -f "$Q5_MARKER_SHZ" ]; then
    log_echo "q5 SZ variance already DONE ($Q5_MARKER_SHZ) — skip"
  else
    cmd="python -m famail_temporal.baselines.run_variance_suite --edit-dir $SHZ_FILTERED --seeds 0,1,2,3,4 --out-dir $RESULTS/variance_suite/supply_lift_shz_5seed"
    ledger_run "Q5-shz" "$RESULTS/variance_suite/supply_lift_shz_5seed" "PRIMARY / shenzhen" "$cmd"
    [ -f "$Q5_MARKER_SHZ" ] || { log_echo "!!! q5 SZ marker missing after run — see $LOG"; exit 1; }
  fi
  if [ -f "$Q5_MARKER_SF" ]; then
    log_echo "q5 SF variance already DONE ($Q5_MARKER_SF) — skip"
  else
    cmd="FAMAIL_CITY=sf12 python -m famail_temporal.baselines.run_variance_suite --edit-dir $SF_FILTERED --seeds 0,1,2,3,4 --out-dir $RESULTS/variance_suite/supply_lift_sf12_5seed"
    ledger_run "Q5-sf12" "$RESULTS/variance_suite/supply_lift_sf12_5seed" "PRIMARY / sf12" "$cmd"
    [ -f "$Q5_MARKER_SF" ] || { log_echo "!!! q5 SF marker missing after run — see $LOG"; exit 1; }
  fi
  log_echo "q5 COMPLETE: $Q5_MARKER_SHZ + $Q5_MARKER_SF"
}

run_editor_stage() {  # $1 qid-prefix, $2 runner --name, $3 config note — edit run + infeasible-trim filter
  local qid="$1" name="$2" note="$3" run_dir cmd
  run_dir=$(find_editor_run "$name")
  if [ -n "$run_dir" ]; then
    log_echo "$STAGE edit run already complete: $run_dir (skip)"
  else
    cmd="python -m famail_temporal.evaluation.runner -k 10000 --name $name --device auto --override ALPHA_SPATIAL=0.2 --override ALPHA_CAUSAL=0.7 --override ALPHA_FIDELITY=0.1"
    ledger_start "$qid-edit" "$STATE_DIR/ledger/$qid-edit" "$note" "$cmd"
    run_logged "$cmd"
    run_dir=$(find_editor_run "$name")
    if [ -z "$run_dir" ]; then log_echo "!!! $STAGE edit run produced no completed dir — see $LOG"; exit 1; fi
    ledger_finish "$qid-edit" "$run_dir"
    log_echo "$STAGE edit run complete: $run_dir"
  fi
  if [ -f "${run_dir}_filtered/metrics.json" ]; then
    log_echo "$STAGE already filtered: ${run_dir}_filtered (skip)"
  else
    cmd="python -m famail_temporal.analysis.filter_infeasible_trims --edit-dir $run_dir"
    ledger_start "$qid-filter" "$STATE_DIR/ledger/$qid-filter" "$note" "$cmd"
    run_logged "$cmd"
    [ -f "${run_dir}_filtered/metrics.json" ] || { log_echo "!!! $STAGE filter failed — see $LOG"; exit 1; }
    ledger_finish "$qid-filter" "${run_dir}_filtered"
  fi
  log_echo "$STAGE COMPLETE: ${run_dir}_filtered/metrics.json"
}

stage_q6a() { require_hgc;   run_editor_stage Q6a supply_lift_v1_shz_hgc   "HGC / shenzhen"; }
stage_q6b() { require_4feat; run_editor_stage Q6b supply_lift_v1_shz_4feat "4FEAT / shenzhen"; }

stage_q7() {
  # Three parts, each guarded by ITS OWN config (Task 15 Step 4: externals run
  # under the matching config per set; the Pareto requires PRIMARY). Run q7
  # once per config state; done parts skip, mismatched parts block (exit 1).
  local blocked=0 run_dir f cmd

  if [ -f "$Q7_MARKER_HGC" ]; then
    log_echo "q7[hgc-ext] already DONE ($Q7_MARKER_HGC) — skip"
  else
    run_dir=$(find_editor_run supply_lift_v1_shz_hgc)
    if [ -z "$run_dir" ] || [ ! -f "${run_dir}_filtered/metrics.json" ]; then
      log_echo "q7[hgc-ext] BLOCKED: q6a's filtered dir not found — run stage q6a first"; blocked=1
    elif ! demo_has GDPperCapita; then
      log_echo "q7[hgc-ext] BLOCKED: config is $(config_label), needs HGC — flip config (commit 'paper-campaign: config -> housing-gdp-comp'), re-run q7"; blocked=1
    else
      f=${run_dir}_filtered
      cmd="python -m famail_temporal.baselines.run_external_fairness --edit-dir $f --dataset shenzhen-hgc-supplylift --bootstrap 1000 --seed 0 --delta-supply $f/delta_supply_3d.npz"
      ledger_run "Q7-ext-hgc" "$EXT_RESULTS/shenzhen-hgc-supplylift" "HGC / shenzhen" "$cmd"
    fi
  fi

  if [ -f "$Q7_MARKER_4FEAT" ]; then
    log_echo "q7[4feat-ext] already DONE ($Q7_MARKER_4FEAT) — skip"
  else
    run_dir=$(find_editor_run supply_lift_v1_shz_4feat)
    if [ -z "$run_dir" ] || [ ! -f "${run_dir}_filtered/metrics.json" ]; then
      log_echo "q7[4feat-ext] BLOCKED: q6b's filtered dir not found — run stage q6b first"; blocked=1
    elif ! demo_has LogPopDensity; then
      log_echo "q7[4feat-ext] BLOCKED: config is $(config_label), needs 4FEAT — flip config (commit 'paper-campaign: config -> 4feat'), re-run q7"; blocked=1
    else
      f=${run_dir}_filtered
      cmd="python -m famail_temporal.baselines.run_external_fairness --edit-dir $f --dataset shenzhen-4feat-supplylift --bootstrap 1000 --seed 0 --delta-supply $f/delta_supply_3d.npz"
      ledger_run "Q7-ext-4feat" "$EXT_RESULTS/shenzhen-4feat-supplylift" "4FEAT / shenzhen" "$cmd"
    fi
  fi

  if [ -f "$Q7_MARKER_PARETO" ]; then
    log_echo "q7[pareto] already DONE ($Q7_MARKER_PARETO) — skip"
  else
    if config_is_primary; then
      cmd="python -m famail_temporal.baselines.run_data_pareto --edit-from-dir $SHZ_FILTERED --out-dir $RESULTS/analysis/pareto_supplylift"
      ledger_run "Q7-pareto" "$RESULTS/analysis/pareto_supplylift" "PRIMARY / shenzhen" "$cmd"
    else
      log_echo "q7[pareto] BLOCKED: config is $(config_label), needs PRIMARY — flip config back (commit 'paper-campaign: config -> PRIMARY (restore)'), re-run q7"; blocked=1
    fi
  fi

  if [ "$blocked" -ne 0 ]; then
    log_echo "q7 incomplete: blocked parts above need a config flip and/or q6a/q6b — re-run q7 after."
    exit 1
  fi
  log_echo "q7 COMPLETE: $Q7_MARKER_HGC + $Q7_MARKER_4FEAT + $Q7_MARKER_PARETO"
}

run_downstream_block() {  # $1 qid-prefix, $2 editor name, $3 config note, $4 l1-marker, $5 wbc-marker, $6 var-marker
  local qid="$1" name="$2" note="$3" m_l1="$4" m_wbc="$5" m_var="$6" run_dir f cmd
  run_dir=$(find_editor_run "$name")
  if [ -z "$run_dir" ] || [ ! -f "${run_dir}_filtered/metrics.json" ]; then
    echo "ERROR: $STAGE needs the $name filtered dir — run its q6 stage first." >&2
    exit 1
  fi
  f=${run_dir}_filtered
  if [ -f "$m_l1" ]; then
    log_echo "$STAGE L1v2 already DONE ($m_l1) — skip"
  else
    cmd="python -m famail_temporal.baselines.run_level1_table_v2 --edit-dir $f --seeds 0,1,2,3,4 --mle-epochs 20 --adv-epochs 3 --gan-loss wgan-gp --n-critic 5 --device auto --out-dir $(dirname "$m_l1")"
    ledger_run "$qid-l1v2" "$(dirname "$m_l1")" "$note" "$cmd"
    [ -f "$m_l1" ] || { log_echo "!!! $STAGE L1v2 marker missing — see $LOG"; exit 1; }
  fi
  if [ -f "$m_wbc" ]; then
    log_echo "$STAGE weighted-BC already DONE ($m_wbc) — skip"
  else
    cmd="python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir $f --seeds 0,1,2,3,4,5 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir $(dirname "$m_wbc")"
    ledger_run "$qid-wbc" "$(dirname "$m_wbc")" "$note" "$cmd"
    [ -f "$m_wbc" ] || { log_echo "!!! $STAGE weighted-BC marker missing — see $LOG"; exit 1; }
  fi
  if [ -f "$m_var" ]; then
    log_echo "$STAGE variance already DONE ($m_var) — skip"
  else
    cmd="python -m famail_temporal.baselines.run_variance_suite --edit-dir $f --seeds 0,1,2,3,4 --out-dir $(dirname "$m_var")"
    ledger_run "$qid-var" "$(dirname "$m_var")" "$note" "$cmd"
    [ -f "$m_var" ] || { log_echo "!!! $STAGE variance marker missing — see $LOG"; exit 1; }
  fi
  log_echo "$STAGE COMPLETE: $m_l1 + $m_wbc + $m_var"
}

stage_q8a() {
  require_hgc
  run_downstream_block Q8a supply_lift_v1_shz_hgc "HGC / shenzhen" \
    "$Q8A_MARKER_L1" "$Q8A_MARKER_WBC" "$Q8A_MARKER_VAR"
}
stage_q8b() {
  require_4feat
  run_downstream_block Q8b supply_lift_v1_shz_4feat "4FEAT / shenzhen" \
    "$Q8B_MARKER_L1" "$Q8B_MARKER_WBC" "$Q8B_MARKER_VAR"
}

# --------------------------------------------------------------- dispatch ----

usage() {
  echo "Usage: bash $0 {q1|q2|q3|q4|q5|q6a|q6b|q7|q8a|q8b|--status}" >&2
  echo "Long stages: nohup setsid bash $0 <stage> >> $STATE_DIR/<stage>.driver.log 2>&1 &" >&2
}

case "${1:-}" in
  --status)
    print_status
    exit 0
    ;;
  q1|q2|q3|q4|q5|q6a|q6b|q7|q8a|q8b)
    STAGE="$1"
    LOG="$STATE_DIR/$1.log"
    trap 'echo "!!! [$(date -Is)] driver aborted in stage ${STAGE:-?}; see ${LOG:-$STATE_DIR}" >&2' ERR
    require_ledger
    echo "=== [$(date -Is)] campaign stage $STAGE start (pid $$; config=$(config_label)) ===" | tee -a "$LOG"
    "stage_$1"
    echo "=== [$(date -Is)] campaign stage $STAGE end ===" | tee -a "$LOG"
    ;;
  *)
    usage
    exit 1
    ;;
esac
