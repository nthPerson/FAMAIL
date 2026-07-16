#!/usr/bin/env bash
# Fairness-baseline GPU chain (plan docs/superpowers/plans/2026-07-16-fairness-baseline.md,
# Tasks 5-7). Waits for the C1 dose-extension to finish (b_chain marker), then:
#   FB-GATE    — full 10-arm seed-0 regression replay; ABORTS THE CHAIN on any mismatch
#                vs the committed sweep (default-off invariant).
#   FB-PENALTY-PILOT — lambda grid {0.1,1,10} at seed 0; scripted selection per the plan
#                criterion (fid_a within 0.02 of raw; n_empty no worse than raw+1);
#                one halved-grid retry; if none stable -> FB-PENALTY skipped (reweigh-only
#                fallback, spec section 7).
#   FB-REWEIGH — 6-seed suite: raw + edited + edited_w30 + fair_reweigh.
#   FB-PENALTY — 6-seed suite: raw + edited + 3 chosen lambda arms.
# Idempotent via markers; relaunch verbatim to resume.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash fb_chain.sh >> fb_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
M=famail_temporal/results/experiments_campaign/markers
mkdir -p "$M"
S10=famail_temporal/results/2026-07-10T02-06-37_alpha_sweep_s10_c80_f10_filtered
COMMITTED=famail_temporal/results/weighted_bc_sweep/alpha_sweep_s10_c80_f10_filtered_6seed/sweep.json
WBC="python -m famail_temporal.baselines.run_weighted_bc_smoke --edit-dir $S10 --device auto"

echo "[fb $(date +%H:%M:%S)] waiting for C1 (marker $M/c1_wbc.done, timeout 15h)"
n=0
until [ -f "$M/c1_wbc.done" ] || [ $n -ge 450 ]; do sleep 120; n=$((n+1)); done
if [ ! -f "$M/c1_wbc.done" ]; then echo "[fb] TIMEOUT waiting for C1 — aborting"; exit 1; fi
echo "[fb $(date +%H:%M:%S)] C1 done — starting FB chain"

run_stage() { # $1 marker $2 qid $3 note $4 outdir $5... cmd
  local marker=$1 qid=$2 note=$3 outdir=$4; shift 4
  if [ -f "$M/$marker" ]; then echo "[fb $(date +%H:%M:%S)] $qid SKIP (marker)"; return 0; fi
  echo "[fb $(date +%H:%M:%S)] $qid START: $*"
  python -m famail_temporal.analysis.run_ledger start --queue-id "$qid" --cmd "$*" \
    --artifact-dir "$outdir" --config-note "$note"
  if "$@"; then
    python -m famail_temporal.analysis.run_ledger finish --queue-id "$qid" --artifact-dir "$outdir"
    touch "$M/$marker"
    echo "[fb $(date +%H:%M:%S)] $qid DONE: $outdir"
    return 0
  else
    echo "[fb $(date +%H:%M:%S)] $qid FAILED rc=$? — chain HALTS (patch ledger row on pickup)"
    exit 1
  fi
}

# ---- Task 5: regression gate (full 10-arm replay, seed 0) ----
GATE_DIR=famail_temporal/results/weighted_bc_sweep/fb_regression_gate_seed0
run_stage fb_gate.done FB-GATE \
  "PRIMARY / shenzhen; FB Task 5 regression gate: full 10-arm seed-0 replay under the Task-1..4 code; MUST equal committed sweep values[0] exactly" \
  "$GATE_DIR" \
  $WBC --seeds 0 --weights 10,20,30 --placebo 10,30 --most-fair 10,20,30 --out-dir "$GATE_DIR"

if [ ! -f "$M/fb_gate_verified.done" ]; then
  python3 - "$GATE_DIR/sweep.json" "$COMMITTED" <<'EOF'
import json, sys
gate = json.load(open(sys.argv[1]))["per_arm"]
ref = json.load(open(sys.argv[2]))["per_arm"]
bad = []
for arm, met in ref.items():
    for metric in ("f_causal", "f_spatial", "fidelity_a", "fidelity_b"):
        r = met[metric]["values"][0]
        g = gate[arm][metric]["values"][0]
        if r != g:
            bad.append(f"{arm}.{metric}: gate {g} != committed {r}")
if bad:
    print("[fb] FB-GATE REGRESSION MISMATCH (" + str(len(bad)) + "):")
    [print("   " + b) for b in bad]
    sys.exit(1)
print("[fb] FB-GATE VERIFIED: all 10 arms x 4 metrics identical to committed seed-0 values")
EOF
  rc=$?
  if [ $rc -ne 0 ]; then echo "[fb] GATE FAILED — chain HALTS, suites NOT run"; exit 1; fi
  touch "$M/fb_gate_verified.done"
fi

# ---- Task 6: lambda pilot + scripted selection ----
pilot() { # $1 grid-csv $2 outdir
  run_stage "fb_pilot_$(echo "$1" | tr ',.' '__').done" FB-PENALTY-PILOT \
    "PRIMARY / shenzhen; FB Task 6 lambda pilot grid {$1} seed 0 (criterion: fid_a within 0.02 of raw; n_empty <= raw+1)" \
    "$2" \
    $WBC --seeds 0 --weights "" --fairness-penalty "$1" --out-dir "$2"
}
select_lambdas() { # $1 pilot-outdir ; writes markers/fb_lambdas.txt or returns 1
  python3 - "$1/sweep.json" "$M/fb_lambdas.txt" <<'EOF'
import json, math, sys
pa = json.load(open(sys.argv[1]))["per_arm"]
raw_fa = pa["raw"]["fidelity_a"]["values"][0]
raw_ne = pa["raw"]["n_empty"][0] if isinstance(pa["raw"]["n_empty"], list) else pa["raw"]["n_empty"]
passing = []
for arm, met in pa.items():
    if not arm.startswith("fair_penalty_l"):
        continue
    lam = float(arm.split("fair_penalty_l")[1])
    fa = met["fidelity_a"]["values"][0]
    ne = met["n_empty"][0] if isinstance(met["n_empty"], list) else met["n_empty"]
    ok = abs(fa - raw_fa) <= 0.02 and ne <= raw_ne + 1
    print(f"[fb-pilot] lambda={lam}: fid_a={fa} (raw {raw_fa}), n_empty={ne} (raw {raw_ne}) -> {'PASS' if ok else 'FAIL'}")
    if ok:
        passing.append(lam)
if not passing:
    sys.exit(1)
hi = max(passing)
lo = hi / 10.0
mid = math.sqrt(lo * hi)
open(sys.argv[2], "w").write(f"{lo:g},{mid:g},{hi:g}\n")
print(f"[fb-pilot] SELECTED lambdas lo={lo:g} mid={mid:g} hi={hi:g}")
EOF
}

PILOT1=famail_temporal/results/weighted_bc_sweep/fairness_penalty_pilot
PILOT2=famail_temporal/results/weighted_bc_sweep/fairness_penalty_pilot_halved
if [ ! -f "$M/fb_lambdas.txt" ] && [ ! -f "$M/fb_penalty_skipped" ]; then
  pilot "0.1,1,10" "$PILOT1"
  if ! select_lambdas "$PILOT1"; then
    echo "[fb] pilot grid 1 unstable — halved retry"
    pilot "0.05,0.5,5" "$PILOT2"
    if ! select_lambdas "$PILOT2"; then
      echo "[fb] BOTH pilot grids unstable — FB-PENALTY SKIPPED (reweigh-only fallback, spec s7)"
      touch "$M/fb_penalty_skipped"
    fi
  fi
fi

# ---- Task 7a: FB-REWEIGH 6-seed suite ----
RW_DIR=famail_temporal/results/weighted_bc_sweep/fairness_baseline_6seed
run_stage fb_reweigh.done FB-REWEIGH \
  "PRIMARY / shenzhen; FB Task 7a: reweigh suite (raw+edited+edited_w30+fair_reweigh, 6 seeds); regression gate PASSED this chain" \
  "$RW_DIR" \
  $WBC --seeds 0,1,2,3,4,5 --weights 30 --fairness-reweigh --out-dir "$RW_DIR"

# ---- Task 7b: FB-PENALTY 6-seed suite (unless skipped) ----
if [ -f "$M/fb_penalty_skipped" ]; then
  echo "[fb $(date +%H:%M:%S)] FB-PENALTY skipped (pilot fallback). FB CHAIN COMPLETE (reweigh-only)."
  exit 0
fi
LAMBDAS=$(cat "$M/fb_lambdas.txt")
PEN_DIR=famail_temporal/results/weighted_bc_sweep/fairness_penalty_6seed
run_stage fb_penalty.done FB-PENALTY \
  "PRIMARY / shenzhen; FB Task 7b: penalty suite (raw+edited+fair_penalty at lambdas $LAMBDAS, 6 seeds)" \
  "$PEN_DIR" \
  $WBC --seeds 0,1,2,3,4,5 --weights "" --fairness-penalty "$LAMBDAS" --out-dir "$PEN_DIR"

echo "[fb $(date +%H:%M:%S)] FB CHAIN COMPLETE"
