#!/usr/bin/env bash
# Overnight config-flip chain (Robert-authorized 2026-07-13 ~01:00 PT: "you've
# got the branch to yourself and you can safely queue the config flip runs").
#
#   wait-for-R5 -> HGC: q6a + q7[hgc-ext] -> 4FEAT: q6b + q7[4feat-ext]
#   -> q8b (4FEAT downstream, ~1 day) -> HGC: q8a (~1 day) -> PRIMARY restore
#
# Every stage is driver.sh (ledger rows + skip-if-done), so a dead chain can be
# relaunched verbatim. Config flips are idempotent (write the target list
# outright) and committed per the campaign protocol. q7 is |-tolerated: its
# not-yet-runnable parts log BLOCKED and exit 1 by design; marker files carry
# the truth and later invocations pick the parts up.
# Launch: cd famail_temporal/results/experiments_campaign && nohup setsid bash q_chain.sh >> q_chain.log 2>&1 &
set -u
cd /home/robert/FAMAIL
DRIVER=famail_temporal/results/experiments_campaign/driver.sh
R5LOG=famail_temporal/results/experiments_campaign/r5.log

say() { echo "[q_chain $(date +%m-%d\ %H:%M:%S)] $*"; }

flip_config() {  # $1 = PRIMARY | HGC | 4FEAT ; $2 = commit message
  python3 - "$1" <<'PY'
import re, sys
from pathlib import Path
target = sys.argv[1]
lists = {
    "PRIMARY": '"AvgHousingPricePerSqM",\n    "CompPerCapita",\n    "MigrantRatio",',
    "HGC": '"AvgHousingPricePerSqM",\n    "GDPperCapita",\n    "CompPerCapita",',
    "4FEAT": '"AvgHousingPricePerSqM",\n    "CompPerCapita",\n    "MigrantRatio",\n    "LogPopDensity",',
}
p = Path("famail_temporal/config.py")
t = p.read_text()
new, n = re.subn(
    r"DEMOGRAPHIC_FEATURES: List\[str\] = \[\n(?:    \"[A-Za-z]+\",\n)+\]",
    f"DEMOGRAPHIC_FEATURES: List[str] = [\n    {lists[target]}\n]",
    t, count=1)
assert n == 1, "DEMOGRAPHIC_FEATURES block not found/matched"
p.write_text(new)
PY
  if git diff --quiet famail_temporal/config.py; then
    say "config already $1 (no commit)"
  else
    git add famail_temporal/config.py && git commit -q -m "$2" \
      && say "config -> $1 (committed: $2)"
  fi
}

stage() {  # $1 = driver stage; abort chain on failure
  say "STAGE $1 start"
  if bash "$DRIVER" "$1"; then say "STAGE $1 done"; else
    say "CHAIN ABORT: stage $1 failed (rc=$?) — fix and relaunch q_chain.sh (skip-if-done resumes)"
    exit 1
  fi
}

# ---- 0. wait for R5 to reach a terminal state (proceed on either outcome) ----
say "waiting for R5 (rollout eval) to finish..."
until grep -q 'R5 DONE\|R5 FAILED' "$R5LOG" 2>/dev/null || ! pgrep -f option_a_rollout_eval >/dev/null; do
  sleep 60
done
grep -q 'R5 FAILED' "$R5LOG" 2>/dev/null && say "NOTE: R5 FAILED — proceeding with flips; triage R5 in the morning"
say "R5 terminal — beginning config-flip sequence"

# ---- 1. HGC: edit run + externals ----
flip_config HGC "paper-campaign: config -> housing-gdp-comp"
stage q6a
bash "$DRIVER" q7 || say "q7 partial (expected: non-HGC parts blocked; markers carry truth)"

# ---- 2. 4FEAT: edit run + externals ----
flip_config 4FEAT "paper-campaign: config -> 4feat"
stage q6b
bash "$DRIVER" q7 || say "q7 partial (expected: any remaining blocked parts; markers carry truth)"

# ---- 3. downstream suites (long): 4FEAT first (already flipped), then HGC ----
stage q8b
flip_config HGC "paper-campaign: config -> housing-gdp-comp (q8a)"
stage q8a

# ---- 4. restore ----
flip_config PRIMARY "paper-campaign: config -> PRIMARY (restore)"
say "CHAIN COMPLETE — config restored to PRIMARY"
