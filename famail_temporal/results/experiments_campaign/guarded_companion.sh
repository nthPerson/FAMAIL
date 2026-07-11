#!/usr/bin/env bash
# Memory-guarded COMPANION launcher (concurrency decision 2026-07-11, Robert-approved).
# Runs ONE companion job alongside the primary GPU chain under a host-memory guard:
#   - refuses to start unless MemAvailable >= START_FLOOR (20 GB)
#   - polls MemAvailable every 30 s; if < KILL_FLOOR (10 GB), kills the COMPANION
#     (never the primary) and logs the event to the run ledger
#   - lockfile enforces at most one companion at a time
# Usage: nohup setsid bash guarded_companion.sh <label> '<command>' >> guarded_<label>.log 2>&1 &
set -uo pipefail
cd /home/robert/FAMAIL
LABEL="${1:?usage: guarded_companion.sh <label> '<command>'}"
CMD="${2:?usage: guarded_companion.sh <label> '<command>'}"
DIR=famail_temporal/results/experiments_campaign
LOCK="$DIR/.companion.lock"
LEDGER=famail_temporal/results/EXPERIMENTS_RUN_LEDGER.md
START_FLOOR_KB=$((20 * 1024 * 1024))
KILL_FLOOR_KB=$((10 * 1024 * 1024))

mem_kb() { awk '/MemAvailable/ {print $2}' /proc/meminfo; }
note() { echo "[guard $(date -u +%H:%M:%S)] $*"; }

# one companion at a time (stale locks cleared)
if [ -f "$LOCK" ]; then
  oldpid=$(cat "$LOCK" 2>/dev/null || echo 0)
  if kill -0 "$oldpid" 2>/dev/null; then
    note "REFUSED: companion already running (pid $oldpid)"; exit 1
  fi
  note "clearing stale lock (pid $oldpid gone)"; rm -f "$LOCK"
fi

avail=$(mem_kb)
if [ "$avail" -lt "$START_FLOOR_KB" ]; then
  note "REFUSED: MemAvailable $((avail/1024/1024)) GB < 20 GB start floor"; exit 1
fi

note "START companion '$LABEL' (MemAvailable $((avail/1024/1024)) GB): $CMD"
setsid bash -c "$CMD" >> "$DIR/${LABEL}.companion.log" 2>&1 &
CPID=$!
echo "$CPID" > "$LOCK"

while kill -0 "$CPID" 2>/dev/null; do
  sleep 30
  avail=$(mem_kb)
  if [ "$avail" -lt "$KILL_FLOOR_KB" ]; then
    note "MEM GUARD TRIPPED: MemAvailable $((avail/1024/1024)) GB < 10 GB — killing companion '$LABEL' (primary protected)"
    kill -TERM -- -"$CPID" 2>/dev/null; sleep 10; kill -KILL -- -"$CPID" 2>/dev/null
    printf '\n> **MEM-GUARD EVENT (%s UTC):** companion `%s` killed at MemAvailable %s GB (kill floor 10 GB); primary chain untouched. Relaunch via guarded_companion.sh when memory recovers.\n' \
      "$(date -u +%FT%T)" "$LABEL" "$((avail/1024/1024))" >> "$LEDGER"
    rm -f "$LOCK"; exit 2
  fi
done
wait "$CPID" 2>/dev/null; rc=$?
note "companion '$LABEL' exited rc=$rc (MemAvailable $(( $(mem_kb)/1024/1024 )) GB)"
rm -f "$LOCK"
exit "$rc"
