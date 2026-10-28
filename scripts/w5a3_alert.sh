#!/bin/bash
# W5 Amendment 3 alert poller (runs ON the instance in its own tmux session, w5a3alert): every 20 s it scans new lines of the driver log, the step logs and ALERTS, and appends one line per event to
# $W/ALERTS.log (timestamped, append-only), touches $W/HEARTBEAT, and raises: DRIVER_SESSION_GONE (tmux w5a3drv missing without an exit marker), STEP_STALE (no new line in any step log or output for
# W5A3_STALE seconds while a step runs; default 2700, the longest D0 per-case time was 1357 s), CUDA/traceback/out-of-memory lines, and GPU_IDLE (once, when the driver has ended (done, hold or exit) and no compute process remains).
# Usage: bash scripts/w5a3_alert.sh [dir, default /lambda/nfs/CTGate/w5a3]
W=${1:-/lambda/nfs/CTGate/w5a3}; IV=${W5A3_INTERVAL:-20}; STALE=${W5A3_STALE:-2700}; SD=${W5A3_SESS:-w5a3drv}
mkdir -p $W; A=$W/ALERTS.log; touch $A
emit() { echo "$(date -u +%FT%TZ) $1" >> $A; }
PAT='A3_DRIVER_HOLD|DRIVER_PROC_EXIT|PREFLIGHT_PASS|HOOK_CHECK_PASS|M3_SMOKE_PASS|M3_FULL_START|M3_DONE|M3_CONTRAST_DONE|EXTRACT_START|EXTRACT_DONE|ERASERS_FITTED|M2_SMOKE_PASS|M2_FULL_START|M2_DONE|A3_ALL_DONE|TIME_CHECK|Traceback|OutOfMemory|out of memory|CUDA error|FAILED'
declare -A seen; for f in $W/driver.log $W/*.log; do [ -f "$f" ] && [ "$(basename $f)" != ALERTS.log ] && seen[$f]=$(wc -l < $f); done      # start from the current end: history is not replayed
emit "ALERT_POLLER_START dir=$W"
dgone=0; stale=0; idle=0
while true; do
  touch $W/HEARTBEAT
  for f in $W/driver.log $W/*.log; do
    [ -f "$f" ] || continue
    [ "$(basename $f)" = ALERTS.log ] && continue          # never scan our own output (it would feed back on itself)
    n=$(wc -l < $f); s=${seen[$f]:-0}
    if [ "$n" -gt "$s" ]; then
      if [ "$f" = "$W/driver.log" ]; then tail -n +$((s+1)) $f | head -n $((n-s)) | while IFS= read -r l; do emit "[driver.log] ${l:0:240}"; done
      else tail -n +$((s+1)) $f | head -n $((n-s)) | grep -E 'Traceback|OutOfMemory|out of memory|CUDA error|FAILED' | head -5 | while IFS= read -r l; do emit "[$(basename $f)] ${l:0:240}"; done; fi
      seen[$f]=$n
    fi
  done
  ended=0; started=0; grep -qE 'DRIVER_PROC_EXIT|A3_DRIVER_HOLD|A3_ALL_DONE' $W/driver.log 2>/dev/null && ended=1; grep -q A3_DRIVER_START $W/driver.log 2>/dev/null && started=1
  if [ $started = 1 ] && [ $ended = 0 ] && ! tmux has-session -t $SD 2>/dev/null; then [ $dgone = 0 ] && emit "DRIVER_SESSION_GONE (tmux $SD missing, no exit marker)"; dgone=1; else dgone=0; fi
  if [ $started = 1 ] && [ $ended = 0 ]; then
    newest=$(find $W -maxdepth 3 -type f ! -name HEARTBEAT ! -name ALERTS.log -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1); age=$(( $(date +%s) - ${newest:-$(date +%s)} ))
    if [ $age -gt $STALE ]; then [ $stale = 0 ] && emit "STEP_STALE (${age}s without a new file or log line under $W)"; stale=1; else stale=0; fi
  fi
  if [ $ended = 1 ] || [ $dgone = 1 ]; then
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then [ $idle = 0 ] && emit "GPU_IDLE (driver ended: $(grep -oE 'A3_ALL_DONE|A3_DRIVER_HOLD[^|]*|DRIVER_PROC_EXIT rc=[0-9]+' $W/driver.log | tail -1))"; idle=1; fi
  else idle=0; fi
  sleep $IV
done
