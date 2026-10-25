#!/bin/bash
# W5 alert poller (runs ON the instance, in its own tmux session): every 20 s it scans the new lines of the driver log and the W5 training logs and appends one line per event to ALERTS.log
# (timestamped, append-only), touches HEARTBEAT, and raises session-gone / stale-log events. A short remote poll of ALERTS.log (plus the HEARTBEAT age) is the alert channel; nothing depends on the researcher's or the assistant's session staying open.
# Usage: bash scripts/w5_alert.sh [dir, default /lambda/nfs/CTGate/w5]. Test overrides: W5_SESS_DRV, W5_SESS_TR (tmux session names), W5_INTERVAL (seconds), W5_STALE (seconds).
W=${1:-/lambda/nfs/CTGate/w5}; SD=${W5_SESS_DRV:-w5drv}; ST=${W5_SESS_TR:-w5B50}; IV=${W5_INTERVAL:-20}; STALE=${W5_STALE:-1200}
mkdir -p $W; A=$W/ALERTS.log; touch $A
emit() { echo "$(date -u +%FT%TZ) $1" >> $A; }
PAT='HOLD|DRIVER_PROC_EXIT|SMOKE_PASS|W5_B50_TRAINING_STARTED|TRAIN_[A-Za-z0-9]*_(DONE|FAILED)|ATTEMPT [0-9]+ exit|Traceback|OutOfMemory|out of memory|FAILED'
declare -A seen; for f in $W/driver.log $W/train_*.log; do [ -f "$f" ] && seen[$f]=$(wc -l < $f); done      # start from the current end: history is not replayed
emit "ALERT_POLLER_START dir=$W"
dgone=0; tgone=0; stale=0
while true; do
  touch $W/HEARTBEAT
  for f in $W/driver.log $W/train_*.log; do
    [ -f "$f" ] || continue
    n=$(wc -l < $f); s=${seen[$f]:-0}
    if [ "$n" -gt "$s" ]; then tail -n +$((s+1)) $f | head -n $((n-s)) | grep -E "$PAT" | head -20 | while IFS= read -r l; do emit "[$(basename $f)] ${l:0:220}"; done; seen[$f]=$n; fi
  done
  if [ -f $W/driver.log ] && ! grep -q DRIVER_PROC_EXIT $W/driver.log && ! grep -q "W5_DRIVER_HOLD" <(tail -n 5 $W/driver.log) && ! tmux has-session -t $SD 2>/dev/null; then
    [ $dgone = 0 ] && emit "DRIVER_SESSION_GONE (tmux $SD missing, no exit marker)"; dgone=1
  else dgone=0; fi
  if [ -f $W/driver.log ] && grep -q W5_B50_TRAINING_STARTED $W/driver.log && ! grep -q -E "TRAIN_B50_(DONE|FAILED)" $W/train_B50.log 2>/dev/null; then
    if ! tmux has-session -t $ST 2>/dev/null; then [ $tgone = 0 ] && emit "TRAINING_SESSION_GONE (tmux $ST missing, no DONE/FAILED)"; tgone=1; else tgone=0
      age=$(( $(date +%s) - $(stat -c %Y $W/train_B50.log 2>/dev/null || date +%s) )); if [ $age -gt $STALE ]; then [ $stale = 0 ] && emit "TRAIN_LOG_STALE (${age}s without a new line)"; stale=1; else stale=0; fi
    fi
  fi
  sleep $IV
done
