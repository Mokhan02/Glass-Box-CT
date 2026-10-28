#!/bin/bash
# W5 alert poller (runs ON the instance, in its own tmux session): every 20 s it scans the new lines of the driver log and the W5 training logs and appends one line per event to ALERTS.log
# (timestamped, append-only), touches HEARTBEAT, and raises session-gone / stale-log events. A short remote poll of ALERTS.log (plus the HEARTBEAT age) is the alert channel; nothing depends on the researcher's or the assistant's session staying open.
# Usage: bash scripts/w5_alert.sh [dir, default /lambda/nfs/CTGate/w5]. Overrides: W5_TAG (training tag, default B50; markers W5_<TAG>_TRAINING_STARTED, TRAIN_<TAG>_DONE/FAILED, log train_<TAG>.log), W5_SESS_DRV, W5_SESS_TR (tmux session names),
# W5_REAL_LOG_DIR (the nnU-Net results fold directory holding training_log_*.txt: when set, the stale check uses the newest REAL training log, since the stdout log does not grow per epoch), W5_POST_DIR (the arm's post-pipeline directory: after TRAIN_<TAG>_DONE and until TRAIN_<TAG>POST_DONE,
# stale = no new file or line under <dir>/pred or post.log), W5_INTERVAL (seconds), W5_STALE (seconds, default 300 when W5_REAL_LOG_DIR is set, else 1200).
W=${1:-/lambda/nfs/CTGate/w5}; TAG=${W5_TAG:-B50}; SD=${W5_SESS_DRV:-w5drv}; ST=${W5_SESS_TR:-w5$TAG}; IV=${W5_INTERVAL:-20}; RL=${W5_REAL_LOG_DIR:-}; PD=${W5_POST_DIR:-}
if [ -n "$RL" ]; then STALE=${W5_STALE:-300}; else STALE=${W5_STALE:-1200}; fi
TL=$W/train_$TAG.log
mkdir -p $W; A=$W/ALERTS.log; touch $A
emit() { echo "$(date -u +%FT%TZ) $1" >> $A; }
PAT='HOLD|DRIVER_PROC_EXIT|SMOKE_PASS|W5_[A-Za-z0-9]*_TRAINING_STARTED|TRAIN_[A-Za-z0-9]*_(DONE|FAILED)|ATTEMPT [0-9]+ exit|Traceback|OutOfMemory|out of memory|FAILED'
declare -A seen; for f in $W/driver.log $W/train_*.log; do [ -f "$f" ] && seen[$f]=$(wc -l < $f); done      # start from the current end: history is not replayed
emit "ALERT_POLLER_START dir=$W"
dgone=0; tgone=0; stale=0; pstale=0
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
  if [ -f $W/driver.log ] && grep -q "W5_${TAG}_TRAINING_STARTED" $W/driver.log && ! grep -q -E "TRAIN_${TAG}_(DONE|FAILED)" $TL 2>/dev/null; then
    if ! tmux has-session -t $ST 2>/dev/null; then [ $tgone = 0 ] && emit "TRAINING_SESSION_GONE (tmux $ST missing, no DONE/FAILED)"; tgone=1; else tgone=0
      if [ -n "$RL" ]; then rl=$(ls -t $RL/training_log_*.txt 2>/dev/null | head -1); mt=$(stat -c %Y ${rl:-$TL} 2>/dev/null || date +%s); what="real training log"; else mt=$(stat -c %Y $TL 2>/dev/null || date +%s); what="stdout log"; fi
      age=$(( $(date +%s) - mt )); if [ $age -gt $STALE ]; then [ $stale = 0 ] && emit "TRAIN_LOG_STALE (${age}s without a new line in the $what)"; stale=1; else stale=0; fi
    fi
  elif [ -n "$PD" ] && grep -q -E "TRAIN_${TAG}_DONE" $TL 2>/dev/null && ! grep -q -E "TRAIN_${TAG}POST_DONE|W5_POST_HOLD" $TL 2>/dev/null; then
    newest=$(find $PD/pred $PD/post.log -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1); age=$(( $(date +%s) - ${newest:-$(date +%s)} ))
    if [ $age -gt $STALE ]; then [ $pstale = 0 ] && emit "POST_PIPELINE_STALE (${age}s without a new file under $PD)"; pstale=1; else pstale=0; fi
  fi
  sleep $IV
done
