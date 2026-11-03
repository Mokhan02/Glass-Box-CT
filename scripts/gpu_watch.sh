#!/bin/bash
# Long-lived GPU watcher (runs on the researcher's Mac, not in a tool session, so it does not expire): polls each instance every GPU_WATCH_INTERVAL seconds (default 60) over ssh and writes one line per event to $GPU_WATCH_LOG,
# plus a macOS desktop notification. Events: GPU_IDLE (a GPU that was busy for at least 5 polls has had no compute process for 3 polls), IDLE_NOW (first seen idle, as long as the instance has been up), IDLE_REMINDER (every 60 minutes while idle),
# GPU_BUSY (busy again), HOST_UNREACHABLE (3 failed polls) and HOST_BACK. A host that is terminated shows up as HOST_UNREACHABLE, which is the expected end of its life.
# Usage: GPU_WATCH_HOSTS=<file with lines "<label> <user@host>"> GPU_WATCH_LOG=<log file> [GPU_WATCH_KEY=<ssh key>] bash scripts/gpu_watch.sh   (the hosts file stays outside the repository: no addresses are committed).
HOSTS=${GPU_WATCH_HOSTS:?hosts file}; LOG=${GPU_WATCH_LOG:?log file}; KEY=${GPU_WATCH_KEY:-$HOME/.ssh/glassbox_lambda}; IV=${GPU_WATCH_INTERVAL:-60}
# state is indexed by the line number of the host (bash 3.2 on macOS has no associative arrays: DEVIATIONS 20)
emit() { local m="$(date -u +%FT%TZ) $1"; echo "$m" >> "$LOG"; command -v osascript >/dev/null && osascript -e "display notification \"${1//\"/}\" with title \"GPU watch\"" >/dev/null 2>&1; }
echo "$(date -u +%FT%TZ) WATCH_START hosts=$(grep -c . "$HOSTS")" >> "$LOG"
while true; do
  i=0
  while read -r label target; do
    i=$(( i + 1 )); [ -n "$label" ] || continue
    out=$(ssh -n -i "$KEY" -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=2 "$target" 'echo $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)' 2>/dev/null)
    if [ -z "$out" ]; then
      fail[$i]=$(( ${fail[$i]:-0} + 1 ))
      [ "${fail[$i]}" = 3 ] && { emit "HOST_UNREACHABLE $label (3 failed polls; terminated or network)"; state[$i]=down; }
      continue
    fi
    [ "${state[$i]}" = down ] && emit "HOST_BACK $label"
    fail[$i]=0; state[$i]=up; n=${out%% *}; now=$(date +%s)
    if [ "$n" -gt 0 ]; then
      busy[$i]=$(( ${busy[$i]:-0} + 1 )); idle[$i]=0
      [ "${busy[$i]}" = 2 ] && [ "${lastrem[$i]:-x}" != x ] && emit "GPU_BUSY $label"
      lastrem[$i]=""
    else
      idle[$i]=$(( ${idle[$i]:-0} + 1 ))
      if [ "${idle[$i]}" = 3 ]; then
        if [ "${busy[$i]:-0}" -ge 5 ]; then emit "GPU_IDLE $label (was busy for ${busy[$i]} polls; no compute process for 3 polls)"; else emit "IDLE_NOW $label (GPU idle)"; fi
        busy[$i]=0; lastrem[$i]=$now
      elif [ "${idle[$i]}" -gt 3 ] && [ -n "${lastrem[$i]}" ] && [ $(( now - ${lastrem[$i]} )) -ge 3600 ]; then
        emit "IDLE_REMINDER $label (GPU idle for $(( (now - ${lastrem[$i]}) / 60 )) min of the last report; the instance is still billing)"; lastrem[$i]=$now
      fi
    fi
  done < "$HOSTS"
  sleep "$IV"
done
