#!/bin/bash
# Long-lived GPU watcher (runs on the researcher's Mac, not in a tool session, so it does not expire): polls each instance every GPU_WATCH_INTERVAL seconds (default 60) over ssh and writes one line per event to $GPU_WATCH_LOG,
# plus a macOS desktop notification. Events: GPU_IDLE (a GPU that was busy for at least 5 polls has had no compute process for 3 polls), IDLE_NOW (first seen idle, as long as the instance has been up), IDLE_REMINDER (every 60 minutes while idle),
# GPU_BUSY (busy again), HOST_UNREACHABLE (3 failed polls) and HOST_BACK. A host that is terminated shows up as HOST_UNREACHABLE, which is the expected end of its life.
# Usage: GPU_WATCH_HOSTS=<file with lines "<label> <user@host>"> GPU_WATCH_LOG=<log file> [GPU_WATCH_KEY=<ssh key>] bash scripts/gpu_watch.sh   (the hosts file stays outside the repository: no addresses are committed).
HOSTS=${GPU_WATCH_HOSTS:?hosts file}; LOG=${GPU_WATCH_LOG:?log file}; KEY=${GPU_WATCH_KEY:-$HOME/.ssh/glassbox_lambda}; IV=${GPU_WATCH_INTERVAL:-60}
declare -A busy idle fail up lastrem state
emit() { local m="$(date -u +%FT%TZ) $1"; echo "$m" >> "$LOG"; command -v osascript >/dev/null && osascript -e "display notification \"${1//\"/}\" with title \"GPU watch\"" >/dev/null 2>&1; }
echo "$(date -u +%FT%TZ) WATCH_START hosts=$(grep -c . "$HOSTS")" >> "$LOG"
while true; do
  while read -r label target; do
    [ -n "$label" ] || continue
    out=$(ssh -i "$KEY" -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=2 "$target" 'echo $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)' 2>/dev/null)
    if [ -z "$out" ]; then
      fail[$label]=$(( ${fail[$label]:-0} + 1 ))
      [ "${fail[$label]}" = 3 ] && { emit "HOST_UNREACHABLE $label (3 failed polls; terminated or network)"; state[$label]=down; }
      continue
    fi
    [ "${state[$label]}" = down ] && emit "HOST_BACK $label"
    fail[$label]=0; state[$label]=up; n=${out%% *}; now=$(date +%s)
    if [ "$n" -gt 0 ]; then
      busy[$label]=$(( ${busy[$label]:-0} + 1 )); idle[$label]=0
      [ "${busy[$label]}" = 2 ] && [ "${lastrem[$label]:-x}" != x ] && emit "GPU_BUSY $label"
      lastrem[$label]=""
    else
      idle[$label]=$(( ${idle[$label]:-0} + 1 ))
      if [ "${idle[$label]}" = 3 ]; then
        if [ "${busy[$label]:-0}" -ge 5 ]; then emit "GPU_IDLE $label (was busy for ${busy[$label]} polls; no compute process for 3 polls)"; else emit "IDLE_NOW $label (GPU idle)"; fi
        busy[$label]=0; lastrem[$label]=$now
      elif [ "${idle[$label]}" -gt 3 ] && [ -n "${lastrem[$label]}" ] && [ $(( now - ${lastrem[$label]} )) -ge 3600 ]; then
        emit "IDLE_REMINDER $label (GPU idle for $(( (now - ${lastrem[$label]}) / 60 )) min of the last report; the instance is still billing)"; lastrem[$label]=$now
      fi
    fi
  done < "$HOSTS"
  sleep "$IV"
done
