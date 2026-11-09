#!/bin/bash
# (v2: also watches the W6b driver) Monitor helper (not part of any frozen analysis): prints one line when (a) an arm has W6_ARM_START but neither W6_ARM_DONE nor W6_HOLD in the queue logs and NO driver process for it exists on either instance (a queue that stopped without a closing line), (b) the TotalSegmentator run has W6_TS_START but neither W6_TS_DONE nor a hold and no driver process, (c) any queue log shows W6_HOLD / W6_QUEUE_STOPPED, (d) an instance is unreachable. Checks every 60 s; each problem is printed once.
# Usage: bash scripts/w6_watch.sh <ssh user@host instance 1> <ssh user@host instance 2>   (host addresses are passed on the command line, never stored)
K=$HOME/.ssh/glassbox_lambda; I1=$1; I2=$2; declare -A seen
FIRST=1
say() { [ -z "${seen[$1]}" ] && { seen[$1]=1; [ "$FIRST" = 1 ] || echo "$1"; }; }      # the first pass only records what is already stopped or held
while true; do
  L=$(ssh -n -i $K -o ConnectTimeout=15 -o BatchMode=yes $I1 'cd /lambda/nfs/CTGate/w6/arms; for f in queue_*.log; do echo "== $f"; grep -E "W6_ARM_START|W6_ARM_DONE|W6_HOLD|W6_QUEUE_STOPPED|W6_TS_START|W6_TS_DONE" $f; done' 2>/dev/null) || { say "WATCH: instance 1 unreachable $(date -u +%H:%M)"; sleep 60; continue; }
  P1=$(ssh -n -i $K -o ConnectTimeout=15 -o BatchMode=yes $I1 'pgrep -af "w6b?_run_arm.sh|w6_run_ts" | cut -c1-200' 2>/dev/null); P2=$(ssh -n -i $K -o ConnectTimeout=15 -o BatchMode=yes $I2 'pgrep -af "w6b?_run_arm.sh|w6_run_ts" | cut -c1-200' 2>/dev/null) || say "WATCH: instance 2 unreachable $(date -u +%H:%M)"
  P="$P1
$P2"
  for A in $(echo "$L" | grep -oE "W6_ARM_START [A-Za-z0-9_.-]+" | awk '{print $2}' | sort -u); do
    echo "$L" | grep -qE "W6_ARM_DONE $A( |$)" && continue
    echo "$P" | grep -qE "w6b?_run_arm.sh $A( |$)" || say "WATCH: ARM STOPPED WITHOUT CLOSING LINE: $A (no driver process on either instance)"
  done
  if echo "$L" | grep -q W6_TS_START && ! echo "$L" | grep -q W6_TS_DONE; then echo "$P" | grep -q w6_run_ts || say "WATCH: TS STOPPED WITHOUT CLOSING LINE (no driver process)"; fi
  while read -r line; do [ -n "$line" ] && say "WATCH: $line"; done <<< "$(echo "$L" | grep -E "W6_HOLD|W6_QUEUE_STOPPED")"
  FIRST=0; sleep 60
done
