#!/bin/bash
# W6 queue for one instance: runs the listed arms one after the other (scripts/w6_run_arm.sh); stops at the first hold; one log per queue tag (the two instances share a filesystem). Usage: bash scripts/w6_queue.sh <tag> ARM1 ARM2 ...
export W6_QUEUE_TAG=$1; shift; LOGQ=/lambda/nfs/CTGate/w6/arms/queue_$W6_QUEUE_TAG.log; mkdir -p /lambda/nfs/CTGate/w6/arms; echo "W6_QUEUE_START $* $(date -u +%FT%TZ)" >> $LOGQ
for A in "$@"; do bash /lambda/nfs/CTGate/w1/repo/scripts/w6_run_arm.sh $A || { echo "W6_QUEUE_STOPPED at $A $(date -u +%FT%TZ)" >> $LOGQ; exit 1; }; done
echo "W6_QUEUE_DONE $* $(date -u +%FT%TZ) (GPU idle)" >> $LOGQ
