#!/bin/bash
# W4b: TotalSegmentator run on one instance (GPU): inference of both halves (scripts/w6_ts_run.py, venv_ts) -> scoring with the W4 scorer (venv_w6) -> marker W6_TS_DONE. Warn-only time rule; holds only on a non-zero exit, an error in a step log, or a 20-minute stall.
# Usage: bash scripts/w6_run_ts.sh. Held-out status of the model: inferred, not author-confirmed (DEVIATIONS 25).
C=/lambda/nfs/CTGate; R=$C/w1/repo; W=$C/w6; O=$W/arms/TS; LOG=$W/arms/queue_ts.log; PYT=$C/venv_ts/bin/python; PY=$C/venv_w6/bin/python; W4=$C/w4_waw; mkdir -p $O
log() { echo "$1 $(date -u +%FT%TZ)" >> $LOG; }; hold() { echo "W6_HOLD: TS $1 $(date -u)" >> $LOG; exit 1; }
cd $R || hold "repo missing"; export TOTALSEG_HOME_DIR=$W/ts_home
run_step() { local lf=$1; shift; : > $lf; "$@" >> $lf 2>&1 & local pid=$!
  while kill -0 $pid 2>/dev/null; do sleep 30; local age=$(( $(date +%s) - $(stat -c %Y $lf) ))
    if [ $age -ge 1200 ]; then kill $pid 2>/dev/null; hold "STALL ($(basename $lf): no new line for $((age/60)) min)"; fi
    if grep -qE "^Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|^RuntimeError" $lf 2>/dev/null; then kill $pid 2>/dev/null; hold "ERROR in $(basename $lf)"; fi
  done; wait $pid || hold "non-zero exit of $(basename $lf)"; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle"
grep -v '^#' results/w6_vista_finetune/code_hashes_ts.txt | sha256sum -c --quiet - >> $O/preflight.log 2>&1 || hold "code hashes differ (code_hashes_ts.txt)"
log "W6_TS_START"
for H in 0 1; do
  run_step $O/ts_half$H.log $PYT scripts/w6_ts_run.py $W4/half$H/gt $W4/half$H/patients.txt $O/pred_half$H --device gpu
  run_step $O/score_half$H.log $PY scripts/w4_waw_score.py $W4/half$H/gt $O/pred_half$H TS $O/score_half$H --workers 12
done
log "W6_TS_DONE (GPU idle after this)"
