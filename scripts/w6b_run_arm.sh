#!/bin/bash
# W6b driver for ONE arm on ONE A10 (results/w6b_distill/PREREG.md); this is scripts/w6_run_arm.sh with only these differences: arm names W6b-FULL-L<lambda> / W6b-DEC-L<lambda> (the lambda is read from the name), training with scripts/w6b_train.py, the W6b hash file in the preflight, and `mkdir -p hcc/analysis` before the HCC analysis (DEVIATIONS 30). Original header: W6 driver for ONE arm on ONE A10 (results/w6_vista_finetune/PREREG.md): preflight -> training (8,000 iterations, final model only) -> WAW-TACE inference of both halves + scoring (W4 route) -> HCC-TACE-Seg inference + W3 analysis -> forgetting-guard inference.
# Usage: bash scripts/w6b_run_arm.sh <W6b-FULL-L<lambda>|W6b-DEC-L<lambda>>. Time rule: WARN ONLY. The only stops (W6_HOLD in $LOG): a non-zero exit, an error in a step's log (Traceback, CUDA out of memory, RuntimeError), a 20-minute stall (no new line in the step's log). Nothing is retried or changed automatically.
ARM=$1; C=/lambda/nfs/CTGate; R=$C/w1/repo; W=$C/w6; O=$W/arms/$ARM; LOG=$W/arms/queue_${W6_QUEUE_TAG:-solo}.log; PY=$C/venv_w6/bin/python; W4=$C/w4_waw; INF=$C/w1/vendor_inferer; BASEW=$C/glassbox/cache/weights/model.pt
mkdir -p $O; log() { echo "$1 $(date -u +%FT%TZ)" >> $LOG; }; hold() { echo "W6_HOLD: $ARM $1 $(date -u)" >> $LOG; exit 1; }
cd $R || hold "repo missing"
case $ARM in
  W6b-FULL-L*) VARIANT=W6b-FULL; LAM=${ARM##*-L}; SEED=20261110;; W6b-DEC-L*) VARIANT=W6b-DEC; LAM=${ARM##*-L}; SEED=20261110;;
  *) echo "unknown arm"; exit 2;;
esac
[[ "$LAM" =~ ^[0-9]+(\.[0-9]+)?$ ]] || hold "bad lambda in the arm name ($ARM)"

run_step() { # LOGFILE cmd... : warn-only time rule; hold on non-zero exit, error in the log, or a 20-minute stall
  local lf=$1; shift; : > $lf; "$@" >> $lf 2>&1 & local pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30; local age=$(( $(date +%s) - $(stat -c %Y $lf) ))
    if [ $age -ge 1200 ]; then kill $pid 2>/dev/null; hold "STALL ($(basename $lf): no new line for $((age/60)) min)"; fi
    if grep -qE "^Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|^RuntimeError" $lf 2>/dev/null; then kill $pid 2>/dev/null; hold "ERROR in $(basename $lf)"; fi
  done
  wait $pid || hold "non-zero exit of $(basename $lf)"
}
# ---- preflight
[ -n "$W6_SHARED_GPU" ] || [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle"  # W6_SHARED_GPU=1: explicit opt-in to share a GPU with another step (DEVIATIONS 32, scheduling only)
for f in results/w6b_distill/code_hashes.txt results/w6_vista_finetune/code_hashes_drift_streaming_v2.txt results/w6_vista_finetune/code_hashes.txt results/w4_waw/code_hashes_w4.txt results/w4b_waw/code_hashes.txt results/w6_vista_finetune/code_hashes_analysis.txt; do [ -f $f ] || continue; grep -v '^#' $f | sha256sum -c --quiet - >> $O/preflight.log 2>&1 || hold "code hashes differ ($f)"; done
MAPPING=$($PY -c "import json;print(json.load(open('$W/liver_rule.json'))['mapping_decision'])" 2>/dev/null); [ "$MAPPING" = "[[2,26]]" ] || hold "mapping decision is not [[2,26]] ($MAPPING)"
log "W6_ARM_START $ARM seed=$SEED mapping=$MAPPING"
# ---- training (not for the baseline guard)
if [ "$ARM" != "BASELINE_GUARD" ]; then
  if [ ! -f $O/train/final_model.pt ]; then
    run_step $O/train.log $PY scripts/w6b_train.py --bundle $W/NV-Segment-CTMR/NV-Segment-CT --arm $VARIANT --lam $LAM --weights $BASEW --dataset-dir $C/w1/msd_task03/Task03_Liver --data-list $W/task03_datalist.json --out $O/train --iterations 8000 --patch 128 --mapping 2,26 --seed $SEED --device cuda --workers 4
  fi
  [ -f $O/train/final_model.pt ] || hold "no final model"; log "W6_TRAINED $ARM"; WTS=$O/train/final_model.pt
else
  WTS=$BASEW
fi
# ---- WAW-TACE inference (W4 route) and scoring, both halves (not for the baseline: scored in W4)
if [ "$ARM" != "BASELINE_GUARD" ]; then
  for H in 0 1; do
    run_step $O/waw_pred_half$H.log $PY scripts/w4_waw_predict.py $INF $WTS $W4/half$H/gt $W4/half$H/patients.txt $O/waw/pred_half$H
    run_step $O/waw_score_half$H.log $PY scripts/w4_waw_score.py $W4/half$H/gt $O/waw/pred_half$H $ARM $O/waw/score_half$H --workers 12
  done
  log "W6_WAW_SCORED $ARM"
  # ---- HCC-TACE-Seg (W3 pipeline), secondary
  mkdir -p $O/hcc/analysis; run_step $O/hcc_run.log $PY scripts/w3_vista_ref_run.py $INF $WTS $O/hcc
  run_step $O/hcc_analyze.log $PY scripts/w3_analyze.py $O/hcc/analysis $O/hcc/w3_lesions_vista_26_auto_hcc.csv
  log "W6_HCC_DONE $ARM"
fi
# ---- forgetting-guard inference (Task09 must exist: scripts/w6_fetch_task09.sh)
[ -d $W/data/Task09_Spleen/imagesTr ] || hold "Task09 missing (run scripts/w6_fetch_task09.sh first)"
run_step $O/forget.log $PY scripts/w6_forget_eval.py $INF $WTS $ARM $O/forget --spleen $W/data/Task09_Spleen --kits $C/glassbox/cache/heldout_nii/c4kc_kits --task03 $C/w1/msd_task03/Task03_Liver --datalist $W/task03_datalist.json --waw $W4 --labels $W/NV-Segment-CTMR/NV-Segment-CT/configs/label_dict.json
log "W6_ARM_DONE $ARM (GPU idle after this arm)"
