#!/bin/bash
# W5 post-training pipeline for one arm, on the instance that trained it (pre-approved plan): wait for the training to finish, run the sanity gate, and if it PASSES run HCC inference on the W3 Stage 1 cohort with
# the Stage 1 model D commands (nnUNetv2_predict defaults, the committed scorer, the Amendment 2 shift gate, the committed W3 analysis). Any failure writes W5_POST_HOLD to the arm's training log (the alert poller reports it) and stops.
# On success it writes TRAIN_<TAG>POST_DONE (GPU idle) to the same log. Usage: bash scripts/w5_post.sh <TAG, e.g. B50> <trainer class, e.g. nnUNetTrainerW5Bright50>
source /lambda/nfs/CTGate/w3/nnunet_env.sh
TAG=$1; TR=$2; LOG=/lambda/nfs/CTGate/w5/train_$TAG.log; R=/lambda/nfs/CTGate/w1/repo; S1=/lambda/nfs/CTGate/w3/stage1; OUT=$S1/W5_$TAG; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python
F=$nnUNet_results/Dataset003_Liver/${TR}__nnUNetPlans__3d_fullres/fold_0
hold() { echo "W5_POST_HOLD: $TAG $1 $(date -u)" >> $LOG; exit 1; }
mkdir -p $OUT $OUT/analysis; cd $R; export PYTHONPATH=$R
until grep -q -E "TRAIN_${TAG}_(DONE|FAILED)" $LOG; do sleep 60; done
grep -q "TRAIN_${TAG}_FAILED" $LOG && hold "training failed"
echo "POST_START $TAG $(date -u)" >> $OUT/post.log
$V scripts/w5_gate.py $F/validation/summary.json $OUT/gate.json >> $OUT/post.log 2>&1 || hold "sanity gate FAILED or not computable (see $OUT/gate.json)"
echo "POST_GATE_PASS $TAG $(date -u)" >> $OUT/post.log
nnUNetv2_predict -i $S1/D_in -o $OUT/pred -d 3 -c 3d_fullres -tr $TR -f 0 -chk checkpoint_final.pth > $OUT/pred.log 2>&1 || hold "HCC inference failed (see pred.log)"
[ "$(ls $OUT/pred | grep -c '\.nii\.gz$')" = "75" ] || hold "HCC inference wrote $(ls $OUT/pred | grep -c '\.nii\.gz$') of 75 cases"
echo "POST_INFERENCE_DONE $TAG $(date -u)" >> $OUT/post.log
$V scripts/w3_modelD_eval.py $OUT/pred $OUT/eval >> $OUT/post.log 2>&1 || hold "scorer failed"
$V scripts/w3_shift_gate.py D $OUT/pred $OUT/eval/shift_gate_D.json >> $OUT/post.log 2>&1 || hold "shift gate failed to run"
grep -q '"status": "OK"' $OUT/eval/shift_gate_D.json || hold "Amendment 2 alignment gate STOP (see shift_gate_D.json)"
$V scripts/w3_analyze.py $OUT/analysis $OUT/eval/w3_lesions_D_nnunet_hcc.csv >> $OUT/post.log 2>&1 || hold "W3 analysis failed"
echo "TRAIN_${TAG}POST_DONE GPU_IDLE $(date -u)" >> $LOG
