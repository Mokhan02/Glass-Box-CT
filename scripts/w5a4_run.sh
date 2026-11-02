#!/bin/bash
# W5 Amendment 4 driver for ONE arm on ONE A10 (results/w5_training/PREREG.md, Amendment 4). Usage: bash scripts/w5a4_run.sh <B25|D0S2>, in tmux session w5a4drv, from any directory (it cd's to the repo checkout on the instance).
#   B25  = BRIGHT-25: trainer nnUNetTrainerW5Bright25, seed 20261020.   D0S2 = D0-seed2 (no augmentation): trainer nnUNetTrainerW5D0Seed2, seed 20261121.
# Steps (the procedure of Amendment 2, generalised): preflight (GPU idle, committed hashes, installed trainer files byte-equal to the committed ones, the reference inputs of the final analysis exist) -> 2-epoch smoke of the real arm with the arm-specific early check
# -> full training (tmux w5<TAG>, scripts/w5_train.sh, automatic resume) -> post pipeline (tmux w5post<TAG>, scripts/w5_post.sh: sanity gate, HCC inference, scorer, alignment gate, W3 analysis) -> for B25 only: the final dose-response analysis waiter (tmux w5a4final, waits for BOTH arms).
# Any failure writes W5A4_DRIVER_HOLD to $WA/driver.log (the alert poller reports it) and stops; nothing is retried or changed automatically. The driver does not start the alert poller: start it first, in its own tmux session, after testing it by forced failure:
#   W5_TAG=<TAG> W5_SESS_DRV=w5a4drv W5_SESS_TR=w5<TAG> W5_REAL_LOG_DIR=<nnU-Net results fold dir of the arm> W5_POST_DIR=/lambda/nfs/CTGate/w3/stage1/W5_<TAG> bash scripts/w5_alert.sh /lambda/nfs/CTGate/w5a4_<TAG>
TAG=$1
case "$TAG" in
  B25)  TR=nnUNetTrainerW5Bright25;  SMK=nnUNetTrainerW5SmokeBright25;  SEED=20261020;;
  D0S2) TR=nnUNetTrainerW5D0Seed2;   SMK=nnUNetTrainerW5SmokeD0Seed2;   SEED=20261121;;
  *) echo "usage: bash scripts/w5a4_run.sh <B25|D0S2>"; exit 2;;
esac
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; WA=/lambda/nfs/CTGate/w5a4_$TAG; R=/lambda/nfs/CTGate/w1/repo; S1=/lambda/nfs/CTGate/w3/stage1; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; L=$WA/driver.log; T=w3_lesions_D_nnunet_hcc.csv
mkdir -p $W $WA; ln -sf $W/train_$TAG.log $WA/train_$TAG.log
hold() { echo "W5A4_DRIVER_HOLD: $TAG $1 $(date -u)" >> $L; exit 1; }
cd $R || hold "repo checkout missing"; echo "W5A4_${TAG}_DRIVER_START $(date -u)" >> $L

# ---- preflight
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle at start"
grep -v '^#' results/w5_training/code_hashes_a4.txt | sha256sum -c --quiet - >> $L 2>&1 || hold "code hashes differ from results/w5_training/code_hashes_a4.txt"
SP=$(dirname $($V -c "import nnunetv2.training.nnUNetTrainer as m; print(m.__file__)"))
[ "$(sha256sum $SP/nnUNetTrainerW5.py | cut -d' ' -f1)" = "$(sha256sum scripts/w5_nnunet_trainer.py | cut -d' ' -f1)" ] || hold "installed nnUNetTrainerW5.py differs from the committed scripts/w5_nnunet_trainer.py"
cp scripts/w5a4_smoke_trainers.py $SP/nnUNetTrainerW5SmokeA4.py
[ "$(sha256sum $SP/nnUNetTrainerW5SmokeA4.py | cut -d' ' -f1)" = "$(sha256sum scripts/w5a4_smoke_trainers.py | cut -d' ' -f1)" ] || hold "smoke trainer file copy differs"
$V -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5SmokeA4 import nnUNetTrainerW5SmokeBright25, nnUNetTrainerW5SmokeD0Seed2; from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5 import nnUNetTrainerW5Bright25, nnUNetTrainerW5D0Seed2" >> $L 2>&1 || hold "nnU-Net cannot import the trainer classes"
for f in $S1/D0_74/$T $S1/W5_B50/eval/$T $S1/W5_B50S2/eval/$T $S1/W5_B50/gate.json $S1/W5_B50S2/gate.json; do [ -f $f ] || hold "reference input of the final analysis missing: $f"; done
echo "W5A4_PREFLIGHT_PASS $TAG $(date -u)" >> $L

# ---- 2-epoch smoke of the real arm (stopped after "Training done"; its slow final validation is skipped, as in Deviation 13)
rm -f $W/edit_$TAG.log; export W5_EDIT_LOG=$W/edit_$TAG.log
D=$nnUNet_results/Dataset003_Liver/${SMK}__nnUNetPlans__3d_fullres/fold_0
echo "SMOKE_START $TAG $(date -u)" >> $L
nnUNetv2_train 3 3d_fullres 0 -tr $SMK > $W/smoke_$TAG.log 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  TLG=$(ls -t $D/training_log_*.txt 2>/dev/null | head -1)
  if [ -n "$TLG" ] && grep -q "Training done" $TLG; then kill $PID; break; fi
  sleep 10
done
wait $PID 2>/dev/null
TLG=$(ls -t $D/training_log_*.txt 2>/dev/null | head -1)
[ -n "$TLG" ] && grep -q "Training done" $TLG || hold "smoke $TAG did not finish training (see smoke_$TAG.log)"
$V scripts/w5a4_early_check.py $TAG $TLG $W5_EDIT_LOG $W/smoke_$TAG.json >> $L 2>&1 || hold "smoke $TAG early check FAILED (see smoke_$TAG.json)"
echo "SMOKE_PASS $TAG $(date -u)" >> $L
sleep 5; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU still busy after the smoke"

# ---- full run: training, post pipeline (and, for B25, the final dose-response waiter)
unset W5_EDIT_LOG
echo "W5_${TAG}_TRAINING_STARTED $(date -u)" >> $L
tmux new-session -d -s w5$TAG "bash $R/scripts/w5_train.sh $TR $TAG"
tmux new-session -d -s w5post$TAG "bash $R/scripts/w5_post.sh $TAG $TR >> $W/post_$TAG.out 2>&1"
[ "$TAG" = "B25" ] && tmux new-session -d -s w5a4final "bash $R/scripts/w5a4_final.sh >> $W/final_a4.out 2>&1"
echo "W5A4_${TAG}_DRIVER_DONE (training and post pipeline waiters started) $(date -u)" >> $L
