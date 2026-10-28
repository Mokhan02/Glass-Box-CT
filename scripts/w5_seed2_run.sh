#!/bin/bash
# W5 Amendment 2 driver (BRIGHT-50-s2, one A10): 2-epoch smoke of the real augmentation with the seed logged -> committed early check -> BRIGHT-50-s2 full training (1000 epochs, tmux w5B50S2, committed w5_train.sh) -> post pipeline (tmux w5post, committed w5_post.sh)
# -> final CPU analysis (tmux w5final, w5_seed2_final.sh). The smoke run is stopped after "Training done" (its slow final validation is skipped, as in Deviation 13). Any failure writes W5_DRIVER_HOLD to driver.log (the alert poller reports it) and stops; nothing is retried.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; R=/lambda/nfs/CTGate/w1/repo; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; L=$W/driver.log; TAG=B50S2; TR=nnUNetTrainerW5Bright50Seed2; SMK=nnUNetTrainerW5SmokeBright50Seed2; SEED=20261028
hold() { echo "W5_DRIVER_HOLD: $1 $(date -u)" >> $L; exit 1; }
mkdir -p $W; cd $R; echo "W5_S2_DRIVER_START $(date -u)" >> $L
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
$V scripts/w5_early_check.py $SEED $TLG $W5_EDIT_LOG $W/smoke_$TAG.json >> $L 2>&1 || hold "smoke $TAG early check FAILED (see smoke_$TAG.json)"
echo "SMOKE_PASS $TAG $(date -u)" >> $L
sleep 5; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU still busy after the smoke"
echo "W5_${TAG}_TRAINING_STARTED $(date -u)" >> $L
tmux new-session -d -s w5$TAG "bash $R/scripts/w5_train.sh $TR $TAG"
tmux new-session -d -s w5post "bash $R/scripts/w5_post.sh $TAG $TR >> $W/post_$TAG.out 2>&1"
tmux new-session -d -s w5final "bash $R/scripts/w5_seed2_final.sh >> $W/final_$TAG.out 2>&1"
echo "W5_S2_DRIVER_DONE (training, post pipeline and final analysis waiters started) $(date -u)" >> $L
