#!/bin/bash
# operator wrapper (host2): SHAM-50 5-epoch smoke -> skip its slow final validation after "Training done" -> committed checker -> if PASS start SHAM-50 full training (committed w5_train.sh); otherwise write a hold marker.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
R=/lambda/nfs/CTGate/w1/repo; W=/lambda/nfs/CTGate/w5; H=/lambda/nfs/CTGate/w5_h2; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; cd $R
rm -f $W/edit_Sham50.log; export W5_EDIT_LOG=$W/edit_Sham50.log; echo "SMOKE_START Sham50 (host2) $(date -u)" >> $H/waiter.log
nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerW5SmokeSham50 > $W/smoke_Sham50.log 2>&1 &
PID=$!
D=$nnUNet_results/Dataset003_Liver/nnUNetTrainerW5SmokeSham50__nnUNetPlans__3d_fullres/fold_0
while kill -0 $PID 2>/dev/null; do
  TL=$(ls -t $D/training_log_*.txt 2>/dev/null | head -1)
  if [ -n "$TL" ] && grep -q "Training done" $TL; then kill $PID; sleep 5; break; fi
  sleep 10
done
wait $PID 2>/dev/null
TL=$(ls -t $D/training_log_*.txt 2>/dev/null | head -1)
if [ -n "$TL" ] && grep -q "Training done" $TL && $V scripts/w5_smoke_check.py SHAM $TL $W/edit_Sham50.log $W/smoke_Sham50.json >> $H/waiter.log 2>&1; then
  echo "SMOKE_PASS Sham50 (host2) $(date -u)" >> $H/waiter.log; echo "SMOKE_PASS Sham50 $(date -u)" >> $H/driver.log
  echo "W5_B50_TRAINING_STARTED (host2: SHAM-50) $(date -u)" >> $H/driver.log
  tmux new-session -d -s w5SHAM "bash $R/scripts/w5_train.sh nnUNetTrainerW5Sham50 SHAM50"
  echo "SHAM50_TRAINING_STARTED $(date -u)" >> $H/waiter.log
else
  echo "W5_DRIVER_HOLD: SHAM-50 smoke failed or did not finish (see smoke_Sham50.log / smoke_Sham50.json) $(date -u)" >> $H/driver.log
fi
