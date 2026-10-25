#!/bin/bash
# W5 overnight driver (pre-approved plan): wait for M3 model D to finish with exit 0 on both shards, validate its outputs (CPU), then run the 5-epoch smoke runs of BRIGHT-50 and SHAM-50 (GPU idle, so epoch times are clean);
# if BOTH pass, start the BRIGHT-50 full training in tmux (session w5B50). Any failure writes W5_DRIVER_HOLD with the reason and stops (no retries, no training). Markers in /lambda/nfs/CTGate/w5/driver.log.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; S=/lambda/nfs/CTGate/w3/stage2; R=/lambda/nfs/CTGate/w1/repo; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; L=$W/driver.log; mkdir -p $W; cd $R
hold() { echo "W5_DRIVER_HOLD: $1 $(date -u)" >> $L; exit 1; }
echo "W5_DRIVER_START $(date -u)" >> $L
while true; do
  l0=$(grep -h -o 'M3_D[01]_EXIT=[0-9]*' $S/m3_D0.log | tail -1); l1=$(grep -h -o 'M3_D[01]_EXIT=[0-9]*' $S/m3_D1.log | tail -1)
  n0=$(grep -c 'M3_D[01]_EXIT' $S/m3_D0.log); n1=$(grep -c 'M3_D[01]_EXIT' $S/m3_D1.log)
  if [ "$n0" -ge 2 ] && [ "$n1" -ge 2 ]; then
    [ "$l0" = "M3_D0_EXIT=0" ] && [ "$l1" = "M3_D1_EXIT=0" ] && break
    hold "M3 D shard exited non-zero ($l0 $l1)"
  fi
  if [ "$n0" -ge 2 ] && [ "$l0" != "M3_D0_EXIT=0" ]; then hold "M3 D0 exited non-zero ($l0)"; fi
  if [ "$n1" -ge 2 ] && [ "$l1" != "M3_D1_EXIT=0" ]; then hold "M3 D1 exited non-zero ($l1)"; fi
  sleep 30
done
echo "M3_D_FINISHED $(date -u)" >> $L
$V scripts/w3s2_m3_validate.py $S/cases_with_027_75.txt $S/m3_D_validate.json $S/m3_D0 $S/m3_D1 >> $L 2>&1 || hold "M3 D validation not OK (see m3_D_validate.json)"
echo "M3_D_VALIDATED $(date -u)" >> $L
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 10; done
echo "GPU_IDLE_FOR_W5_SMOKE $(date -u)" >> $L
for A in Bright50:BRIGHT Sham50:SHAM; do
  N=${A%%:*}; ARM=${A##*:}; rm -f $W/edit_$N.log; export W5_EDIT_LOG=$W/edit_$N.log
  echo "SMOKE_START $N $(date -u)" >> $L
  nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerW5Smoke$N > $W/smoke_$N.log 2>&1 || hold "smoke $N training exited non-zero"
  TL=$(ls -t $nnUNet_results/Dataset003_Liver/nnUNetTrainerW5Smoke${N}__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt | head -1)
  $V scripts/w5_smoke_check.py $ARM $TL $W/edit_$N.log $W/smoke_$N.json >> $L 2>&1 || hold "smoke $N check FAILED (see smoke_$N.json)"
  echo "SMOKE_PASS $N $(date -u)" >> $L
done
tmux new-session -d -s w5B50 "bash $R/scripts/w5_train.sh nnUNetTrainerW5Bright50 B50"
echo "W5_B50_TRAINING_STARTED $(date -u)" >> $L
