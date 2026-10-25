#!/bin/bash
# W5 full training of one arm (nnU-Net v2, Task03, fold 0, 1000 epochs, committed seed), unattended with automatic resume from checkpoint_latest.pth (every 50 epochs, persistent filesystem), up to 5 attempts.
# Usage: bash scripts/w5_train.sh <trainer class, e.g. nnUNetTrainerW5Bright50> <tag, e.g. B50>. Log: /lambda/nfs/CTGate/w5/train_<tag>.log with ATTEMPT n exit <rc> lines and TRAIN_<tag>_DONE or TRAIN_<tag>_FAILED. No HCC inference here (the sanity gate comes first).
source /lambda/nfs/CTGate/w3/nnunet_env.sh
TR=$1; TAG=$2; mkdir -p /lambda/nfs/CTGate/w5; LOG=/lambda/nfs/CTGate/w5/train_$TAG.log
D=$nnUNet_results/Dataset003_Liver/${TR}__nnUNetPlans__3d_fullres/fold_0
for attempt in 1 2 3 4 5; do
  echo "ATTEMPT $attempt start $(date -u)" >> $LOG
  if [ -f $D/checkpoint_latest.pth ]; then nnUNetv2_train 3 3d_fullres 0 -tr $TR --c >> $LOG 2>&1; else nnUNetv2_train 3 3d_fullres 0 -tr $TR >> $LOG 2>&1; fi
  echo "ATTEMPT $attempt exit $? $(date -u)" >> $LOG
  if [ -f $D/checkpoint_final.pth ]; then echo "TRAIN_${TAG}_DONE $(date -u)" >> $LOG; exit 0; fi
done
echo "TRAIN_${TAG}_FAILED $(date -u)" >> $LOG
