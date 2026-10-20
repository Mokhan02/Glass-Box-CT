#!/bin/bash
# W3 model D (nnU-Net v2, Task03, fold 0, 1000 epochs, fixed seed), unattended: runs the trainer and, if the process dies, resumes from checkpoint_latest.pth (saved every 50 epochs on the persistent filesystem), up to 5 attempts.
# Writes TRAIN_D_DONE or TRAIN_D_FAILED to the log. No HCC inference is started here (PREREG: the fold-0 sanity gate comes first).
source /lambda/nfs/CTGate/w3/nnunet_env.sh
LOG=/lambda/nfs/CTGate/w3/train_d.log
D=$nnUNet_results/Dataset003_Liver/nnUNetTrainerW3Seed__nnUNetPlans__3d_fullres/fold_0
for attempt in 1 2 3 4 5; do
  echo "ATTEMPT $attempt start $(date -u)" >> $LOG
  if [ -f $D/checkpoint_latest.pth ]; then nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerW3Seed --c >> $LOG 2>&1; else nnUNetv2_train 3 3d_fullres 0 -tr nnUNetTrainerW3Seed >> $LOG 2>&1; fi
  echo "ATTEMPT $attempt exit $? $(date -u)" >> $LOG
  if [ -f $D/checkpoint_final.pth ]; then echo "TRAIN_D_DONE $(date -u)" >> $LOG; exit 0; fi
done
echo "TRAIN_D_FAILED $(date -u)" >> $LOG
