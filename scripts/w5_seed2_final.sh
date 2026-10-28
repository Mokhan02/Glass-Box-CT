#!/bin/bash
# W5 Amendment 2 final CPU step: when the BRIGHT-50-s2 post pipeline has finished, re-apply the unchanged W5 verdict rule (BRIGHT-50-s2 against D0 and SHAM-50, the BRIGHT arm passed to w5_analyze.py as B50) and write the replication report.
# Writes TRAIN_B50S2FINAL_DONE to the arm's training log, or W5_POST_HOLD on any failure. Outputs: /lambda/nfs/CTGate/w5/seed2_analysis and seed2_replication.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; R=/lambda/nfs/CTGate/w1/repo; S1=/lambda/nfs/CTGate/w3/stage1; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; LOG=$W/train_B50S2.log; T=w3_lesions_D_nnunet_hcc.csv
hold() { echo "W5_POST_HOLD: B50S2 final analysis $1 $(date -u)" >> $LOG; exit 1; }
cd $R
until grep -q -E "TRAIN_B50S2POST_DONE|W5_POST_HOLD" $LOG; do sleep 60; done
grep -q "TRAIN_B50S2POST_DONE" $LOG || hold "not run (post pipeline held)"
$V scripts/w5_analyze.py $W/seed2_analysis --arm D0=$S1/D0_74/$T --arm B50=$S1/W5_B50S2/eval/$T --arm S50=$S1/W5_SHAM50/eval/$T --gate B50=$S1/W5_B50S2/gate.json --gate S50=$S1/W5_SHAM50/gate.json > $W/seed2_analysis.log 2>&1 || hold "contrast analysis failed"
$V scripts/w5_replication_report.py $W/w5_results_b50_original.json $W/seed2_analysis/w5_results.json $W/seed2_replication >> $W/seed2_analysis.log 2>&1 || hold "replication report failed"
echo "TRAIN_B50S2FINAL_DONE $(date -u)" >> $LOG
