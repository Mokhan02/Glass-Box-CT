#!/bin/bash
# W5 Amendment 4 final CPU step (started by the BRIGHT-25 driver, tmux w5a4final): waits until BOTH new arms' post pipelines have finished (TRAIN_B25POST_DONE in train_B25.log and TRAIN_D0S2POST_DONE in train_D0S2.log; a W5_POST_HOLD in either stops it),
# then runs the dose-response analysis (scripts/w5a4_dose_response.py) over the five runs D0, D0-seed2, BRIGHT-25, BRIGHT-50, BRIGHT-50-s2. Writes TRAIN_A4FINAL_DONE to both logs, or W5_POST_HOLD on any failure. Output: /lambda/nfs/CTGate/w5a4/analysis.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; R=/lambda/nfs/CTGate/w1/repo; S1=/lambda/nfs/CTGate/w3/stage1; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; T=w3_lesions_D_nnunet_hcc.csv; O=/lambda/nfs/CTGate/w5a4
LB=$W/train_B25.log; LD=$W/train_D0S2.log
note() { echo "$1 $(date -u)" | tee -a $LB >> $LD; }
hold() { note "W5_POST_HOLD: A4 final analysis $1"; exit 1; }
mkdir -p $O; cd $R
until { grep -q "TRAIN_B25POST_DONE" $LB 2>/dev/null && grep -q "TRAIN_D0S2POST_DONE" $LD 2>/dev/null; } || grep -q "W5_POST_HOLD" $LB $LD 2>/dev/null; do sleep 60; done
{ grep -q "TRAIN_B25POST_DONE" $LB && grep -q "TRAIN_D0S2POST_DONE" $LD; } || hold "not run (a post pipeline held)"
$V scripts/w5a4_dose_response.py $O/analysis --arm D0=$S1/D0_74/$T --arm D0S2=$S1/W5_D0S2/eval/$T --arm B25=$S1/W5_B25/eval/$T --arm B50=$S1/W5_B50/eval/$T --arm B50S2=$S1/W5_B50S2/eval/$T \
  --gate D0S2=$S1/W5_D0S2/gate.json --gate B25=$S1/W5_B25/gate.json --gate B50=$S1/W5_B50/gate.json --gate B50S2=$S1/W5_B50S2/gate.json > $O/analysis.log 2>&1 || hold "dose-response analysis failed (see $O/analysis.log)"
note "TRAIN_A4FINAL_DONE"
