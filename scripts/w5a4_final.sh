#!/bin/bash
# W5 Amendment 4 final CPU step (started by the BRIGHT-25 driver, tmux w5a4final). D0-seed2 was NOT run (Amendment 4 note, 2026-11-02): by default it waits for BRIGHT-25 only (TRAIN_B25POST_DONE in train_B25.log; a W5_POST_HOLD stops it) and runs the dose-response analysis
# (scripts/w5a4_dose_response.py) over D0, BRIGHT-25, BRIGHT-50, BRIGHT-50-s2 (the p = 0 side is D0 alone). With W5A4_WITH_D0S2=1 it also waits for TRAIN_D0S2POST_DONE and includes D0-seed2 (the original two-arm plan). Writes TRAIN_A4FINAL_DONE to both logs, or W5_POST_HOLD on any failure. Output: /lambda/nfs/CTGate/w5a4/analysis.
source /lambda/nfs/CTGate/w3/nnunet_env.sh
W=/lambda/nfs/CTGate/w5; R=/lambda/nfs/CTGate/w1/repo; S1=/lambda/nfs/CTGate/w3/stage1; V=/lambda/nfs/CTGate/w3/venv_nnunet/bin/python; T=w3_lesions_D_nnunet_hcc.csv; O=/lambda/nfs/CTGate/w5a4
LB=$W/train_B25.log; LD=$W/train_D0S2.log; WITH=${W5A4_WITH_D0S2:-0}
note() { if [ "$WITH" = 1 ]; then echo "$1 $(date -u)" | tee -a $LB >> $LD; else echo "$1 $(date -u)" >> $LB; fi; }
hold() { note "W5_POST_HOLD: A4 final analysis $1"; exit 1; }
mkdir -p $O; cd $R
done_all() { grep -q "TRAIN_B25POST_DONE" $LB 2>/dev/null && { [ "$WITH" != 1 ] || grep -q "TRAIN_D0S2POST_DONE" $LD 2>/dev/null; }; }
until done_all || grep -q "W5_POST_HOLD" $LB $LD 2>/dev/null; do sleep 60; done
done_all || hold "not run (a post pipeline held)"
ARMS="--arm D0=$S1/D0_74/$T --arm B25=$S1/W5_B25/eval/$T --arm B50=$S1/W5_B50/eval/$T --arm B50S2=$S1/W5_B50S2/eval/$T"; GATES="--gate B25=$S1/W5_B25/gate.json --gate B50=$S1/W5_B50/gate.json --gate B50S2=$S1/W5_B50S2/gate.json"
[ "$WITH" = 1 ] && { ARMS="$ARMS --arm D0S2=$S1/W5_D0S2/eval/$T"; GATES="$GATES --gate D0S2=$S1/W5_D0S2/gate.json"; }
$V scripts/w5a4_dose_response.py $O/analysis $ARMS $GATES > $O/analysis.log 2>&1 || hold "dose-response analysis failed (see $O/analysis.log)"
note "TRAIN_A4FINAL_DONE"
