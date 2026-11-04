#!/bin/bash
# W4 (WAW-TACE), DEVIATIONS 23: re-predict the one non-LAS series (540_1, half 1) with D0 and BRIGHT-50 on its LAS-reoriented CT, restore the labels to the original grid, keep the original predictions and scores (sensitivity), re-score both arms of half 1.
# Waits for half 0's driver to finish (the GPU is in use until then). Usage (on the instance, in tmux): bash scripts/w4_waw_reorient_run.sh. Log: $C/w4_waw/reorient.log. Stops at the first failure (REORIENT_HOLD).
C=/lambda/nfs/CTGate; R=$C/w1/repo; H=$C/w4_waw/half1; W=$C/w4_waw/reorient_half1; L=$C/w4_waw/reorient.log; PYV=$C/w1/venv_gpu/bin/python
log() { echo "$1 $(date -u +%FT%TZ)" >> $L; }; hold() { echo "REORIENT_HOLD: $1 $(date -u)" >> $L; exit 1; }
cd $R || hold "repo missing"; log "REORIENT_START (waiting for half 0)"
until grep -qE "W4_HALF0_DONE|W4_DRIVER_HOLD: half0" $C/w4_waw/half0/driver.log && ! tmux has-session -t w4drv0 2>/dev/null; do sleep 30; done
grep -q W4_HALF0_DONE $C/w4_waw/half0/driver.log || hold "half 0 did not finish cleanly"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle"
grep -v '^#' results/w4_waw/code_hashes_w4.txt | sha256sum -c --quiet - >> $L 2>&1 || hold "code hashes differ"
source $C/w3/nnunet_env.sh
$PYV scripts/w4_waw_reorient.py prepare $H/gt $H $W >> $L 2>&1 || hold "prepare failed"
[ -s $W/cases.txt ] || hold "no non-LAS series found"
mkdir -p $H/score_original_orientation; cp -p $H/score/scores_D0.csv $H/score/scores_B50.csv $H/score/alignment_D0.json $H/score/alignment_B50.json $H/score_original_orientation/ || hold "could not keep the original scores"
for pair in D0:nnUNetTrainerW3Seed B50:nnUNetTrainerW5Bright50; do
  arm=${pair%%:*}; tr=${pair##*:}; mkdir -p $W/pred_$arm
  nnUNetv2_predict -i $W/in -o $W/pred_$arm -d 3 -c 3d_fullres -tr $tr -f 0 -chk checkpoint_final.pth > $W/pred_$arm.log 2>&1 || hold "prediction $arm failed (see $W/pred_$arm.log)"
  $PYV scripts/w4_waw_reorient.py restore $H/gt $W $arm $H >> $L 2>&1 || hold "restore $arm failed"
  log "REORIENT_PREDICTED $arm"
done
for arm in D0 B50; do $PYV scripts/w4_waw_score.py $H/gt $H/pred_$arm $arm $H/score --workers 12 >> $L 2>&1 || hold "re-scoring $arm failed"; done
log "REORIENT_DONE (D0 and B50 of half 1 re-scored; originals in score_original_orientation/ and pred_<arm>_original/)"
