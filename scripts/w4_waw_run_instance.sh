#!/bin/bash
# W4 (WAW-TACE) driver for ONE half of the patients on ONE A10 (results/w4_waw/PREREG.md). Usage: bash scripts/w4_waw_run_instance.sh <0|1>, in tmux session w4drv<half>, from any directory (it cd's to the repo checkout on the instance).
# Both instances run the same hashed code on their own half (scripts/w4_waw_split.py; seed name w4_waw_instance_split). Steps: preflight (GPU idle, committed hashes, installed nnU-Net trainer files byte-equal to the committed ones, checkpoints present)
# -> ground truth of the half (scripts/w4_waw_paired_contrast.py --save-dir, same registration seed / threads / workers as the cohort run) and a check that its tumour table equals the committed host1 table -> 3-series smoke of every model, scored
# -> full predictions, model by model (vista, A, D0, B50, B25), each scored on completion with the alignment gate. The PREREG time rule (stop if a model's projected wall time exceeds 1.25 x its estimate from the measured per-scan times) is checked every 30 min.
# Any failure writes W4_DRIVER_HOLD to $WA/driver.log (the alert poller reports it) and stops; nothing is retried or changed automatically. Outputs under $WA: gt/, paired/, pred_<arm>/, score/ (scores_<arm>.csv, alignment_<arm>.json), driver.log.
H=$1; [ "$H" = 0 ] || [ "$H" = 1 ] || { echo "usage: bash scripts/w4_waw_run_instance.sh <0|1>"; exit 2; }
C=/lambda/nfs/CTGate; B=$C/waw_tace; R=$C/w1/repo; WA=$C/w4_waw/half$H; L=$WA/driver.log
PYW=$C/venv_w4/bin/python; PYV=$C/w1/venv_gpu/bin/python; PYA=$C/w3/venv_a/bin/python; PYN=$C/w3/venv_nnunet/bin/python
INF=$C/w1/vendor_inferer; VW=$C/glassbox/cache/weights/model.pt; AR=$C/w3/modelA/repo_A; AW=$C/w3/modelA/weights/swin_unetr.pth
mkdir -p $WA; log() { echo "$1 $(date -u +%FT%TZ)" >> $L; }
hold() { echo "W4_DRIVER_HOLD: half$H $1 $(date -u)" >> $L; exit 1; }
cd $R || hold "repo checkout missing"; log "W4_DRIVER_START half$H"
alive() { local p; for p in "$@"; do [ -d /proc/$p ] && ! grep -q '^State:.*Z' /proc/$p/status 2>/dev/null && return 0; done; return 1; }
# watch NAME EST_S_PER_SERIES TOTAL COUNTFN PIDS...: HOLD (and kill) if the projected wall time for the series still to do at the start exceeds 1.25 x the estimate
watch() {
  local name=$1 per=$2 total=$3 cnt=$4; shift 4
  local t0=$(date +%s) d0=$($cnt) lim nxt=0 d el dn proj
  lim=$(awk -v e=$per -v n=$total 'BEGIN{printf "%d", e*n*1.25}')
  while alive "$@"; do
    sleep 60; d=$($cnt); el=$(( $(date +%s) - t0 )); dn=$(( d - d0 ))
    if [ $dn -ge 5 ] && [ $el -ge $nxt ]; then
      proj=$(( el * (total - d0) / dn )); nxt=$(( el + 1800 ))
      log "TIME_CHECK $name done=$d/$total elapsed=${el}s projected_wall=${proj}s limit=${lim}s (estimate ${per}s x ${total} x 1.25)"
      if [ $proj -gt $lim ]; then kill "$@" 2>/dev/null; hold "TIME_RULE_EXCEEDED $name (projected ${proj}s > ${lim}s)"; fi
    fi
  done
}

# ---- preflight
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle at start"
grep -v '^#' results/w4_waw/code_hashes_w4.txt | sha256sum -c --quiet - >> $L 2>&1 || hold "code hashes differ from results/w4_waw/code_hashes_w4.txt"
grep -q FETCH_DONE $B/fetch.log || hold "FETCH_DONE missing"
[ -f $WA/patients.txt ] || { $PYW scripts/w4_waw_split.py results/w4_waw/paired_host1/patient_qa.csv $WA/split > /dev/null || hold "split failed"; cp $WA/split/patients_half$H.txt $WA/patients.txt; }
grep -q . $WA/patients.txt || hold "empty patient list"
source $C/w3/nnunet_env.sh
SP=$(dirname $($PYN -c "import nnunetv2.training.nnUNetTrainer as m; print(m.__file__)"))
[ "$(sha256sum $SP/nnUNetTrainerW5.py | cut -d' ' -f1)" = "$(sha256sum scripts/w5_nnunet_trainer.py | cut -d' ' -f1)" ] || hold "installed nnUNetTrainerW5.py differs from the committed scripts/w5_nnunet_trainer.py"
F3=$(grep -l "^class nnUNetTrainerW3Seed" $SP/*.py | head -1); [ -n "$F3" ] && [ "$(sha256sum $F3 | cut -d' ' -f1)" = "$(sha256sum scripts/w3_nnunet_trainer.py | cut -d' ' -f1)" ] || hold "installed nnUNetTrainerW3Seed file differs from (or is missing for) the committed scripts/w3_nnunet_trainer.py"
for TR in nnUNetTrainerW3Seed nnUNetTrainerW5Bright50 nnUNetTrainerW5Bright25; do
  [ -f $nnUNet_results/Dataset003_Liver/${TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth ] || hold "checkpoint missing for $TR"
done
[ -f $VW ] && [ -f $AW ] || hold "VISTA or Universal Model weights missing"
log "PREFLIGHT_PASS half$H ($(wc -l < $WA/patients.txt) patients)"

cases() { ls $WA/gt | grep -E '^[0-9]+_[0-9]+$' | sort -t_ -k1,1n -k2,2n; }
# ---- ground truth of this half (registration identical to the cohort run: seed 1 in the code, 2 ITK threads per worker, 12 workers)
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
if [ ! -f $WA/paired/summary.json ]; then
  $PYW -I scripts/w4_waw_paired_contrast.py $B/extracted/ct_scans $B/extracted/tumor_masks $B/extracted/organ_masks $B/raw/ct_hcc_metadata_v2.csv $WA/paired --workers 12 --save-dir $WA/gt --patients $(cat $WA/patients.txt | tr '\n' ' ') > $WA/gt_build.log 2>&1 || hold "ground-truth build failed (see gt_build.log)"
fi
$PYW - <<PYEOF || hold "ground-truth tumour table differs from the committed host1 table"
import csv
ref = {(r["patient"], r["rank"]): r for r in csv.DictReader(open("results/w4_waw/paired_host1/tumours_paired.csv"))}; new = list(csv.DictReader(open("$WA/paired/tumours_paired.csv"))); assert new
for r in new:
    k = (r["patient"], r["rank"]); assert k in ref and ref[k]["category"] == r["category"], k
    for c in ("c_arterial", "c_portal", "c_drawn", "c_other", "diam_mm"):
        a, b = ref[k][c], r[c]; assert (a == b) or abs(float(a) - float(b)) <= 0.01, (k, c, a, b)
print("ground truth table equals host1 on", len(new), "tumours")
PYEOF
ls -d $WA/gt/*_* | wc -l | grep -q -v '^0$' || hold "no ground-truth series written"
log "GT_BUILT_AND_MATCHES_HOST1 half$H ($(ls -d $WA/gt/*_* | wc -l) series)"
N=$(ls -d $WA/gt/*_* | wc -l); cases | head -3 | tr '\n' ' ' > $WA/smoke_cases.txt

# ---- one model: predict (resumable), then score
predict_vista() { cd $R && $PYV scripts/w4_waw_predict.py $INF $VW $WA/gt $WA/patients.txt $WA/pred_vista "$@" ; }
predict_A()     { cd $R && $PYA scripts/w4_waw_predict_A.py $AR $AW $WA/gt $WA/patients.txt $WA/pred_A "$@" ; }
predict_nn()    { # $1 arm, $2 trainer, [--limit n] : inputs _0000 symlinks for the series without an output yet
  local arm=$1 tr=$2 lim=${3:-100000}; mkdir -p $WA/pred_$arm
  mkdir -p $WA/in_nn_$arm; rm -f $WA/in_nn_$arm/*; cases | head -n $lim | while read c; do [ -f $WA/pred_$arm/$c.nii.gz ] || ln -s $WA/gt/$c/ct.nii.gz $WA/in_nn_$arm/${c}_0000.nii.gz; done
  [ -z "$(ls $WA/in_nn_$arm)" ] && return 0
  nnUNetv2_predict -i $WA/in_nn_$arm -o $WA/pred_$arm -d 3 -c 3d_fullres -tr $tr -f 0 -chk checkpoint_final.pth
}
score() { cd $R && $PYV scripts/w4_waw_score.py $WA/gt $WA/pred_$1 $1 $WA/score "${@:2}" ; }
nfiles() { ls $WA/pred_$1 2>/dev/null | grep -c -E '^[0-9]+_[0-9]+\.(npz|nii\.gz)$'; }

# ---- 3-series smoke of every model, scored (pipeline test only, never a result; its outputs are deleted)
SM=$(cat $WA/smoke_cases.txt); log "SMOKE_START"
predict_vista --limit 3 > $WA/smoke_vista.log 2>&1 || hold "smoke vista failed"; predict_A --limit 3 > $WA/smoke_A.log 2>&1 || hold "smoke A failed"
for a in D0:nnUNetTrainerW3Seed B50:nnUNetTrainerW5Bright50 B25:nnUNetTrainerW5Bright25; do predict_nn ${a%%:*} ${a##*:} 3 > $WA/smoke_${a%%:*}.log 2>&1 || hold "smoke ${a%%:*} failed"; done
for a in vista A D0 B50 B25; do score $a --cases $SM > $WA/smoke_score_$a.log 2>&1 || hold "smoke scoring $a failed (see smoke_score_$a.log)"; done
$PYV - <<PYEOF || hold "smoke outputs look wrong"
import csv
for a in ("vista", "A", "D0", "B50", "B25"):
    rows = list(csv.DictReader(open("$WA/score/scores_%s.csv" % a))); assert rows and any(r["detected"] != "" for r in rows), a
print("smoke ok")
PYEOF
for a in vista A D0 B50 B25; do rm -rf $WA/pred_$a $WA/score; done; rm -rf $WA/in_nn_*; mkdir -p $WA/score
log "SMOKE_PASS"

# ---- full predictions, model by model
run_model() { # arm est_s_per_series launcher...
  local arm=$1 per=$2; shift 2
  log "MODEL_START $arm"; "$@" > $WA/pred_$arm.log 2>&1 &
  local pid=$!; watch $arm $per $N "nfiles $arm" $pid; wait $pid || hold "prediction $arm failed (see pred_$arm.log)"
  [ "$(nfiles $arm)" -eq "$N" ] || hold "prediction $arm produced $(nfiles $arm) of $N series"
  score $arm > $WA/score_$arm.log 2>&1 || hold "scoring $arm failed (see score_$arm.log)"; log "MODEL_DONE $arm"
}
run_model vista 7 predict_vista
run_model A 28 predict_A
run_model D0 56 predict_nn D0 nnUNetTrainerW3Seed
run_model B50 56 predict_nn B50 nnUNetTrainerW5Bright50
run_model B25 56 predict_nn B25 nnUNetTrainerW5Bright25
log "W4_HALF${H}_DONE (all five models predicted and scored; GPU idle)"
