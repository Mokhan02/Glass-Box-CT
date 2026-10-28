#!/bin/bash
# W5 Amendment 3 driver (BRIGHT-50 mechanism, one A10; results/w5_training/PREREG.md, Amendment 3). Runs ON the instance from the repo checkout, in tmux session w5a3drv:
#   preflight (checkpoint, validation numbers, code hashes) -> hook check -> M3 smoke (3 cases) -> M3 full -> M3 validation + paired contrast (CPU)
#   -> token extraction (smoke 3 cases, then 2 shards) -> erasers -> M2 smoke (3 cases, own folder) -> M2 full (2 shards, merged) -> CPU analyses (M2, M3 projection, M2 by brightness).
# Every step stops at the first failure with A3_DRIVER_HOLD in driver.log (the alert poller reports it); nothing is retried or changed automatically. The PREREG time rule (stop if the projection
# exceeds 1.25 x the estimate) is applied DURING each long step from its real throughput (watch(): projected wall time after the first cases, re-checked every 30 min), not from the 3-case smoke,
# whose cases are not representative. Create $W/HOLD_M2 to stop before the M2 steps.
# The unhooked-output reference is the W5 BRIGHT-50 lesion table (W5A3_STAGE1_D_TABLE) and the trainer class is nnUNetTrainerW5Bright50 (W5A3_D_TRAINER); both are read only by the Stage 2 scripts' overrides.
W=/lambda/nfs/CTGate/w5a3; R=/lambda/nfs/CTGate/w1/repo; W3=/lambda/nfs/CTGate/w3; L=$W/driver.log
MF=$W3/nnunet/results/Dataset003_Liver/nnUNetTrainerW5Bright50__nnUNetPlans__3d_fullres
VG=/lambda/nfs/CTGate/w1/venv_gpu/bin/python
CKPT_SHA=0783e42fe5f3bbbe070cfa32e1715e88f6a37e1268ff3d73546c9d8b820621c4
C74=results/w3_models/stage2/m1/cases_primary74.txt; C75=results/w3_models/stage2/m1/cases_with_027_75.txt
SMOKE_M3="HCC_017_372380 HCC_023_800523 HCC_052_425020"     # the BRIGHT-50 Arm B lesions (HCC_017, HCC_023) plus one Arm A case, so the smoke check sees both arms
EST_M3=4.0; EST_EXT=2.0; EST_M2=6.5                          # hours; corrected before launch from D0's logged shard timings (PREREG, Amendment 3, time correction)
mkdir -p $W; cd $R || exit 2
log() { echo "$1 $(date -u +%FT%TZ)" >> $L; }
hold() { log "A3_DRIVER_HOLD: $1"; exit 1; }
trap 'rc=$?; log "DRIVER_PROC_EXIT rc=$rc"' EXIT
source $W3/nnunet_env.sh
export PYTHONPATH=. W5A3_D_TRAINER=nnUNetTrainerW5Bright50 W5A3_STAGE1_D_TABLE=results/w5_training/results/BRIGHT50/w3_lesions_D_nnunet_hcc.csv
log "A3_DRIVER_START"

alive() { local p; for p in "$@"; do [ -d /proc/$p ] && ! grep -q '^State:.*Z' /proc/$p/status 2>/dev/null && return 0; done; return 1; }
# watch NAME EST_H TOTAL MINDONE COUNTFN PIDS...: until the processes end, re-check every 30 min the projected wall time for the cases still to do at the start; HOLD (and kill them) if > 1.25 x EST_H
watch() {
  local name=$1 est=$2 total=$3 mind=$4 cnt=$5; shift 5
  local t0=$(date +%s) d0=$($cnt) lim=$(awk -v e=$est 'BEGIN{printf "%d", e*1.25*3600}') nxt=0 d el dn proj
  while alive "$@"; do
    sleep 60; d=$($cnt); el=$(( $(date +%s) - t0 )); dn=$(( d - d0 ))
    if [ $dn -ge $mind ] && [ $el -ge $nxt ]; then
      proj=$(( el * (total - d0) / dn )); nxt=$(( el + 1800 ))
      log "TIME_CHECK $name done=$d/$total elapsed=${el}s projected_wall=${proj}s limit=${lim}s (estimate ${est}h x 1.25)"
      if [ $proj -gt $lim ]; then kill "$@" 2>/dev/null; hold "TIME_RULE_EXCEEDED $name (projected ${proj}s > ${lim}s)"; fi
    fi
  done
}
n_m3() { wc -l < $W/m3_B50/done_cases.txt 2>/dev/null || echo 0; }
n_ext() { ls $W/m1_B50/tokens/*.npz 2>/dev/null | wc -l; }
n_m2() { cat $W/m2_B50_s0/done_cases.txt $W/m2_B50_s1/done_cases.txt 2>/dev/null | wc -l; }

# ---- preflight
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || hold "GPU not idle at start"
[ "$(sha256sum $MF/fold_0/checkpoint_final.pth | cut -d' ' -f1)" = "$CKPT_SHA" ] || hold "checkpoint_final.pth sha256 differs from the recorded one"
python - <<PYEOF || hold "validation summary does not match the logged sanity gate"
import json
m = json.load(open("$MF/fold_0/validation/summary.json"))["mean"]; g = json.load(open("results/w5_training/results/BRIGHT50/gate.json"))
assert abs(m["1"]["Dice"] - g["liver_dice"]) < 1e-9 and abs(m["2"]["Dice"] - g["tumour_dice"]) < 1e-9, (m["1"]["Dice"], m["2"]["Dice"])
PYEOF
grep -v '^#' results/w5_training/code_hashes_a3.txt | sha256sum -c --quiet - >> $L 2>&1 || hold "code hashes differ from results/w5_training/code_hashes_a3.txt"
log "PREFLIGHT_PASS (checkpoint sha256 ok, validation Dice equals the logged gate, code hashes ok)"

# ---- hook check (real cases; propagation to both consumers; unhooked output reproduces the W5 BRIGHT-50 table on 3 cases)
python scripts/w3s2_hook_check.py D $W/hook_check_B50 $MF > $W/hook_check_B50.log 2>&1 || hold "hook check FAILED (see hook_check_B50.log)"
log "HOOK_CHECK_PASS"

# ---- M3
if [ ! -f $W/m3_B50/m3_lesions.csv ]; then
  python scripts/w3s2_m3_run.py D $W/m3_B50 $MF --cases $SMOKE_M3 > $W/m3_smoke.log 2>&1 || hold "M3 smoke run failed (see m3_smoke.log)"
fi
if [ "$(n_m3)" -le 3 ]; then     # the smoke check applies to the 3-case smoke only (a restart after the full run skips it)
  $VG scripts/w3s2_m3_smoke_check.py D $W/m3_B50 $W/m3_smoke_check.json >> $L 2>&1 || hold "M3 smoke check FAILED (see m3_smoke_check.json)"
  log "M3_SMOKE_PASS"
fi
log "M3_FULL_START"
python scripts/w3s2_m3_run.py D $W/m3_B50 $MF >> $W/m3_full.log 2>&1 &
P=$!; watch M3 $EST_M3 75 15 n_m3 $P; wait $P || hold "M3 full run failed (see m3_full.log; the runner resumes per case)"
$VG scripts/w3s2_m3_validate.py $C75 $W/m3_B50_validate.json $W/m3_B50 >> $L 2>&1 || hold "M3 validation found problems (see m3_B50_validate.json)"
log "M3_DONE (validated)"
$VG scripts/w5a3_m3_contrast.py results/w3_models/stage2/m3/D/m3_lesions.csv $W/m3_B50/m3_lesions.csv $C74 $W/a3_m3_contrast >> $W/a3_m3_contrast.log 2>&1 || hold "M3 contrast failed (see a3_m3_contrast.log)"
log "M3_CONTRAST_DONE"

# ---- token extraction (to fit the erasers and the probe direction; M1 probes are NOT analysed here)
[ "$(n_ext)" -gt 0 ] || { python scripts/w3s2_m1_extract.py D $W/m1_B50 $MF --limit 3 > $W/m1_smoke.log 2>&1 || hold "extraction smoke failed (see m1_smoke.log)"; }
sort $C75 | awk 'NR%2==1' > $W/cases_even.txt; sort $C75 | awk 'NR%2==0' > $W/cases_odd.txt
log "EXTRACT_START"
python scripts/w3s2_m1_extract.py D $W/m1_B50 $MF --cases $(cat $W/cases_even.txt) >> $W/m1_s0.log 2>&1 &
P0=$!
python scripts/w3s2_m1_extract.py D $W/m1_B50 $MF --cases $(cat $W/cases_odd.txt) >> $W/m1_s1.log 2>&1 &
P1=$!
watch EXTRACTION $EST_EXT 75 10 n_ext $P0 $P1
wait $P0; R0=$?; wait $P1; R1=$?
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] || hold "extraction shard failed (exit $R0 / $R1; see m1_s0.log, m1_s1.log; per-case outputs are kept, re-run resumes)"
[ "$(n_ext)" -ge 75 ] || hold "token files missing after extraction ($(n_ext) of 75)"
log "EXTRACT_DONE (75 token files)"
python scripts/w3s2_m2_fit.py $W/m1_B50 $C74 $W/erasers_B50.npz > $W/erasers_B50.log 2>&1 || hold "eraser fit failed (see erasers_B50.log)"
log "ERASERS_FITTED"

# ---- M2
[ ! -f $W/HOLD_M2 ] || hold "HOLD_M2 present: stopped before M2 as requested"
if [ ! -f $W/m2_smoke_B50/determinism.json ]; then
  python scripts/w3s2_m2_run.py D $W/m2_smoke_B50 $W/erasers_B50.npz $MF --limit 3 > $W/m2_smoke.log 2>&1 || hold "M2 smoke run failed (see m2_smoke.log)"
fi
python - <<PYEOF || hold "M2 smoke determinism check failed (base must reproduce the BRIGHT-50 table within 1e-3 with no flag mismatch; identity must equal base)"
import json
s = json.load(open("$W/m2_smoke_B50/determinism.json"))["summary"]; print(s)
assert s["max_base_vs_stage1_cov_diff"] is not None and s["max_base_vs_stage1_cov_diff"] <= 1e-3 and s["detection_flag_mismatches"] == 0 and (s["max_identity_vs_base_cov_diff"] or 0) == 0
PYEOF
log "M2_SMOKE_PASS (smoke folder is separate and not part of the results)"
log "M2_FULL_START"
python scripts/w3s2_m2_run.py D $W/m2_B50_s0 $W/erasers_B50.npz $MF --shard 0/2 >> $W/m2_s0.log 2>&1 &
P0=$!
python scripts/w3s2_m2_run.py D $W/m2_B50_s1 $W/erasers_B50.npz $MF --shard 1/2 >> $W/m2_s1.log 2>&1 &
P1=$!
watch M2 $EST_M2 75 8 n_m2 $P0 $P1
wait $P0; R0=$?; wait $P1; R1=$?
[ $R0 -eq 0 ] && [ $R1 -eq 0 ] || hold "M2 shard failed (exit $R0 / $R1; see m2_s0.log, m2_s1.log; the runner resumes per case)"
python scripts/w5a3_merge_m2.py $W/m2_B50 $W/m2_B50_s0 $W/m2_B50_s1 > $W/m2_merge.log 2>&1 || hold "M2 shard merge failed (see m2_merge.log)"
python - <<PYEOF || hold "M2 full-run determinism or completeness check failed (see m2_B50/determinism.json)"
import csv, json
d = json.load(open("$W/m2_B50/determinism.json"))["summary"]; n = len({r["case"] for r in csv.DictReader(open("$W/m2_B50/cases.csv"))}); print(d, n)
assert n == 75 and d["max_base_vs_stage1_cov_diff"] <= 1e-3 and d["detection_flag_mismatches"] == 0 and (d["max_identity_vs_base_cov_diff"] or 0) == 0
PYEOF
log "M2_DONE (75 cases merged and checked)"

# ---- CPU analyses (committed Stage 2 analyses, unchanged, plus the exploratory brightness split)
$VG scripts/w3s2_m2_analyze.py D $W/m2_B50 $W/m1_B50 $W/erasers_B50.npz $C74 $W/a3_m2_analysis > $W/a3_m2_analysis.log 2>&1 || hold "M2 analysis failed (see a3_m2_analysis.log)"
$VG scripts/w3s2_m3_analyze.py D $W/m3_B50 $W/m1_B50 $C74 $W/a3_m3_analysis > $W/a3_m3_analysis.log 2>&1 || hold "M3 projection analysis failed (see a3_m3_analysis.log)"
$VG scripts/w5a3_m2_by_brightness.py $W/m2_B50 $C74 $W/a3_m2_by_brightness > $W/a3_m2_by_brightness.log 2>&1 || hold "M2 by-brightness analysis failed (see a3_m2_by_brightness.log)"
log "A3_ALL_DONE"
