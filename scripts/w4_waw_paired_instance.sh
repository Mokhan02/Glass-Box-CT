#!/bin/bash
# W4 scouting: the paired-contrast cohort run (all patients with an arterial and a portal-venous series and a tumour mask on one of them) on a many-core instance, with the SAME hashed code as the Mac run
# (results/w4_waw/code_hashes_paired.txt). Fixed for reproducibility and recorded in run_meta.json: the registration sampling seed (the second argument of SetMetricSamplingPercentage in scripts/w4_waw_register.py, 1),
# the ITK thread count per worker (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2) and the number of workers (12; 12 x 2 = 24 of the 30 vCPUs). Output: $O (tumours_paired.csv, patient_qa.csv, summary.json, run.log, run_meta.json).
# Usage: bash scripts/w4_waw_paired_instance.sh   (run in tmux; the fetch of the data must have printed FETCH_DONE)
B=/lambda/nfs/CTGate/waw_tace; R=/lambda/nfs/CTGate/w1/repo; O=/lambda/nfs/CTGate/w4_waw/paired; PY=/lambda/nfs/CTGate/venv_w4/bin/python; WORKERS=12
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
cd $R || exit 2; mkdir -p $O
grep -q FETCH_DONE $B/fetch.log || { echo "FETCH_DONE missing" >> $O/run.log; exit 3; }
grep -v '^#' results/w4_waw/code_hashes_paired.txt | sha256sum -c --quiet - || { echo "code hashes differ" >> $O/run.log; exit 4; }
$PY - <<PYEOF
import json, os, platform, re, hashlib, time, numpy, scipy, SimpleITK
src = open("scripts/w4_waw_register.py").read()
json.dump({"started_utc": time.strftime("%FT%TZ", time.gmtime()), "python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__, "scipy": scipy.__version__, "SimpleITK": SimpleITK.Version_VersionString(),
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"], "workers": $WORKERS, "cpu_count": os.cpu_count(), "registration_sampling_calls": re.findall(r"SetMetricSamplingPercentage\([^)]*\)", src),
           "code_hashes_file_sha256": hashlib.sha256(open("results/w4_waw/code_hashes_paired.txt", "rb").read()).hexdigest(), "wrapper_sha256": hashlib.sha256(open("scripts/w4_waw_paired_instance.sh", "rb").read()).hexdigest()}, open("$O/run_meta.json", "w"), indent=1)
PYEOF
$PY -I scripts/w4_waw_paired_contrast.py $B/extracted/ct_scans $B/extracted/tumor_masks $B/extracted/organ_masks $B/raw/ct_hcc_metadata_v2.csv $O --workers $WORKERS > $O/run.log 2>&1
echo "PAIRED_DONE rc=$? $(date -u +%FT%TZ)" >> $O/run.log
