"""W3 Stage 1, model D (nnU-Net v2 Task03 fold 0, `nnUNetv2_predict` default settings) scored on the HCC UID cohort through the same lesion metrics and the same Amendment 2 alignment check as model A. The nnU-Net labels (1 liver, 2 tumour) are on the input grid;
tumour = label 2, predicted organ = label >= 1. Corrected cases (HCC_019_773068) are scored against the corrected label and, under arms `asconv__*`, the as-converted one. Writes w3_lesions_D_nnunet_hcc.csv and w3_alignment_D_hcc.json (status STOP if more than 2 cases fail the check).
Usage (cwd = repo checkout): python scripts/w3_modelD_eval.py <nnU-Net prediction dir> <outdir>"""
import csv
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

pdir, outdir = sys.argv[1:3]; os.makedirs(outdir, exist_ok=True)
table = L.lesion_table("hcc"); table_asconv = L.lesion_table("hcc", as_converted=True); corrected = L.corrected_cases(); cases = sorted({k[0] for k in table}); rows, aligns, failures, low = [], [], [], []
for c in cases:
    lab = np.asarray(nib.load(os.path.join(pdir, f"{c}.nii.gz")).dataobj); root = L.case_root("hcc", c, L.HCC_ROOT); in_shape = nib.load(os.path.join(root, c, "ct.nii.gz")).shape
    arms = {"D_nnunet": lab == 2}; r, geo = L.lesion_rows("hcc", c, root, table, arms); rows += r
    if c in corrected:
        rows += L.lesion_rows("hcc", c, L.HCC_ROOT, table_asconv, {"asconv__D_nnunet": lab == 2})[0]
    al = L.alignment_report(lab >= 1, geo["seg"], geo["ann"], geo["spacing"], in_shape); aligns.append({"case": c, **al}); failures += [] if al["pass"] else [c]; low += [c] if al["low_dice_lt_0.85"] else []
    print(c, al["organ_dice"], al["centroid_offset_mm"], "PASS" if al["pass"] else "FAIL", flush=True)
with open(os.path.join(outdir, "w3_lesions_D_nnunet_hcc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=L.FIELDS); w.writeheader(); w.writerows(rows)
json.dump({"model": "D_nnunet", "gate": "Amendment 2: shape equals input, no axis flip raises Dice, centroid offset <= %.0f mm" % L.ALIGN_MAX_OFFSET_MM, "cases": len(cases), "n_failures": len(failures), "failures": failures, "low_dice_lt_0.85": low,
           "status": "STOP" if len(failures) > L.ALIGN_MAX_FAILURES else "OK", "per_case": aligns}, open(os.path.join(outdir, "w3_alignment_D_hcc.json"), "w"), indent=1)
print(f"DONE D: {len(cases)} cases, {len(rows)} rows, alignment failures {failures}, organ Dice < 0.85: {low}")
