"""Pre-specified within-patient test (results/exp8_lesions/PHASE_PAIRS_PREREG.md). For each candidate acquisition volume of the 11 ambiguous cases:
signed tumour-liver contrast (mean HU of GT tumour minus mean HU of non-tumour GT liver, annotated slices) and tumour Dice (from the clean evaluation).
Sign test over cases whose candidate acquisitions differ in tumour Dice by >= 0.10: does the highest-contrast acquisition have the LOWER Dice?
Usage (instance): python scripts/exp8_phase_pairs.py cache/heldout_eval_clean/hcc_tace_seg.csv cache/phase_pairs.csv"""
import collections
import csv
import os
import re
import sys

import nibabel as nib
import numpy as np
from scipy.stats import binomtest

root = "cache/heldout_nii_clean/hcc_tace_seg"
dice = {r["case"]: float(r["dice"]) for r in csv.DictReader(open(sys.argv[1])) if r["target"] == "tumor"}
rows = []
for case in sorted(os.listdir(root)):
    if not re.search(r"_acq\w+$", case):
        continue
    ct = np.asarray(nib.load(os.path.join(root, case, "ct.nii.gz")).dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1)); sl = ann[None, None, :]
    t, l, a = ct[(seg == 2) & sl], ct[(seg == 1) & sl], ct[(seg == 4) & sl]
    rows.append({"case": case, "base": re.sub(r"_acq\w+$", "", case), "acq": case.split("_acq")[-1], "tumour_hu": float(t.mean()) if len(t) else np.nan, "liver_hu": float(l.mean()) if len(l) else np.nan,
                 "contrast": float(t.mean() - l.mean()) if len(t) and len(l) else np.nan, "aorta_hu": float(a.mean()) if len(a) > 100 else np.nan, "dice": dice.get(case, np.nan)})
with open(sys.argv[2], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by = collections.defaultdict(list)
for r in rows:
    by[r["base"]].append(r)
agree = total = 0; print(f"{'case':14s} acquisitions: acq | signed contrast HU | aorta HU | tumour Dice")
for b, rs in sorted(by.items()):
    d = [r["dice"] for r in rs]; used = max(d) - min(d) >= 0.10
    print(f"{b[:11]:12s} " + " | ".join(f"acq{r['acq']}: {r['contrast']:+6.1f} HU, aorta {r['aorta_hu']:5.0f}, Dice {r['dice']:.2f}" for r in rs) + ("   <- used (Dice differs >= 0.10)" if used else "   (Dice differs < 0.10; not used)"))
    if used:
        hi = max(rs, key=lambda r: r["contrast"]); lo = min(rs, key=lambda r: r["contrast"]); total += 1; agree += int(hi["dice"] < lo["dice"])
p = binomtest(agree, total, 0.5).pvalue if total else float("nan")
print(f"\nSIGN TEST: higher-contrast acquisition has the lower tumour Dice in {agree} of {total} qualifying cases; two-sided exact binomial p = {p:.3f}")
