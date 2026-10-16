"""W1 reproduction test (PREREG): the shared contrast function must reproduce results/exp8_lesions/lesion_table.csv on the HCC-TACE-Seg cases.
Needs the converted HCC volumes (cache/heldout_nii/hcc_tace_seg/<case>/ct.nii.gz, seg.nii.gz; on the Lambda instance, not on the laptop).
Also writes hcc_lesions_variants.csv (portal vein put back into the reference) and counts class flips. Checks the 86 UID-route lesions (primary HCC reference): same (case, rank) set, diameter within 0.01 mm, |delta contrast_hu| <= 0.1 HU (PREREG) and reports the max (reviewer bar: < ~1 HU).
Usage: python scripts/w1_repro_hcc.py [--root cache/heldout_nii/hcc_tace_seg] [--route uid]"""
import argparse
import csv
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="cache/heldout_nii/hcc_tace_seg"); ap.add_argument("--route", default="uid")
ap.add_argument("--table", default="results/exp8_lesions/lesion_table.csv"); a = ap.parse_args()
ref = [r for r in csv.DictReader(open(a.table)) if r["route"] == a.route]
cases = sorted({r["case"] for r in ref}); got = {}; variant = {}
for case in cases:
    nii = nib.load(os.path.join(a.root, case, "ct.nii.gz")); sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(a.root, case, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1)); idx = np.flatnonzero(ann)          # exp8 works on annotated slices only
    sub = lambda m: m[:, :, idx]
    for r, v in zip(case_lesions(sub(ct), sub(seg == 2), sub(seg == 1), sp), case_lesions(sub(ct), sub(seg == 2), sub((seg == 1) | (seg == 3)), sp)):
        got[(case, r["rank"])] = r; variant[(case, r["rank"])] = v["contrast_hu"]      # variant: portal vein (label 3) put back into the reference
want = {(r["case"], int(r["rank"])): r for r in ref}
missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
dc = [abs(got[k]["contrast_hu"] - float(want[k]["contrast_hu"])) for k in want if k in got]
dd = [abs(got[k]["diam_mm"] - float(want[k]["diam_mm"])) for k in want if k in got]
print(f"reference lesions {len(want)}, reproduced {len(got)}, missing {len(missing)}, extra {len(extra)}")
print(f"max |delta contrast| {max(dc):.4f} HU, max |delta diameter| {max(dd):.4f} mm")
ok = not missing and not extra and max(dc) <= 0.1 and max(dd) <= 0.01
print("REPRODUCTION", "PASSED" if ok else "FAILED", "(strict 0.1 HU bar)" if ok else f"missing={missing[:5]} extra={extra[:5]}")
with open("results/w1_training_audit/hcc_lesions_variants.csv", "w", newline="") as f:      # primary (exp8 parity) and no_vessel_removal (DEVIATIONS #3)
    w = csv.writer(f); w.writerow(["case", "rank", "diam_mm", "contrast_hu", "contrast_hu_no_vessel_removal"])
    for k in sorted(got):
        w.writerow([k[0], k[1], f"{got[k]['diam_mm']:.3f}", f"{got[k]['contrast_hu']:.3f}", f"{variant[k]:.3f}"])
flips = sum((got[k]["contrast_hu"] > 0) != (variant[k] > 0) for k in got)
print(f"no_vessel_removal variant: {flips} of {len(got)} HCC lesions change bright/dark class")
sys.exit(0 if ok else 1)
