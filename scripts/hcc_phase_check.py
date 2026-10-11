"""Do any labelled HCC-TACE-Seg CT series contain more than one acquisition (phase) at the same slice positions?
My converter (scripts/convert_heldout.py) sorts ALL files of a series by position into one volume, so a series that mixes phases would be
interleaved. For each converted case: files, unique slice positions, duplicates, distinct acquisition times, series description, and how the
SEG's labelled slices relate. Reads DICOM headers only. Usage: python scripts/hcc_phase_check.py [out.csv]"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pydicom

root, held = "cache/heldout_nii/hcc_tace_seg", "cache/heldout/hcc_tace_seg"
ev = {(r["case"], r["target"]): r for r in csv.DictReader(open("cache/heldout_eval/hcc_tace_seg.csv"))}
rows = []
for case in sorted(os.listdir(root)):
    if not os.path.exists(os.path.join(root, case, ".done")):
        continue
    meta = json.load(open(os.path.join(root, case, "segments.json"))); pid = re.match(r"^(HCC_\d+)", case).group(1)
    d = glob.glob(f"{held}/{pid}/*/{meta['ct_series']}")[0]
    hs = [pydicom.dcmread(f, stop_before_pixels=True) for f in sorted(glob.glob(d + "/*.dcm"))]
    n = np.cross(*np.array(hs[0].ImageOrientationPatient, float).reshape(2, 3))
    z = np.array([float(np.dot(np.array(h.ImagePositionPatient, float), n)) for h in hs]); zr = np.round(z, 1)
    acq = collections.Counter(str(h.get("AcquisitionTime", "")).split(".")[0] for h in hs)
    uz = len(set(zr)); sd = {str(h.get("SeriesDescription", "")) for h in hs}
    rows.append({"case": case, "route": meta.get("label_method", "uid"), "n_files": len(hs), "n_unique_z": uz, "n_duplicate_slices": len(hs) - uz,
                 "n_acq_times": len(acq), "acq_counts": "/".join(str(v) for v in acq.values()), "series_desc": "|".join(sorted(sd))[:50],
                 "n_slices_in_nii": meta.get("n_slices"), "frames_matched": meta.get("frames_matched"),
                 "tumour_dice": float(ev[(case, "tumor")]["dice"]) if (case, "tumor") in ev else np.nan, "liver_dice": float(ev[(case, "liver")]["dice"]) if (case, "liver") in ev else np.nan})
with open(sys.argv[1] if len(sys.argv) > 1 else "cache/hcc_phase_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
mixed = [r for r in rows if r["n_duplicate_slices"] > 0]
print(f"{len(rows)} cases; series with duplicate slice positions (more than one acquisition at the same z): {len(mixed)}")
for rt in ("uid", "geometry"):
    rr = [r for r in rows if r["route"] == rt]; mm = [r for r in rr if r["n_duplicate_slices"] > 0]
    print(f"  route {rt:8s}: {len(rr)} cases, {len(mm)} with duplicates; mean tumour Dice dup {np.mean([r['tumour_dice'] for r in mm]) if mm else float('nan'):.3f} vs no-dup {np.mean([r['tumour_dice'] for r in rr if r['n_duplicate_slices'] == 0]):.3f}")
print(f"all: tumour Dice with duplicates {np.mean([r['tumour_dice'] for r in mixed]) if mixed else float('nan'):.3f} (n={len(mixed)}) vs without {np.mean([r['tumour_dice'] for r in rows if r['n_duplicate_slices'] == 0]):.3f}; liver Dice {np.mean([r['liver_dice'] for r in mixed]) if mixed else float('nan'):.3f} vs {np.mean([r['liver_dice'] for r in rows if r['n_duplicate_slices'] == 0]):.3f}")
print("distinct acquisition times per series:", dict(collections.Counter(r["n_acq_times"] for r in rows)))
print("examples with duplicates:", [(r["case"][:11], r["n_files"], r["n_unique_z"], r["acq_counts"], r["route"]) for r in mixed[:6]])
