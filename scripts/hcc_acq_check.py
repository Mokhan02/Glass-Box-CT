"""Count distinct acquisitions (AcquisitionNumber) in each labelled HCC series, and how irregular the slice spacing is when all files are sorted by position.
Usage: python scripts/hcc_acq_check.py cache/hcc_acq_check.csv"""
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
rows = []
for case in sorted(os.listdir(root)):
    if not os.path.exists(os.path.join(root, case, ".done")):
        continue
    meta = json.load(open(os.path.join(root, case, "segments.json"))); pid = re.match(r"^(HCC_\d+)", case).group(1)
    d = glob.glob(f"{held}/{pid}/*/{meta['ct_series']}")[0]
    hs = [pydicom.dcmread(f, stop_before_pixels=True) for f in sorted(glob.glob(d + "/*.dcm"))]
    n = np.cross(*np.array(hs[0].ImageOrientationPatient, float).reshape(2, 3))
    z = np.sort([float(np.dot(np.array(h.ImagePositionPatient, float), n)) for h in hs]); dz = np.diff(z)
    acq = collections.Counter(str(h.get("AcquisitionNumber", "")) for h in hs)
    xy = {tuple(np.round(np.array(h.ImagePositionPatient, float)[:2], 1)) for h in hs}
    rows.append({"case": case, "route": meta.get("label_method", "uid"), "n_files": len(hs), "n_acq_numbers": len(acq), "acq_counts": "/".join(str(v) for v in acq.values()),
                 "n_unique_xy_origins": len(xy), "dz_median": round(float(np.median(dz)), 2), "dz_irregular_frac": round(float((np.abs(dz - np.median(dz)) > 0.2 * max(np.median(dz), 0.1)).mean()), 3)})
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("distinct AcquisitionNumber per labelled series:", dict(sorted(collections.Counter(r["n_acq_numbers"] for r in rows).items())))
for rt in ("uid", "geometry"):
    rr = [r for r in rows if r["route"] == rt]
    print(f"  {rt:8s}: {len(rr)} cases; with >1 acquisition: {sum(r['n_acq_numbers'] > 1 for r in rr)}; with irregular slice spacing (>5% of gaps off by >20%): {sum(r['dz_irregular_frac'] > 0.05 for r in rr)}")
print("series with >1 acquisition that do NOT have duplicate positions:", sum(1 for r in rows if r["n_acq_numbers"] > 1) - 13)
