"""[DESCRIPTIVE, requested by the researcher 2026-11-09, specified before it was run; no pass/fail criterion] W6 over-segmentation check on the WAW-TACE predictions (all series, W4 route).
Per series and per arm: predicted tumour volume (mL), ground-truth tumour volume (mL), predicted voxels strictly outside every ground-truth tumour voxel (mL, "FP outside GT"), and the ratio predicted / GT volume (series with a ground-truth tumour only).
Summary per arm: median FP outside GT over all series, median ratio, median of the pooled ratio is NOT used. The 10 mm false-positive measure of the forgetting guard is computed separately by scripts/w6_forget_analyze*.py.
Usage: python scripts/w6_overseg.py <out dir> <w4_waw root> NAME=<pred dir half0>,<pred dir half1> [...]"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

WAW, ARMS = None, {}


def one(job):
    half, case = job; g = os.path.join(WAW, half, "gt", case)
    lab = np.asarray(nib.load(os.path.join(g, "tumour_labels.nii.gz")).dataobj) > 0; sp = nib.load(os.path.join(g, "ct.nii.gz")).header.get_zooms()[:3]; v = float(np.prod(sp)) / 1000.0; gt = float(lab.sum()) * v; out = []
    for name, dirs in ARMS.items():
        f = os.path.join(dirs[int(half[-1])], case + ".npz")
        if not os.path.exists(f):
            continue
        p = np.load(f)["tumour"].astype(bool); pv = float(p.sum()) * v; fp = float((p & ~lab).sum()) * v
        out.append({"arm": name, "half": half, "case": case, "pred_ml": round(pv, 3), "gt_ml": round(gt, 3), "fp_outside_gt_ml": round(fp, 3), "ratio": (round(pv / gt, 4) if gt > 0 else "")})
    return out


def main(argv):
    global WAW, ARMS
    out, WAW = argv[0], argv[1]; os.makedirs(out, exist_ok=True)
    for a in argv[2:]:
        n, p = a.split("=", 1); ARMS[n] = p.split(",")
    jobs = []
    for half in ("half0", "half1"):
        d = os.path.join(WAW, half, "gt"); jobs += [(half, c) for c in sorted(os.listdir(d)) if c[0].isdigit() and os.path.isdir(os.path.join(d, c))]
    with Pool(int(os.environ.get("W6_WORKERS", "6"))) as pool:
        rows = [r for rs in pool.imap(one, jobs, chunksize=4) for r in rs]
    with open(os.path.join(out, "overseg_per_series.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for n in ARMS:
        R = [r for r in rows if r["arm"] == n]; ra = [r["ratio"] for r in R if r["ratio"] != ""]
        summ[n] = {"n_series": len(R), "n_series_with_gt_tumour": len(ra), "median_fp_outside_gt_ml": float(np.median([r["fp_outside_gt_ml"] for r in R])), "mean_fp_outside_gt_ml": float(np.mean([r["fp_outside_gt_ml"] for r in R])),
                   "median_pred_over_gt_volume": float(np.median(ra)), "median_pred_ml": float(np.median([r["pred_ml"] for r in R]))}
    json.dump(summ, open(os.path.join(out, "overseg_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
