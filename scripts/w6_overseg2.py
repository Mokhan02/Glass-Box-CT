"""[DESCRIPTIVE, requested by the researcher 2026-11-09, specified before it was run; no pass/fail criterion; no frozen file touched] Over-segmentation detail on the WAW-TACE predictions (all series, W4 route, class-26 mask `tumour`).
(1) Predicted connected components (26-connectivity) with NO overlap with any ground-truth tumour voxel and size >= 0.5 mL: count per series (median, mean) and % of series with at least one.
(2) False-positive volume (predicted voxels outside every GT tumour voxel) split by Euclidean distance (mm) to the nearest GT tumour voxel: 0-5, 5-10 and >10 mm (a series with no GT tumour: everything is >10 mm). Bands are (0,5], (5,10], (10,inf).
Usage: python scripts/w6_overseg2.py <out dir> <w4_waw root> NAME=<pred dir half0>,<pred dir half1> [...]"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

WAW, ARMS, MIN_ML = None, {}, 0.5


def bands(p, lab, sp):
    out = np.zeros(3)
    fp = p & ~lab
    if not fp.any():
        return out
    v = float(np.prod(sp)) / 1000.0
    if not lab.any():
        out[2] = fp.sum() * v; return out
    idx = np.argwhere(p); m = [int(np.ceil(10.0 / s)) + 1 for s in sp]
    lo = np.maximum(idx.min(0) - m, 0); hi = np.minimum(idx.max(0) + 1 + m, p.shape); sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    d = ndimage.distance_transform_edt(~lab[sl], sampling=sp); f = fp[sl]
    out[0] = (f & (d <= 5)).sum() * v; out[1] = (f & (d > 5) & (d <= 10)).sum() * v; out[2] = (f & (d > 10)).sum() * v
    return out


def one(job):
    half, case = job; g = os.path.join(WAW, half, "gt", case)
    lab = np.asarray(nib.load(os.path.join(g, "tumour_labels.nii.gz")).dataobj) > 0; sp = tuple(float(z) for z in nib.load(os.path.join(g, "ct.nii.gz")).header.get_zooms()[:3]); v = float(np.prod(sp)) / 1000.0; res = []
    for name, dirs in ARMS.items():
        f = os.path.join(dirs[int(half[-1])], case + ".npz")
        if not os.path.exists(f):
            continue
        p = np.load(f)["tumour"].astype(bool); n_fp = 0
        if p.any():
            L, n = ndimage.label(p, structure=np.ones((3, 3, 3)))
            if n:
                size = np.bincount(L.ravel(), minlength=n + 1)[1:]; hit = np.bincount(L[lab].ravel(), minlength=n + 1)[1:] > 0 if lab.any() else np.zeros(n, bool)
                n_fp = int(((size * v >= MIN_ML) & ~hit).sum())
        b = bands(p, lab, sp); res.append({"arm": name, "half": half, "case": case, "n_fp_components_ge_0p5ml": n_fp, "fp_ml_0_5mm": round(b[0], 3), "fp_ml_5_10mm": round(b[1], 3), "fp_ml_gt10mm": round(b[2], 3)})
    return res


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
    with open(os.path.join(out, "overseg2_per_series.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for n in ARMS:
        R = [r for r in rows if r["arm"] == n]; c = np.array([r["n_fp_components_ge_0p5ml"] for r in R]); B = {k: np.array([r[k] for r in R]) for k in ("fp_ml_0_5mm", "fp_ml_5_10mm", "fp_ml_gt10mm")}
        summ[n] = {"n_series": len(R), "components_median": float(np.median(c)), "components_mean": float(c.mean()), "pct_series_with_ge1": float(100 * (c >= 1).mean()),
                   **{k + "_median": float(np.median(x)) for k, x in B.items()}, **{k + "_mean": float(x.mean()) for k, x in B.items()}}
    json.dump(summ, open(os.path.join(out, "overseg2_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
