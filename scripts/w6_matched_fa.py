"""[EXPLORATORY, descriptive, requested by the researcher 2026-11-09, specified before it was run; no pass/fail criterion; no frozen file touched] Matched false-alarm comparison from MASKS (the W4-route predictions keep only boolean masks, no probability maps).
Reference: the baseline's false-alarm lesion rate = share of WAW-TACE series (all 396) with at least one predicted class-26 connected component (26-connectivity) of >= 0.5 mL that has no overlap with any ground-truth tumour voxel.
For each other arm the minimum component size s (mL) is chosen blind to detection: the smallest s such that the share of series with a no-overlap component LARGER than s is at most the baseline's share. Predictions are then filtered (every component of size <= s removed, with or without overlap) and rescored with the unchanged W4 scorer (scripts/w4_waw_score.py), then detection by bright/dark is tabulated (scripts/w6_detection_table.py).
Usage: python scripts/w6_matched_fa.py sizes <out dir> <w4_waw root> NAME=<dir0>,<dir1> [...]   -> fp_component_sizes.csv, matched_s.json
       python scripts/w6_matched_fa.py filter <matched_s.json> <out root> NAME=<dir0>,<dir1> [...]   -> <out root>/<NAME>/pred_half{0,1} (filtered npz copies)"""
import csv
import json
import os
import shutil
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

WAW, ARMS, S = None, {}, {}
ST = np.ones((3, 3, 3))


def sizes(job):
    half, case = job; g = os.path.join(WAW, half, "gt", case)
    lab = np.asarray(nib.load(os.path.join(g, "tumour_labels.nii.gz")).dataobj) > 0; v = float(np.prod(nib.load(os.path.join(g, "ct.nii.gz")).header.get_zooms()[:3])) / 1000.0; out = []
    for name, dirs in ARMS.items():
        f = os.path.join(dirs[int(half[-1])], case + ".npz")
        if not os.path.exists(f):
            continue
        p = np.load(f)["tumour"].astype(bool); fp = []
        if p.any():
            L, n = ndimage.label(p, structure=ST)
            if n:
                size = np.bincount(L.ravel(), minlength=n + 1)[1:]; hit = (np.bincount(L[lab].ravel(), minlength=n + 1)[1:] > 0) if lab.any() else np.zeros(n, bool); fp = [float(x) * v for x in size[~hit]]
        out.append({"arm": name, "half": half, "case": case, "max_fp_component_ml": round(max(fp), 4) if fp else 0.0, "n_fp_ge_0p5": int(sum(x >= 0.5 for x in fp))})
    return out


def filt(job):
    name, half, case = job; f = os.path.join(ARMS[name][int(half[-1])], case + ".npz"); z = np.load(f); p = z["tumour"].astype(bool); g = os.path.join(WAW, half, "gt", case)
    v = float(np.prod(nib.load(os.path.join(g, "ct.nii.gz")).header.get_zooms()[:3])) / 1000.0
    if p.any():
        L, n = ndimage.label(p, structure=ST)
        if n:
            size = np.bincount(L.ravel(), minlength=n + 1) * v; keep = size > S[name]; keep[0] = False; p = keep[L]
    np.savez_compressed(os.path.join(OUT, name, f"pred_{half}", case + ".npz"), **{**{k: z[k] for k in z.files if k != "tumour"}, "tumour": p})
    return name, half, case


def cases(waw):
    jobs = []
    for half in ("half0", "half1"):
        d = os.path.join(waw, half, "gt"); jobs += [(half, c) for c in sorted(os.listdir(d)) if c[0].isdigit() and os.path.isdir(os.path.join(d, c))]
    return jobs


def parse(args):
    for a in args:
        n, p = a.split("=", 1); ARMS[n] = p.split(",")


def main(argv):
    global WAW, OUT, S
    mode = argv[0]
    if mode == "sizes":
        out, WAW = argv[1], argv[2]; parse(argv[3:]); os.makedirs(out, exist_ok=True)
        with Pool(int(os.environ.get("W6_WORKERS", "6"))) as pool:
            rows = [r for rs in pool.imap(sizes, cases(WAW), chunksize=4) for r in rs]
        with open(os.path.join(out, "fp_component_sizes.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        base = [r for r in rows if r["arm"] == "BASELINE"]; K = int(sum(r["n_fp_ge_0p5"] >= 1 for r in base)); res = {"waw": WAW, "baseline_series": len(base), "baseline_series_with_fp_component_ge_0p5ml": K, "baseline_rate_pct": round(100 * K / len(base), 2), "arms": {}}
        for n in ARMS:
            m = sorted((r["max_fp_component_ml"] for r in rows if r["arm"] == n), reverse=True)
            s = 0.0 if n == "BASELINE" else (m[K] if len(m) > K else 0.0)      # keep components LARGER than s: at most K series keep one
            res["arms"][n] = {"s_ml": s, "series_with_fp_component_gt_s": int(sum(x > s for x in m)), "series_with_fp_component_ge_0p5ml_unfiltered": int(sum(r["n_fp_ge_0p5"] >= 1 for r in rows if r["arm"] == n))}
        json.dump(res, open(os.path.join(out, "matched_s.json"), "w"), indent=1); print(json.dumps(res, indent=1))
    else:
        spec, root = argv[1], argv[2]; WAW = json.load(open(spec))["waw"]; OUT = root; S = {n: v["s_ml"] for n, v in json.load(open(spec))["arms"].items()}; parse(argv[3:])
        jobs = []
        for n in ARMS:
            for h in ("half0", "half1"):
                os.makedirs(os.path.join(root, n, f"pred_{h}"), exist_ok=True)
                for c in sorted(os.listdir(ARMS[n][int(h[-1])])):
                    if c.endswith(".npz") and c[0].isdigit():
                        jobs.append((n, h, c[:-4]))
        with Pool(int(os.environ.get("W6_WORKERS", "6"))) as pool:
            for _ in pool.imap(filt, jobs, chunksize=4):
                pass
        print("filtered", len(jobs))


if __name__ == "__main__":
    main(sys.argv[1:])
