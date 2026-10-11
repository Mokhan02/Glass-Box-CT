"""EXPLORATORY, descriptive (no tests, no tuning). Does the cross-validated threshold shift from the voxel-probe run recover bright or dark tumours?
Inputs: the saved class-26 logits of the 89-case clean cohort (cache/rr/*.npz). Per-fold thresholds are the voxel-probe run's
(results/voxel_probe/head_check.json, thresholds_chosen_per_fold, all-103 run) applied ONLY to the cases of that fold; fold = index of the case in the sorted
103-case voxel-probe list, mod 5 (the same assignment the voxel-probe analysis used). The shift is applied to the whole logit map, as in the voxel-probe recovery.
Lesions = GT RECIST targets (>= 10 mm longest axial diameter, 2 largest per case); coverage = fraction of the lesion's GT voxels predicted positive; detected = coverage >= 10%.
Bright / dark = case-level tumour-minus-liver contrast (raw HU, GT masks) > 0 / < 0. Usage: python scripts/bright_threshold_check.py"""
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

VOX = 1.5
RR = "cache/rr"
thr = {int(k): v for k, v in json.load(open("results/voxel_probe/head_check.json"))["thresholds_chosen_per_fold"].items()}
all103 = sorted(f[:-4] for f in os.listdir("cache/vp/trained") if f.endswith(".npz") and not f.endswith(".part.npz"))
fold103 = {c: i % 5 for i, c in enumerate(all103)}
cases = sorted(f[:-4] for f in os.listdir(RR) if f.endswith(".npz") and not f.endswith(".part.npz"))


def feret(m2):
    pts = np.argwhere(m2).astype(float) * VOX
    if len(pts) < 2:
        return 0.0
    try:
        hh = pts[ConvexHull(pts).vertices]
    except Exception:
        hh = pts
    return float(np.sqrt(((hh[:, None] - hh[None]) ** 2).sum(-1)).max())


def job(c):
    z = np.load(os.path.join(RR, c + ".npz")); L = z["L"]; shape = tuple(z["shape"])
    lg = z["logit_full"].astype(np.float32).ravel(); lg[z["roi_idx"].astype(np.int64)] = z["logit_roi"]; lg = lg.reshape(shape)
    ann = (L != 3).any(axis=(0, 1)); gt = (L == 2)[:, :, ann]; lg = lg[:, :, ann]; t = thr[fold103[c]]
    lab, n = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    les = sorted([(max(feret(lab[:, :, k] == i) for k in np.flatnonzero((lab == i).any(axis=(0, 1)))), i) for i in range(1, n + 1)], reverse=True)
    les = [x for x in les if x[0] >= 10.0][:2]; out = []
    for d, i in les:
        m = lab == i; out.append({"case": c, "contrast_hu": float(z["contrast_hu"]), "diam_mm": d, "cov_orig": float((lg[m] > 0).mean()), "cov_shift": float((lg[m] > t).mean()), "t": t})
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = [r for rs in ex.map(job, cases) for r in rs]
    for r in rows:
        r["det_orig"] = r["cov_orig"] >= 0.10; r["det_shift"] = r["cov_shift"] >= 0.10
    os.makedirs("results/readout_refit", exist_ok=True)
    with open("results/readout_refit/bright_threshold_lesions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines = ["EXPLORATORY / descriptive. clean-89 cohort; per-fold voxel-probe thresholds applied to held-out folds only.", "",
             "| group (case contrast) | lesions | cases | mean coverage orig | mean coverage shifted | detected orig | detected shifted | missed->detected | detected->missed |", "|---|---|---|---|---|---|---|---|---|"]
    for name, sel in (("bright (> 0 HU)", lambda r: r["contrast_hu"] > 0), ("dark (< 0 HU)", lambda r: r["contrast_hu"] < 0), ("all", lambda r: True)):
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        lines.append(f"| {name} | {len(g)} | {len({r['case'] for r in g})} | {np.mean([r['cov_orig'] for r in g]):.3f} | {np.mean([r['cov_shift'] for r in g]):.3f} | "
                     f"{sum(r['det_orig'] for r in g)}/{len(g)} ({100 * np.mean([r['det_orig'] for r in g]):.0f}%) | {sum(r['det_shift'] for r in g)}/{len(g)} ({100 * np.mean([r['det_shift'] for r in g]):.0f}%) | "
                     f"{sum((not r['det_orig']) and r['det_shift'] for r in g)} | {sum(r['det_orig'] and not r['det_shift'] for r in g)} |")
    lines += ["", "Named cases (the two bright, fully missed tumours the re-fit failed on):", "", "| case | contrast HU | lesion mm | coverage orig | coverage shifted (t) |", "|---|---|---|---|---|"]
    for r in rows:
        if r["case"] in ("HCC_031_552418", "HCC_078_958325"):
            lines.append(f"| {r['case']} | {r['contrast_hu']:+.1f} | {r['diam_mm']:.0f} | {r['cov_orig']:.3f} | {r['cov_shift']:.3f} ({r['t']}) |")
    open("results/readout_refit/bright_threshold_check.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
