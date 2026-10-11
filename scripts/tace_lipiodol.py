"""TACE hypothesis: do very bright voxels inside the ground-truth tumour (a lipiodol proxy) go with how badly the model under-segments it?
Lipiodol fraction = share of GT tumour voxels (on annotated slices) above a HU threshold on the CT the label was drawn on.
Thresholds come from the literature, not from tuning: 241 HU = high-density lipiodol on 24 h post-cTACE CT (PMC7582153, three tiers
87/155/241 HU); 200 HU = the high-density cutoff in a Yale thesis (elischolar.library.yale.edu/ymtdl/3934). 241 is primary; 200 is the
robustness check. Caveats: both come from abstracts/summaries; contrast-enhanced vessels and calcification can also exceed these values.
Outcomes (from eval_heldout.py): Dice and predicted/GT volume ratio for the tumour target. Tumour size confounds both, so we also
report the Spearman correlation after removing log GT volume (rank-based partial correlation) and by size tercile.
Usage: python scripts/tace_lipiodol.py cache/heldout_eval/hcc_tace_seg.csv"""
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy.stats import rankdata, spearmanr

THRESHOLDS = (241.0, 200.0)
root = "cache/heldout_nii/hcc_tace_seg"
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["target"] == "tumor"]
data = []
for r in rows:
    d = os.path.join(root, r["case"])
    ct = np.asarray(nib.load(os.path.join(d, "ct.nii.gz")).dataobj).astype(np.float32)
    seg = np.asarray(nib.load(os.path.join(d, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1))
    tumor = (seg == 2)[:, :, ann]
    hu = ct[:, :, ann][tumor]
    if len(hu) < 50:
        continue
    data.append({"case": r["case"], "patient": r["case"].rsplit("_", 1)[0], "dice": float(r["dice"]), "ratio": int(r["pred_voxels"]) / max(int(r["gt_voxels"]), 1),
                 "gt": int(r["gt_voxels"]), "median_hu": float(np.median(hu)), **{f"frac{int(t)}": float((hu > t).mean()) for t in THRESHOLDS}})
print(f"{len(data)} HCC cases with a tumour label")
print("fraction of cases with any tumour voxel above threshold: " + ", ".join(f"{int(t)} HU: {np.mean([d[f'frac{int(t)}'] > 0 for d in data]):.2f}" for t in THRESHOLDS))
y = {k: np.array([d[k] for d in data]) for k in ("dice", "ratio", "gt", "median_hu", "frac241", "frac200")}
logv = np.log(y["gt"])


def partial_spearman(x, z, ctrl):
    rx, rz, rc = rankdata(x), rankdata(z), rankdata(ctrl)
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)  # noqa: E731
    return spearmanr(res(rx, rc), res(rz, rc))[0]


def boot(f, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pats = sorted({d["patient"] for d in data}); idx = {p: [i for i, d in enumerate(data) if d["patient"] == p] for p in pats}
    v = []
    for _ in range(n):
        sel = np.concatenate([idx[p] for p in rng.choice(pats, len(pats), replace=True)])
        try:
            v.append(f(sel))
        except Exception:
            pass
    return np.percentile(v, [2.5, 97.5])


print("\nSpearman rho between lipiodol fraction and model performance (95% patient-bootstrap CI):")
for t in THRESHOLDS:
    k = f"frac{int(t)}"
    for out in ("dice", "ratio"):
        rho = spearmanr(y[k], y[out])[0]; lo, hi = boot(lambda s: spearmanr(y[k][s], y[out][s])[0])
        prho = partial_spearman(y[k], y[out], logv); plo, phi = boot(lambda s: partial_spearman(y[k][s], y[out][s], logv[s]))
        print(f"  >{int(t)} HU  vs {out:5s}: rho {rho:+.2f} [{lo:+.2f},{hi:+.2f}]   controlling for tumour size: {prho:+.2f} [{plo:+.2f},{phi:+.2f}]")
rho = spearmanr(y["median_hu"], y["dice"])[0]
print(f"  median tumour HU vs dice: rho {rho:+.2f}   | tumour size vs dice: rho {spearmanr(y['gt'], y['dice'])[0]:+.2f}")
q = np.quantile(logv, [0, 1 / 3, 2 / 3, 1])
print("\nDice by size tercile, split by high lipiodol fraction (> 5% of tumour voxels above 241 HU):")
for i in range(3):
    m = (logv >= q[i]) & (logv <= q[i + 1]); hi = y["frac241"] > 0.05
    a, b = m & hi, m & ~hi
    print(f"  size tercile {i + 1}: lipiodol-rich n={a.sum():2d} Dice {y['dice'][a].mean() if a.any() else float('nan'):.3f} | other n={b.sum():2d} Dice {y['dice'][b].mean() if b.any() else float('nan'):.3f}")
