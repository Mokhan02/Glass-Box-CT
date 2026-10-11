"""Do any SAE features explain tumour failures BEYOND size? Residual search, as follows.
 1. size = log predicted tumour voxels (what a user can observe); a second pass uses log ground-truth size.
 2. residual Dice = rank(Dice) minus its linear fit on rank(size). Each feature is residualised on size the same way
    (so a feature that is just 'big mask' cannot score).
 3. partial Spearman between feature and residual Dice, with a permutation null for the MAXIMUM over features (shuffling the
    residual Dice), because hundreds of features are tested on ~100 rows.
The model's own confidence (mean sigmoid probability in the mask) goes through the same test as a reference.
Usage: python scripts/sae_residual_features.py cache/pulled"""
import csv
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, "rq2_hcc_tace_seg.csv"))))
X = np.load(os.path.join(d, "rq2_hcc_tace_seg_feats.npz"))["sae_mid"].astype(np.float32)
st = np.load(os.path.join(d, "feature_stats.npz"))
LAB = ["background", "spleen", "liver", "aorta", "psoas_L", "psoas_R", "autoch_L", "autoch_R", "L3"]
keep = [i for i, r in enumerate(rows) if r["target"] == "tumor" and not np.isnan(X[i]).any() and r["mean_prob"] not in ("", "nan")]
Xt = X[keep]; dice = np.array([float(rows[i]["dice"]) for i in keep]); prob = np.array([float(rows[i]["mean_prob"]) for i in keep])
logpred = np.log1p([int(rows[i]["pred_voxels"]) for i in keep]); loggt = np.log1p([int(rows[i]["gt_voxels"]) for i in keep])
act = (Xt > 0).mean(0) > 0.10; J = np.flatnonzero(act)


def resid(v, size):
    r, s = rankdata(v), rankdata(size)
    return r - np.polyval(np.polyfit(s, r, 1), s)


rng = np.random.default_rng(0)
for name, size in (("log PREDICTED volume (observable)", logpred), ("log GROUND-TRUTH volume (oracle, for reference)", loggt)):
    rd = resid(dice, size)
    F = np.stack([resid(Xt[:, j], size) for j in J], 1)
    rho = np.array([spearmanr(F[:, k], rd)[0] for k in range(len(J))])
    null = np.array([np.abs(np.array([spearmanr(F[:, k], rng.permutation(rd))[0] for k in range(len(J))])).max() for _ in range(300)])
    pc = spearmanr(resid(prob, size), rd)[0]
    print(f"\n=== residualised on {name}: {len(keep)} tumour rows, {len(J)} features tested")
    print(f"how much of Dice does size explain? Spearman(size, Dice) = {spearmanr(size, dice)[0]:+.2f}  (R^2 of rank fit ~ {spearmanr(size, dice)[0] ** 2:.2f})")
    print(f"best |partial rho| over features: {np.abs(rho).max():.3f}; null for the max: median {np.median(null):.3f}, 95th pct {np.percentile(null, 95):.3f}; p = {(null >= np.abs(rho).max()).mean():.3f}")
    print(f"features beyond the null 95th pct: {(np.abs(rho) > np.percentile(null, 95)).sum()}   |   model confidence (mean prob), same test: partial rho {pc:+.2f}")
    for k in np.argsort(-np.abs(rho))[:6]:
        j = J[k]; cf = st["class_frac"][j]; a = int(np.argmax(cf))
        print(f"   feat {j:4d} partial rho {rho[k]:+.2f}  | normal anatomy: {LAB[a]} ({cf[a]:.2f}), TS mean HU {st['mean_hu'][j]:.0f}, TS freq {st['freq'][j]:.3f}")
