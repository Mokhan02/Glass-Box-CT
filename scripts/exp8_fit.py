"""Experiment 8: does anything beyond size predict whether a target lesion is detected? Design fixed before the covariate table was read.
Outcome: detected (predicted voxels cover >= 10% of the lesion). Baseline: log diameter. Blocks added to size ONE AT A TIME:
  contrast  |tumour - local liver HU|  (a tumour may be darker or brighter than liver)
  phase     aorta HU and whole-liver HU (continuous enhancement proxies; no phase cut-offs invented)
  position  distance to window edge (mm), number of covering windows, contained-in-one-window flag
  content   log liver volume and log tumour volume in the best window
  all       every block together (exploratory: ~9 parameters on ~30 events, so overfit is expected and flagged)
Evaluation: L2 logistic regression (C = 1, standardised, NaN imputed with the training median), 5-fold CV grouped by case, repeated 20 times; the repeated
out-of-fold predictions are averaged per lesion; AUROC with 95% CIs from resampling CASES; delta-AUROC over size alone, paired on the same resamples.
Specific prediction (30-60 mm lesions): missed vs detected on contrast, phase proxies and window variables, with the literature's 20 HU low-contrast cut (LiTS), not tuned.
Subsets: all lesions, UID-route only (labels verifiably linked), single-acquisition series only. Usage: python scripts/exp8_fit.py cache/pulled/exp8_lesions.csv results/exp8_lesions"""
import csv
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
rows = list(csv.DictReader(open(src)))
num = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan  # noqa: E731
D = {k: np.array([num(r, k) for r in rows]) for k in ("diam_mm", "volume_mm3", "contrast_hu", "aorta_hu", "liver_hu_all", "liver_hu_shell", "tumour_hu", "best_edge_mm", "n_windows", "win_liver_mm3", "win_tumour_mm3", "cover", "n_acq")}
D["detected"] = np.array([r["detected"] == "True" for r in rows]).astype(int); D["contained"] = np.array([r["contained_in_a_window"] == "True" for r in rows]).astype(float)
case = np.array([r["case"] for r in rows]); route = np.array([r["route"] for r in rows])
feat = {"size": np.log(D["diam_mm"])[:, None],
        "contrast": np.abs(D["contrast_hu"])[:, None],
        "phase": np.stack([D["aorta_hu"], D["liver_hu_all"]], 1),
        "position": np.stack([D["best_edge_mm"], D["n_windows"], D["contained"]], 1),
        "content": np.stack([np.log1p(D["win_liver_mm3"]), np.log1p(D["win_tumour_mm3"])], 1)}
MODELS = {"size only": ["size"], "size + contrast": ["size", "contrast"], "size + phase": ["size", "phase"], "size + position": ["size", "position"], "size + window content": ["size", "content"],
          "all blocks (exploratory)": ["size", "contrast", "phase", "position", "content"]}


def oof(X, y, groups, reps=20):
    P = np.zeros(len(y))
    for rep in range(reps):
        rng = np.random.default_rng(rep); ug = np.unique(groups); perm = {g: rng.permutation(len(ug))[i] for i, g in enumerate(ug)}
        g2 = np.array([perm[g] for g in groups])
        for tr, te in GroupKFold(n_splits=5).split(X, y, g2):
            med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0, med)
            Xtr, Xte = np.where(np.isnan(X[tr]), med, X[tr]), np.where(np.isnan(X[te]), med, X[te])
            sc = StandardScaler().fit(Xtr); P[te] += LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(Xtr), y[tr]).predict_proba(sc.transform(Xte))[:, 1]
    return P / reps


def boot_auc(y, p, groups, base=None, n=1000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}; a, d = [], []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); ys = y[s]
        if 0 < ys.sum() < len(ys):
            a.append(roc_auc_score(ys, p[s])); d.append(roc_auc_score(ys, p[s]) - roc_auc_score(ys, base[s]) if base is not None else 0.0)
    return np.percentile(a, [2.5, 97.5]), np.percentile(d, [2.5, 97.5])


subsets = {"all lesions": np.ones(len(rows), bool), "UID route only": route == "uid", "single-acquisition only": D["n_acq"] == 1}
res = []
print("Detection model: cross-validated AUROC (case-bootstrap 95% CI) and gain over size alone")
for sname, m in subsets.items():
    y, g = D["detected"][m], case[m]
    print(f"\n[{sname}] lesions {m.sum()}, detected {y.sum()}, missed {len(y) - y.sum()}, cases {len(set(g))}")
    base = None
    for mname, blocks in MODELS.items():
        X = np.hstack([feat[b][m] for b in blocks]); p = oof(X, y, g)
        if base is None:
            base = p
        (alo, ahi), (dlo, dhi) = boot_auc(y, p, g, base)
        res.append({"subset": sname, "model": mname, "n": int(m.sum()), "events_missed": int(len(y) - y.sum()), "auroc": roc_auc_score(y, p), "auroc_ci_low": alo, "auroc_ci_high": ahi,
                    "gain_over_size": roc_auc_score(y, p) - roc_auc_score(y, base), "gain_ci_low": dlo, "gain_ci_high": dhi})
        r = res[-1]; print(f"  {mname:28s} AUROC {r['auroc']:.3f} [{alo:.2f},{ahi:.2f}]   gain over size {r['gain_over_size']:+.3f} [{dlo:+.3f},{dhi:+.3f}]")
with open(os.path.join(out, "models.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in res])

# The specific prediction: 30-60 mm lesions, missed vs detected
print("\nSPECIFIC PREDICTION: among 30-60 mm lesions, do the missed ones have low contrast or a different phase?")
strata = []
for sname, m0 in subsets.items():
    m = m0 & (D["diam_mm"] >= 30) & (D["diam_mm"] < 60); miss, det = m & (D["detected"] == 0), m & (D["detected"] == 1)
    print(f"\n[{sname}] 30-60 mm: {m.sum()} lesions, {miss.sum()} missed, {det.sum()} detected")
    for lab, v in (("|contrast| (HU)", np.abs(D["contrast_hu"])), ("tumour HU", D["tumour_hu"]), ("liver HU (shell)", D["liver_hu_shell"]), ("aorta HU (arterial proxy)", D["aorta_hu"]), ("whole-liver HU", D["liver_hu_all"]),
                   ("best edge distance (mm)", D["best_edge_mm"]), ("covering windows", D["n_windows"]), ("liver volume in window (cm3)", D["win_liver_mm3"] / 1000), ("diameter (mm)", D["diam_mm"])):
        a, b = v[miss][np.isfinite(v[miss])], v[det][np.isfinite(v[det])]
        if len(a) < 4 or len(b) < 4:
            continue
        u = mannwhitneyu(a, b); rb = 1 - 2 * u.statistic / (len(a) * len(b))  # rank-biserial: positive = missed lower
        strata.append({"subset": sname, "variable": lab, "n_missed": len(a), "n_detected": len(b), "median_missed": float(np.median(a)), "median_detected": float(np.median(b)), "mannwhitney_p": float(u.pvalue), "rank_biserial": float(rb)})
        print(f"  {lab:30s} missed median {np.median(a):9.1f} (n={len(a):2d})  detected median {np.median(b):9.1f} (n={len(b):2d})   Mann-Whitney p={u.pvalue:.3f}  rank-biserial {rb:+.2f}")
    ac = np.abs(D["contrast_hu"])
    if miss.sum() and det.sum():
        print(f"  share with |contrast| < 20 HU (LiTS cut, not tuned): missed {np.mean(ac[miss] < 20):.2f}, detected {np.mean(ac[det] < 20):.2f}")
        strata.append({"subset": sname, "variable": "share |contrast| < 20 HU", "n_missed": int(miss.sum()), "n_detected": int(det.sum()), "median_missed": float(np.mean(ac[miss] < 20)), "median_detected": float(np.mean(ac[det] < 20)), "mannwhitney_p": np.nan, "rank_biserial": np.nan})
with open(os.path.join(out, "strata_30_60mm.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(strata[0])); w.writeheader(); w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in strata])
print("\nsize alone, for orientation: median diameter missed vs detected:", f"{np.median(D['diam_mm'][D['detected'] == 0]):.0f} vs {np.median(D['diam_mm'][D['detected'] == 1]):.0f} mm")
