"""First look at the mid-layer SAE (3072 features) on held-out HCC cases: which features fire inside predicted tumours, and which of
those track whether the model gets the tumour right?  Inputs are per-(case, target) mean SAE activations inside the predicted mask
(build_rq2.py) and per-feature statistics from TotalSegmentator tokens (train_sae.py).
  tumour-vs-liver : AUROC separating tumour-mask rows from liver-mask rows, and the paired within-case mean difference
  failure         : among tumour rows with a non-empty prediction, AUROC for Dice < 0.7 and Spearman with Dice
The feature_stats side shows what each feature does on ordinary anatomy (labels 0..8: background, spleen, liver, aorta, psoas L/R,
autochthon L/R, L3) so a 'tumour' feature that is really a 'bright vessel' or 'bone' feature can be spotted.
Usage: python scripts/sae_tumor_features.py cache/pulled"""
import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, "rq2_hcc_tace_seg.csv"))))
X = np.load(os.path.join(d, "rq2_hcc_tace_seg_feats.npz"))["sae_mid"].astype(np.float32)
st = np.load(os.path.join(d, "feature_stats.npz"))
LAB = ["background", "spleen", "liver", "aorta", "psoas_L", "psoas_R", "autoch_L", "autoch_R", "L3"]
tgt = np.array([r["target"] for r in rows]); case = np.array([r["case"] for r in rows])
dice = np.array([float(r["dice"]) for r in rows]); ok = ~np.isnan(X).any(1)
print(f"{len(rows)} rows, {ok.sum()} with a usable (non-empty) mask; tumour rows {int(((tgt == 'tumor') & ok).sum())}, liver rows {int(((tgt == 'liver') & ok).sum())}")
m = ok & np.isin(tgt, ["tumor", "liver"])
y = (tgt[m] == "tumor").astype(int); Xm = X[m]
active = (Xm > 0).mean(0) > 0.05  # features that fire in at least 5% of rows
auc = np.array([roc_auc_score(y, Xm[:, j]) if active[j] else np.nan for j in range(X.shape[1])])
# paired within-case difference (tumour minus liver) on cases having both masks
cs = [c for c in sorted(set(case[m])) if ((case == c) & (tgt == "tumor") & ok).any() and ((case == c) & (tgt == "liver") & ok).any()]
T = np.stack([X[(case == c) & (tgt == "tumor") & ok][0] for c in cs]); Lv = np.stack([X[(case == c) & (tgt == "liver") & ok][0] for c in cs])
paired = (T - Lv).mean(0)
order = np.argsort(-np.nan_to_num(auc, nan=0.5) + 0)  # highest tumour-vs-liver AUROC first
print(f"\n{int(active.sum())} features active in >=5% of rows; paired cases: {len(cs)}")
print("\nTop features that fire MORE in tumour masks than liver masks:")
print(f"{'feat':>5s} {'AUROC':>6s} {'paired diff':>11s} {'TS freq':>8s} {'TS mean HU':>10s}  strongest normal-anatomy label (TotalSegmentator)")
for j in order[:12]:
    cf = st["class_frac"][j]; k = int(np.argmax(cf))
    print(f"{j:5d} {auc[j]:6.3f} {paired[j]:11.4f} {st['freq'][j]:8.4f} {st['mean_hu'][j]:10.1f}  {LAB[k]} ({cf[k]:.2f})")
tm = ok & (tgt == "tumor"); Xt, dt = X[tm], dice[tm]; fail = (dt < 0.7).astype(int)
print(f"\nAmong tumour rows ({tm.sum()}), failures (Dice<0.7): {fail.sum()}")
act = (Xt > 0).mean(0) > 0.10
fa = np.array([roc_auc_score(fail, Xt[:, j]) if act[j] and 0 < fail.sum() < len(fail) else np.nan for j in range(X.shape[1])])
rho = np.array([spearmanr(Xt[:, j], dt)[0] if act[j] else np.nan for j in range(X.shape[1])])
print("Features whose activation inside the predicted tumour is higher when the model FAILS (AUROC for failure), with Spearman vs Dice:")
for j in np.argsort(-np.nan_to_num(fa, nan=0.5))[:8]:
    cf = st["class_frac"][j]; k = int(np.argmax(cf))
    print(f"  feat {j:4d}  failure AUROC {fa[j]:.3f}  rho(Dice) {rho[j]:+.2f}   tumour-vs-liver AUROC {auc[j]:.3f}   normal anatomy: {LAB[k]} ({cf[k]:.2f}), TS mean HU {st['mean_hu'][j]:.0f}")
print("Features higher when the model SUCCEEDS:")
for j in np.argsort(np.nan_to_num(fa, nan=0.5))[:5]:
    cf = st["class_frac"][j]; k = int(np.argmax(cf))
    print(f"  feat {j:4d}  failure AUROC {fa[j]:.3f}  rho(Dice) {rho[j]:+.2f}   tumour-vs-liver AUROC {auc[j]:.3f}   normal anatomy: {LAB[k]} ({cf[k]:.2f}), TS mean HU {st['mean_hu'][j]:.0f}")
print(f"\nnote: {int(act.sum())} features tested for failure on {len(fail)} rows; with this many tests some AUROCs this far from 0.5 occur by chance.")
