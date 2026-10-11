"""Mechanistic experiment C, step 1 (results/mech/PREREG.md section C): steering directions per fold and layer, from training-fold tokens only. CPU.
probe direction u: logistic probe on standardised tokens (tumour vs liver), direction = w/sigma, unit-normalised; dmu = mean projection of tumour tokens minus liver tokens on u.
mean-difference direction m: unit(mean of detected GT-tumour tokens - mean of missed GT-tumour tokens) (pred_cov >= 0.5 vs < 0.5), moved with the same step size.
Usage: python scripts/mech_c_fit.py"""
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VP, RR, OUT, K = "cache/vp/trained", "cache/rr", "cache/mech_c", 5
os.makedirs(OUT, exist_ok=True)
cases = sorted(f[:-4] for f in os.listdir(RR) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(cases)}
save = {}
for layer in ("mid", "bottleneck"):
    X, y, pc, cid = [], [], [], []
    for i, c in enumerate(cases):
        z = np.load(os.path.join(VP, c + ".npz")); X.append(z[f"{layer}_x"].astype(np.float32)); y.append(z[f"{layer}_is_tum"].astype(bool)); pc.append(z[f"{layer}_pred_cov"].astype(np.float32)); cid.append(np.full(len(y[-1]), i))
    X, y, pc, cid = np.concatenate(X), np.concatenate(y), np.concatenate(pc), np.concatenate(cid); fold = np.array([fold_of[cases[i]] for i in cid])
    for f in range(K):
        tr = fold != f; sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(X[tr]), y[tr])
        d = m.coef_[0] / sc.scale_; u = d / np.linalg.norm(d); proj = X[tr] @ u; dmu = float(proj[y[tr]].mean() - proj[~y[tr]].mean())
        det, mis = tr & y & (pc >= 0.5), tr & y & (pc < 0.5)
        md = X[det].mean(0) - X[mis].mean(0); md /= np.linalg.norm(md)
        save[f"{layer}_f{f}_u"], save[f"{layer}_f{f}_dmu"], save[f"{layer}_f{f}_md"] = u, np.array(dmu), md
        print(layer, f, "dmu", round(dmu, 3), "cos(u, meandiff)", round(float(u @ md), 3), flush=True)
np.savez(os.path.join(OUT, "dirs.npz"), **save)
