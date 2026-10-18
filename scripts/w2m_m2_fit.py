"""W2-M M2/M3 step 0 (CPU; results/w2_mech/PREREG.md, M1 branch = ORGAN-SPECIFIC): fit the erasers and the kidney-native probes, and write the seeded organ-check subset.
- Kidney-native rank-1 LEACE per fold (5-fold, case index mod 5 over the 201 sorted main-set kidney cases, as M1) on TRAINING-fold mid tokens, concept = tumour (1) vs kidney-parenchyma (0) token; HU control = LEACE for the local-HU concept
  (cell mean HU, `mid_hu14[:, 0]`, continuous) on the same tokens; training-token mean saved (the random-direction control is built per case at run time through that mean, seed 1000 + case index).
- HCC-direction LEACE (secondary): the concept 'token is tumour vs liver' fitted ONCE on all HCC clean-cohort cases' trained mid tokens.
- Kidney-native probes per fold (same probe spec as M1: StandardScaler + L2 LogisticRegression(C=0.1, balanced, max_iter 2000), <= 200 tokens per (patient, group), seed 0) for the exploratory M3 projection.
- Organ-check subset: sorted main-set case ids, numpy default_rng(133).choice(201, 60, replace=False) -> results/w2_mech/m2_organ_subset.csv.
Eraser x' = x - d (a.(x - mu)) / c, d = Sigma_xz, a = Sigma^-1 Sigma_xz, c = a.d (covariance ridge 1e-4 x mean eigenvalue), as scripts/mech_b_fit.py.
Usage: python scripts/w2m_m2_fit.py --kid <kid tokens dir with trained/> --hcc-vp <cache/vp> --rr <cache/rr> --out <dir>"""
import argparse
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

K, CAP, SEED = 5, 200, 0


def leace(X, z):
    mu = X.mean(0); Xc = X - mu; S = Xc.T @ Xc / len(X); S += np.eye(S.shape[0]) * 1e-4 * np.trace(S) / S.shape[0]
    d = Xc.T @ (z - z.mean()) / len(X); a = np.linalg.solve(S, d); return {"mu": mu, "d": d, "a": a, "c": np.float64(a @ d)}


def cap_idx(pid, mask, rng):
    out = []
    for p in np.unique(pid[mask]):
        idx = np.flatnonzero(mask & (pid == p)); out.append(idx if len(idx) <= CAP else rng.choice(idx, CAP, replace=False))
    return np.concatenate(out)


def fit_probe(X, y, pid):
    rng = np.random.default_rng(SEED); idx = np.r_[cap_idx(pid, y == 1, rng), cap_idx(pid, y == 0, rng)]
    sc = StandardScaler().fit(X[idx]); clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(X[idx]), y[idx]); return sc, clf


def main(a):
    os.makedirs(a.out, exist_ok=True)
    cases = sorted(f[:-4] for f in os.listdir(os.path.join(a.kid, "trained")) if f.endswith(".npz") and not f.endswith(".part.npz")); assert len(cases) == 201, len(cases)
    X, y, hu, pid = [], [], [], []
    for i, c in enumerate(cases):
        z = np.load(os.path.join(a.kid, "trained", c + ".npz")); keep = z["mid_is_tum"] | z["mid_is_kid"]
        X.append(z["mid_x"][keep].astype(np.float32)); y.append(z["mid_is_tum"][keep].astype(np.float32)); hu.append(z["mid_hu14"][keep][:, 0].astype(np.float32)); pid.append(np.full(int(keep.sum()), i))
    X, y, hu, pid = np.concatenate(X), np.concatenate(y), np.concatenate(hu), np.concatenate(pid); fold = pid % K
    save = {}
    for f in range(K):
        tr = fold != f
        for name, concept in (("leace", y), ("hu", hu)):
            p = leace(X[tr], concept[tr])
            for k, v in p.items():
                save[f"native_f{f}_{name}_{k}"] = np.asarray(v)
        sc, clf = fit_probe(X[tr], y[tr].astype(int), pid[tr])
        save[f"probe_f{f}_mean"], save[f"probe_f{f}_scale"], save[f"probe_f{f}_coef"], save[f"probe_f{f}_intercept"] = sc.mean_, sc.scale_, clf.coef_[0], clf.intercept_
        print(f"fold {f}: train tokens {int(tr.sum())}, tumour fraction {y[tr].mean():.3f}", flush=True)
    rr = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz")); hx, hy = [], []
    for c in rr:
        z = np.load(os.path.join(a.hcc_vp, "trained", c + ".npz")); hx.append(z["mid_x"].astype(np.float32)); hy.append(z["mid_is_tum"].astype(np.float32))
    p = leace(np.concatenate(hx), np.concatenate(hy))
    for k, v in p.items():
        save[f"hcc_leace_{k}"] = np.asarray(v)
    np.savez(os.path.join(a.out, "erasers.npz"), **save)
    subset = np.random.default_rng(133).choice(201, 60, replace=False)
    with open(os.path.join("results", "w2_mech", "m2_organ_subset.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_index", "case"]); [w.writerow([int(i), cases[i]]) for i in sorted(subset)]
    print(f"HCC cases {len(rr)}, HCC LEACE c = {float(save['hcc_leace_c']):.4g}; saved erasers and probes; organ subset of {len(subset)} cases written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--kid", required=True); ap.add_argument("--hcc-vp", required=True); ap.add_argument("--rr", required=True); ap.add_argument("--out", required=True)
    main(ap.parse_args())
