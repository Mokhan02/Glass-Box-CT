"""Mechanistic experiment B, step 1 (results/mech/PREREG.md, section B): fit the erasers on training folds and run the probe check. CPU.
Eraser x' = x - d * (a.(x - mu)) / (a.d) with d = Sigma_xz and a = Sigma^-1 Sigma_xz (rank-1 LEACE, closed form; Sigma ridge 1e-4 x mean eigenvalue).
Controls: orthogonal projection of a random unit direction (through mu); LEACE for the local-HU concept (cell mean HU, continuous).
Usage: python scripts/mech_b_fit.py [--out cache/mech_b] [--res results/mech]"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="cache/mech_b"); ap.add_argument("--res", default="results/mech"); ap.add_argument("--vp", default="cache/vp/trained"); ap.add_argument("--rr", default="cache/rr")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.res, exist_ok=True)
K = 5
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(cases)}
X, y, hu, cid = [], [], [], []
for i, c in enumerate(cases):
    z = np.load(os.path.join(a.vp, c + ".npz")); X.append(z["mid_x"].astype(np.float32)); y.append(z["mid_is_tum"].astype(np.float32)); hu.append(z["mid_hu14"][:, 0].astype(np.float32)); cid.append(np.full(len(y[-1]), i))
X, y, hu, cid = np.concatenate(X), np.concatenate(y), np.concatenate(hu), np.concatenate(cid); fold = np.array([fold_of[cases[i]] for i in cid])
print("tokens", X.shape, "tumour fraction", y.mean())


def leace(Xt, z):
    mu = Xt.mean(0); Xc = Xt - mu; S = Xc.T @ Xc / len(Xt); S += np.eye(S.shape[0]) * 1e-4 * np.trace(S) / S.shape[0]
    d = Xc.T @ (z - z.mean()) / len(Xt); a_ = np.linalg.solve(S, d); return {"mu": mu, "d": d, "a": a_, "c": float(a_ @ d)}


def apply(p, Xe):
    return Xe - np.outer((Xe - p["mu"]) @ p["a"], p["d"]) / p["c"]


def rand_proj(mu, seed):
    v = np.random.default_rng(seed).normal(size=mu.shape); v /= np.linalg.norm(v); return {"mu": mu, "d": v, "a": v, "c": 1.0}  # orthogonal projection: x - v v.(x-mu)


save, check = {}, []
for f in range(5):
    tr, te = fold != f, fold == f
    P = {"leace": leace(X[tr], y[tr]), "hu": leace(X[tr], hu[tr]), "random": rand_proj(X[tr].mean(0), 100 + f)}
    for n, p in P.items():
        for k, v in p.items():
            save[f"f{f}_{n}_{k}"] = np.asarray(v)
    # probe check: before and after erasure (LEACE), held-out AUROC
    row = {"fold": f}
    for tag, T, E in (("before", X[tr], X[te]), ("after_leace", apply(P["leace"], X[tr]), apply(P["leace"], X[te])), ("after_random", apply(P["random"], X[tr]), apply(P["random"], X[te])), ("after_hu", apply(P["hu"], X[tr]), apply(P["hu"], X[te]))):
        sc = StandardScaler().fit(T); m = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(T), y[tr]); s = m.decision_function(sc.transform(E)); row[tag] = float(roc_auc_score(y[te], s)); row[tag + "_scores"] = s
    check.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not k.endswith("scores")})
np.savez(os.path.join(a.out, "erasers.npz"), **save)
# pooled AUROC with patient-bootstrap CI (held-out tokens, resample cases)
rng = np.random.default_rng(1); res = {}
tests = {f: np.flatnonzero(fold == f) for f in range(5)}
for tag in ("before", "after_leace", "after_random", "after_hu"):
    sc = np.zeros(len(y)); 
    for f in range(5):
        sc[tests[f]] = check[f][tag + "_scores"]
    point = float(roc_auc_score(y, sc)); P_ = len(cases); bs = []
    for _ in range(300):
        pick = rng.integers(0, P_, P_); idx = np.concatenate([np.flatnonzero(cid == i) for i in pick]); bs.append(roc_auc_score(y[idx], sc[idx]))
    res[tag] = {"auroc_heldout_pooled": point, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
res["near_chance_rule"] = "after_leace upper CI <= 0.60"; res["probe_near_chance_after_leace"] = bool(res["after_leace"]["ci"][1] <= 0.60)
json.dump(res, open(os.path.join(a.res, "B_probe_check.json"), "w"), indent=1); print(json.dumps(res, indent=1))
