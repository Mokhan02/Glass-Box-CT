"""RQ2: can internal activations predict segmentation failure (Dice < 0.7) better than the model's own confidence?
Reads the outputs of build_rq2.py. Every number is an out-of-fold prediction (GroupKFold by PATIENT, so a patient's degraded
copies never straddle train/test) with a patient-level bootstrap 95% CI. AUROC is reported WITHIN each target class, because
failure rates differ hugely by class (hepatic tumor fails often, liver rarely) and a predictor that only recognises the class
would score well pooled. Methods:
  class_only   class identity alone (the baseline the plan flagged as hardest to beat)
  confidence   mean sigmoid prob + mean binary entropy in the mask (+ class, + empty flag)
  volume       log predicted volume (+ class, + empty flag)
  conf+volume  both of the above (+ class, + empty flag)
  acts_mean    mean mid-layer activations in the mask (+ class, + empty flag)
  acts_cov     covariance-pooled mid-layer activations (+ class, + empty flag)
  sae          mean SAE feature activations in the mask (+ class, + empty flag)
  acts+conf    acts_mean + confidence + volume (+ class, + empty flag)
SIZE IS THE BASELINE TO BEAT. Dice punishes small structures, and held-out tumours fail mostly when they are small, so a predicted-volume
model is strong. The headline numbers are therefore the GAINS (with a patient-bootstrap 95% CI) of volume + features over volume alone
and over confidence + volume; a gain whose interval includes 0 means the internals only rediscover size. Methods prefixed vol+ include
predicted volume. `--failure` swaps the failure definition: dice (Dice < threshold), relvol (|pred - GT| / GT volume >= 0.5, how wrong
the SIZE is), or miss (Dice < 0.1: the tumour was effectively not found). The last two are far less tied to tumour size than Dice.
Usage: python scripts/run_rq2.py --dirs cache/rq2_none cache/rq2_pert --out cache/rq2_results [--failure dice|relvol|miss]"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from glassbox.probes.core import oof_predict  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", default=["cache/rq2_none", "cache/rq2_pert"])
ap.add_argument("--out", default="cache/rq2_results")
ap.add_argument("--threshold", type=float, default=0.7)
ap.add_argument("--failure", default="dice", choices=["dice", "relvol", "miss"])
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--sae-pca", type=int, default=64, help="PCA components for the SAE features (3072 dims, few hundred rows)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

rows, F = [], {"mean_mid": [], "cov_mid": [], "sae_mid": []}
for d in a.dirs:
    for f in sorted(glob.glob(os.path.join(d, "rq2_*[!s].csv"))):
        if f.endswith("_feats.npz"):
            continue
        r = list(csv.DictReader(open(f)))
        z = np.load(f.replace(".csv", "_feats.npz"))
        n = min(len(r), len(z["mean_mid"]))  # a checkpoint may be mid-write; keep the aligned prefix
        rows += r[:n]
        for k in F:
            F[k].append(z[k][:n].astype(np.float32))
F = {k: np.concatenate(v) for k, v in F.items()}
def failed(r):
    if a.failure == "dice":
        return float(r["dice"]) < a.threshold
    if a.failure == "miss":
        return float(r["dice"]) < 0.1
    return abs(int(r["pred_voxels"]) - int(r["gt_voxels"])) / max(int(r["gt_voxels"]), 1) >= 0.5


y = np.array([failed(r) for r in rows]).astype(int)
tgt = np.array([r["target"] for r in rows]); pat = np.array([r["patient"] for r in rows]); pert = np.array([r["perturbation"] for r in rows])
empty = np.array([r["empty"] == "True" for r in rows]).astype(float)
col = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])  # noqa: E731
prob, ent, vol = col("mean_prob"), col("mean_entropy"), np.log1p(col("pred_voxels"))
print(f"{len(rows)} rows, {len(set(pat))} patients; failure definition: {a.failure}; failure rate by class:")
for t in sorted(set(tgt)):
    m = tgt == t
    print(f"  {t:7s} n={m.sum():4d} failures={y[m].sum():4d} ({y[m].mean():.2f})   by perturbation: "
          + " ".join(f"{p}:{y[m & (pert == p)].sum()}/{(m & (pert == p)).sum()}" for p in sorted(set(pert))))


def impute(X):
    X = X.copy()
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    bad = np.isnan(X)
    X[bad] = np.take(med, np.where(bad)[1])
    return X


onehot = np.stack([(tgt == t).astype(float) for t in sorted(set(tgt))], 1)
base = np.hstack([onehot, empty[:, None]])
mean_pca8 = PCA(8, random_state=0).fit_transform(impute(F["mean_mid"]))  # low-dimensional variant: fewer features than rows by a wide margin
sae_X = impute(F["sae_mid"]); sae_X = PCA(min(a.sae_pca, sae_X.shape[1], len(sae_X) - 1), random_state=0).fit_transform(sae_X)
conf = np.stack([prob, ent], 1)
methods = {
    "class_only": base,
    "confidence": np.hstack([impute(conf), base]),
    "volume": np.hstack([impute(vol[:, None]), base]),
    "conf+volume": np.hstack([impute(np.hstack([conf, vol[:, None]])), base]),
    "acts_mean": np.hstack([impute(F["mean_mid"]), base]),
    "acts_cov": np.hstack([impute(F["cov_mid"]), base]),
    "sae": np.hstack([sae_X, base]),
    "acts+conf": np.hstack([impute(F["mean_mid"]), impute(np.hstack([conf, vol[:, None]])), base]),
    "vol+acts_mean": np.hstack([impute(vol[:, None]), impute(F["mean_mid"]), base]),
    "vol+acts_cov": np.hstack([impute(vol[:, None]), impute(F["cov_mid"]), base]),
    "vol+sae": np.hstack([impute(vol[:, None]), sae_X, base]),
    "vol+conf+acts": np.hstack([impute(np.hstack([conf, vol[:, None]])), impute(F["mean_mid"]), base]),
    "vol+acts_pca8": np.hstack([impute(vol[:, None]), mean_pca8, base]),
    "vol+conf+pca8": np.hstack([impute(np.hstack([conf, vol[:, None]])), mean_pca8, base]),
}


def auroc(mask, p):
    yy = y[mask]
    return roc_auc_score(yy, p[mask]) if 0 < yy.sum() < len(yy) else np.nan


def boot(mask, p, n, seed=0):
    rng = np.random.default_rng(seed)
    ps = np.unique(pat[mask]); idx = {q: np.flatnonzero(mask & (pat == q)) for q in ps}
    v = []
    for _ in range(n):
        sel = np.concatenate([idx[q] for q in rng.choice(ps, len(ps), replace=True)])
        yy = y[sel]
        if 0 < yy.sum() < len(yy):
            v.append(roc_auc_score(yy, p[sel]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) > 20 else (np.nan, np.nan)


results, preds = [], {}
for name, X in methods.items():
    X = np.nan_to_num(X)
    k = min(5, len(set(pat)))
    p = oof_predict(X, y, pat, "logistic", n_splits=k)[:, 1]
    preds[name] = p
    for scope in sorted(set(tgt)) + ["pooled"]:
        m = np.ones(len(y), bool) if scope == "pooled" else tgt == scope
        lo, hi = boot(m, p, a.n_boot)
        results.append({"method": name, "scope": scope, "auroc": auroc(m, p), "ci_low": lo, "ci_high": hi,
                        "n": int(m.sum()), "failures": int(y[m].sum())})
with open(os.path.join(a.out, f"rq2_auroc_{a.failure}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
scopes = sorted(set(tgt)) + ["pooled"]
print("\nAUROC [95% CI] for flagging failures (out-of-fold, patient-grouped):")
print(f"{'method':14s}" + "".join(f"{s:>26s}" for s in scopes))
for name in methods:
    line = f"{name:14s}"
    for s in scopes:
        r = next(x for x in results if x["method"] == name and x["scope"] == s)
        line += f"{r['auroc']:>11.3f} [{r['ci_low']:.2f},{r['ci_high']:.2f}]" if not np.isnan(r["auroc"]) else f"{'n/a':>26s}"
    print(line)


def boot_delta(mask, p1, p2, n, seed=0):
    rng = np.random.default_rng(seed)
    ps = np.unique(pat[mask]); idx = {q: np.flatnonzero(mask & (pat == q)) for q in ps}
    d = []
    for _ in range(n):
        sel = np.concatenate([idx[q] for q in rng.choice(ps, len(ps), replace=True)])
        yy = y[sel]
        if 0 < yy.sum() < len(yy):
            d.append(roc_auc_score(yy, p1[sel]) - roc_auc_score(yy, p2[sel]))
    return (np.percentile(d, 2.5), np.percentile(d, 97.5)) if len(d) > 20 else (np.nan, np.nan)


gains = []
print("\nGAIN in AUROC of volume+features over the size baselines (95% patient-bootstrap CI; interval including 0 = no demonstrated gain):")
for base_name in ("volume", "conf+volume"):
    print(f"  over '{base_name}':")
    for m_name in ("vol+acts_mean", "vol+acts_cov", "vol+sae", "vol+conf+acts", "vol+acts_pca8", "vol+conf+pca8"):
        line = f"    {m_name:14s}"
        for scope in scopes:
            msk = np.ones(len(y), bool) if scope == "pooled" else tgt == scope
            a1, a0 = auroc(msk, preds[m_name]), auroc(msk, preds[base_name])
            lo, hi = boot_delta(msk, preds[m_name], preds[base_name], a.n_boot)
            gains.append({"baseline": base_name, "method": m_name, "scope": scope, "failure": a.failure, "gain": a1 - a0, "ci_low": lo, "ci_high": hi})
            line += f"  {scope}: {a1 - a0:+.3f} [{lo:+.3f},{hi:+.3f}]" if not np.isnan(a1 - a0) else f"  {scope}: n/a"
        print(line)
with open(os.path.join(a.out, f"rq2_gains_{a.failure}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gains[0])); w.writeheader(); w.writerows(gains)
print("\nAUROC by perturbation (pooled over classes), best baseline vs activations:")
for pt in sorted(set(pert)):
    m = pert == pt
    print(f"  {pt:6s} n={m.sum():4d} fail={y[m].sum():3d} " + " ".join(f"{n}={auroc(m, preds[n]):.3f}" for n in ("class_only", "conf+volume", "acts_mean", "sae", "acts+conf")))
