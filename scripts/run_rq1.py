"""RQ1: what does the encoder encode beyond its output classes? Probes on pooled activations vs baselines.
Usage: python scripts/run_rq1.py --acts cache/acts --rand cache/acts_rand --targets cache/targets --hook mid --out cache/rq1
Feature sets: trained / random-init encoder (mean + covariance pooled), global HU histogram, local L3-muscle HU stats.
Targets: sex, age, kVp, manufacturer (balanced accuracy), muscle area / mean HU / low-attenuation fraction (R^2).
For the muscle targets the informative number is the INCREMENTAL R^2 of [local HU stats + activations] over local alone:
local stats determine mean muscle HU by construction, so beating them on that target is impossible and is not claimed."""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.data.results import ResultsTable  # noqa: E402
from glassbox.probes.core import cov_pool, evaluate, fit_basis, mean_pool  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--acts", default="cache/acts")
ap.add_argument("--rand", default="cache/acts_rand")
ap.add_argument("--targets", default="cache/targets")
ap.add_argument("--hook", default="mid", choices=["bottleneck", "mid", "decoder_F"])
ap.add_argument("--out", default="cache/rq1")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--subset", default="all", choices=["all", "l3"],
                help="l3: only scans with L3 muscle measures (abdomen covered), which reduces the body-region confound")
ap.add_argument("--n-boot", type=int, default=1000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
MUSCLE = (4, 5, 6, 7)  # label ids: iliopsoas L/R, autochthon L/R (see extract_activations.LABELS)
L3 = 8
L3_HALF_WINDOW = 12  # voxels (18 mm) either side of the L3 centroid


def load(path):
    z = np.load(path)
    h = a.hook
    return z[f"{h}_x"], z[f"{h}_xyz"], z[f"{h}_label"]


def coverage_features(path):
    """Baseline for the body-region confound: what the scan covers, with no learned features.
    Fraction of tokens per label (0 = background, 1..8 = spleen, liver, aorta, psoas L/R, autochthon L/R, L3),
    log token count and the preprocessed scan extent in voxels."""
    z = np.load(path)
    lab = z[f"{a.hook}_label"].astype(int)
    frac = np.bincount(np.clip(lab, 0, 8), minlength=9) / max(len(lab), 1)
    return np.concatenate([frac, [np.log(max(len(lab), 1))], z["shape"].astype(float)]).astype(np.float32)


def l3_tokens(x, xyz, lab):
    if not (lab == L3).any():
        return None
    zc = xyz[lab == L3, 2].astype(float).mean()
    m = np.isin(lab, MUSCLE) & (np.abs(xyz[:, 2] - zc) <= L3_HALF_WINDOW)
    return x[m] if m.sum() >= 5 else None


rows = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(a.targets, "targets.csv")))}
base = np.load(os.path.join(a.targets, "baselines.npz"))
hist = dict(zip(map(str, base["hist_ids"]), base["hist"]))
local = dict(zip(map(str, base["local_ids"]), base["local"]))
ids = sorted(i for i in rows if os.path.exists(f"{a.acts}/{i}.npz") and os.path.exists(f"{a.rand}/{i}.npz")
             and (a.subset == "all" or rows[i].get("muscle_mean_hu")))
print(f"{len(ids)} scans with trained + random activations", flush=True)

# Basis for covariance pooling is fit on a token sample from the FIRST 60% of scans (a fixed prefix) and applied to all.
# The grouped CV folds are random over scans, so this is a mild approximation; the leakage-free refit is a TODO.
feats = {}
for tag, d in (("trained", a.acts), ("random", a.rand)):
    cache = {i: load(f"{d}/{i}.npz") for i in ids}
    sample = np.concatenate([cache[i][0][::20] for i in ids[: int(len(ids) * 0.6)]])
    mu, B = fit_basis(sample, a.rank)
    for region in ("all", "l3"):
        for pool in ("mean", "cov"):
            X, ok = [], []
            for i in ids:
                x, xyz, lab = cache[i]
                t = x if region == "all" else l3_tokens(x, xyz, lab)
                if t is None:
                    X.append(None)
                    continue
                X.append(mean_pool(t) if pool == "mean" else cov_pool(t, mu, B))
            feats[(tag, region, pool)] = X
feats[("hu_hist", "all", "-")] = [hist[i] for i in ids]
feats[("coverage", "all", "-")] = [coverage_features(f"{a.acts}/{i}.npz") for i in ids]
feats[("local_hu", "l3", "-")] = [local.get(i) for i in ids]


def col(name, cast=float):
    out = []
    for i in ids:
        v = rows[i].get(name, "")
        out.append(cast(v) if v not in ("", None) else None)
    return out


T = {
    "sex": (col("gender", lambda v: {"m": 1, "f": 0}.get(v.lower(), np.nan)), "logistic", "auroc", "all"),
    "age": (col("age"), "ridge", "r2", "all"),
    "kvp": (col("kvp", lambda v: round(float(v))), "logistic", "balanced_acc", "all"),
    "manufacturer": (col("manufacturer", lambda v: v.lower()), "logistic", "balanced_acc", "all"),
    "muscle_area_cm2": (col("muscle_area_cm2"), "ridge", "r2", "l3"),
    "muscle_mean_hu": (col("muscle_mean_hu"), "ridge", "r2", "l3"),
    "lama_fraction": (col("lama_fraction"), "ridge", "r2", "l3"),
}
table = ResultsTable(["hook", "target", "features", "pooling", "metric", "estimate", "ci_low", "ci_high"])
for tname, (y, kind, metric, region) in T.items():
    for (tag, reg, pool), X in feats.items():
        if tag in ("trained", "random") and reg != region:
            continue
        if tag == "local_hu" and region != "l3":
            continue
        ok = [j for j, (xi, yi) in enumerate(zip(X, y)) if xi is not None and yi is not None
              and not (isinstance(yi, float) and np.isnan(yi))]
        if len(ok) < 40:
            print(f"skip {tname}/{tag}/{reg}/{pool}: only {len(ok)} scans")
            continue
        Xa, ya, ga = np.stack([X[j] for j in ok]), np.array([y[j] for j in ok]), np.array([ids[j] for j in ok])
        if kind == "logistic":
            vals, counts = np.unique(ya, return_counts=True)
            keep = np.isin(ya, vals[counts >= 10])  # drop classes with fewer than 10 scans (reported in the log)
            Xa, ya, ga = Xa[keep], ya[keep], ga[keep]
            if len(np.unique(ya)) < 2:
                continue
        r = evaluate(Xa, ya, ga, kind, metric, a.n_boot)
        table.add("totalsegmentator", r["n_scans"], hook=a.hook, target=tname, features=tag, pooling=pool,
                  metric=metric, estimate=round(r["estimate"], 4), ci_low=round(r["ci_low"], 4), ci_high=round(r["ci_high"], 4))
        print(f"{tname:16s} {tag:9s} {reg:3s} {pool:4s} {metric:12s} {r['estimate']:.3f} [{r['ci_low']:.3f},{r['ci_high']:.3f}] n={r['n_scans']}", flush=True)

# Incremental R^2 on muscle targets: [local HU stats + trained activations] vs local alone, same scans, same folds.
for tname in ("muscle_area_cm2", "lama_fraction"):
    y = T[tname][0]
    ok = [j for j in range(len(ids)) if y[j] is not None and local.get(ids[j]) is not None
          and feats[("trained", "l3", "cov")][j] is not None]
    if len(ok) < 40:
        continue
    g = np.array([ids[j] for j in ok]); yy = np.array([y[j] for j in ok])
    L = np.stack([local[ids[j]] for j in ok]); A = np.stack([feats[("trained", "l3", "cov")][j] for j in ok])
    r0 = evaluate(L, yy, g, "ridge", "r2", a.n_boot); r1 = evaluate(np.hstack([L, A]), yy, g, "ridge", "r2", a.n_boot)
    print(f"INCREMENTAL {tname}: local R2 {r0['estimate']:.3f} -> local+acts {r1['estimate']:.3f} (gain {r1['estimate'] - r0['estimate']:+.3f}) n={r0['n_scans']}")
name = f"rq1_{a.hook}_{a.subset}.csv"
table.to_csv(os.path.join(a.out, name))
print("wrote", os.path.join(a.out, name))
