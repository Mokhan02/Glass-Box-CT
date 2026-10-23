"""W3 Stage 2, M2 analysis (CPU; PREREG_stage2.md; mech B margins). Inputs: the runner's lesions.csv / cases.csv, the M1 token tables, the cross-fitted erasers. Population = lesions the UNHOOKED model detects (coverage >= 0.10) in the cohort given by --cases.
Per lesion: change in lesion Dice (dice_within_10mm, the mech B definition) under leace / random / hu relative to base; paired differences (leace - random), (leace - hu); fraction still detected; per case: change in organ (liver) Dice under leace; probe check: a probe (the M1 spec) fitted on ERASED training-fold tokens and tested on ERASED held-out tokens, AUROC with a case-bootstrap CI (near chance iff the CI upper bound <= 0.60).
Verdict from glassbox.seg.w3s2_stats.m2_verdict. Usage: python scripts/w3s2_m2_analyze.py <model A|D> <run dir> <tokens dir> <erasers.npz> <cases file> <out dir>"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from glassbox.seg import w3s2_stats as S  # noqa: E402

model, rundir, tdir, erasers, cases_file, out = sys.argv[1:7]; os.makedirs(out, exist_ok=True)
cases = sorted(l.strip() for l in open(cases_file) if l.strip()); cset = set(cases); primary = sorted(l.strip() for l in open("results/w3_models/stage2/m1/cases_primary74.txt") if l.strip()); fold = S.fold_of_cases(primary)
les = [r for r in csv.DictReader(open(os.path.join(rundir, "lesions.csv"))) if r["case"] in cset]; cs = [r for r in csv.DictReader(open(os.path.join(rundir, "cases.csv"))) if r["case"] in cset]
by = {}
for r in les:
    by.setdefault((r["case"], r["rank"]), {})[r["arm"].split("_", 1)[1]] = r
keys = sorted(k for k, v in by.items() if "base" in v and "leace" in v and "random" in v and "hu" in v and float(v["base"]["coverage"]) >= 0.10)
ca = np.array([k[0] for k in keys]); d = {c: np.array([float(by[k][c]["dice_within_10mm"]) - float(by[k]["base"]["dice_within_10mm"]) for k in keys]) for c in ("leace", "random", "hu")}
dcov = {c: np.array([float(by[k][c]["coverage"]) - float(by[k]["base"]["coverage"]) for k in keys]) for c in ("leace", "random", "hu")}; still = {c: float(np.mean([float(by[k][c]["coverage"]) >= 0.10 for k in keys])) for c in ("leace", "random", "hu")}
ob = {(r["case"], r["cond"]): float(r["organ_dice"]) for r in cs}; lc = sorted({r["case"] for r in cs if (r["case"], "base") in ob and (r["case"], "leace") in ob}); liver_d = (np.array(lc), np.array([ob[(c, "leace")] - ob[(c, "base")] for c in lc]))
# ---- probe check on erased tokens (the M1 probe spec, 5 folds) ----
E = np.load(erasers); X, y, pid = [], [], []
for i, c in enumerate(primary):
    z = np.load(os.path.join(tdir, "tokens", c + ".npz")); X.append(z["mid_x"].astype(np.float64)); y.append(z["mid_is_tum"].astype(int)); pid.append(np.full(len(y[-1]), i))
X, y, pid = np.concatenate(X), np.concatenate(y), np.concatenate(pid); f_of = np.array([fold[primary[p]] for p in pid]); rng = np.random.default_rng(S.arm_seed(f"M2_probe_{model}")); sc_pre = np.full(len(y), np.nan); sc_post = np.full(len(y), np.nan)


def cap(idx_mask):
    o = []
    for p in np.unique(pid[idx_mask]):
        i = np.flatnonzero(idx_mask & (pid == p)); o.append(rng.choice(i, 200, replace=False) if len(i) > 200 else i)
    return np.concatenate(o)


for f in range(5):
    e = {k: E[f"{f}_leace_{k}"] for k in ("mu", "d", "a", "c")}; tr, te = f_of != f, f_of == f
    for Xs, dest in ((X, sc_pre), (S.apply_eraser(X, e), sc_post)):
        i1, i0 = cap(tr & (y == 1)), cap(tr & (y == 0)); sc = StandardScaler().fit(Xs[np.r_[i1, i0]]); clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(Xs[np.r_[i1, i0]]), np.r_[np.ones(len(i1)), np.zeros(len(i0))])
        dest[te] = clf.decision_function(sc.transform(Xs[te]))


def auc(s, yy):
    r = rankdata(s); n1 = int(yy.sum()); return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def auc_ci(score):
    by1 = [score[(pid == p) & (y == 1)] for p in range(len(primary))]; by0 = [score[(pid == p) & (y == 0)] for p in range(len(primary))]; rr = np.random.default_rng(S.arm_seed(f"M2_probe_ci_{model}")); v = []
    for b in rr.integers(0, len(primary), (200, len(primary))):
        s1, s0 = np.concatenate([by1[i] for i in b]), np.concatenate([by0[i] for i in b]); v.append(auc(np.r_[s1, s0], np.r_[np.ones(len(s1)), np.zeros(len(s0))]))
    return float(auc(score, y)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


probe = {"pre_erasure_auroc": auc_ci(sc_pre), "post_erasure_auroc": auc_ci(sc_post)}; v = S.m2_verdict({c: (ca, d[c]) for c in ("leace", "random", "hu")}, liver_d, probe["post_erasure_auroc"][2])
res = {"model": model, "cases": len(cases), "n_lesions_detected_at_baseline": len(keys), "n_cases_with_lesions": len(set(ca)), "mean_change_lesion_dice": {k: v[k] for k in ("leace", "random", "hu")}, "paired_leace_minus_random": v["leace_minus_random"], "paired_leace_minus_hu": v["leace_minus_hu"],
       "mean_change_liver_dice_under_leace": v["liver"], "fraction_still_detected": still, "mean_change_coverage": {c: float(np.mean(dcov[c])) for c in dcov}, "probe_check": probe, "verdict": v["verdict"], "flags": v["flags"], "meets_M2": v["meets_M2"]}
json.dump(res, open(os.path.join(out, "m2_results.json"), "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
