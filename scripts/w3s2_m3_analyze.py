"""W3 Stage 2, M3 analysis (CPU; PREREG_stage2.md; contrast_cf rule). Primary outcome = change in coverage: Arm A D_A = delta_cov(edit) - delta_cov(control A) (expected < 0), Arm B D_B = delta_cov(edit) - delta_cov(control B) (expected > 0); medians with case-bootstrap CIs, exact paired sign tests, verdict SUPPORTED / NOT SUPPORTED / INCONCLUSIVE
(glassbox.seg.w3s2_stats.m3_verdict; SUPPORTED needs at least 10 lesions in each arm). Secondary: the same statistics for the lesion Dice (dice_within_10mm), the count of flagged edits. EXPLORATORY (no verdict): paired projection drop of the lesion's mean mid token (edited minus baseline) on the model's own cross-fitted probe direction
(StandardScaler + logistic probe, tumour vs liver tokens, direction = coef / scale, unit-normalised; the fold of the case) and its Spearman correlation with the coverage drop, case-bootstrap CI.
Usage: python scripts/w3s2_m3_analyze.py <model A|D> <run dir> <tokens dir (M1)> <cases file> <out dir>"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from glassbox.seg import w3s2_stats as S  # noqa: E402

model, rundir, tdir, cases_file, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True); cases = sorted(l.strip() for l in open(cases_file) if l.strip()); cset = set(cases)
primary = sorted(l.strip() for l in open("results/w3_models/stage2/m1/cases_primary74.txt") if l.strip()); fold = S.fold_of_cases(primary)
rows = [r for r in csv.DictReader(open(os.path.join(rundir, "m3_lesions.csv"))) if r["case"] in cset]; by = {}
for r in rows:
    by.setdefault((r["case"], r["rank"]), {})[r["condition"]] = r
res = {"model": model, "cases": len(cases)}; arms = {}
for arm, direction in (("A", -1), ("B", +1)):
    ks = sorted(k for k, v in by.items() if "base" in v and "edit" in v and "control" in v and v["edit"]["arm"] == arm); ca = np.array([k[0] for k in ks])
    de = {m: np.array([float(by[k]["edit"][m]) - float(by[k]["base"][m]) for k in ks]) for m in ("coverage", "dice_within_10mm")}; dc = {m: np.array([float(by[k]["control"][m]) - float(by[k]["base"][m]) for k in ks]) for m in ("coverage", "dice_within_10mm")}
    if len(ks) < 2:
        arms[arm] = {"n": len(ks), "median_edit": (np.nan,) * 3, "median_D": (np.nan,) * 3, "sign_test": {"n": 0, "k": 0, "p": 1.0}}; continue
    prim = S.m3_arm_stats(ca, de["coverage"], de["coverage"] - dc["coverage"], f"{model}_{arm}", direction); prim["flagged_edits"] = int(sum(str(by[k]["edit"]["flagged"]) == "True" for k in ks)); prim["keys"] = ks
    prim["secondary_dice"] = S.m3_arm_stats(ca, de["dice_within_10mm"], de["dice_within_10mm"] - dc["dice_within_10mm"], f"{model}_{arm}_dice", direction); arms[arm] = prim; prim["_ca"], prim["_dcov"] = ca, de["coverage"]
res["verdict"] = S.m3_verdict(arms["A"], arms["B"])
# ---- exploratory projection on the model's own cross-fitted probe direction ----
X, y, pid = [], [], []
for i, c in enumerate(primary):
    z = np.load(os.path.join(tdir, "tokens", c + ".npz")); X.append(z["mid_x"].astype(np.float64)); y.append(z["mid_is_tum"].astype(int)); pid.append(np.full(len(y[-1]), i))
X, y, pid = np.concatenate(X), np.concatenate(y), np.concatenate(pid); f_of = np.array([fold[primary[p]] for p in pid]); rng = np.random.default_rng(0); U = {}


def cap(mask):
    o = []
    for p in np.unique(pid[mask]):
        i = np.flatnonzero(mask & (pid == p)); o.append(rng.choice(i, 200, replace=False) if len(i) > 200 else i)
    return np.concatenate(o)


for f in list(range(5)) + ["all"]:
    tr = np.ones(len(y), bool) if f == "all" else f_of != f; i1, i0 = cap(tr & (y == 1)), cap(tr & (y == 0)); sc = StandardScaler().fit(X[np.r_[i1, i0]])
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000).fit(sc.transform(X[np.r_[i1, i0]]), np.r_[np.ones(len(i1)), np.zeros(len(i0))]); u = clf.coef_[0] / sc.scale_; U[f] = u / np.linalg.norm(u)
expl = {}
for arm in ("A", "B"):
    a = arms[arm]
    if a.get("keys") is None:
        continue
    drops, dcovs, cs_ = [], [], []
    for k, dcv in zip(a["keys"], a["_dcov"]):
        fp = os.path.join(rundir, "tokens", f"{k[0]}_r{k[1]}.npz")
        if os.path.exists(fp):
            z = np.load(fp); drops.append(float(U[fold.get(k[0], "all")] @ (z["edit"].astype(np.float64) - z["base"].astype(np.float64)))); dcovs.append(float(dcv)); cs_.append(k[0])
    if len(drops) >= 5:
        r_ = float(spearmanr(drops, dcovs)[0]); cs_ = np.array(cs_); idx = np.arange(len(drops)); rr = np.random.default_rng(S.arm_seed(f"M3_proj_{model}_{arm}")); uc = sorted(set(cs_)); per = {c: idx[cs_ == c] for c in uc}; v = []
        for b in rr.integers(0, len(uc), (1000, len(uc))):
            ii = np.concatenate([per[uc[j]] for j in b]); v.append(spearmanr(np.array(drops)[ii], np.array(dcovs)[ii])[0])
        expl[arm] = {"n": len(drops), "mean_projection_drop": float(np.mean(drops)), "spearman_drop_vs_coverage_change": r_, "ci": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))], "label": "EXPLORATORY"}
for a in arms.values():
    for k in ("keys", "_ca", "_dcov"):
        a.pop(k, None)
res.update({"arm_A": arms["A"], "arm_B": arms["B"], "exploratory_projection": expl}); json.dump(res, open(os.path.join(out, "m3_results.json"), "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
