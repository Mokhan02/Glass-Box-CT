"""Fit the deployable tumour quality flag on DEPLOYMENT-STYLE features: whole-scan predicted class-26 volume (cm3), mean probability and mean binary entropy inside the
predicted mask (p > 0.5 over the entire scan, including any false positives elsewhere). Label: tumour Dice < 0.7 against the GT, scored on annotated slices as in the held-out evaluation.
Training cohort: the 75 UID-route HCC cases (labels verifiably linked to images). Empty predictions are handled by a rule, not the model.
Output: glassbox/quality/flag_model.json (coefficients, scaler, threshold) and a report. Usage (instance): python scripts/fit_quality_flag.py glassbox/quality/flag_model.json"""
import csv
import json
import sys

import nibabel as nib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}
ev = {r["case"]: r for r in csv.DictReader(open("cache/heldout_eval/hcc_tace_seg.csv")) if r["target"] == "tumor"}
X, y, cases, empty_rows = [], [], [], []
for c in sorted(acq):
    if acq[c]["route"] != "uid" or c not in ev:
        continue
    p = np.load(f"cache/vp_prob/{c}.npz")["p"].astype(np.float32); z = nib.load(f"cache/heldout_nii/hcc_tace_seg/{c}/ct.nii.gz").header.get_zooms()[:3]
    m = p > 0.5; vol_cm3 = float(m.sum() * np.prod(z) / 1000.0); dice = float(ev[c]["dice"])
    if m.sum() == 0:
        empty_rows.append((c, dice)); continue
    q = np.clip(p[m], 1e-6, 1 - 1e-6); ent = float((-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean())
    X.append([np.log1p(vol_cm3), float(p[m].mean()), ent]); y.append(int(dice < 0.7)); cases.append(c)
X, y, g = np.array(X), np.array(y), np.array(cases)
print(f"cases with a non-empty prediction: {len(y)} (failures {y.sum()}); empty predictions: {len(empty_rows)} (their Dice: {[round(d, 2) for _, d in empty_rows]})")
oof = np.zeros(len(y))
for rep in range(20):
    rng = np.random.default_rng(rep); perm = dict(zip(g, rng.permutation(len(g)))); g2 = np.array([perm[x] for x in g])
    for tr, te in GroupKFold(5).split(X, y, g2):
        sc = StandardScaler().fit(X[tr]); oof[te] += LogisticRegression(C=1.0).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1] / 20
auc = roc_auc_score(y, oof); aucv = roc_auc_score(y, -X[:, 0]); rng = np.random.default_rng(0)
b = [roc_auc_score(y[i], oof[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if 0 < y[i].sum() < len(i)]
print(f"cross-validated AUROC (5-fold grouped, repeated 20x): confidence + volume {auc:.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]  | predicted volume alone {aucv:.3f}")
order = np.argsort(-oof); thr = None
for t in np.unique(oof)[::-1]:
    if (oof[y == 1] >= t).mean() >= 0.90:
        thr = float(t); break
sens, spec = (oof[y == 1] >= thr).mean(), (oof[y == 0] < thr).mean()
print(f"operating point chosen on out-of-fold scores for >= 90% sensitivity: threshold {thr:.3f} -> sensitivity {sens:.2f}, specificity {spec:.2f}, flagged {np.mean(oof >= thr):.2f} of non-empty predictions")
sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0).fit(sc.transform(X), y)
json.dump({"features": ["log1p(predicted_volume_cm3)", "mean_probability_in_mask", "mean_binary_entropy_in_mask"], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
           "flag_threshold_probability": thr, "empty_prediction_rule": f"if no voxel exceeds p = 0.5 the flag cannot assess the structure; report 'no tumour found, which is not the same as no tumour'. In the training data {len(empty_rows)} of the {len(y) + len(empty_rows)} cases had an empty prediction, so this rule was not exercised.", "base_failure_rate_in_training_data": float(y.mean()),
           "trained_on": f"{len(y)} UID-route HCC-TACE-Seg cases with a non-empty prediction (+{len(empty_rows)} empty); cross-validated AUROC {auc:.3f}", "label": "tumour Dice < 0.7", "research_use_only": True}, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
