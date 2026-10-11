"""Statistics and decision for the contrast-sign counterfactual, implementing results/contrast_cf/PREREG.md mechanically.
Input: edits.csv from scripts/contrast_cf_run.py (one row per lesion x {edit, control}). Output: results/contrast_cf/{results.csv, DECISION.json, STATS.txt}.
Usage: python scripts/contrast_cf_stats.py cache/contrast_cf/edits.csv results/contrast_cf"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
R = list(csv.DictReader(open(src)))
for r in R:
    for k in ("cov0", "cov", "dice0", "dice"):
        r[k] = float(r[k])
    r["flagged"] = r["flagged"] == "True"
by = {}
for r in R:
    by.setdefault((r["case"], r["rank"], r["arm"]), {})[r["cond"]] = r
L = [{"case": k[0], "rank": k[1], "arm": k[2], "edit": v["edit"], "control": v["control"]} for k, v in by.items() if "edit" in v and "control" in v]
rng = np.random.default_rng(0)


def stats(arm, key, direction):
    """direction -1: expect decrease (Arm A); +1: expect increase (Arm B)."""
    ls = [l for l in L if l["arm"] == arm]; n = len(ls)
    if n == 0:
        return {"arm": arm, "n": 0}
    de = np.array([l["edit"][key] - l["edit"][key + "0" if key == "cov" else "dice0"] for l in ls]); dc = np.array([l["control"][key] - l["control"][key + "0" if key == "cov" else "dice0"] for l in ls]); D = de - dc
    cases = np.array([l["case"] for l in ls]); ug = np.unique(cases); idx = {c: np.flatnonzero(cases == c) for c in ug}
    me, mD = [], []
    for _ in range(1000):
        s = np.concatenate([idx[c] for c in rng.choice(ug, len(ug))]); me.append(np.median(de[s])); mD.append(np.median(D[s]))
    nz = D[D != 0]; k = int((nz * direction > 0).sum()); p = binomtest(k, len(nz), 0.5).pvalue if len(nz) else 1.0
    return {"arm": arm, "n": n, "median_delta_edit": float(np.median(de)), "ci_delta_edit": [float(x) for x in np.percentile(me, [2.5, 97.5])], "median_delta_control": float(np.median(dc)),
            "median_D": float(np.median(D)), "ci_D": [float(x) for x in np.percentile(mD, [2.5, 97.5])], "n_D_in_hypothesised_direction": k, "n_D_nonzero": int(len(nz)), "sign_test_p": float(p),
            "n_flagged_edits": int(sum(l["edit"]["flagged"] for l in ls))}


res = {key: {arm: stats(arm, key, -1 if arm == "A" else 1) for arm in ("A", "B")} for key in ("cov", "dice")}
A, B = res["cov"]["A"], res["cov"]["B"]


def meets(s, direction):
    if s["n"] == 0:
        return False
    ci_e, ci_d = s["ci_delta_edit"], s["ci_D"]
    ok_e = ci_e[1] < 0 if direction < 0 else ci_e[0] > 0
    ok_d = ci_d[1] < 0 if direction < 0 else ci_d[0] > 0
    return bool(ok_e and ok_d and s["sign_test_p"] < 0.05 and s["n_D_in_hypothesised_direction"] > s["n_D_nonzero"] / 2)


wrong = lambda s, d: s["n"] > 0 and (s["median_D"] * d <= 0)  # noqa: E731
if min(A.get("n", 0), B.get("n", 0)) < 10:
    verdict = "INCONCLUSIVE (an arm has fewer than 10 lesions)"
elif meets(A, -1) and meets(B, +1):
    verdict = "SUPPORTED"
elif not meets(A, -1) and not meets(B, +1) and wrong(A, -1) and wrong(B, +1):
    verdict = "NOT SUPPORTED"
else:
    verdict = "INCONCLUSIVE"
dec = {"rule": "results/contrast_cf/PREREG.md", "verdict": verdict, "arm_A_meets_all_criteria": meets(A, -1), "arm_B_meets_all_criteria": meets(B, +1), "coverage": res["cov"], "lesion_dice_secondary": res["dice"]}
json.dump(dec, open(os.path.join(out, "DECISION.json"), "w"), indent=1)
with open(os.path.join(out, "results.csv"), "w", newline="") as f:
    cols = ["case", "rank", "arm", "T", "L", "contrast", "cov0", "dice0", "edit_target", "edit_realised", "edit_flagged", "edit_cov", "edit_dice", "control_target", "control_realised", "control_cov", "control_dice"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for l in L:
        e, c = l["edit"], l["control"]
        w.writerow({"case": l["case"], "rank": l["rank"], "arm": l["arm"], "T": e["T"], "L": e["L"], "contrast": e["contrast"], "cov0": e["cov0"], "dice0": e["dice0"], "edit_target": e["target_mean"], "edit_realised": e["realised_mean"],
                    "edit_flagged": e["flagged"], "edit_cov": e["cov"], "edit_dice": e["dice"], "control_target": c["target_mean"], "control_realised": c["realised_mean"], "control_cov": c["cov"], "control_dice": c["dice"]})
lines = [f"VERDICT (pre-registered rule): {verdict}", ""]
for key, name in (("cov", "COVERAGE (decision outcome)"), ("dice", "LESION DICE (secondary)")):
    for arm, label in (("A", "Arm A: dark tumours made bright (expect decrease)"), ("B", "Arm B: bright, missed tumours made dark (expect increase)")):
        s = res[key][arm]
        if s["n"] == 0:
            lines.append(f"{name} {label}: no lesions"); continue
        lines.append(f"{name} {label}: n={s['n']} | median change with edit {s['median_delta_edit']:+.3f} [{s['ci_delta_edit'][0]:+.3f},{s['ci_delta_edit'][1]:+.3f}] | median change with sign-keeping control {s['median_delta_control']:+.3f} | "
                     f"edit minus control: median {s['median_D']:+.3f} [{s['ci_D'][0]:+.3f},{s['ci_D'][1]:+.3f}] | sign test {s['n_D_in_hypothesised_direction']}/{s['n_D_nonzero']} in the hypothesised direction, p={s['sign_test_p']:.4f} | edits flagged (realised mean >5 HU off target): {s['n_flagged_edits']}")
    lines.append("")
open(os.path.join(out, "STATS.txt"), "w").write("\n".join(lines)); print("\n".join(lines))
