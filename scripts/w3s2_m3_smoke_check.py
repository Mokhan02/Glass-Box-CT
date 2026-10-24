"""W3 Stage 2, M3 smoke-test checker (CPU). On the 3-case smoke run of scripts/w3s2_m3_run.py it verifies, and exits non-zero on any failure: (1) the unedited pass reproduces Stage 1 (lesion coverage within 1e-3, same detection flag); (2) the edit and the sign-keeping control hit their intended contrast SIGN
(realised lesion mean minus L: Arm A edit > 0, control < 0; Arm B edit < 0, control > 0); (3) the `flagged` column equals the pre-registered rule (|realised mean - target| > 5 HU) on every edited row, and the number of flagged edits is reported; (4) both arms were exercised.
Usage: python scripts/w3s2_m3_smoke_check.py <model A|D> <m3 run dir> <out json>"""
import csv
import json
import sys

model, run, out = sys.argv[1:4]
s1 = {(r["case"], r["rank"]): r for r in csv.DictReader(open("results/w3_models/stage1/A_swin_hcc/w3_lesions_A_swin_hcc.csv" if model == "A" else "results/w3_models/stage1/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv")) if r["arm"] == ("A_swin_shipped" if model == "A" else "D_nnunet")}
rows = list(csv.DictReader(open(f"{run}/m3_lesions.csv"))); R = {"model": model, "n_rows": len(rows), "problems": []}; arms = set(); flagged = 0; edited = 0; base_diffs = []
for r in rows:
    key = (r["case"], r["rank"])
    if r["condition"] == "base":
        d = abs(float(r["coverage"]) - float(s1[key]["coverage"])); base_diffs.append(d)
        if d > 1e-3 or (float(r["coverage"]) >= 0.1) != (float(s1[key]["coverage"]) >= 0.1):
            R["problems"].append(f"base differs from Stage 1 at {key}: {d:.5f}")
        continue
    edited += 1; arms.add(r["arm"]); L, T, c, tgt, real = float(r["L"]), float(r["T"]), float(r["c"]), float(r["target"]), float(r["realised_mean"]); cr = real - L
    ok = {("A", "edit"): cr > 0, ("A", "control"): cr < 0, ("B", "edit"): cr < 0, ("B", "control"): cr > 0}[(r["arm"], r["condition"])]
    if not ok:
        R["problems"].append(f"contrast sign wrong at {key} {r['arm']} {r['condition']}: realised - L = {cr:.1f} (c = {c:.1f})")
    if (abs(real - tgt) > 5.0) != (str(r["flagged"]) == "True"):
        R["problems"].append(f"flag disagrees with the rule at {key} {r['condition']}: |real - target| = {abs(real - tgt):.1f}, flagged {r['flagged']}")
    flagged += str(r["flagged"]) == "True"
if arms != {"A", "B"}:
    R["problems"].append(f"smoke run exercised arms {sorted(arms)}, need both")
R.update({"max_base_vs_stage1_cov_diff": max(base_diffs, default=None), "edited_rows": edited, "flagged_edits": flagged, "arms_exercised": sorted(arms), "pass": not R["problems"]}); json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1)); sys.exit(0 if R["pass"] else 1)
