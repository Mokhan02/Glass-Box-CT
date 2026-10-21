"""Impact of excluding HCC_027_967743 (source-label displacement, rule: exclude from ALL cohorts, sensitivity including it) on top of the HCC_019 correction (CPU, from stored tables). Baseline = HCC_019 corrected, HCC_027 included (what is currently reported);
EXCLUDED = HCC_027 dropped everywhere. Quantities: UID-route and clean-89 sum-of-diameters bias with its case bootstrap CI, detection, W1 HCC bright share and verdicts (scripts/w1_compare.py), W2 v3 verdicts (scripts/w2v3_analyze.py, HCC_019 re-run rows in place),
W3 verdicts for A, D and the VISTA reference row (scripts/w3_analyze.py). Rule: bias moves > 2 pts or any verdict category changes -> tell the researcher. Usage: python scripts/dq_impact_027.py <outdir>"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

out = sys.argv[1]; os.makedirs(out, exist_ok=True); CASE = "HCC_027_967743"; rd = lambda p: list(csv.DictReader(open(p)))      # noqa: E731
uid_rows = [r for r in rd("results/exp8_lesions/lesion_table.csv") if r["route"] == "uid"]; uid = {r["case"] for r in uid_rows}; c89 = [l.strip() for l in open("results/dq_hcc_labels/clean89_cases.txt")]; non_uid = [c for c in c89 if c not in uid]
recist = rd("cache/pulled/recist_lesions.csv"); recist_clean = rd("cache/pulled/clean_recist_lesions.csv"); imp = {r["conversion"]: r for r in rd("results/dq_hcc_labels/impact_HCC_019.csv")}; R = {}


def stats(rows):
    cases = sorted({r["case"] for r in rows}); rel = np.array([(sum(float(r["pred"]) for r in rows if r["case"] == c) - sum(float(r["gt"]) for r in rows if r["case"] == c)) / sum(float(r["gt"]) for r in rows if r["case"] == c) for c in cases])
    d = np.array([np.median(rel[b]) for b in np.random.default_rng(0).integers(0, len(cases), size=(1000, len(cases)))]); det = [r for r in rows if str(r["detected"]) == "True"]
    return {"n_cases": len(cases), "n_lesions": len(rows), "sld_median": float(np.median(rel)), "sld_lo": float(np.percentile(d, 2.5)), "sld_hi": float(np.percentile(d, 97.5)), "n_detected": len(det), "detection": len(det) / len(rows), "detected_median_error_mm": float(np.median([float(r["pred"]) - float(r["gt"]) for r in det]))}


def corr(rows):      # HCC_019 corrected row (the current baseline)
    c = imp["corrected"]; return [({**r, "gt": c["diam_mm"], "pred": c["pred_diam_mm"], "detected": c["detected"], "cover": c["cover"]} if r["case"] == "HCC_019_773068" else r) for r in rows]


for name in ("baseline_027_included", "excluded_027"):
    drop = (lambda rs: [r for r in rs if r["case"] != CASE]) if name.startswith("excl") else (lambda rs: rs)      # noqa: E731
    R.setdefault("recist", {})[name] = {"uid75": stats(drop(corr([r for r in recist if r["case"] in uid]))), "clean89": stats(drop(corr([r for r in recist if r["case"] in uid]) + [r for r in recist_clean if r["case"] in non_uid]))}
    tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/results/exp8_lesions"); os.makedirs(f"{tmp}/w1"); [shutil.copy(f"results/w1_training_audit/{f}", f"{tmp}/w1/") for f in os.listdir("results/w1_training_audit") if f.endswith(".csv")]
    rows = rd("results/exp8_lesions/lesion_table.csv")
    for r in rows:
        if r["case"] == "HCC_019_773068" and r["route"] == "uid":
            r["contrast_hu"] = imp["corrected"]["contrast_hu"]
    rows = [r for r in rows if not (name.startswith("excl") and r["case"] == CASE and r["route"] == "uid")]
    with open(f"{tmp}/results/exp8_lesions/lesion_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    subprocess.run([sys.executable, os.path.abspath("scripts/w1_compare.py"), f"{tmp}/w1"], cwd=tmp, check=True, capture_output=True); txt = open(f"{tmp}/w1/w1_results.md").read().split("\n")
    R.setdefault("w1", {})[name] = {"p_HCC": next(l for l in txt if l.startswith("- p_HCC")), "verdict": next(l for l in txt if "Verdict" in l)}
    tmp = tempfile.mkdtemp()
    for f in os.listdir("results/w2_kidney/v3_hcc019_corrected"):
        if f.startswith("w2v2_") and f.endswith(".csv"):
            rs = rd(f"results/w2_kidney/v3_hcc019_corrected/{f}"); rs = [r for r in rs if not (name.startswith("excl") and r["case"] == CASE)]
            with open(f"{tmp}/{f}", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    subprocess.run([sys.executable, "scripts/w2v3_analyze.py", tmp, tmp], check=True, capture_output=True); j = json.load(open(f"{tmp}/w2v3_results.json"))
    R.setdefault("w2v3", {})[name] = {a: {"gap": j[a]["primary_dice_gap"]["gap"], "verdict": j[a]["verdict"], "incl_pilot_verdict": j[a]["sensitivity_including_pilot"]["verdict"]} for a in ("kidney_133_1", "hcc_26_3", "hcc_133_1")}
    tmp = tempfile.mkdtemp(); S = "results/w3_models/stage1"; parts = []
    for src in (f"{S}/A_swin_hcc/w3_lesions_A_swin_hcc.csv", f"{S}/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv", f"{S}/reference_row/w3_lesions_vista_26_auto_hcc.csv"):
        parts += rd(src)
    parts = [r for r in parts if not (name.startswith("excl") and r["case"] == CASE)]
    with open(f"{tmp}/all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(parts[0])); w.writeheader(); w.writerows(parts)
    subprocess.run([sys.executable, "scripts/w3_analyze.py", tmp, f"{tmp}/all.csv"], check=True, capture_output=True); j = json.load(open(f"{tmp}/w3_results.json"))
    R.setdefault("w3", {})[name] = {a: {"label": j[a]["label"], "det": j[a]["detection_gap"]["gap"], "dice": j[a]["dice_gap"]["gap"], "n_bright": j[a]["detection_gap"]["n_bright"], "n_dark": j[a]["detection_gap"]["n_dark"]} for a in ("A_swin_shipped", "D_nnunet", "vista_26_auto")}
b, e = R["recist"]["baseline_027_included"], R["recist"]["excluded_027"]
R["rule"] = {"bias_move_uid_pts": 100 * (e["uid75"]["sld_median"] - b["uid75"]["sld_median"]), "bias_move_clean89_pts": 100 * (e["clean89"]["sld_median"] - b["clean89"]["sld_median"]), "w1_verdict_changed": R["w1"]["baseline_027_included"]["verdict"] != R["w1"]["excluded_027"]["verdict"],
             "w2_verdict_changed": any(R["w2v3"]["baseline_027_included"][a][k] != R["w2v3"]["excluded_027"][a][k] for a in R["w2v3"]["excluded_027"] for k in ("verdict", "incl_pilot_verdict")), "w3_label_changed": any(R["w3"]["baseline_027_included"][a]["label"] != R["w3"]["excluded_027"][a]["label"] for a in R["w3"]["excluded_027"])}
R["rule"]["triggered"] = bool(abs(R["rule"]["bias_move_uid_pts"]) > 2 or abs(R["rule"]["bias_move_clean89_pts"]) > 2 or R["rule"]["w1_verdict_changed"] or R["rule"]["w2_verdict_changed"] or R["rule"]["w3_label_changed"])
json.dump(R, open(f"{out}/dq_impact_027.json", "w"), indent=1, default=float)
f = lambda s: f"{100 * s['sld_median']:+.1f}% [{100 * s['sld_lo']:+.1f}, {100 * s['sld_hi']:+.1f}] ({s['n_cases']} cases), detection {s['n_detected']}/{s['n_lesions']} = {100 * s['detection']:.1f}%, detected median error {s['detected_median_error_mm']:+.1f} mm"      # noqa: E731
L = ["# Impact of excluding HCC_027_967743 (generated by scripts/dq_impact_027.py; baseline = HCC_019 corrected, HCC_027 included)", ""]
for n in ("baseline_027_included", "excluded_027"):
    L += [f"## {n}", f"- UID route: {f(R['recist'][n]['uid75'])}", f"- clean-89: {f(R['recist'][n]['clean89'])}", f"- W1: {R['w1'][n]['p_HCC']}; {R['w1'][n]['verdict']}", "- W2 v3: " + "; ".join(f"{a}: Dice gap {100 * v['gap']:+.1f} {v['verdict']} (incl. pilot {v['incl_pilot_verdict']})" for a, v in R["w2v3"][n].items()),
          "- W3: " + "; ".join(f"{a}: {v['n_bright']}/{v['n_dark']} detection {100 * v['det']:+.1f}, Dice {100 * v['dice']:+.1f} {v['label']}" for a, v in R["w3"][n].items()), ""]
L += ["## Pre-committed rule", f"- bias move: UID {R['rule']['bias_move_uid_pts']:+.2f} pts, clean-89 {R['rule']['bias_move_clean89_pts']:+.2f} pts; W1 verdict changed {R['rule']['w1_verdict_changed']}; W2 verdict changed {R['rule']['w2_verdict_changed']}; W3 label changed {R['rule']['w3_label_changed']}", f"- **triggered (> 2 pts or any verdict change): {R['rule']['triggered']}**"]
open(f"{out}/dq_impact_027.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
