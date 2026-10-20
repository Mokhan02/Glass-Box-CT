"""Impact of the HCC_019_773068 label displacement on the headline numbers (CPU only, from stored tables; the corrected-case rows come from scripts/dq_impact_case.py and scripts/dq_ref_corrected.py).
Scenarios: ORIGINAL (as published), EXCLUDED (the case dropped), CORRECTED (the case with the SEG-declared offset honoured: its contrast and ground truth change, its VISTA-3D automatic prediction does not).
Quantities: UID-route (75 cases / 86 lesions) and clean-89 sum-of-diameters bias with its case bootstrap CI, detection (coverage >= 0.10), detected-lesion median error; the W1 HCC bright share and verdicts (scripts/w1_compare.py unchanged);
the W2 v3 HCC verdicts under EXCLUSION (scripts/w2v3_analyze.py unchanged; the CORRECTED point-prompt arms need new GPU runs and are not computed here); the W3 reference row under all three scenarios (scripts/w3_analyze.py unchanged).
Usage: python scripts/dq_impact.py <dir with impact_HCC_019.csv, ref_corrected_HCC_019.csv> <outdir>"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True); CASE = "HCC_019_773068"
rd = lambda p: list(csv.DictReader(open(p)))      # noqa: E731
uid_rows = [r for r in rd("results/exp8_lesions/lesion_table.csv") if r["route"] == "uid"]; uid = {r["case"] for r in uid_rows}
c89 = [l.strip() for l in open("results/dq_hcc_labels/clean89_cases.txt")]; non_uid = [c for c in c89 if c not in uid]
recist = rd("cache/pulled/recist_lesions.csv"); recist_clean = rd("cache/pulled/clean_recist_lesions.csv")
imp = {r["conversion"]: r for r in rd(f"{src}/impact_HCC_019.csv")}
assert abs(float(imp["original"]["contrast_hu"]) - float(next(r for r in uid_rows if r["case"] == CASE)["contrast_hu"])) < 0.1, "original contrast does not reproduce the stored table"
R = {}


def recist_stats(rows):
    cases = sorted({r["case"] for r in rows}); rel = np.array([(sum(float(r["pred"]) for r in rows if r["case"] == c) - sum(float(r["gt"]) for r in rows if r["case"] == c)) / sum(float(r["gt"]) for r in rows if r["case"] == c) for c in cases])
    d = np.array([np.median(rel[b]) for b in np.random.default_rng(0).integers(0, len(cases), size=(1000, len(cases)))]); det = [r for r in rows if str(r["detected"]) == "True"]
    return {"n_cases": len(cases), "n_lesions": len(rows), "sld_median": float(np.median(rel)), "sld_lo": float(np.percentile(d, 2.5)), "sld_hi": float(np.percentile(d, 97.5)), "detection": len(det) / len(rows), "n_detected": len(det),
            "detected_median_error_mm": float(np.median([float(r["pred"]) - float(r["gt"]) for r in det]))}


def scenario_rows(base, name):
    if name == "excluded":
        return [r for r in base if r["case"] != CASE]
    if name == "corrected":     # same prediction, same GT diameter; coverage and detection from the corrected label
        c = imp["corrected"]; return [({**r, "gt": c["diam_mm"], "pred": c["pred_diam_mm"], "detected": c["detected"], "cover": c["cover"]} if r["case"] == CASE else r) for r in base]
    return base


for name in ("original", "excluded", "corrected"):
    u = scenario_rows([r for r in recist if r["case"] in uid], name); c = scenario_rows([r for r in recist if r["case"] in uid] + [r for r in recist_clean if r["case"] in non_uid], name)
    R.setdefault("recist", {})[name] = {"uid75": recist_stats(u), "clean89": recist_stats(c)}
# W1: bright share and verdicts (unchanged script, run in a temporary working copy)
for name in ("original", "excluded", "corrected"):
    tmp = tempfile.mkdtemp(); os.makedirs(f"{tmp}/results/exp8_lesions"); os.makedirs(f"{tmp}/w1"); [shutil.copy(f"results/w1_training_audit/{f}", f"{tmp}/w1/") for f in os.listdir("results/w1_training_audit") if f.endswith(".csv")]
    rows = rd("results/exp8_lesions/lesion_table.csv")
    if name == "excluded":
        rows = [r for r in rows if not (r["case"] == CASE and r["route"] == "uid")]
    if name == "corrected":
        for r in rows:
            if r["case"] == CASE and r["route"] == "uid":
                r["contrast_hu"] = imp["corrected"]["contrast_hu"]
    with open(f"{tmp}/results/exp8_lesions/lesion_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    subprocess.run([sys.executable, os.path.abspath("scripts/w1_compare.py"), f"{tmp}/w1"], cwd=tmp, check=True, capture_output=True)
    txt = open(f"{tmp}/w1/w1_results.md").read().split("\n"); R.setdefault("w1", {})[name] = {"p_HCC": next(l for l in txt if l.startswith("- p_HCC")), "d": next(l for l in txt if l.startswith("- d =")), "verdict": next(l for l in txt if "Verdict" in l)}
# W2 v3 under EXCLUSION
for name in ("original", "excluded"):
    tmp = tempfile.mkdtemp()
    for f in os.listdir("results/w2_kidney/v3"):
        if f.endswith(".csv"):
            rs = rd(f"results/w2_kidney/v3/{f}"); rs = [r for r in rs if not (name == "excluded" and r["case"] == CASE)]
            with open(f"{tmp}/{f}", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    subprocess.run([sys.executable, "scripts/w2v3_analyze.py", tmp, tmp], check=True, capture_output=True); j = json.load(open(f"{tmp}/w2v3_results.json"))
    R.setdefault("w2v3", {})[name] = {a: {"gap": j[a]["primary_dice_gap"], "verdict": j[a]["verdict"], "sens_incl_pilot_verdict": j[a]["sensitivity_including_pilot"]["verdict"], "size_adjusted_gone": j[a]["size_adjusted"]["adjusted_effect_gone"]} for a in ("kidney_133_1", "hcc_26_3", "hcc_133_1")}
# W3 reference row, three scenarios
ref = rd("results/w3_models/stage1/reference_row/w3_lesions_vista_26_auto_hcc.csv"); corr = rd(f"{src}/ref_corrected_HCC_019.csv")
for name in ("original", "excluded", "corrected"):
    rs = [r for r in ref if not (name == "excluded" and r["case"] == CASE)]
    if name == "corrected":
        rs = [r for r in rs if r["case"] != CASE] + corr
    tmp = tempfile.mkdtemp()
    with open(f"{tmp}/ref.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
    subprocess.run([sys.executable, "scripts/w3_analyze.py", tmp, f"{tmp}/ref.csv"], check=True, capture_output=True); j = json.load(open(f"{tmp}/w3_results.json"))["vista_26_auto"]
    R.setdefault("w3_reference_row", {})[name] = {"detection_gap": j["detection_gap"], "detection_verdict": j["detection_verdict"], "dice_gap": j["dice_gap"], "dice_verdict": j["dice_verdict"], "label": j["label"], "n_bright": j["detection_gap"]["n_bright"], "n_dark": j["detection_gap"]["n_dark"]}
json.dump(R, open(f"{out}/dq_impact.json", "w"), indent=1, default=float)
f = lambda s: f"{100 * s['sld_median']:+.1f}% [{100 * s['sld_lo']:+.1f}, {100 * s['sld_hi']:+.1f}] ({s['n_cases']} cases), detection {s['n_detected']}/{s['n_lesions']} = {100 * s['detection']:.1f}%, detected median error {s['detected_median_error_mm']:+.1f} mm"      # noqa: E731
L = ["# Impact of the HCC_019_773068 label displacement (generated by scripts/dq_impact.py)", ""]
for name in ("original", "excluded", "corrected"):
    L += [f"## {name}", f"- UID route: {f(R['recist'][name]['uid75'])}", f"- clean-89: {f(R['recist'][name]['clean89'])}", f"- W1: {R['w1'][name]['p_HCC']}; {R['w1'][name]['d']}; {R['w1'][name]['verdict']}"]
    if name in R["w2v3"]:
        L.append("- W2 v3 (excluded = case dropped; corrected not computed, needs GPU): " + "; ".join(f"{a}: Dice gap {100 * v['gap']['gap']:+.1f} [{100 * v['gap']['lo']:+.1f}, {100 * v['gap']['hi']:+.1f}] {v['verdict']}" for a, v in R["w2v3"][name].items()))
    w3 = R["w3_reference_row"][name]; L += [f"- W3 reference row (VISTA 26 auto): {w3['n_bright']} bright / {w3['n_dark']} dark; detection gap {100 * w3['detection_gap']['gap']:+.1f} [{100 * w3['detection_gap']['lo']:+.1f}, {100 * w3['detection_gap']['hi']:+.1f}] {w3['detection_verdict']}; Dice gap {100 * w3['dice_gap']['gap']:+.1f} [{100 * w3['dice_gap']['lo']:+.1f}, {100 * w3['dice_gap']['hi']:+.1f}] {w3['dice_verdict']}; {w3['label']}", ""]
open(f"{out}/dq_impact.md", "w").write("\n".join(L)); print("\n".join(L))
