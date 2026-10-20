"""Would the corrected HCC_019_773068 change the demo selection? (CPU only; demos and main are NOT changed.) Rebuilds the candidate rows exactly as scripts/make_demo_results.py (select branch) does from the instance caches, runs
glassbox.pipeline.select.select_demos on (a) the stored rows (must reproduce the five shipped picks) and (b) the rows with HCC_019's contrast replaced by the corrected contrast and its coverage by the corrected coverage (baseline.csv rank-1 lesion).
The shift_matters slot depends on a cross-validated shifted-threshold coverage that needs the probability map, so it is evaluated only as 'unchanged unless the case qualifies' and reported as not evaluable for the corrected label.
Usage (cwd = repo checkout): python scripts/dq_demo_check.py <cache root> <impact csv> <out json>"""
import csv
import json
import sys

sys.path.insert(0, ".")
from glassbox.pipeline.select import select_demos  # noqa: E402

cache, impact, out = sys.argv[1:4]; CASE = "HCC_019_773068"
cohort = [l.strip() for l in open("results/dq_hcc_labels/clean89_cases.txt")]
shiftm = {}
for r in csv.DictReader(open("results/readout_refit/bright_threshold_lesions.csv")):
    if r["case"] not in shiftm or float(r["diam_mm"]) > shiftm[r["case"]][0]:
        shiftm[r["case"]] = (float(r["diam_mm"]), float(r["cov_orig"]) < 0.10 and float(r["cov_shift"]) >= 0.5)
base = {r["case"]: r for r in csv.DictReader(open(f"{cache}/contrast_cf/baseline.csv")) if r["rank"] == "1"}
rows = []
for case in cohort:
    d = f"{cache}/demo_all/{case}"
    try:
        qa = json.load(open(f"{d}/quality_audit.json"))["structures"][0]; les = json.load(open(f"{d}/lesions.json")); bc = json.load(open(f"{d}/body_composition.json"))
    except FileNotFoundError:
        continue
    if case not in base:
        continue
    rows.append({"id": case, "coverage": float(base[case]["coverage"]), "contrast": float(base[case]["contrast"]), "diam_mm": float(base[case]["diam_mm"]), "flag": qa["flag"], "assessed": bool(qa.get("assessed")), "l3_available": bool(bc.get("available")),
                 "n_lesions": les["n_lesions"], "p": qa.get("failure_probability"), "shift_matters": shiftm.get(case, (0, False))[1]})
corr = next(r for r in csv.DictReader(open(impact)) if r["conversion"] == "corrected"); r19 = next(r for r in rows if r["id"] == CASE)
picks = lambda rs: {p["slot"]: (p["row"]["id"] if p["row"] else None) for p in select_demos(rs)}      # noqa: E731
rows2 = [({**r, "contrast": float(corr["contrast_hu"]), "coverage": float(corr["cover"])} if r["id"] == CASE else r) for r in rows]
cand = lambda rs: [r["id"] for r in rs if r["n_lesions"] and r["assessed"] and r["contrast"] > 0 and r["coverage"] < 0.3 and r["flag"] == "likely wrong"]      # noqa: E731
R = {"n_candidate_rows": len(rows), "hcc_019_row_stored": r19, "hcc_019_row_corrected": {k: (float(corr["contrast_hu"]) if k == "contrast" else float(corr["cover"]) if k == "coverage" else v) for k, v in r19.items()},
     "picks_stored": picks(rows), "picks_corrected": picks(rows2), "bright_missed_flagged_candidates_stored": cand(rows), "bright_missed_flagged_candidates_corrected": cand(rows2)}
R["demo_2_pick_changes"] = R["picks_stored"]["bright_missed_flagged"] != R["picks_corrected"]["bright_missed_flagged"]; R["any_pick_changes_excluding_shift_slot"] = any(R["picks_stored"][s] != R["picks_corrected"][s] for s in R["picks_stored"] if s != "shift_matters")
json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
