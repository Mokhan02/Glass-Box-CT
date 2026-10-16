"""W2 validity check: fixed lesion-selection rule, applied to the existing Gate 3 CSVs and written to a file BEFORE any rerun.
Per organ (kidney class 129 = w2_kidney_lesions.csv; HCC class 130 = w2_hcc_lesions.csv), rel_err = (pred_diam_mm - diam_mm) / diam_mm per lesion; bright = contrast_hu > 0.
  largest: the 2 bright + the 2 dark lesions with the largest rel_err
  typical: the 2 bright + the 2 dark lesions closest to the median rel_err of their own brightness group (within that organ), not already selected
16 lesions in total (ties broken by case name, then rank). Usage: python scripts/w2_validity_select.py results/w2_kidney"""
import csv
import sys

import numpy as np

d = sys.argv[1]; out = []
for organ, f in (("kidney_129", "w2_kidney_lesions.csv"), ("hcc_130", "w2_hcc_lesions.csv")):
    rs = list(csv.DictReader(open(f"{d}/{f}")))
    for r in rs:
        r["rel_err"] = (float(r["pred_diam_mm"]) - float(r["diam_mm"])) / float(r["diam_mm"]); r["bright"] = float(r["contrast_hu"]) > 0
    chosen = set()
    for bright in (True, False):
        g = sorted([r for r in rs if r["bright"] == bright], key=lambda r: (-r["rel_err"], r["case"], int(r["rank"])))
        for r in g[:2]:
            out.append({"organ": organ, "case": r["case"], "rank": int(r["rank"]), "group": "bright" if bright else "dark", "stratum": "largest_error", "diam_mm": r["diam_mm"], "pred_diam_mm": r["pred_diam_mm"], "rel_err": round(r["rel_err"], 4)}); chosen.add((r["case"], r["rank"]))
    for bright in (True, False):
        g = [r for r in rs if r["bright"] == bright]; med = float(np.median([r["rel_err"] for r in g]))
        g = sorted([r for r in g if (r["case"], r["rank"]) not in chosen], key=lambda r: (abs(r["rel_err"] - med), r["case"], int(r["rank"])))
        for r in g[:2]:
            out.append({"organ": organ, "case": r["case"], "rank": int(r["rank"]), "group": "bright" if bright else "dark", "stratum": f"typical(group median {med:+.3f})", "diam_mm": r["diam_mm"], "pred_diam_mm": r["pred_diam_mm"], "rel_err": round(r["rel_err"], 4)})
with open(f"{d}/validity_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out:
    print(r)
