"""W2 v2 pilot set, stratified, chosen by seed from the GROUND-TRUTH lesion tables only (no model outcome and no v1 result is read).
Pool: primary lesions (two largest >= 10 mm per case) with a defined contrast sign: kidney = results/w2_kidney/gt_lesions.csv (rank <= 2); HCC = results/exp8_lesions/lesion_table.csv, route == uid.
Per organ 10 lesions, half bright (contrast_hu > 0) and half dark: size stratum <30 mm: 2 bright + 1 dark; 30-60 mm: 2 bright + 2 dark; >60 mm: 1 bright + 2 dark (so >= 3 / >= 3 / >= 2 lesions per stratum, 5 bright, 5 dark).
Draw order fixed: organ (kidney, hcc), stratum (<30, 30-60, >60), group (bright, dark); within each cell a numpy default_rng(SEED) permutation of the case-sorted pool, taking the first lesions whose case has not been used in the pilot.
Writes results/w2_kidney/v2_pilot_selection.csv. Usage: python scripts/w2v2_pilot_select.py"""
import csv

import numpy as np

SEED = 133
STRATA = (("<30 mm", lambda d: d < 30, (2, 1)), ("30-60 mm", lambda d: 30 <= d <= 60, (2, 2)), (">60 mm", lambda d: d > 60, (1, 2)))      # (n bright, n dark)
rng = np.random.default_rng(SEED); out = []
pools = {"kidney": [r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2],
         "hcc": [r for r in csv.DictReader(open("results/exp8_lesions/lesion_table.csv")) if r["route"] == "uid"]}
for organ, pool in pools.items():
    used = set()
    for sname, f, (nb, nd) in STRATA:
        for bright, n in ((True, nb), (False, nd)):
            g = sorted([r for r in pool if f(float(r["diam_mm"])) and (float(r["contrast_hu"]) > 0) == bright], key=lambda r: (r["case"], int(r["rank"])))
            taken = 0
            for i in rng.permutation(len(g)):
                r = g[i]
                if r["case"] in used:
                    continue
                used.add(r["case"]); taken += 1
                out.append({"organ": organ, "case": r["case"], "rank": int(r["rank"]), "stratum": sname, "group": "bright" if bright else "dark", "diam_mm": round(float(r["diam_mm"]), 3), "contrast_hu": round(float(r["contrast_hu"]), 2)})
                if taken == n:
                    break
            assert taken == n, f"pool too small: {organ} {sname} {'bright' if bright else 'dark'}"
with open("results/w2_kidney/v2_pilot_selection.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for r in out:
    print(r)
