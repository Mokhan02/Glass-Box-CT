"""W5 Amendment 3, EXPLORATORY (no verdict; CPU): the M2 erasure effect on lesion Dice for bright and for dark lesions separately (glassbox/seg/w5a3.py, m2_by_group).
Bright = lesion contrast (lesion mean minus liver shell mean) > 0 HU, the W3 definition (`glassbox/seg/w3_lib.lesion_table`).
Usage: python scripts/w5a3_m2_by_brightness.py <M2 run dir (lesions.csv)> <cases file> <out dir>"""
import csv
import json
import os
import sys

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w5a3 as W  # noqa: E402

rundir, cases_file, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); cases = {l.strip() for l in open(cases_file) if l.strip()}
table = L.lesion_table("hcc", include_excluded=True); group_of = {(c, str(r)): ("bright" if float(v["contrast_hu"]) > 0 else "dark") for (c, r), v in table.items()}
by = {}
for r in csv.DictReader(open(os.path.join(rundir, "lesions.csv"))):
    if r["case"] in cases:
        by.setdefault((r["case"], str(r["rank"])), {})[r["arm"].split("_", 1)[1]] = r
res = W.m2_by_group(by, group_of); res["label"] = "EXPLORATORY, no verdict"; json.dump(res, open(os.path.join(out, "m2_by_brightness.json"), "w"), indent=1, default=float)
f = lambda d: f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"      # noqa: E731
L_ = ["# W5 Amendment 3, M2 by brightness (EXPLORATORY, no verdict)", ""]
for g in ("bright", "dark"):
    r = res[g]; L_.append(f"- **{g}**: {r['n_lesions']} detected lesions in {r['n_cases']} cases" + ("" if "leace" not in r else "; " + "; ".join(f"{c}: {f(r[c]['mean_dice_change'])}, still detected {r[c]['still_detected']:.0%}" for c in ("leace", "random", "hu"))))
open(os.path.join(out, "m2_by_brightness.md"), "w").write("\n".join(L_) + "\n"); print("\n".join(L_))
