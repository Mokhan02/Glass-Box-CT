"""W3 reference-row rows (VISTA-3D class 26, stored automatic prediction) for HCC_019_773068 under the CORRECTED label and the corrected contrast (CPU only; the prediction does not depend on the label). Uses glassbox.seg.w3_lib.lesion_rows unchanged.
Usage (cwd = repo checkout): python scripts/dq_ref_corrected.py <cache root> <corrected root> <corrected contrast csv from dq_impact_case.py> <out csv>"""
import csv
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

cache, fixed_root, impact, out = sys.argv[1:5]; case = "HCC_019_773068"
table = {k: dict(v) for k, v in L.lesion_table("hcc").items() if k[0] == case}
for r in csv.DictReader(open(impact)):
    if r["conversion"] == "corrected":
        table[(case, int(r["rank"]))]["contrast_hu"] = r["contrast_hu"]
pred = np.load(f"{cache}/heldout_pred/hcc_tace_seg/{case}.npz")["pred"]
rows, _ = L.lesion_rows("hcc", case, f"{fixed_root}/hcc_tace_seg", table, {"vista_26_auto": pred})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=L.FIELDS); w.writeheader(); w.writerows(rows)
print(open(out).read())
