"""W3 Stage 2, M4 analysis (CPU; PREREG_stage2.md): readout-fix generality by a nested-CV threshold shift on the saved tumour decision-margin maps (the M2 runner's unhooked pass). Groups (bright / dark per lesion rank) from the cohort lesion table
(HCC_019 corrected; the grid's lesion ranks are matched to the table ranks by diameter, mismatches above 25% are reported). Usage: python scripts/w3s2_m4_analyze.py <model A|D> <maps dir> <cases file> <out dir>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w3s2_m4 as M  # noqa: E402

model, mapdir, cases_file, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True); cases = sorted(l.strip() for l in open(cases_file) if l.strip()); table = L.lesion_table("hcc", include_excluded=True)
maps, group, mism = {}, {}, []
for c in cases:
    z = np.load(os.path.join(mapdir, f"{c}.npz")); cm = M.CaseMap(z["margin"], z["label"], z["spacing"]); rk = sorted(r for (cc, r) in table if cc == c)
    if not cm.lesions:
        continue
    g = []
    for i, les in enumerate(cm.lesions):
        r = rk[i] if i < len(rk) else None; row = table.get((c, r)) if r else None
        if row is None:
            g.append("dark"); mism.append((c, i + 1, "no table lesion")); continue
        if abs(les["diam"] / float(row["diam_mm"]) - 1) > 0.25:
            mism.append((c, i + 1, round(les["diam"], 1), round(float(row["diam_mm"]), 1)))
        g.append("bright" if float(row["contrast_hu"]) > 0 else "dark")
    maps[c], group[c] = cm, g
used = sorted(maps); res = M.m4_run(used, maps, group, arm=f"M4_{model}"); res.update({"model": model, "n_cases": len(used), "n_lesions": int(sum(len(maps[c].lesions) for c in used)), "diameter_mismatches_gt_25pct": mism[:30], "n_mismatch": len(mism)})
json.dump(res, open(os.path.join(out, "m4_results.json"), "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
