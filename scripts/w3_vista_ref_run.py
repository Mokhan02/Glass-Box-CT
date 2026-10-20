"""W3 Stage 1 reference row: VISTA-3D class 26 (automatic, whole-volume sliding window, probability > 0.5; the route of the earlier RECIST run and of the W2-M positive control) on the same HCC UID lesions, written in the
W3 per-lesion format so scripts/w3_analyze.py can reproduce the stored detection / size numbers. Usage (cwd = repo checkout): python scripts/w3_vista_ref_run.py <vendored_inferer_dir> <weights> <outdir> [--limit N]"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg.w2_lib import Runner  # noqa: E402

inferer_dir, weights, outdir = sys.argv[1:4]; limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
table = L.lesion_table("hcc"); table_asconv = L.lesion_table("hcc", as_converted=True); cases = sorted({k[0] for k in table})[:limit]; os.makedirs(outdir, exist_ok=True)
R = Runner(inferer_dir, weights); rows = []; t0 = time.time()
for i, c in enumerate(cases):
    data, img, M = R.prep(os.path.join(L.HCC_ROOT, c, "ct.nii.gz")); pred = R.predict_auto(data, img, 26)
    r, _ = L.lesion_rows("hcc", c, L.case_root("hcc", c, L.HCC_ROOT), table, {"vista_26_auto": pred}); rows += r
    if c in L.corrected_cases():          # sensitivity: as-converted (displaced) label and contrast
        rows += L.lesion_rows("hcc", c, L.HCC_ROOT, table_asconv, {"asconv__vista_26_auto": pred})[0]
    print(f"[vista ref] {i + 1}/{len(cases)} {c} elapsed {time.time() - t0:.0f}s", flush=True)
with open(os.path.join(outdir, "w3_lesions_vista_26_auto_hcc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=L.FIELDS); w.writeheader(); w.writerows(rows)
print(f"DONE vista ref: {len(cases)} cases, {len(rows)} lesions, {time.time() - t0:.0f}s", flush=True)
