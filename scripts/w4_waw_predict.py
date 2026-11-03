"""W4 (WAW-TACE) VISTA-3D predictions (class 26 = liver tumour, class 1 = liver; automatic, whole-volume sliding window, probability > 0.5: the route of the W3 reference row and the W2 positive control) for every series of the listed patients
in a ground-truth root. Writes <out>/<pid>_<series>.npz (bool arrays `tumour`, `liver`, on the CT grid); series already present are skipped (resume). Python of the VISTA venv, cwd = repo checkout.
Usage: python scripts/w4_waw_predict.py <vendored_inferer_dir> <weights> <gt root> <patients file> <out dir> [--limit N]"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg.w2_lib import Runner  # noqa: E402

inf, wts, gt, pfile, out = sys.argv[1:6]; limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
pats = set(open(pfile).read().split()); cases = sorted((d for d in os.listdir(gt) if d.split("_")[0] in pats), key=lambda d: (int(d.split("_")[0]), d))[:limit]; os.makedirs(out, exist_ok=True)
R = Runner(inf, wts); t0 = time.time(); n = 0
for i, c in enumerate(cases):
    dst = os.path.join(out, c + ".npz")
    if os.path.exists(dst):
        continue
    data, img, M = R.prep(os.path.join(gt, c, "ct.nii.gz")); tum = np.asarray(R.predict_auto(data, img, 26)).astype(bool); liv = np.asarray(R.predict_auto(data, img, 1)).astype(bool)
    np.savez_compressed(dst + ".tmp.npz", tumour=tum, liver=liv); os.replace(dst + ".tmp.npz", dst); n += 1
    print(f"[vista w4] {i + 1}/{len(cases)} {c} elapsed {time.time() - t0:.0f}s", flush=True)
print(f"DONE vista w4: {n} new series of {len(cases)}, {time.time() - t0:.0f}s", flush=True)
