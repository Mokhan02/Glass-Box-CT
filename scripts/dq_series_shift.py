"""Is the HCC_027 label displaced relative to the series it references because the other 2.5 mm series of the same study (different acquisition number, same z grid) is anatomically shifted? (CPU only.) Loads the CT series used and the other series with the
same slice grid, and finds the through-plane shift (in slices) maximising the normalised cross-correlation of the two volumes (soft-tissue window, 4x in-plane downsampling). Usage (cwd = cache root): python scripts/dq_series_shift.py <case> <used series suffix> <other series suffix> <out json>"""
import glob
import json
import re
import sys

import numpy as np
import pydicom
import SimpleITK as sitk

case, used, other, out = sys.argv[1:5]; pid = re.match(r"^(HCC_\d+)", case).group(1)


def vol(suffix):
    d = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/CT_*{suffix}")[0]; fs = sorted(glob.glob(d + "/*.dcm")); hs = [(pydicom.dcmread(f, stop_before_pixels=True), f) for f in fs]; n = np.cross(*np.array(hs[0][0].ImageOrientationPatient, float).reshape(2, 3))
    hs.sort(key=lambda m: float(np.dot(np.array(m[0].ImagePositionPatient, float), n))); r = sitk.ImageSeriesReader(); r.SetFileNames([f for _, f in hs]); a = sitk.GetArrayFromImage(r.Execute()).astype(np.float32)
    return np.clip(a, -50, 250)[:, ::4, ::4]


A, B = vol(used), vol(other); res = {}
for s in range(-12, 13):
    if s >= 0:
        a, b = A[s:], B[:len(B) - s]
    else:
        a, b = A[:len(A) + s], B[-s:]
    a, b = a - a.mean(), b - b.mean(); res[s] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
best = max(res, key=res.get); R = {"case": case, "used_series": used, "other_series": other, "shape_used": list(A.shape), "shape_other": list(B.shape), "ncc_by_shift_slices": {str(k): round(v, 4) for k, v in res.items()}, "best_shift_slices": best, "best_ncc": round(res[best], 4), "ncc_at_zero": round(res[0], 4)}
json.dump(R, open(out, "w"), indent=1); print({k: R[k] for k in ("best_shift_slices", "best_ncc", "ncc_at_zero")})
