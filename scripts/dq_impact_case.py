"""Per-lesion quantities of ONE HCC case under the ORIGINAL conversion and under the CORRECTED conversion (CPU only): contrast (the W1 / exp8 function, portal vein excluded from the reference as in w1_repro_hcc.py) and the RECIST / detection quantities of
scripts/recist_shift.py computed from the STORED VISTA-3D class-26 prediction (the prediction does not depend on the label). Writes one CSV row per lesion and conversion.
Usage (cwd = repo checkout, cache root given): python scripts/dq_impact_case.py <cache root> <case> <corrected root> <out csv>"""
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.audit.contrast import case_lesions, select_lesions  # noqa: E402
from glassbox.seg.w2_lib import axial_diam  # noqa: E402

cache, case, fixed_root, out = sys.argv[1:5]
pred_full = np.load(f"{cache}/heldout_pred/hcc_tace_seg/{case}.npz")["pred"]; rows = []
for name, root in (("original", f"{cache}/heldout_nii/hcc_tace_seg"), ("corrected", f"{fixed_root}/hcc_tace_seg")):
    nii = nib.load(f"{root}/{case}/ct.nii.gz"); sp = tuple(float(z) for z in nii.header.get_zooms()[:3]); ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(f"{root}/{case}/seg.nii.gz").dataobj)
    ann = (seg > 0).any(axis=(0, 1)); idx = np.flatnonzero(ann); sub = lambda m: m[:, :, idx]      # noqa: E731
    les = case_lesions(sub(ct), sub(seg == 2), sub(seg == 1), sp, exclude=sub(seg == 3)); lab, sel = select_lesions(sub(seg == 2), sp); pred = pred_full[:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, idx]
    plab, _ = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    for r, (d, cid) in zip(les, sel):
        g = lab == cid; cover = float((pred & g).sum()) / int(g.sum()); ov = np.unique(plab[g & (plab > 0)]); det = cover >= 0.10
        pd_ = max([axial_diam(plab == o, sp[0], sp[1]) for o in ov], default=0.0) if det else 0.0
        rows.append({"conversion": name, "case": case, "rank": r["rank"], "diam_mm": round(r["diam_mm"], 3), "contrast_hu": round(r["contrast_hu"], 3), "bright": r["contrast_hu"] > 0, "cover": round(cover, 5), "detected": det, "pred_diam_mm": round(pd_, 3),
                     "n_annotated_slices": int(ann.sum())})
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(open(out).read())
