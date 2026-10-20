"""Diagnostic for the model D alignment-check failures (no tumour metric): for each failing case, is the large centroid offset caused by spurious predicted-organ blobs away from the liver (nnU-Net default has no connected-component post-processing)?
Reports per case: organ Dice and centroid offset as scored, the same with only the largest connected component of the predicted organ (annotated slices), the number of components and the volume share outside the largest one.
Usage (cwd = repo checkout): python scripts/w3_modelD_gate_diag.py <nnU-Net prediction dir> <alignment json> <out json>"""
import json
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402

pdir, aj, out = sys.argv[1:4]; al = json.load(open(aj)); R = {}
for c in al["failures"]:
    lab = np.asarray(nib.load(f"{pdir}/{c}.nii.gz").dataobj) >= 1; root = L.case_root("hcc", c, L.HCC_ROOT); _, geo = L.lesion_rows("hcc", c, root, {}, {}); seg, ann, sp = geo["seg"], geo["ann"], geo["spacing"]
    pa = lab[:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, ann]; cc, n = ndimage.label(pa, structure=np.ones((3, 3, 3))); sizes = np.bincount(cc.ravel())[1:]; big = cc == (1 + int(sizes.argmax())) if n else pa
    rep = L.alignment_report(np.pad(big, ((0, 0), (0, 0), (0, 0))), seg[:, :, ann], np.ones(int(ann.sum()), bool), sp, big.shape)
    R[c] = {"scored": next(x for x in al["per_case"] if x["case"] == c) | {}, "n_components": int(n), "share_outside_largest": round(1 - float(sizes.max() / sizes.sum()), 3) if n else None, "largest_component_only": {k: rep[k] for k in ("organ_dice", "centroid_offset_mm", "pass")}}
    print(c, "scored dice", R[c]["scored"]["organ_dice"], "offset", R[c]["scored"]["centroid_offset_mm"], "| comps", n, "outside-largest share", R[c]["share_outside_largest"], "| largest only dice", rep["organ_dice"], "offset", rep["centroid_offset_mm"], flush=True)
json.dump(R, open(out, "w"), indent=1, default=float)
