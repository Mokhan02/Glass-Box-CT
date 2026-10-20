"""Re-runs the LABEL stage of the fixed UID-route converter (glassbox.data.seg_offset.read_uid_label, as wired into scripts/convert_heldout.py) on every UID-route HCC case and compares the label with the stored original conversion (heldout_nii/.../seg.nii.gz).
Expected: identical for every case except HCC_019_773068, which must equal the corrected conversion (heldout_nii_fixed). Logs case ID, offset (mm) and shift for every non-"matched" case. CPU only.
Usage (cwd = cache root, repo on PYTHONPATH): python scripts/dq_verify_fixed_converter.py <out json> [<case> ...]   (default: all cases with label_method uid)"""
import glob
import json
import os
import re
import sys

import nibabel as nib
import numpy as np
import pydicom

from glassbox.data.seg_offset import read_uid_label

out, cases = sys.argv[1], sys.argv[2:]
root = "heldout_nii/hcc_tace_seg"; cases = cases or sorted(c for c in os.listdir(root) if json.load(open(f"{root}/{c}/segments.json")).get("label_method", "uid") == "uid"); R = {}


def sorted_ct(d):
    ms = [(pydicom.dcmread(f, stop_before_pixels=True), f) for f in sorted(glob.glob(d + "/*.dcm"))]; n = np.cross(*np.array(ms[0][0].ImageOrientationPatient, float).reshape(2, 3))
    ms.sort(key=lambda m: float(np.dot(np.array(m[0].ImagePositionPatient, float), n))); return ms


for case in cases:
    meta = json.load(open(f"{root}/{case}/segments.json")); pid = re.match(r"^(HCC_\d+)", case).group(1)
    seg_dir = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/{meta['seg_series']}")[0]; ct_dir = glob.glob(f"heldout/hcc_tace_seg/{pid}/*/{meta['ct_series']}")[0]
    label, info = read_uid_label(sorted(glob.glob(seg_dir + "/*.dcm"))[0], sorted_ct(ct_dir)); new = label.transpose(2, 1, 0)
    stored = np.asarray(nib.load(f"{root}/{case}/seg.nii.gz").dataobj); same = bool(new.shape == stored.shape and np.array_equal(new, stored))
    rec = {"mode": info["mode"], "offset_mm": [round(v, 3) for v in info["offset_mm"]], "shift_voxels": list(info["shift_voxels"]), "identical_to_stored_original": same}
    if info["mode"] != "matched":
        fixed = np.asarray(nib.load(f"heldout_nii_fixed/hcc_tace_seg/{case}/seg.nii.gz").dataobj); rec["identical_to_corrected_conversion"] = bool(np.array_equal(new, fixed)); print(f"SEG-OFFSET {case}: offset mm {rec['offset_mm']}, shift {rec['shift_voxels']}", flush=True)
    R[case] = rec; print(case, rec["mode"], same, flush=True)
json.dump(R, open(out, "w"), indent=1)
print(f"{len(R)} cases; changed vs stored original: {[c for c, r in R.items() if not r['identical_to_stored_original']]}; non-matched: {[c for c, r in R.items() if r['mode'] != 'matched']}")
