"""Instance-side: export the dataset's own tumour annotation (label 2 of the manual SEG) for each demo scan, on the scan's original grid, to <out>/<demo_id>/reference_tumour.nii.gz.
Run after make_demo_results.py --stage rebuild; the local side then puts it on the viewer grid and draws the dashed outline (glassbox.pipeline.viewer.make_outline_layers).
The SEG sometimes labels only some slices; reference.json records how many, and the viewer says so. Usage: python scripts/export_reference.py [--demo cache/demo] [--out cache/demo_ref]"""
import argparse
import json
import os

import nibabel as nib
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--demo", default="cache/demo"); ap.add_argument("--out", default="cache/demo_ref"); a = ap.parse_args()
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
for sc in json.load(open(os.path.join(a.demo, "index.json")))["scans"]:
    case = sc["source"]["case"]; route = json.load(open(os.path.join(a.demo, sc["id"], "lesions.json")))["provenance"]["label_route"]
    seg = nib.load(os.path.join(ROOTS[route], case, "seg.nii.gz")); arr = np.asarray(seg.dataobj); ann = (arr > 0).any(axis=(0, 1))
    d = os.path.join(a.out, sc["id"]); os.makedirs(d, exist_ok=True)
    nib.save(nib.Nifti1Image((arr == 2).astype(np.uint8), seg.affine), os.path.join(d, "reference_tumour.nii.gz"))
    json.dump({"case": case, "annotated_slices": int(ann.sum()), "total_slices": int(arr.shape[2]), "reference_tumour_voxels": int((arr == 2).sum())}, open(os.path.join(d, "reference.json"), "w"), indent=1)
    print(sc["id"], case, route, "annotated slices", int(ann.sum()), "/", arr.shape[2])
