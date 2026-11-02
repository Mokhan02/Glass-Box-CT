"""W4 scouting (CPU): verify the mask-selection rules the contrast and registration scripts rely on, on the real files (results/w4_waw/SCOUTING.md).
Rules under test: tumour = voxels of a `*_tumor_seg.nrrd` file with value > 0 (the files should hold exactly {0, 1}); liver = TotalSegmentator label 5 of the dataset's organ mask for the same patient and phase.
For --patients: unique values with voxel counts of every tumour mask file of the patient, and of the organ mask of each phase used (the full label histogram), the volume (litres) and the physical centroid of label 5, whether label 5 lies to the patient's right of label 1 (spleen),
and the share of each tumour mask inside label 5 (tumour resampled onto the organ grid). Without --patients it also runs the cohort-wide checks (every tumour mask file and every phase-1/2 organ mask).
Usage (python -I): python -I scripts/w4_waw_mask_check.py <tumour mask dir> <organ mask dir> <out json> [--patients 14 21 455 514 521]"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np
import SimpleITK as sitk

LIVER, SPLEEN, PORTAL = 5, 1, 64      # TotalSegmentator `total` (117 classes, official map_to_binary.py): 5 liver, 1 spleen, 64 portal_vein_and_splenic_vein


def organ_path(odir, pid, ph):
    h = glob.glob(os.path.join(odir, pid, f"{pid}_{ph}_total_segmentator*")); return h[0] if h else None


def hist(arr):
    v, c = np.unique(arr, return_counts=True); return {int(a): int(b) for a, b in zip(v, c)}


def centroid_mm(img, label):
    arr = sitk.GetArrayFromImage(img) == label
    if not arr.any():
        return None
    idx = np.argwhere(arr).mean(0)[::-1]; return [float(x) for x in img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in idx))]


def check_organ(path):
    img = sitk.ReadImage(path); arr = sitk.GetArrayFromImage(img); sp = img.GetSpacing(); h = hist(arr); vox_l = float(np.prod(sp)) / 1e6
    cl, cs = centroid_mm(img, LIVER), centroid_mm(img, SPLEEN)
    return {"file": os.path.basename(path), "dtype": str(arr.dtype), "spacing_mm": [round(x, 3) for x in sp], "n_labels_present": len(h) - (1 if 0 in h else 0), "labels_min_max": [min(h), max(h)], "liver_label5_voxels": h.get(LIVER, 0), "liver_litres": round(h.get(LIVER, 0) * vox_l, 3),
            "spleen_label1_litres": round(h.get(SPLEEN, 0) * vox_l, 3), "portal_splenic_vein_label64_mL": round(h.get(PORTAL, 0) * vox_l * 1000, 1), "label64_present": h.get(PORTAL, 0) > 0, "liver_centroid_mm_LPS": cl, "spleen_centroid_mm_LPS": cs, "liver_right_of_spleen": (cl[0] < cs[0]) if (cl and cs) else None, "direction": [round(x, 3) for x in img.GetDirection()]}, img


def check_tumour(path, organ_img):
    t = sitk.ReadImage(path); arr = sitk.GetArrayFromImage(t); h = hist(arr)
    on_organ_grid = sitk.Resample(t, organ_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8); ta = sitk.GetArrayFromImage(on_organ_grid) > 0; la = sitk.GetArrayFromImage(organ_img) == LIVER
    return {"file": os.path.basename(path), "dtype": str(arr.dtype), "unique_values_voxel_counts": h, "tumour_voxels_gt0": int((arr > 0).sum()), "values_exactly_0_1": sorted(h) == [0, 1], "tumour_volume_cm3": round(float((arr > 0).sum() * np.prod(t.GetSpacing()) / 1000), 2),
            "inside_label5_fraction": round(float((ta & la).sum() / max(ta.sum(), 1)), 4), "tumour_voxels_on_organ_grid": int(ta.sum())}


def main():
    tdir, odir, out = sys.argv[1:4]; pats = sys.argv[sys.argv.index("--patients") + 1:] if "--patients" in sys.argv else None
    files = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(tdir, "*", "*", "*_tumor_seg.nrrd"))):
        pid, ph, idx = re.match(r"(\d+)_(\d)_(\d+)_tumor_seg\.nrrd", os.path.basename(f)).groups(); files[(pid, int(ph))].append(f)
    res = {"patients": {}}
    todo = sorted({p for p, _ in files}, key=int) if pats is None else pats
    organ_summary = []; tum_summary = []
    for pid in todo:
        entry = {"organ": {}, "tumour": []}
        for ph in (1, 2):
            op = organ_path(odir, pid, ph)
            if not op:
                continue
            o, img = check_organ(op); entry["organ"][str(ph)] = o; organ_summary.append(o)
            for f in files.get((pid, ph), []):
                tcheck = check_tumour(f, img); tcheck["phase"] = ph; entry["tumour"].append(tcheck); tum_summary.append(tcheck)
        res["patients"][pid] = entry
        if pats is not None:
            print(json.dumps({"patient": pid, **entry}, indent=1), flush=True)
    pv = np.array([o["portal_splenic_vein_label64_mL"] for o in organ_summary]); liv = np.array([o["liver_litres"] for o in organ_summary]); ins = np.array([t["inside_label5_fraction"] for t in tum_summary])
    res["summary"] = {"patients_checked": len(todo), "organ_masks_checked": len(organ_summary), "tumour_files_checked": len(tum_summary), "tumour_values_exactly_0_1": sum(t["values_exactly_0_1"] for t in tum_summary), "organ_label_max_over_masks": max(o["labels_min_max"][1] for o in organ_summary),
                      "liver_litres_min_median_max": [float(liv.min()), float(np.median(liv)), float(liv.max())], "liver_right_of_spleen": f"{sum(bool(o['liver_right_of_spleen']) for o in organ_summary)} of {len(organ_summary)}",
                      "tumour_inside_label5_fraction_min_median": [float(ins.min()), float(np.median(ins))], "masks_with_inside_fraction_below_0.9": int((ins < 0.9).sum()),
                      "label64_present_in": f"{sum(o['label64_present'] for o in organ_summary)} of {len(organ_summary)}", "label64_mL_min_median_max": [float(pv.min()), float(np.median(pv)), float(pv.max())], "organ_dtype_uint8": all(o["dtype"] == "uint8" for o in organ_summary)}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
