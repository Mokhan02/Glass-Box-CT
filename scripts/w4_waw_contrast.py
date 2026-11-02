"""W4 scouting (CPU): tumour-minus-liver contrast on the phase each WAW-TACE tumour mask was drawn on (results/w4_waw/SCOUTING.md).
Shared function and shell definition of the project (glassbox.audit.contrast.case_lesions): lesion = 26-connected component with longest axial (Feret) diameter >= 10 mm, contrast = mean HU of the lesion minus the mean HU of non-tumour liver in the 5-20 mm shell, bright = contrast > 0.
Liver = TotalSegmentator label 5 of the same phase (the dataset's own organ masks); reference parenchyma = liver minus ALL tumour voxels of that scan. Every lesion >= 10 mm is kept (n_lesions=None), the two largest per patient-phase are flagged `top2` (the W1/W3 convention).
Usage (python -I): python -I scripts/w4_waw_contrast.py <ct root dir (extracted scans)> <tumour mask dir> <organ mask dir> <out csv> [--patients 10 527 ...]"""
import collections
import csv
import glob
import os
import re
import sys

import nrrd  # noqa: F401  (SimpleITK reads NRRD; pynrrd is used by the inventory script)
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.getcwd())
from glassbox.audit.contrast import case_lesions  # noqa: E402
from glassbox.audit.waw import LIVER, PORTAL_SPLENIC_VEIN  # noqa: E402



def find_ct(root, pid, ph):
    hits = glob.glob(os.path.join(root, "**", f"{pid}_{ph}_scan.nii*"), recursive=True)
    return hits[0] if hits else None


def load_aligned(ct_path, other_path, interp=sitk.sitkNearestNeighbor):
    """(CT array [x, y, z], other array on the CT grid [x, y, z], spacing). The other image is resampled (nearest neighbour, identity transform in physical space) only if its grid differs."""
    ct = sitk.ReadImage(ct_path); o = sitk.ReadImage(other_path)
    same = o.GetSize() == ct.GetSize() and np.allclose(o.GetSpacing(), ct.GetSpacing(), atol=1e-4) and np.allclose(o.GetOrigin(), ct.GetOrigin(), atol=1e-2) and np.allclose(o.GetDirection(), ct.GetDirection(), atol=1e-4)
    if not same:
        o = sitk.Resample(o, ct, sitk.Transform(), interp, 0, o.GetPixelID())
    return sitk.GetArrayFromImage(ct).transpose(2, 1, 0).astype(np.float32), sitk.GetArrayFromImage(o).transpose(2, 1, 0), ct.GetSpacing(), (not same)


def main():
    ctroot, tdir, odir, out = sys.argv[1:5]; only = sys.argv[sys.argv.index("--patients") + 1:] if "--patients" in sys.argv else None
    masks = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(tdir, "*", "*", "*_tumor_seg.nrrd"))):
        pid, ph, idx = re.match(r"(\d+)_(\d)_(\d+)_tumor_seg\.nrrd", os.path.basename(f)).groups(); masks[(pid, int(ph))].append(f)
    rows, skipped = [], []
    for (pid, ph), files in sorted(masks.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
        if only and pid not in only:
            continue
        ctp = find_ct(ctroot, pid, ph)
        if ctp is None:
            skipped.append((pid, ph, "ct missing")); continue
        organ = glob.glob(os.path.join(odir, pid, f"{pid}_{ph}_total_segmentator*"))
        if not organ:
            skipped.append((pid, ph, "organ mask missing")); continue
        ct, liver_lab, sp, res_l = load_aligned(ctp, organ[0]); tumour = np.zeros(ct.shape, bool); resampled_t = False
        for f in files:
            _, t, _, r = load_aligned(ctp, f); tumour |= t > 0; resampled_t |= r
        les = case_lesions(ct, tumour, liver_lab == LIVER, sp, n_lesions=None, min_diam_mm=10.0, exclude=(liver_lab == PORTAL_SPLENIC_VEIN))      # reference = label 5 minus all tumour voxels minus label 64 (glassbox/audit/waw.py)
        for r in les:
            rows.append({"patient": pid, "phase": ph, "rank": r["rank"], "top2": int(r["rank"] <= 2), "diam_mm": round(r["diam_mm"], 2), "volume_mm3": round(r["volume_mm3"], 1), "tumour_hu": round(r["tumour_hu"], 2), "liver_hu": round(r["liver_hu"], 2),
                         "contrast_hu": round(r["contrast_hu"], 2), "bright": int(r["contrast_hu"] > 0), "shell_fallback": int(r["fallback"]), "n_shell_vox": r["n_shell_vox"], "mask_files": len(files), "mask_resampled": int(resampled_t), "organ_resampled": int(res_l),
                         "slice_thickness_mm": round(sp[2], 3)})
        print(pid, ph, len(les), "lesions", flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} lesions from {len({(r['patient'], r['phase']) for r in rows})} patient-phases; skipped {skipped}")


if __name__ == "__main__":
    main()
