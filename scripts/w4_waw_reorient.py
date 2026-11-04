"""W4 (WAW-TACE): nnU-Net input-orientation handling (DEVIATIONS 23). nnU-Net v2 reads the array axes as given and was trained on LAS volumes; VISTA-3D and the Universal Model reorient on load. For the series whose CT orientation code is not LAS (one of 396: 540_1, LSP):
  prepare <gt root> <half dir> <work dir>   write <work>/in/<case>_0000.nii.gz (the CT reoriented to LAS: a permutation / flip of the array axes, no resampling) and <work>/cases.txt (the non-LAS series of the half)
  restore <gt root> <work dir> <arm> <half dir>   reorient <work>/pred_<arm>/<case>.nii.gz (LAS grid) back to the ORIGINAL grid of the CT (same array shape and affine as ct.nii.gz; labels are exact, a pure axis permutation / flip), keep the original prediction in <half>/pred_<arm>_original/ and put the restored one in <half>/pred_<arm>/
Usage: python scripts/w4_waw_reorient.py prepare|restore ..."""
import os
import shutil
import sys

import nibabel as nib
import numpy as np
from nibabel import orientations as O

TARGET = ("L", "A", "S")


def to_target(img):
    """(data in the target orientation, its affine, the transform used)."""
    start = O.io_orientation(img.affine); end = O.axcodes2ornt(TARGET); t = O.ornt_transform(start, end)
    return O.apply_orientation(np.asarray(img.dataobj), t), img.affine @ O.inv_ornt_aff(t, img.shape), t


def back_to_original(arr, ct_img):
    """arr: array on the target-orientation grid derived from ct_img; returns it on ct_img's own grid."""
    _, aff2, _ = to_target(ct_img); t = O.ornt_transform(O.io_orientation(aff2), O.io_orientation(ct_img.affine)); out = O.apply_orientation(arr, t)
    assert out.shape == tuple(ct_img.shape), (out.shape, ct_img.shape)
    return out


def non_las(gt, cases):
    return [c for c in cases if "".join(nib.aff2axcodes(nib.load(os.path.join(gt, c, "ct.nii.gz")).affine)) != "LAS"]


def main(argv):
    cmd = argv[0]
    if cmd == "prepare":
        gt, half, work = argv[1:4]; cases = sorted(d for d in os.listdir(gt) if os.path.isdir(os.path.join(gt, d)) and d[0].isdigit()); odd = non_las(gt, cases); os.makedirs(os.path.join(work, "in"), exist_ok=True)
        for c in odd:
            img = nib.load(os.path.join(gt, c, "ct.nii.gz")); data, aff, _ = to_target(img); hdr = img.header.copy(); hdr.set_data_dtype(img.get_data_dtype())
            nib.save(nib.Nifti1Image(data, aff, hdr), os.path.join(work, "in", f"{c}_0000.nii.gz")); print(c, "".join(nib.aff2axcodes(img.affine)), "->", "".join(nib.aff2axcodes(aff)), img.shape, "->", data.shape)
        open(os.path.join(work, "cases.txt"), "w").write("\n".join(odd) + "\n" if odd else "")
    elif cmd == "restore":
        gt, work, arm, half = argv[1:5]; cases = open(os.path.join(work, "cases.txt")).read().split(); orig = os.path.join(half, f"pred_{arm}_original"); os.makedirs(orig, exist_ok=True)
        for c in cases:
            ct = nib.load(os.path.join(gt, c, "ct.nii.gz")); lab = nib.load(os.path.join(work, f"pred_{arm}", f"{c}.nii.gz")); arr = back_to_original(np.asarray(lab.dataobj), ct)
            cur = os.path.join(half, f"pred_{arm}", f"{c}.nii.gz")
            if not os.path.exists(os.path.join(orig, f"{c}.nii.gz")):
                shutil.copy2(cur, os.path.join(orig, f"{c}.nii.gz"))
            hdr = lab.header.copy(); nib.save(nib.Nifti1Image(arr.astype(lab.get_data_dtype()), ct.affine, hdr), cur); print("restored", arm, c, arr.shape, "labels", np.unique(arr).tolist())


if __name__ == "__main__":
    main(sys.argv[1:])
