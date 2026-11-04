"""W4 (WAW-TACE) diagnosis of the alignment-gate failures (CPU; no detection outcome is read, only the predicted liver against the ground-truth organ, labels 1|2 of seg.nii.gz on the annotated slices).
Per (arm, series) failing the gate in ANY arm of the half: the full shift test (unshifted Dice, best translation in mm per axis, best Dice, gain), the array-index centroid offset of the predicted organ and of its LARGEST connected component against the GT organ, the volumes of the excess (predicted, not GT)
and the missing (GT, not predicted) organ with their centroid offsets from the GT centroid, the flip Dice, and whether the prediction file's shape / orientation code / spacing / affine equal the input CT's (nnU-Net label maps; for the npz arms only the shape can be checked). Series covariates: phase, patient, voxel spacing, shape, orientation code, GT organ and tumour volumes.
Classification (the gate's own definition): MISALIGNED = best translation > 5 mm on some axis AND Dice gain >= 0.05; MODEL-SIDE = otherwise. Writes <out>/align_diag.json and small npz slices for overlays.
Usage (python with nibabel, scipy): python scripts/w4_waw_align_diag.py <gt root> <half dir (pred_<arm>/, score/alignment_<arm>.json)> <out dir> [--overlay case1 case2 ...]"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glassbox.seg import w3_lib as L  # noqa: E402

ARMS = ["vista", "A", "D0", "B50"]


def load_pred(pdir, case):
    npz = os.path.join(pdir, case + ".npz")
    if os.path.exists(npz):
        z = np.load(npz); return z["liver"].astype(bool), None
    import nibabel as nib
    im = nib.load(os.path.join(pdir, case + ".nii.gz")); return np.asarray(im.dataobj) >= 1, im


def cent(m, sp):
    return (np.argwhere(m).mean(0) * np.array(sp)) if m.any() else np.full(3, np.nan)


def one(gt, half, arm, case, keep=None):
    import nibabel as nib
    d = os.path.join(gt, case); ct = nib.load(os.path.join(d, "ct.nii.gz")); sp = tuple(float(z) for z in ct.header.get_zooms()[:3]); seg = np.asarray(nib.load(os.path.join(d, "seg.nii.gz")).dataobj)
    organ, im = load_pred(os.path.join(half, "pred_" + arm), case); ann = (seg > 0).any(axis=(0, 1)); rep = L.alignment_report(organ, seg, ann, sp, tuple(ct.shape)); sh = L.shift_test(organ, seg, ann, sp)
    gto = np.isin(seg[:, :, ann], (1, 2)); pa = organ[:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, ann]; g = cent(gto, sp)
    lab, n = ndimage.label(pa); big = (lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)) if n else pa
    exc, mis = pa & ~gto, gto & ~pa; vox = float(np.prod(sp)) / 1000.0
    hdr = None
    if im is not None:
        hdr = {"shape_equal": tuple(im.shape) == tuple(ct.shape), "orientation_equal": nib.aff2axcodes(im.affine) == nib.aff2axcodes(ct.affine), "spacing_equal": bool(np.allclose(im.header.get_zooms()[:3], ct.header.get_zooms()[:3], atol=1e-3)), "affine_equal": bool(np.allclose(im.affine, ct.affine, atol=1e-3))}
    r = {"arm": arm, "case": case, "phase": case.split("_")[1], "patient": case.split("_")[0], "spacing_mm": [round(v, 3) for v in sp], "shape": list(ct.shape), "orientation": "".join(nib.aff2axcodes(ct.affine)),
         "gt_organ_ml": round(float(gto.sum()) * vox, 1), "gt_tumour_ml": round(float((seg[:, :, ann] == 2).sum()) * vox, 1), "pred_organ_ml": round(float(pa.sum()) * vox, 1), "organ_dice": rep["organ_dice"], "centroid_offset_mm": rep["centroid_offset_mm"], "centroid_offset_xyz_mm": rep["centroid_offset_xyz_mm"],
         "largest_component_offset_mm": [round(float(v), 1) for v in (cent(big, sp) - g)], "n_components": int(n), "largest_component_fraction": round(float(big.sum() / max(pa.sum(), 1)), 3), "excess_ml": round(float(exc.sum()) * vox, 1), "missing_ml": round(float(mis.sum()) * vox, 1),
         "excess_centroid_offset_mm": [round(float(v), 1) for v in (cent(exc, sp) - g)], "missing_centroid_offset_mm": [round(float(v), 1) for v in (cent(mis, sp) - g)], "flip_dice": rep["flip_dice"], "flip_improves": rep["flip_improves_dice"], "shape_ok": rep["shape_ok"],
         "dice_unshifted": sh["dice_unshifted"], "best_dice": sh["best_dice"], "gain": sh["gain"], "best_shift_mm": sh["best_shift_mm"], "header_vs_input": hdr, "gate_pass": bool(rep["pass"] and sh["verdict"] in ("ALIGNED", "MODEL-SIDE")),
         "class": "MISALIGNED" if (sh["max_axis_shift_mm"] > 5.0 and sh["gain"] >= 0.05) else "MODEL-SIDE"}
    if keep is not None:
        zs = np.flatnonzero(ann); mid = zs[len(zs) // 2]; cy = int(np.argwhere(gto.any(axis=(0, 2))).mean()) if gto.any() else seg.shape[1] // 2
        ctv = np.asarray(ct.dataobj); np.savez_compressed(os.path.join(keep, f"{arm}__{case}.npz"), ct_ax=ctv[:, :, mid].astype(np.int16), gt_ax=(np.isin(seg[:, :, mid], (1, 2))), pr_ax=organ[:seg.shape[0], :seg.shape[1], mid],
                                                         ct_cor=ctv[:, cy, :].astype(np.int16), gt_cor=np.isin(seg[:, cy, :], (1, 2)), pr_cor=organ[:seg.shape[0], cy, :seg.shape[2]], spacing=np.array(sp), z=mid, y=cy)
    return r


def main(argv):
    gt, half, out = argv[:3]; ov = argv[argv.index("--overlay") + 1:] if "--overlay" in argv else []; os.makedirs(out, exist_ok=True); fails = {}
    for a in ARMS:
        p = os.path.join(half, "score", f"alignment_{a}.json")
        if os.path.exists(p):
            j = json.load(open(p)); fails[a] = j["series_gate_fail"]; allser = [x["case"] for x in j["per_series"]]
    union = sorted({c for v in fails.values() for c in v}, key=lambda c: (int(c.split("_")[0]), c)); rows = []
    for a in ARMS:
        if a not in fails:
            continue
        for c in union:
            rows.append(one(gt, half, a, c, out if (c in ov) else None)); print(a, c, rows[-1]["class"], rows[-1]["gain"], flush=True)
    # covariates of ALL series (passing ones too): spacing, shape, orientation, organ and tumour volumes (ground truth only)
    import nibabel as nib
    cov = []
    for c in allser:
        d = os.path.join(gt, c); ct = nib.load(os.path.join(d, "ct.nii.gz")); seg = np.asarray(nib.load(os.path.join(d, "seg.nii.gz")).dataobj); sp = tuple(float(z) for z in ct.header.get_zooms()[:3]); vox = float(np.prod(sp)) / 1000.0
        cov.append({"case": c, "phase": c.split("_")[1], "patient": c.split("_")[0], "spacing_mm": [round(v, 3) for v in sp], "shape": list(ct.shape), "orientation": "".join(nib.aff2axcodes(ct.affine)), "gt_organ_ml": round(float(np.isin(seg, (1, 2)).sum()) * vox, 1), "gt_tumour_ml": round(float((seg == 2).sum()) * vox, 1)})
    json.dump({"fails": fails, "union": union, "rows": rows, "covariates_all_series": cov}, open(os.path.join(out, "align_diag.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
