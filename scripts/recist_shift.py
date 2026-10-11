"""RECIST shift: how far do the model's tumour diameters sit from the ground truth, and could that cross a response threshold?
Longest axial diameter of a lesion = largest in-plane Feret distance (max distance between convex-hull vertices) over its axial slices,
in mm using the CT's in-plane voxel size. Lesions are 3D connected components restricted to annotated slices (same as the Dice rule).
Per RECIST 1.1 a measurable target is >= 10 mm, up to 2 targets in one organ (here: the 2 largest GT lesions).
A GT target is 'detected' if any predicted component overlaps it; its predicted diameter is the largest among overlapping components
(a miss counts as 0 mm). Per case we compare the SUM of target diameters, which is what RECIST response is judged on.
Threshold-crossing proxy (no baseline/follow-up pairs exist in this data): RECIST calls partial response at a >= 30% decrease in the sum
and progression at a >= 20% (and >= 5 mm) increase. A case whose measurement error alone is that large could change category from
measurement error alone, so we count cases with relative error beyond +/-20% and +/-30%. This is a proxy, not a response analysis.
Usage: python scripts/recist_shift.py cache/heldout_pred/hcc_tace_seg"""
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

predd = sys.argv[1]
root = "cache/heldout_nii/hcc_tace_seg"


def feret(mask2d, sx, sy):
    pts = np.argwhere(mask2d).astype(float)
    if len(pts) < 2:
        return 0.0
    pts = pts * np.array([sx, sy]) + 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:  # collinear points
        h = pts
    d = np.sqrt(((h[:, None, :] - h[None, :, :]) ** 2).sum(-1))
    return float(d.max() + 0.0)


def lesions(mask3d, sx, sy):
    lab, n = ndimage.label(mask3d, structure=np.ones((3, 3, 3)))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        zs = np.flatnonzero(comp.any(axis=(0, 1)))
        out.append({"id": i, "diam": max(feret(comp[:, :, z], sx, sy) for z in zs), "nvox": int(comp.sum())})
    return lab, out


per_lesion, per_case = [], []
for f in sorted(os.listdir(predd)):
    case = f[:-4]
    img = nib.load(os.path.join(root, case, "ct.nii.gz")); sx, sy = (float(z) for z in img.header.get_zooms()[:2])
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1))
    gt = (seg == 2)[:, :, ann]
    pred = np.load(os.path.join(predd, f))["pred"][:, :, ann]
    glab, gl = lesions(gt, sx, sy)
    gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]  # RECIST targets: >=10 mm, max 2 per organ
    if not gl:
        continue
    plab, pl = lesions(pred, sx, sy)
    pdiam = {l["id"]: l["diam"] for l in pl}
    sg = sp = 0.0
    for l in gl:
        ov = np.unique(plab[(glab == l["id"]) & (plab > 0)])
        pd_ = max([pdiam[o] for o in ov], default=0.0)
        per_lesion.append({"case": case, "gt": l["diam"], "pred": pd_, "detected": len(ov) > 0})
        sg += l["diam"]; sp += pd_
    per_case.append({"case": case, "gt_sum": sg, "pred_sum": sp, "rel": (sp - sg) / sg, "n_targets": len(gl), "missed": any(not p["detected"] for p in per_lesion[-len(gl):])})
    print(f"{case[:16]:17s} targets {len(gl)} GT sum {sg:6.1f} mm  pred sum {sp:6.1f} mm  rel err {(sp - sg) / sg:+.2f}", flush=True)

L = per_lesion; det = [l for l in L if l["detected"]]
err = np.array([l["pred"] - l["gt"] for l in det])
print(f"\n{len(per_case)} cases with >=1 measurable (>=10 mm) GT target; {len(L)} target lesions")
print(f"lesions detected (any overlap): {len(det)}/{len(L)} = {len(det) / len(L):.2f}")
print(f"diameter error on detected lesions (pred - GT): median {np.median(err):+.1f} mm, median |error| {np.median(np.abs(err)):.1f} mm, IQR of |error| {np.percentile(np.abs(err), 25):.1f}-{np.percentile(np.abs(err), 75):.1f} mm")
rel = np.array([c["rel"] for c in per_case]); gsum = np.array([c["gt_sum"] for c in per_case])
print(f"sum-of-diameters relative error per case: median {np.median(rel):+.2f}, median |rel| {np.median(np.abs(rel)):.2f}")
for t in (0.20, 0.30):
    print(f"  cases with |relative error| >= {int(t * 100)}%: {(np.abs(rel) >= t).sum()}/{len(rel)} = {(np.abs(rel) >= t).mean():.2f}"
          f"   (of which underestimates: {((rel <= -t)).sum()}, overestimates: {(rel >= t).sum()})")
full = [c for c in per_case if not c["missed"]]
r2 = np.array([c["rel"] for c in full])
print(f"restricted to cases where every target was detected (n={len(full)}): median |rel| {np.median(np.abs(r2)):.2f}; >=20%: {(np.abs(r2) >= 0.2).mean():.2f}; >=30%: {(np.abs(r2) >= 0.3).mean():.2f}")

# By lesion size: is the bias concentrated in small tumours? (Dice punishes small structures, but mm error and relative error need not.)
import csv as _csv
with open(os.path.join(predd, "..", "recist_lesions.csv"), "w", newline="") as f:
    w = _csv.DictWriter(f, fieldnames=["case", "gt", "pred", "detected"]); w.writeheader(); w.writerows(L)
print("\nby GT lesion diameter (target lesions):")
print(f"{'bin':>10s} {'n':>4s} {'detected':>9s} {'median err (mm)':>16s} {'median rel err':>15s} {'rel err <= -30%':>16s}")
for lo, hi in ((10, 20), (20, 50), (50, 1e9)):
    sel = [l for l in L if lo <= l["gt"] < hi]
    if not sel:
        continue
    dt = [l for l in sel if l["detected"]]
    e = np.array([l["pred"] - l["gt"] for l in dt]); rl = np.array([(l["pred"] - l["gt"]) / l["gt"] for l in sel])
    print(f"{lo:>4d}-{('inf' if hi > 1e8 else int(hi)):>5} {len(sel):4d} {len(dt) / len(sel):9.2f} {np.median(e):+16.1f} {np.median(rl):+15.2f} {(rl <= -0.3).mean():16.2f}")
