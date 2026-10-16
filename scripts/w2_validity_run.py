"""[helpers copied verbatim from scripts/w2_point_run.py for the validity rerun] W2 Gate 3 runner (results/w2_kidney/PREREG.md). One positive point per lesion at its interior point (voxel of maximum 3D distance to the lesion boundary), label prompt = one class.
Arms: kidney 129 (C4KC-KiTS), hcc 130 (HCC UID route), hcc26 (class 26 + the same point, sensitivity). The VISTA-3D point path is the official one (Vista3dInferer with use_point_window=True, as in the HF pipeline);
points are given in the loaded image's voxel indices and mapped with inv(preprocessed affine) @ original affine, exactly as the pipeline's transform_points does.
Usage: python scripts/w2_validity_run.py x <vendored_inferer_dir> <weights> <outdir>"""
import argparse
import csv
import math
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.audit.contrast import select_lesions  # noqa: E402
from glassbox.data.preprocess import build_transforms  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("inferer_dir"); ap.add_argument("weights"); ap.add_argument("outdir")
ap.add_argument("--arms", default="kidney,hcc,hcc26"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--kits", default="/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits")
ap.add_argument("--hcc", default="/lambda/nfs/CTGate/glassbox/cache/heldout_nii/hcc_tace_seg"); a = ap.parse_args()
sys.path.insert(0, a.inferer_dir); from inferer import Vista3dInferer  # noqa: E402  (NVIDIA's point-window inferer, vendored at run time, not committed)

dev = torch.device("cuda"); net = load_network(a.weights, dev)
inferer = Vista3dInferer(roi_size=(128, 128, 128), overlap=0.3, use_point_window=True, sw_batch_size=1)
ARMS = {"kidney": ("kidney", 129, a.kits), "hcc": ("hcc", 130, a.hcc), "hcc26": ("hcc", 26, a.hcc)}
pre = build_transforms("image")


def feret(m2, sx, sy):
    from scipy.spatial import ConvexHull
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def axial_diam(mask, sx, sy):
    zs = np.flatnonzero(mask.any(axis=(0, 1)))
    return max((feret(mask[:, :, z], sx, sy) for z in zs), default=0.0)


def prep(path):
    data = pre({"image": path}); img = data["image"].unsqueeze(0).to(dev)
    return data, img, np.linalg.inv(np.array(data["image"].affine, float)) @ np.array(data["image"].meta["original_affine"], float)


def predict(data, img, M, cls, point_vox):
    """Binary prediction on the ORIGINAL grid (prob > 0.5) for one class prompt and one positive point (original voxel indices)."""
    from monai.data import MetaTensor
    from monai.transforms import Invertd
    p = M @ np.array([*point_vox, 1.0]); pts = torch.tensor(p[:3][None, None, :], dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = inferer(inputs=img, network=net, point_coords=pts, point_labels=torch.ones(1, 1, device=dev), class_vector=torch.tensor([[cls]], device=dev))
    probs = torch.sigmoid(logits.reshape(1, *img.shape[2:]).float().cpu())
    pred = MetaTensor(probs, meta=data["image"].meta, applied_operations=data["image"].applied_operations)
    out = Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True)({"pred": pred, "image": data["image"]})
    return out["pred"].as_tensor()[0].numpy() > 0.5


def bbox(mask, pad, shape):
    idx = np.nonzero(mask); lo = [max(int(i.min()) - pad, 0) for i in idx]; hi = [min(int(i.max()) + pad + 1, n) for i, n in zip(idx, shape)]
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def interior_point(comp, sp):
    """Voxel of maximum distance to the boundary, ties to the lowest flat index. Computed on the component's bounding box (+1 voxel), which gives the same distances and the same tie order as the full volume."""
    bb = bbox(comp, 1, comp.shape); c = comp[bb]
    d = ndimage.distance_transform_edt(np.pad(c, 1), sampling=sp)[1:-1, 1:-1, 1:-1]
    loc = np.unravel_index(np.argmax(np.where(c, d, -1)), c.shape)
    return tuple(int(l + b.start) for l, b in zip(loc, bb))


def lesions_for(organ, case, root):
    nii = nib.load(os.path.join(root, case, "ct.nii.gz")); sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann)
    lab, sel = select_lesions((seg == 2)[:, :, ann], sp)                     # annotated slices only, as exp8 and the W2 ground-truth table
    return nii, sp, seg, ann, zidx, lab, sel



import json
sel = list(csv.DictReader(open("results/w2_kidney/validity_selection.csv")))
os.makedirs(a.outdir, exist_ok=True)
CSV = {"kidney_129": ("results/w2_kidney/w2_kidney_lesions.csv", 129, a.kits, "kidney"), "hcc_130": ("results/w2_kidney/w2_hcc_lesions.csv", 130, a.hcc, "hcc")}
mismatch = []; by_case = {}
for r in sel:
    by_case.setdefault((r["organ"], r["case"]), []).append(int(r["rank"]))
for (org, case), ranks in by_case.items():
    csvp, cls, root, organ = CSV[org]; ref_rows = {(x["case"], int(x["rank"])): x for x in csv.DictReader(open(csvp))}
    nii, sp, seg, ann, zidx, lab, sel_l = lesions_for(organ, case, root); ct = np.asarray(nii.dataobj).astype(np.float32)
    seg_nii = nib.load(os.path.join(root, case, "seg.nii.gz"))
    data, img, M = prep(os.path.join(root, case, "ct.nii.gz"))
    for rank in ranks:
        diam, cid = sel_l[rank - 1]; comp_c = lab == cid; pt_c = interior_point(comp_c, sp)
        comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
        pred = predict(data, img, M, cls, pt)[:seg.shape[0], :seg.shape[1], :seg.shape[2]]; pred_a = pred[:, :, ann]
        cov = float((pred_a & comp_c).sum() / comp_c.sum())
        bb = bbox(comp_c, int(math.ceil(10.0 / min(sp))) + 2, comp_c.shape); dist = ndimage.distance_transform_edt(~comp_c[bb], sampling=sp); near = pred_a[bb] & (dist <= 10.0)
        dice10 = float(2 * (near & comp_c[bb]).sum() / max(near.sum() + comp_c.sum(), 1)); pdiam = 0.0
        if (pred_a & comp_c).any():
            plab, pn = ndimage.label(pred_a, structure=np.ones((3, 3, 3))); ov = [i for i in np.unique(plab[comp_c]) if i > 0]
            pdiam = max((axial_diam(plab == i, sp[0], sp[1]) for i in ov), default=0.0)
        ref = ref_rows[(case, rank)]
        diffs = {"coverage": abs(cov - float(ref["coverage"])), "dice_within_10mm": abs(dice10 - float(ref["dice_within_10mm"])), "pred_diam_mm": abs(pdiam - float(ref["pred_diam_mm"]))}
        ok = diffs["coverage"] <= 2e-5 and diffs["dice_within_10mm"] <= 2e-5 and diffs["pred_diam_mm"] <= 2e-3 and (pt == (int(ref["point_x"]), int(ref["point_y"]), int(ref["point_z"])))
        print(f"{org} {case} rank {rank}: coverage {cov:.5f} (csv {ref['coverage']}), dice10 {dice10:.5f} (csv {ref['dice_within_10mm']}), pred_diam {pdiam:.3f} (csv {ref['pred_diam_mm']}) -> {'MATCH' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            mismatch.append((org, case, rank, diffs))
        z = pt[2]
        np.savez_compressed(os.path.join(a.outdir, f"{org}__{case}__r{rank}.npz"), pred=pred, seg=seg.astype(np.uint8), lesion=comp, point=np.array(pt), ct_slice=ct[:, :, z].astype(np.float32),
                            ct_affine=nii.affine, seg_affine=seg_nii.affine, ct_zooms=np.array(nii.header.get_zooms()[:3]), seg_zooms=np.array(seg_nii.header.get_zooms()[:3]), ct_shape=np.array(ct.shape), seg_shape=np.array(seg.shape),
                            ann=ann, aff_pre=np.array(data["image"].affine, float), orig_affine=np.array(data["image"].meta["original_affine"], float), cls=cls, csv_pred_diam=float(ref["pred_diam_mm"]))
print("DETERMINISM CHECK", "PASSED" if not mismatch else f"FAILED: {mismatch}", f"({sum(len(v) for v in by_case.values())} lesions)")
sys.exit(1 if mismatch else 0)
