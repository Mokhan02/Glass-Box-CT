"""W2 Gate 3 runner (results/w2_kidney/PREREG.md). One positive point per lesion at its interior point (voxel of maximum 3D distance to the lesion boundary), label prompt = one class.
Arms: kidney 129 (C4KC-KiTS), hcc 130 (HCC UID route), hcc26 (class 26 + the same point, sensitivity). The VISTA-3D point path is the official one (Vista3dInferer with use_point_window=True, as in the HF pipeline);
points are given in the loaded image's voxel indices and mapped with inv(preprocessed affine) @ original affine, exactly as the pipeline's transform_points does.
Usage: python scripts/w2_point_run.py convention|run <vendored_inferer_dir> <weights> <outdir> [--arms kidney,hcc,hcc26] [--limit N]"""
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


if a.stage == "convention":
    # Organ-level check, no tumour outcome: point inside each kidney component of the Kidney label with class 5 and 14; the better-matching class must reach Dice > 0.7 with that kidney.
    rows = []
    for case in sorted(os.listdir(a.kits))[: (a.limit or 3)]:
        t0 = time.time(); path = os.path.join(a.kits, case, "ct.nii.gz"); nii = nib.load(path); sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
        seg = np.asarray(nib.load(os.path.join(a.kits, case, "seg.nii.gz")).dataobj); data, img, M = prep(path); t_pre = time.time() - t0
        kid = (seg == 1) | (seg == 2); lab, n = ndimage.label(kid); sizes = ndimage.sum(kid, lab, range(1, n + 1)); big = [i + 1 for i in np.argsort(sizes)[::-1][:2] if sizes[i] > 5000]
        for comp_id in big:
            comp = lab == comp_id; pt = interior_point(comp, sp); best = (-1, None)
            for cls in (5, 14):
                t1 = time.time(); pred = predict(data, img, M, cls, pt); dt = time.time() - t1
                d = 2 * (pred & comp).sum() / max(pred.sum() + comp.sum(), 1)
                if d > best[0]:
                    best = (d, cls)
            rows.append({"case": case, "component_voxels": int(comp.sum()), "point": pt, "best_class": best[1], "dice_with_kidney": round(float(best[0]), 3), "t_preprocess_s": round(t_pre, 1), "t_inference_s": round(dt, 1)})
            print(rows[-1], flush=True)
    ok = all(r["dice_with_kidney"] > 0.7 for r in rows)
    print("CONVENTION TEST", "PASSED" if ok else "FAILED", f"({len(rows)} kidney components)")
    sys.exit(0 if ok else 1)

# run stage
os.makedirs(a.outdir, exist_ok=True)
for arm in a.arms.split(","):
    organ, cls, root = ARMS[arm]
    if organ == "kidney":
        table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2}
    else:
        table = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/exp8_lesions/lesion_table.csv")) if r["route"] == "uid"}
    cases = sorted({k[0] for k in table})[: (a.limit or None)]; out_rows = []; t_arm = time.time()
    for ci, case in enumerate(cases):
        nii, sp, seg, ann, zidx, lab, sel = lesions_for(organ, case, root)
        data, img, M = prep(os.path.join(root, case, "ct.nii.gz"))
        preds = []
        for rank, (diam, cid) in enumerate(sel[:2], 1):
            ref = table.get((case, rank)); assert ref is not None and abs(float(ref["diam_mm"]) - diam) < 0.01, f"lesion mismatch {case} {rank}"
            comp_c = lab == cid; pt_c = interior_point(comp_c, sp)                     # point in annotated-slice coordinates
            comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
            t1 = time.time(); pred = predict(data, img, M, cls, pt)[:seg.shape[0], :seg.shape[1], :seg.shape[2]]; dt = time.time() - t1
            pred_a = pred[:, :, ann]; cov = float((pred_a & comp_c).sum() / comp_c.sum())
            bb = bbox(comp_c, int(math.ceil(10.0 / min(sp))) + 2, comp_c.shape)                   # distances <= 10 mm only need the lesion's neighbourhood (identical result)
            dist = ndimage.distance_transform_edt(~comp_c[bb], sampling=sp); near = pred_a[bb] & (dist <= 10.0)
            dice10 = float(2 * (near & comp_c[bb]).sum() / max(near.sum() + comp_c.sum(), 1))
            pdiam = 0.0
            if (pred_a & comp_c).any():                                                         # a miss counts as 0 mm; otherwise the largest overlapping predicted component
                plab, pn = ndimage.label(pred_a, structure=np.ones((3, 3, 3))); ov = [i for i in np.unique(plab[comp_c]) if i > 0]
                pdiam = max((axial_diam(plab == i, sp[0], sp[1]) for i in ov), default=0.0)
            out_rows.append({"arm": arm, "organ": organ, "class": cls, "case": case, "rank": rank, "diam_mm": round(diam, 3), "contrast_hu": float(ref["contrast_hu"]), "coverage": round(cov, 5), "dice_within_10mm": round(dice10, 5),
                             "detected_cov_ge_0.10": int(cov >= 0.10), "pred_diam_mm": round(pdiam, 3), "point_x": pt[0], "point_y": pt[1], "point_z": pt[2], "t_inference_s": round(dt, 2)})
        if ci % 10 == 0:
            print(f"[{arm}] {ci + 1}/{len(cases)} {case} elapsed {time.time() - t_arm:.0f}s", flush=True)
    with open(os.path.join(a.outdir, f"w2_{arm}_lesions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(f"ARM {arm} done: {len(out_rows)} lesions, {time.time() - t_arm:.0f}s", flush=True)
