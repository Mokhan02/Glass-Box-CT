"""W2 v2 runner (results/w2_kidney/PREREG_v2.md). Arms (glassbox/seg/w2_arms.py): kidney 133/point label 1; HCC 26/3; HCC 133/1; HCC 26/1 (v1-equivalent, secondary).
Single object, one positive point at the lesion's interior point (voxel of max 3D distance to the boundary), the point label is a per-arm parameter and is checked at the point head on every call.
Stages: pilot (v2_pilot_selection.csv lesions) and main (all other primary lesions). Saves ALL predicted masks (npz per lesion and arm) and a CSV per organ with the pilot-gate quantities.
Usage: python scripts/w2v2_run.py pilot|main <vendored_inferer_dir> <weights> <outdir>"""
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
from glassbox.seg.w2_arms import ARMS  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("inferer_dir"); ap.add_argument("weights"); ap.add_argument("outdir")
ap.add_argument("--arms", default="all", help="comma-separated arm names; default all arms of each organ (the v3 main run passes the three primary arms)")
ap.add_argument("--kits", default="/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"); ap.add_argument("--hcc", default="/lambda/nfs/CTGate/glassbox/cache/heldout_nii/hcc_tace_seg"); a = ap.parse_args()
sys.path.insert(0, a.inferer_dir); from inferer import Vista3dInferer  # noqa: E402  (NVIDIA's point-window inferer, vendored at run time, not committed)

dev = torch.device("cuda"); net = load_network(a.weights, dev)
inferer = Vista3dInferer(roi_size=(128, 128, 128), overlap=0.3, use_point_window=True, sw_batch_size=1)
pre = build_transforms("image")
SEEN = []      # point labels that actually reach the point head


def _hook(module, args, kwargs):
    pl = args[2] if len(args) > 2 else kwargs["point_labels"]
    SEEN.append(sorted({float(v) for v in torch.unique(pl).tolist() if v != -1}))


net.point_head.register_forward_pre_hook(_hook, with_kwargs=True)


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


def predict(data, img, M, label, point_label, point_vox):
    """Binary prediction on the ORIGINAL grid (prob > 0.5): one class/zero-shot label prompt, one positive point carrying `point_label` (original voxel indices)."""
    from monai.data import MetaTensor
    from monai.transforms import Invertd
    p = M @ np.array([*point_vox, 1.0]); pts = torch.tensor(p[:3][None, None, :], dtype=torch.float32, device=dev)
    SEEN.clear()
    with torch.no_grad():
        logits = inferer(inputs=img, network=net, point_coords=pts, point_labels=torch.full((1, 1), float(point_label), device=dev), class_vector=torch.tensor([[label]], device=dev))
    seen = {v for s in SEEN for v in s}
    assert seen == {float(point_label)}, f"point label at the point head was {seen}, expected {point_label}"      # the label sent is the label the head received
    probs = torch.sigmoid(logits.reshape(1, *img.shape[2:]).float().cpu())
    pred = MetaTensor(probs, meta=data["image"].meta, applied_operations=data["image"].applied_operations)
    out = Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True)({"pred": pred, "image": data["image"]})
    return out["pred"].as_tensor()[0].numpy() > 0.5


def stratum(d):
    return "<30 mm" if d < 30 else ("30-60 mm" if d <= 60 else ">60 mm")


pilot = {(r["organ"], r["case"], int(r["rank"])) for r in csv.DictReader(open("results/w2_kidney/v2_pilot_selection.csv"))}
tables = {"kidney": {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/w2_kidney/gt_lesions.csv")) if int(r["rank"]) <= 2},
          "hcc": {(r["case"], int(r["rank"])): r for r in csv.DictReader(open("results/exp8_lesions/lesion_table.csv")) if r["route"] == "uid"}}
roots = {"kidney": a.kits, "hcc": a.hcc}
os.makedirs(os.path.join(a.outdir, "masks"), exist_ok=True)
for organ in ("kidney", "hcc"):
    want = {k for k in tables[organ] if ((organ,) + k in pilot) == (a.stage == "pilot")}
    cases = sorted({k[0] for k in want}); arms = [n for n, c in ARMS.items() if c["organ"] == organ and (a.arms == "all" or n in a.arms.split(","))]; rows = []; t0 = time.time()
    for ci, case in enumerate(cases):
        nii, sp, seg, ann, zidx, lab, sel_l = lesions_for(organ, case, roots[organ]); ct = np.asarray(nii.dataobj).astype(np.float32); seg_nii = nib.load(os.path.join(roots[organ], case, "seg.nii.gz"))
        data, img, M = prep(os.path.join(roots[organ], case, "ct.nii.gz"))
        for rank, (diam, cid) in enumerate(sel_l[:2], 1):
            if (case, rank) not in want:
                continue
            ref = tables[organ][(case, rank)]; assert abs(float(ref["diam_mm"]) - diam) < 0.01, f"lesion mismatch {case} {rank}"
            comp_c = lab == cid; pt_c = interior_point(comp_c, sp); comp = np.zeros(seg.shape, bool); comp[:, :, ann] = comp_c; pt = (pt_c[0], pt_c[1], int(zidx[pt_c[2]]))
            for arm in arms:
                cfg = ARMS[arm]; t1 = time.time(); pred = predict(data, img, M, cfg["label"], cfg["point_label"], pt)[:seg.shape[0], :seg.shape[1], :seg.shape[2]]; dt = time.time() - t1
                pa = pred[:, :, ann]; n_p, n_l = int(pa.sum()), int(comp_c.sum()); inter = int((pa & comp_c).sum())
                bb = bbox(comp_c, int(math.ceil(10.0 / min(sp))) + 2, comp_c.shape); dist = ndimage.distance_transform_edt(~comp_c[bb], sampling=sp); near = pa[bb] & (dist <= 10.0)
                pdiam = pcdiam = 0.0
                if n_p:
                    plab, pn = ndimage.label(pa, structure=np.ones((3, 3, 3))); ov = [i for i in np.unique(plab[comp_c]) if i > 0]
                    pdiam = max((axial_diam(plab == i, sp[0], sp[1]) for i in ov), default=0.0)
                    pz = int(np.searchsorted(zidx, pt[2])); pc = int(plab[pt[0], pt[1], pz]); pcdiam = axial_diam(plab == pc, sp[0], sp[1]) if pc > 0 else 0.0
                rows.append({"stage": a.stage, "arm": arm, "organ": organ, "label_prompt": cfg["label"], "point_label": cfg["point_label"], "case": case, "rank": rank, "group": "bright" if float(ref["contrast_hu"]) > 0 else "dark",
                             "stratum": stratum(diam), "diam_mm": round(diam, 3), "contrast_hu": float(ref["contrast_hu"]), "coverage": round(inter / n_l, 5), "dice_full": round(2 * inter / max(n_p + n_l, 1), 5),
                             "dice_within_10mm": round(float(2 * (near & comp_c[bb]).sum() / max(near.sum() + n_l, 1)), 5), "pred_diam_mm": round(pdiam, 3), "point_comp_diam_mm": round(pcdiam, 3),
                             "pred_over_lesion_volume": round(n_p / n_l, 4), "pred_frac_in_lesion": round(inter / max(n_p, 1), 4), "point_x": pt[0], "point_y": pt[1], "point_z": pt[2], "t_inference_s": round(dt, 2)})
                np.savez_compressed(os.path.join(a.outdir, "masks", f"{a.stage}__{arm}__{case}__r{rank}.npz"), pred=pred, lesion=comp, seg=seg.astype(np.uint8), point=np.array(pt), point_label=cfg["point_label"], label_prompt=cfg["label"],
                                    ct_slice=ct[:, :, pt[2]].astype(np.float32), ct_affine=nii.affine, seg_affine=seg_nii.affine, ct_zooms=np.array(nii.header.get_zooms()[:3]), ann=ann,
                                    aff_pre=np.array(data["image"].affine, float), orig_affine=np.array(data["image"].meta["original_affine"], float))
        if ci % 10 == 0:
            print(f"[{a.stage} {organ}] {ci + 1}/{len(cases)} {case} elapsed {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.outdir, f"w2v2_{a.stage}_{organ}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"DONE {a.stage} {organ}: {len(rows)} predictions, {time.time() - t0:.0f}s", flush=True)
