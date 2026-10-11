"""Contrast-sign counterfactual (results/contrast_cf/PREREG.md). Two stages:
  --stage baseline : unedited inference for the 89 clean-cohort cases, lesion table (T, L, c, baseline coverage and Dice), arm assignment. No edited volume exists at this stage.
  --stage edits    : for each arm lesion, the edit and its sign-keeping control; inference on edited COPIES only (scratch dir, deleted after use); coverage and Dice per run.
Usage (instance): python scripts/contrast_cf_run.py --stage baseline|edits --out cache/contrast_cf"""
import argparse
import csv
import json
import os
import re
import sys

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.bodycomp.counterfactual import arm_targets, edit_lesion  # noqa: E402
from glassbox.seg.infer import load_network, predict_probs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["baseline", "edits"])
ap.add_argument("--out", default="cache/contrast_cf")
ap.add_argument("--limit-cases", type=int, default=0)
a = ap.parse_args()
os.makedirs(os.path.join(a.out, "slices"), exist_ok=True); os.makedirs(os.path.join(a.out, "scratch"), exist_ok=True)
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}
conv = {r["case"]: r["status"] for r in csv.DictReader(open("results/exp1_audit/hcc_clean_conversion.csv"))}
cohort = [(c, "uid") for c in sorted(acq) if acq[c]["route"] == "uid"] + [(c, "resolved") for c in sorted(conv) if conv[c] == "RESOLVED"]
cohort = cohort[: a.limit_cases or None]
dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev)


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def infer(path):
    probs, _ = predict_probs(path, [26], net, dev)
    return probs[0].numpy() > 0.5


def measure(pred, ctx):
    """coverage and lesion Dice for one lesion context."""
    cov = float(pred[ctx["xs"], ctx["ys"], ctx["zs"]].mean())
    P = pred[ctx["box"]] & (ctx["dist"] <= 10.0); L = ctx["les3"]
    return cov, float(2 * (P & L).sum() / max(P.sum() + L.sum(), 1))


rows, base_rows = [], []
for case, src in cohort:
    root = ROOTS[src]; nii = nib.load(os.path.join(root, case, "ct.nii.gz")); sx, sy, sz = (float(z) for z in nii.header.get_zooms()[:3])
    ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann); gt_c = (seg == 2)[:, :, ann]
    lab, n = ndimage.label(gt_c, structure=np.ones((3, 3, 3))); les = []
    for i in range(1, n + 1):
        comp = lab == i; zs_ = np.flatnonzero(comp.any(axis=(0, 1))); les.append((max(feret(comp[:, :, z], sx, sy) for z in zs_), i))
    les = sorted([l for l in les if l[0] >= 10.0], key=lambda l: -l[0])[:2]
    if not les:
        continue
    pred0 = infer(os.path.join(root, case, "ct.nii.gz")); liver = (seg == 1) & ann[None, None, :]; ctxs = []
    for rank, (diam, cid) in enumerate(les, 1):
        xs, ys, ks = np.nonzero(lab == cid); zs = zidx[ks]; X, Y, Z = ct.shape; pad = int(25 / min(sx, sy, sz))
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X); y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y); z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
        box = (slice(x0, x1), slice(y0, y1), slice(z0, z1)); les3 = np.zeros((x1 - x0, y1 - y0, z1 - z0), bool); les3[xs - x0, ys - y0, zs - z0] = True
        dist = ndimage.distance_transform_edt(~les3, sampling=(sx, sy, sz)); shell = liver[box] & (dist >= 5) & (dist <= 20)
        Lm = float(ct[box][shell].mean()) if shell.sum() >= 200 else float(ct[liver].mean()); T = float(ct[xs, ys, zs].mean())
        ctx = {"xs": xs, "ys": ys, "zs": zs, "box": box, "les3": les3, "dist": dist, "T": T, "L": Lm, "rank": rank, "diam": diam}
        cov0, dice0 = measure(pred0, ctx); ctx.update({"cov0": cov0, "dice0": dice0}); ctxs.append(ctx)
        at = arm_targets(T, Lm); arm = ""
        if at and at[0] == "A" and cov0 >= 0.10:
            arm = "A"
        elif at and at[0] == "B" and cov0 < 0.10:
            arm = "B"
        ctx["arm"], ctx["targets"] = arm, at
        zstar = int(np.bincount(zs).argmax())  # the slice holding most of the lesion, used for the overlay figures
        ctx["zstar"] = zstar; lm = np.zeros((X, Y), bool); sel = zs == zstar; lm[xs[sel], ys[sel]] = True
        base_rows.append({"case": case, "source": src, "rank": rank, "diam_mm": round(diam, 1), "T": round(T, 1), "L": round(Lm, 1), "contrast": round(T - Lm, 1), "coverage": round(cov0, 4), "dice": round(dice0, 4), "arm": arm})
        np.savez_compressed(os.path.join(a.out, "slices", f"{case}_L{rank}_baseline.npz"), ct=ct[:, :, zstar].astype(np.float16), pred=pred0[:, :, zstar], lesion_mask=lm, z=zstar)
    print(f"{case} {src}: " + " ".join(f"L{c['rank']} d={c['diam']:.0f}mm c={c['T'] - c['L']:+.0f}HU cov={c['cov0']:.2f} arm={c['arm'] or '-'}" for c in ctxs), flush=True)
    if a.stage == "edits":
        for ctx in ctxs:
            if not ctx["arm"]:
                continue
            _, edit_t, ctrl_t = ctx["targets"]
            for cond, target in (("edit", edit_t), ("control", ctrl_t)):
                m = np.zeros(ct.shape, bool); m[ctx["xs"], ctx["ys"], ctx["zs"]] = True
                edited, info = edit_lesion(ct, m, target)
                assert np.array_equal(edited[~m], ct[~m]), "edit changed voxels outside the mask"
                tmp = os.path.join(a.out, "scratch", f"{case}_L{ctx['rank']}_{cond}.nii.gz"); nib.save(nib.Nifti1Image(edited, nii.affine), tmp)
                try:
                    pred = infer(tmp)
                finally:
                    os.remove(tmp)
                cov, dice = measure(pred, ctx)
                rows.append({"case": case, "source": src, "rank": ctx["rank"], "arm": ctx["arm"], "cond": cond, "T": round(ctx["T"], 1), "L": round(ctx["L"], 1), "contrast": round(ctx["T"] - ctx["L"], 1),
                             "target_mean": round(target, 1), "realised_mean": round(info["realised_mean"], 1), "flagged": info["flagged"], "cov0": round(ctx["cov0"], 4), "dice0": round(ctx["dice0"], 4), "cov": round(cov, 4), "dice": round(dice, 4)})
                zs_ = ctx["zstar"]
                np.savez_compressed(os.path.join(a.out, "slices", f"{case}_L{ctx['rank']}_{cond}.npz"), ct=edited[:, :, zs_].astype(np.float16), pred=pred[:, :, zs_], z=zs_)
        if rows:  # rewrite after every case so a stopped run keeps its results
            with open(os.path.join(a.out, "edits.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(os.path.join(a.out, "baseline.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(base_rows[0])); w.writeheader(); w.writerows(base_rows)
armA = sum(r["arm"] == "A" for r in base_rows); armB = sum(r["arm"] == "B" for r in base_rows)
print(f"\nBASELINE: {len(cohort)} cases, {len(base_rows)} lesions; Arm A (detected, dark) {armA}; Arm B (missed/poorly detected, bright) {armB}")
