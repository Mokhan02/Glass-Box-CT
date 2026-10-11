"""EXPLORATORY, post hoc (results/mech/DEVIATIONS.md #5): projection of mid-encoder tumour cells onto the LEACE tumour-vs-liver functional, dark vs bright (D's counterfactual pairs). GPU.
Usage: python scripts/mech_bridge.py --out cache/mech_bridge"""
import argparse
import csv
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as Fn
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import LoadImage
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.bodycomp.counterfactual import edit_lesion  # noqa: E402
from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_pair_transforms, build_transforms  # noqa: E402
from glassbox.extract.hooks import window_pads  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="cache/mech_bridge"); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
os.makedirs(os.path.join(a.out, "scratch"), exist_ok=True)
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
K, VOX, STRIDE = 5, 1.5, 4
dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev); load = LoadImage(image_only=True)
E = np.load("cache/mech_b/erasers.npz")
allcases = sorted(f[:-4] for f in os.listdir("cache/rr") if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(allcases)}
st = {"p": -1, "coords": [], "core": None, "par": None, "acc": []}


def _pre(module, args, kwargs):
    slices = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]; st["p"] += 1; st["coords"].append(tuple(slices[-3:])); return None


def _post(module, args, out):
    if st["core"] is None:
        return None
    sl = st["coords"][st["p"]]; frac = Fn.avg_pool3d(st["core"][sl].float()[None, None], STRIDE)[0, 0]; sel = frac >= 0.5
    if sel.any():
        mu, aa, d, c = st["par"]; t = torch.einsum("bcxyz,c->bxyz", out - mu[None, :, None, None, None], aa)[0] / c; st["acc"].append(t[sel].detach().cpu())
    return None


net.register_forward_pre_hook(_pre, with_kwargs=True); net.image_encoder.encoder.layers[2]["blocks"].register_forward_hook(_post)


def run(x, core_pad, par):
    st.update({"p": -1, "coords": [], "core": core_pad, "par": par, "acc": []})
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    with torch.no_grad():
        sw(x, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[26]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
    v = torch.cat(st["acc"]) if st["acc"] else torch.zeros(0); st["core"] = None
    return (float(v.mean()) if len(v) else float("nan")), int(len(v))


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


edits = sorted([r for r in csv.DictReader(open("cache/contrast_cf/edits.csv")) if r["arm"] == "A" and r["cond"] == "edit"], key=lambda r: (r["case"], int(r["rank"])))
src_of = {r["case"]: r["source"] for r in edits}; my = sorted({r["case"] for r in edits})[: a.limit or None]; rows, t0 = [], time.time()
for ci, case in enumerate(my, 1):
    root = ROOTS[src_of[case]]; nii = nib.load(os.path.join(root, case, "ct.nii.gz")); sx, sy, sz = (float(z) for z in nii.header.get_zooms()[:3]); ct = np.asarray(nii.dataobj).astype(np.float32)
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann); lab, n = ndimage.label((seg == 2)[:, :, ann], structure=np.ones((3, 3, 3)))
    les = sorted([(max(feret(lab[:, :, z] == i, sx, sy) for z in np.flatnonzero((lab == i).any(axis=(0, 1)))), i) for i in range(1, n + 1)], reverse=True); les = [l for l in les if l[0] >= 10.0][:2]
    f = fold_of[case]; par = tuple(torch.from_numpy(np.asarray(E[f"f{f}_leace_{k}"], dtype=np.float32)).to(dev) for k in ("mu", "a", "d")) + (float(E[f"f{f}_leace_c"]),)
    pre = build_transforms(); data = pre({"image": os.path.join(root, case, "ct.nii.gz")}); shape = tuple(data["image"].shape[1:]); before, after = window_pads(shape); img_meta = load(os.path.join(root, case, "ct.nii.gz"))
    xd = data["image"].unsqueeze(0).to(dev)
    for e in [r for r in edits if r["case"] == case]:
        rank = int(e["rank"]); cid = les[rank - 1][1]; xs, ys, ks = np.nonzero(lab == cid); zs = zidx[ks]; m = np.zeros(ct.shape, bool); m[xs, ys, zs] = True
        edited, _ = edit_lesion(ct, m, float(e["target_mean"])); tmp = os.path.join(a.out, "scratch", f"{case}_L{rank}.nii.gz"); nib.save(nib.Nifti1Image(edited, nii.affine), tmp)
        xb = build_transforms()({"image": tmp})["image"].unsqueeze(0).to(dev); assert tuple(xb.shape[2:]) == shape
        pair = build_pair_transforms()({"image": img_meta, "label": MetaTensor(torch.from_numpy(m.astype(np.int16)), meta=img_meta.meta.copy(), affine=img_meta.affine)})
        core = pair["label"][0].as_tensor().numpy() > 0
        core_pad = Fn.pad(torch.from_numpy(core.astype(np.float32)).to(dev), [p for k in (2, 1, 0) for p in (before[k], after[k])]) > 0.5
        td, nd_ = run(xd, core_pad, par); tb, nb = run(xb, core_pad, par); os.remove(tmp)
        rows.append({"case": case, "rank": rank, "fold": f, "n_cells": nd_, "t_dark": td, "t_bright": tb})
        print(f"[{ci}/{len(my)}] {case} L{rank}: cells {nd_} t dark {td:.3f} bright {tb:.3f}; {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(a.out, "bridge.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
