"""Mechanistic experiment B, step 2 (results/mech/PREREG.md, section B): erase the tumour-vs-liver concept at the mid-encoder on held-out folds and run the full inference. GPU.
Conditions: none (hooked, identity), leace, random (same rank), hu (local-HU concept). Hook: encoder.layers[2]["blocks"] output (192 ch, every window, every voxel).
Outputs cache/mech_b/run_shard{k}_lesions.csv and _cases.csv. Usage: python scripts/mech_b_run.py --shard 0 --nshards 2"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from monai.inferers import SlidingWindowInfererAdapt
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.data.preprocess import PATCH_SIZE, SW_OVERLAP, build_transforms  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--out", default="cache/mech_b"); ap.add_argument("--rr", default="cache/rr"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
VOX, K = 1.5, 5
dev = torch.device("cuda"); net = load_network("cache/weights/model.pt", dev)
E = np.load(os.path.join(a.out, "erasers.npz"))
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(cases)}
mine = [c for i, c in enumerate(cases) if i % a.nshards == a.shard][: a.limit or None]
state = {"p": None}


def hook(module, args, out):
    p = state["p"]
    if p is None:
        return None
    mu, d, aa, c = p
    t = torch.einsum("bcxyz,c->bxyz", out - mu[None, :, None, None, None], aa) / c
    return out - d[None, :, None, None, None] * t[:, None]


net.image_encoder.encoder.layers[2]["blocks"].register_forward_hook(hook)


def params(f, name):
    g = lambda k: torch.from_numpy(np.asarray(E[f"f{f}_{name}_{k}"], dtype=np.float32)).to(dev)  # noqa: E731
    return g("mu"), g("d"), g("a"), float(E[f"f{f}_{name}_c"])


def feret(m2):
    pts = np.argwhere(m2).astype(float) * VOX
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def infer(x):
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    with torch.no_grad():
        lg = sw(x, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[1], [26]], device=dev), prompt_class=None, prev_mask=None, labels=None, label_set=None)
    return lg.reshape(2, *x.shape[2:]).float().cpu().numpy()


les_rows, case_rows = [], []
t0 = time.time()
for ci, c in enumerate(mine, 1):
    z = np.load(os.path.join(a.rr, c + ".npz")); L = z["L"]; shape = tuple(z["shape"]); f = fold_of[c]
    x = build_transforms()({"image": os.path.join(ROOTS[str(z["route"])], c, "ct.nii.gz")})["image"].unsqueeze(0).to(dev); assert tuple(x.shape[2:]) == shape
    ann = (L != 3).any(axis=(0, 1)); Lc = L[:, :, ann]; gt = Lc == 2; gl = (Lc == 1) | (Lc == 2)
    lab, n = ndimage.label(gt, structure=np.ones((3, 3, 3))); les = sorted([(max(feret(lab[:, :, k] == i) for k in np.flatnonzero((lab == i).any(axis=(0, 1)))), i) for i in range(1, n + 1)], reverse=True)
    les = [l for l in les if l[0] >= 10.0][:2]
    ref = z["logit_full"].astype(np.float32).ravel(); ref[z["roi_idx"].astype(np.int64)] = z["logit_roi"]; ref = ref.reshape(shape)
    for cond in ("none", "leace", "random", "hu"):
        state["p"] = None if cond == "none" else params(f, cond)
        lg = infer(x); l26, l1 = lg[0][:, :, ann], lg[1][:, :, ann]; pred = l26 > 0; liv = l1 > 0
        diff = float(np.abs(lg[0] - ref).max()) if cond == "none" else float("nan")
        dice = lambda p, g: float(2 * (p & g).sum() / max(p.sum() + g.sum(), 1))  # noqa: E731
        case_rows.append({"case": c, "fold": f, "cond": cond, "tumour_dice": dice(pred, gt), "liver_dice": dice(liv, gl), "max_abs_diff_vs_stored": diff, "contrast_hu": float(z["contrast_hu"])})
        for rank, (diam, i) in enumerate(les, 1):
            m = lab == i; xs, ys, zs = np.nonzero(m); lo = np.maximum([xs.min(), ys.min(), zs.min()] - np.array([8, 8, 8]), 0); hi = np.minimum([xs.max(), ys.max(), zs.max()] + np.array([9, 9, 9]), gt.shape)
            sl = tuple(slice(l, h) for l, h in zip(lo, hi)); mm = m[sl]; dist = ndimage.distance_transform_edt(~mm) * VOX; P = pred[sl] & (dist <= 10.0)
            les_rows.append({"case": c, "fold": f, "rank": rank, "cond": cond, "diam_mm": round(diam, 1), "coverage": float(pred[m].mean()), "lesion_dice": float(2 * (P & mm).sum() / max(P.sum() + mm.sum(), 1)), "contrast_hu": float(z["contrast_hu"])})
    state["p"] = None
    print(f"[{ci}/{len(mine)}] {c} f{f}; none-vs-stored max diff {[r['max_abs_diff_vs_stored'] for r in case_rows if r['case'] == c and r['cond'] == 'none'][0]:.3f}; {time.time() - t0:.0f}s", flush=True)
    for nm, rows in (("lesions", les_rows), ("cases", case_rows)):
        with open(os.path.join(a.out, f"run_shard{a.shard}_{nm}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
