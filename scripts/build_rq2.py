"""RQ2 dataset: per (case, target, perturbation) -> Dice (label) + features that could predict it.
For every held-out case (and optionally degraded copies of its CT) one forward pass gives probabilities and activation tokens.
Per target (same targets/prompts as eval_heldout.py) we store:
  dice            computed on annotated slices, as in eval_heldout.py
  baselines       mean sigmoid prob in mask, mean binary entropy in mask, predicted volume (native voxels), empty-prediction flag
  mean_mid        mean of mid-layer tokens inside the predicted mask [192]
  cov_mid         covariance-pooled mid tokens (fixed PCA basis from TotalSegmentator train tokens) [136]
  sae_mid         mean SAE feature activation over mask tokens [n_features]
Perturbations keep the voxel grid (so GT stays valid): noise (sigma 250 HU), thick (slice-averaging to ~5 mm),
crop (FOV truncation to 30-62% of the slices), shift (+300 HU). Milder settings (sigma 120, +150 HU, 25-80%) left the liver at Dice 0.87-0.95.
Usage: python scripts/build_rq2.py --collection hcc_tace_seg --perturb none noise thick crop shift --out cache/rq2 [--limit N]"""
import argparse
import os
import re
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, ".")
from monai.transforms import LoadImage  # noqa: E402

from glassbox.data.registry import is_in_distribution  # noqa: E402
from glassbox.probes.core import cov_pool, fit_basis  # noqa: E402
from glassbox.sae.batchtopk import BatchTopKSAE  # noqa: E402
from glassbox.seg.infer import dice, load_network, predict_with_tokens  # noqa: E402

TARGETS = {
    "c4kc_kits": {"kidney": ((1, 2), (5, 14))},
    "hcc_tace_seg": {"liver": ((1, 2), (1,)), "tumor": ((2,), (26,))},
}
ap = argparse.ArgumentParser()
ap.add_argument("--collection", required=True, choices=list(TARGETS))
ap.add_argument("--root", default="cache/heldout_nii")
ap.add_argument("--out", default="cache/rq2")
ap.add_argument("--perturb", nargs="+", default=["none"], choices=["none", "noise", "thick", "crop", "shift"])
ap.add_argument("--sae", default="cache/sae/mid_x16_k32")
ap.add_argument("--acts", default="cache/acts")
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
net = load_network("cache/weights/model.pt", dev)
sae = BatchTopKSAE(192, 16, 32)
sae.load_state_dict(torch.load(os.path.join(a.sae, "sae.pt"), map_location="cpu"))
sae = sae.to(dev).eval()

basis_path = os.path.join(a.out, "basis_mid.npz")
if not os.path.exists(basis_path):
    import glob
    fs = sorted(glob.glob(os.path.join(a.acts, "*.npz")))[:100]
    mu, B = fit_basis(np.concatenate([np.load(f)["mid_x"][::20] for f in fs]), 16)
    np.savez(basis_path, mu=mu, B=B)
bz = np.load(basis_path)
mu_b, B_b = bz["mu"], bz["B"]


def patient_of(case):
    return re.match(r"^(KiTS-\d+|HCC_\d+)", case).group(1)


def perturbed(ct_path, kind, rng, tmp):
    """Write a degraded copy of the CT on the same grid, or return None when the perturbation would be a no-op."""
    img = nib.load(ct_path)
    x = np.asarray(img.dataobj).astype(np.float32)
    if kind == "noise":
        x = x + rng.normal(0, 250, x.shape).astype(np.float32)
    elif kind == "shift":
        x = x + 300.0
    elif kind == "crop":
        n = x.shape[2]
        keep = np.zeros(n, bool); keep[int(0.30 * n):int(0.62 * n)] = True
        x[:, :, ~keep] = -1000.0
    elif kind == "thick":
        dz = float(img.header.get_zooms()[2])
        k = int(round(5.0 / dz))
        if k < 2:
            return None
        n = x.shape[2]
        y = x.copy()
        for s in range(0, n, k):
            y[:, :, s:s + k] = x[:, :, s:s + k].mean(2, keepdims=True)
        x = y
    nib.save(nib.Nifti1Image(x.astype(np.int16 if kind != "noise" else np.float32), img.affine), tmp)
    return tmp


load = LoadImage(image_only=True)
base = os.path.join(a.root, a.collection)
cases = sorted(c for c in os.listdir(base) if os.path.exists(os.path.join(base, c, ".done")))
if a.limit:
    cases = cases[: a.limit]
classes = sorted({c for _, cs in TARGETS[a.collection].values() for c in cs})
rng = np.random.default_rng(a.seed)
rows, feats = [], {"mean_mid": [], "cov_mid": [], "sae_mid": []}
tmp = os.path.join(a.out, f"_tmp_{a.collection}.nii.gz")
t0 = time.time()
for ci, case in enumerate(cases, 1):
    seg = load(os.path.join(base, case, "seg.nii.gz")).numpy()
    ann = (seg > 0).any(axis=(0, 1))
    for kind in a.perturb:
        ct_path = os.path.join(base, case, "ct.nii.gz")
        if kind != "none":
            ct_path = perturbed(ct_path, kind, rng, tmp)
            if ct_path is None:
                continue
        try:
            po, pn, tok = predict_with_tokens(ct_path, classes, net, dev)
        except Exception as e:
            print(f"FAILED {case}/{kind}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        po, pn = po.numpy(), pn.numpy()
        xyz = tok["mid_xyz"].astype(int)
        ok = (xyz >= 0).all(1) & (xyz < np.array(pn.shape[1:])).all(1)
        X = tok["mid_x"].astype(np.float32)
        for target, (seg_ids, cls) in TARGETS[a.collection].items():
            gt_full = np.isin(seg, seg_ids)
            if gt_full.sum() == 0:
                continue
            p_full = np.max([po[classes.index(c)] for c in cls], axis=0)
            gt, p = gt_full[:, :, ann], p_full[:, :, ann]
            pred = p > 0.5
            pin = p[pred]
            pn_t = np.max([pn[classes.index(c)] for c in cls], axis=0)
            in_mask = np.zeros(len(xyz), bool)
            in_mask[ok] = pn_t[tuple(xyz[ok].T)] > 0.5
            empty = (not pred.any()) or in_mask.sum() < 5
            if empty:
                m, cv, sv = (np.full(192, np.nan), np.full(136, np.nan), np.full(sae.n, np.nan))
            else:
                T = X[in_mask]
                m, cv = T.mean(0), cov_pool(T, mu_b, B_b)
                with torch.no_grad():
                    z = sae.encode(torch.from_numpy(T).to(dev) * sae.scale, use_threshold=True)
                sv = z.mean(0).cpu().numpy()
            ent = -(pin * np.log(pin + 1e-8) + (1 - pin) * np.log(1 - pin + 1e-8)) if pred.any() else np.array([np.nan])
            rows.append({"collection": a.collection, "in_distribution": is_in_distribution(a.collection), "case": case,
                         "patient": patient_of(case), "target": target, "perturbation": kind,
                         "dice": float(dice(pred, gt)), "gt_voxels": int(gt.sum()), "pred_voxels": int(pred.sum()),
                         "mean_prob": float(pin.mean()) if pred.any() else np.nan, "mean_entropy": float(np.nanmean(ent)),
                         "n_mask_tokens": int(in_mask.sum()), "empty": bool(empty)})
            feats["mean_mid"].append(m.astype(np.float32)); feats["cov_mid"].append(cv.astype(np.float32))
            feats["sae_mid"].append(sv.astype(np.float16))
    print(f"[{ci}/{len(cases)}] {case} rows={len(rows)} {time.time() - t0:.0f}s", flush=True)
    if ci % 10 == 0 or ci == len(cases):  # checkpoint
        import csv
        with open(os.path.join(a.out, f"rq2_{a.collection}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        np.savez(os.path.join(a.out, f"rq2_{a.collection}_feats.npz"), **{k: np.stack(v) for k, v in feats.items()})
if os.path.exists(tmp):
    os.remove(tmp)
print("done", len(rows), "rows")
