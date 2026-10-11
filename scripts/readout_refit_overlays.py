"""Two overlays of bright (tumour-minus-liver contrast > 0 HU) missed tumours before/after the re-fit. Selection rule (fixed, not outcome-based):
the two bright cases with the lowest original-readout held-out Dice. Axial slice with the largest GT tumour area.
Usage: python scripts/readout_refit_overlays.py [--out results/readout_refit]"""
import argparse
import csv
import os
import sys

import matplotlib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import LoadImage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")
from glassbox.data.preprocess import HU_MAX, HU_MIN, build_pair_transforms  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/readout_refit"); ap.add_argument("--rr", default="cache/rr"); a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.out, "per_case.csv"))))
bright = sorted([r for r in rows if float(r["contrast_hu"]) > 0 and r["orig_n_targets"] != "0"], key=lambda r: float(r["orig_dice"]))[:2]
M = np.load("cache/rr_models.npz"); dec = __import__("json").load(open(os.path.join(a.out, "DECISION.json")))
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
load = LoadImage(image_only=True)
for k, r in enumerate(bright, 1):
    c, f = r["case"], int(r["fold"]); z = np.load(os.path.join(a.rr, c + ".npz")); L = z["L"]; H, W, D = L.shape; idx = z["roi_idx"].astype(np.int64)
    base = z["logit_full"].astype(np.float32).ravel(); base[idx] = z["logit_roi"]
    new = base.copy(); new[idx] = z["F"].astype(np.float32) @ M["w"][f] + M["b"][f]
    t = dec["matched_t_per_fold"][str(f)]; thr = np.where(np.isin(np.arange(base.size), idx), base > t, base > 0)  # threshold arm (matched), same rule as in the run
    d = os.path.join(ROOTS[str(z["route"])], c); img, seg = load(os.path.join(d, "ct.nii.gz")), load(os.path.join(d, "seg.nii.gz")); sn = seg.numpy()
    ann = (sn > 0).any(axis=(0, 1)); lab = np.zeros(sn.shape, np.int16); lab[sn == 1] = 1; lab[sn == 2] = 2; lab[:, :, ~ann] = 3
    pair = build_pair_transforms()({"image": img, "label": MetaTensor(torch.from_numpy(lab), meta=img.meta.copy(), affine=img.affine)})
    hu = pair["image"][0].as_tensor().numpy() * (HU_MAX - HU_MIN) + HU_MIN
    assert tuple(hu.shape) == (H, W, D) and (pair["label"][0].as_tensor().numpy().astype(np.int8) == L).all()
    zs = int((L == 2).sum(axis=(0, 1)).argmax())
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.3))
    for axx, (ttl, m) in zip(ax, (("original readout", base > 0), ("threshold shift (matched)", thr.reshape(H, W, D) if thr.ndim == 1 else thr), ("re-fit readout", new > 0))):
        m = m.reshape(H, W, D)
        axx.imshow(np.clip(hu[:, :, zs], -160, 240).T, cmap="gray", origin="lower")
        axx.contour((L[:, :, zs] == 2).T, levels=[0.5], colors="lime", linewidths=1.2)
        axx.contour(m[:, :, zs].T, levels=[0.5], colors="red", linewidths=1.2)
        inter = (m[:, :, zs] & (L[:, :, zs] == 2)).sum(); dsl = 2 * inter / max(m[:, :, zs].sum() + (L[:, :, zs] == 2).sum(), 1)
        axx.set_title(f"{ttl}\nslice Dice {dsl:.2f}", fontsize=9); axx.axis("off")
    fig.suptitle(f"{c} (fold {f}) bright tumour, contrast {float(r['contrast_hu']):+.0f} HU; green = reference, red = prediction. Case Dice: orig {float(r['orig_dice']):.2f}, thr {float(r['thr_dice']):.2f}, re-fit {float(r['refit_dice']):.2f}", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f"overlay_bright_missed_{k}.png"), dpi=140); plt.close(fig)
    print("saved", c, r["orig_dice"], r["thr_dice"], r["refit_dice"])
