"""Readout re-fit, step 1 (results/readout_refit/PREREG.md): decoder_F at voxel level inside the predicted liver dilated by 10 mm.
Per case of the 89-case clean cohort: one single-window-set inference prompting [1, 26]; ROI = (class-1 logit > 0) dilated 10 mm;
store window-averaged F (fp16) at ROI voxels, the exact class-26 logit there, the full class-26 logit map (fp16), the native-space label map
(0 bg, 1 liver, 2 tumour, 3 unknown/unannotated slice), and tumour-minus-liver contrast (raw HU, GT masks).
Also checks that w_orig . F reproduces the network's class-26 logit.
Usage: python scripts/readout_refit_extract.py --out cache/rr [--limit N] [--cases A B]"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInfererAdapt
from monai.transforms import LoadImage
from scipy import ndimage

sys.path.insert(0, ".")
from glassbox.data.preprocess import HU_MAX, HU_MIN, PATCH_SIZE, SW_OVERLAP, build_pair_transforms  # noqa: E402
from glassbox.extract.voxel_f import VoxelFAccumulator  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="cache/rr")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cases", nargs="*")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
ROOTS = {"uid": "cache/heldout_nii/hcc_tace_seg", "resolved": "cache/heldout_nii_clean/hcc_tace_seg"}
acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}
conv = {r["case"]: r["status"] for r in csv.DictReader(open("results/exp1_audit/hcc_clean_conversion.csv"))}
cohort = [(c, "uid") for c in sorted(acq) if acq[c]["route"] == "uid"] + [(c, "resolved") for c in sorted(conv) if conv[c] == "RESOLVED"]
cohort = sorted(cohort)  # fold = index % 5 over the sorted case ids
if a.cases:
    cohort = [x for x in cohort if x[0] in a.cases]
cohort = cohort[: a.limit or None]
dev = torch.device("cuda")
net = load_network("cache/weights/model.pt", dev)
with torch.no_grad():
    w_orig = net.class_head.mlp(net.class_head.class_embeddings(torch.tensor([[26]], device=dev))).reshape(-1).float().cpu().numpy()
np.save(os.path.join(a.out, "w_orig.npy"), w_orig)
load = LoadImage(image_only=True)
VOX = 1.5
for ci, (case, src) in enumerate(cohort, 1):
    dst = os.path.join(a.out, f"{case}.npz")
    if os.path.exists(dst):
        continue
    t0 = time.time()
    d = os.path.join(ROOTS[src], case)
    img, seg = load(os.path.join(d, "ct.nii.gz")), load(os.path.join(d, "seg.nii.gz"))
    segn = seg.numpy(); ann = (segn > 0).any(axis=(0, 1))
    lab = np.zeros(segn.shape, np.int16); lab[segn == 1] = 1; lab[segn == 2] = 2; lab[:, :, ~ann] = 3
    pair = build_pair_transforms()({"image": img, "label": MetaTensor(torch.from_numpy(lab), meta=img.meta.copy(), affine=img.affine)})
    x = pair["image"].unsqueeze(0).to(dev)
    L = pair["label"][0].as_tensor().numpy().astype(np.int8)
    H, W, D = L.shape
    rec = VoxelFAccumulator(net, (H, W, D), dev)
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    try:
        with torch.no_grad():
            logits = sw(x, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[1], [26]], device=dev),
                        prompt_class=None, prev_mask=None, labels=None, label_set=None)
        Fm = rec.result()
    finally:
        rec.close()
    logits = logits.reshape(2, H, W, D).float()
    liver_pred = (logits[0] > 0).cpu().numpy()
    if liver_pred.sum() == 0:
        print(f"[{ci}/{len(cohort)}] {case} SKIP (no predicted liver)", flush=True); continue
    roi = ndimage.distance_transform_edt(~liver_pred) * VOX <= 10.0
    idx = np.flatnonzero(roi.ravel())
    it = torch.from_numpy(idx).to(dev)
    Froi = Fm.reshape(48, -1)[:, it].T  # [N,48]
    lg26 = logits[1].reshape(-1)[it]
    recon = Froi @ torch.from_numpy(w_orig).to(dev)
    err = float((recon - lg26).abs().max())
    hu = x[0, 0].cpu().numpy() * (HU_MAX - HU_MIN) + HU_MIN
    known = L != 3
    tm, lm = (L == 2) & known, (L == 1) & known
    contrast = float(hu[tm].mean() - hu[lm].mean()) if tm.sum() and lm.sum() else float("nan")
    np.savez_compressed(dst + ".part.npz", shape=np.array([H, W, D]), roi_idx=idx.astype(np.int32), F=Froi.half().cpu().numpy(),
                        logit_roi=lg26.cpu().numpy().astype(np.float32), logit_full=logits[1].cpu().numpy().astype(np.float16), L=L,
                        liver_pred_vox=int(liver_pred.sum()), contrast_hu=contrast, recon_maxerr=err, route=src)
    os.replace(dst + ".part.npz", dst)
    print(f"[{ci}/{len(cohort)}] {case} roi {len(idx)} vox; GT tumour in ROI {int(roi[L == 2].sum())}/{int((L == 2).sum())}; recon max err {err:.4f}; contrast {contrast:.1f} HU; {time.time() - t0:.0f}s", flush=True)
    del Fm, logits, Froi
    torch.cuda.empty_cache()
