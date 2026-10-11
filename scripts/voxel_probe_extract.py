"""Voxel-probe experiment, step 1 (see results/voxel_probe/PREREG.md): per-token activations and masks on held-out HCC-TACE-Seg.
Per case: tokens inside the GT liver (+tumour) mask dilated by 10 mm, each with GT tumour/liver coverage over ANNOTATED voxels,
predicted-tumour coverage, distance to the GT tumour edge (mm), and 14 local raw-HU features; tumour connected components and whether each
survives at each layer's grid; class-26 logits of interior GT-tumour voxels; the original-space tumour probability map (trained model only).
Usage: python scripts/voxel_probe_extract.py --model trained|random --out cache/vp [--limit N] [--cases A B]"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from monai.data import MetaTensor  # noqa: E402
from monai.inferers import SlidingWindowInfererAdapt  # noqa: E402
from monai.transforms import Invertd, LoadImage  # noqa: E402
from scipy import ndimage  # noqa: E402

from glassbox.data.preprocess import HU_MAX, HU_MIN, PATCH_SIZE, SW_OVERLAP, build_pair_transforms, build_transforms  # noqa: E402
from glassbox.extract.hooks import HOOKS, RoiTokenRecorder  # noqa: E402
from glassbox.seg.infer import load_network  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="trained", choices=["trained", "random"])
ap.add_argument("--root", default="cache/heldout_nii/hcc_tace_seg")
ap.add_argument("--out", default="cache/vp")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--cases", nargs="*")
a = ap.parse_args()
dev = torch.device("cuda")
out_dir = os.path.join(a.out, a.model); os.makedirs(out_dir, exist_ok=True)
if a.model == "trained":
    net = load_network("cache/weights/model.pt", dev); os.makedirs("cache/vp_prob", exist_ok=True)
else:
    from monai.networks.nets import vista3d132
    torch.manual_seed(0)
    net = vista3d132(encoder_embed_dim=48, in_channels=1).to(dev).eval()
load = LoadImage(image_only=True)
VOX_MM = 1.5
cases = a.cases or sorted(c for c in os.listdir(a.root) if os.path.exists(os.path.join(a.root, c, ".done")))
cases = cases[: a.limit or None]


def box_sums(II, lo, hi):
    """Sum over [lo, hi) boxes from a zero-padded integral image II (shape +1 per axis); lo/hi [N,3] already clipped."""
    x0, y0, z0 = lo.T; x1, y1, z1 = hi.T
    return (II[x1, y1, z1] - II[x0, y1, z1] - II[x1, y0, z1] - II[x1, y1, z0] + II[x0, y0, z1] + II[x0, y1, z0] + II[x1, y0, z0] - II[x0, y0, z0])


def integral(v):
    II = np.zeros([s + 1 for s in v.shape], np.float64)
    II[1:, 1:, 1:] = v.astype(np.float64).cumsum(0).cumsum(1).cumsum(2)
    return II


for ci, case in enumerate(cases, 1):
    dst = os.path.join(out_dir, f"{case}.npz")
    if os.path.exists(dst):
        continue
    t0 = time.time()
    d = os.path.join(a.root, case)
    img, seg = load(os.path.join(d, "ct.nii.gz")), load(os.path.join(d, "seg.nii.gz"))
    segn = seg.numpy()
    ann = (segn > 0).any(axis=(0, 1))
    lab = np.zeros(segn.shape, np.int16); lab[segn == 1] = 1; lab[segn == 2] = 2; lab[:, :, ~ann] = 3  # 3 = unannotated slice ("unknown")
    pair = build_pair_transforms()({"image": img, "label": MetaTensor(torch.from_numpy(lab), meta=img.meta.copy(), affine=img.affine)})
    x = pair["image"].unsqueeze(0).to(dev)
    L = pair["label"][0].as_tensor().numpy().astype(np.int16)  # native-space label map 0 bg, 1 liver, 2 tumour, 3 unknown
    H, W, D = L.shape
    hu = x[0, 0].cpu().numpy() * (HU_MAX - HU_MIN) + HU_MIN
    tum, liv, known = L == 2, L == 1, L != 3
    if tum.sum() == 0 or liv.sum() == 0:
        print(f"[{ci}/{len(cases)}] {case} SKIP (no tumour or liver after resampling)", flush=True); continue
    roi = (ndimage.distance_transform_edt(~(tum | liv)) * VOX_MM) <= 10.0
    rec = RoiTokenRecorder(net, torch.from_numpy(roi), seed=0); rec.label = rec.label.to(dev)
    sw = SlidingWindowInfererAdapt(roi_size=PATCH_SIZE, sw_batch_size=1, overlap=SW_OVERLAP, with_coord=True, padding_mode="replicate")
    try:
        with torch.no_grad():
            logits = sw(x, net, transpose=True, point_coords=None, point_labels=None, class_vector=torch.tensor([[26]], device=dev),
                        prompt_class=None, prev_mask=None, labels=None, label_set=None)
        tokens = rec.collect()
    finally:
        rec.close()
    logit = logits.reshape(H, W, D).float().cpu().numpy()
    pred = logit > 0.0
    II = {k: integral(v) for k, v in (("tum", tum), ("liv", liv), ("known", known), ("pred", pred), ("one", np.ones_like(tum)))}
    Ihu1, Ihu2 = integral(hu), integral(hu.astype(np.float64) ** 2)
    unknown = ~known
    edt = ndimage.distance_transform_edt(tum | unknown) * VOX_MM  # distance to the nearest KNOWN non-tumour voxel (DEVIATIONS.md #1)
    bridged = tum | (ndimage.binary_closing(tum, structure=np.ones((1, 1, 7))) & unknown)  # join tumour across unannotated slabs
    comp, ncomp = ndimage.label(bridged, structure=np.ones((3, 3, 3)))
    comp = np.where(tum, comp, 0)
    keep_ids = [i for i in range(1, ncomp + 1) if (comp == i).any()]
    out = {"n_comp": len(keep_ids), "comp_mm3": np.array([(comp == i).sum() * VOX_MM ** 3 for i in keep_ids]), "gt_tumour_mm3": float(tum.sum() * VOX_MM ** 3)}
    for h, (stride, _) in HOOKS.items():
        xyz = tokens[f"{h}_xyz"].astype(np.int64); xs = tokens[f"{h}_x"]
        c = np.clip(xyz, 0, np.array([H, W, D]) - 1)
        lo = np.clip(xyz - stride // 2, 0, [H, W, D]); hi = np.clip(lo + stride, 0, [H, W, D])
        n_cell = box_sums(II["one"], lo, hi); n_known = box_sums(II["known"], lo, hi)
        s_tum, s_liv, s_pred = (box_sums(II[k], lo, hi) for k in ("tum", "liv", "pred"))
        full = n_cell >= 0.5 * stride ** 3  # drop cells mostly outside the image
        kf = n_known / np.maximum(n_cell, 1)
        ct_ = s_tum / np.maximum(n_known, 1); cl_ = s_liv / np.maximum(n_known, 1); pc = s_pred / np.maximum(n_cell, 1)
        is_t = full & (kf >= 0.5) & (ct_ >= 0.5)
        is_l = full & (kf >= 0.5) & (s_tum == 0) & (cl_ >= 0.5)
        sel = np.flatnonzero(is_t | is_l)
        # 14 local HU features: cell and 3x-cell cube (mean, std, p5, p25, p50, p75, p95)
        hf = np.zeros((len(sel), 14), np.float32)
        for scale, off in ((1, 0), (3, 7)):
            half = stride * scale // 2
            lo2 = np.clip(xyz[sel] - half, 0, [H, W, D]); hi2 = np.clip(xyz[sel] + (stride * scale - half), 0, [H, W, D])
            n2 = np.maximum(box_sums(II["one"], lo2, hi2), 1)
            m = box_sums(Ihu1, lo2, hi2) / n2; v = box_sums(Ihu2, lo2, hi2) / n2 - m ** 2
            hf[:, off] = m; hf[:, off + 1] = np.sqrt(np.maximum(v, 0))
            for k, (l_, h_) in enumerate(zip(lo2, hi2)):
                hf[k, off + 2:off + 7] = np.percentile(hu[l_[0]:h_[0], l_[1]:h_[1], l_[2]:h_[2]], [5, 25, 50, 75, 95])
        out[f"{h}_x"] = xs[sel]; out[f"{h}_xyz"] = xyz[sel].astype(np.int16); out[f"{h}_patch"] = tokens[f"{h}_patch"][sel]
        out[f"{h}_cov_tum"] = ct_[sel].astype(np.float16); out[f"{h}_cov_liv"] = cl_[sel].astype(np.float16)
        out[f"{h}_pred_cov"] = pc[sel].astype(np.float16); out[f"{h}_edt"] = edt[c[sel, 0], c[sel, 1], c[sel, 2]].astype(np.float16)
        out[f"{h}_is_tum"] = is_t[sel]; out[f"{h}_hu14"] = hf.astype(np.float16)
        tum_ids = comp[c[is_t, 0], c[is_t, 1], c[is_t, 2]]
        out[f"{h}_comp_retained"] = np.array([(tum_ids == i).any() for i in keep_ids])
    # head check: class-26 logits of interior GT-tumour voxels (annotated slices, > 2 mm from the GT edge)
    interior = tum & (edt > 2.0)
    rng = np.random.default_rng(0)
    fn_v, tp_v = interior & ~pred, interior & pred
    for nm, m in (("logit_fn", fn_v), ("logit_tp", tp_v)):
        v = logit[m]; out[nm] = (rng.choice(v, 20000, replace=False) if len(v) > 20000 else v).astype(np.float16); out[nm + "_n"] = int(m.sum())
    zc = fn_v.sum(axis=(0, 1)); z = int(zc.argmax()) if zc.max() > 0 else int(tum.sum(axis=(0, 1)).argmax())
    out.update({"prev_hu": hu[:, :, z].astype(np.float16), "prev_lab": L[:, :, z], "prev_pred": pred[:, :, z], "prev_z": z})
    np.savez_compressed(dst + ".part.npz", **out); os.replace(dst + ".part.npz", dst)
    if a.model == "trained":  # original-space tumour probability map for threshold recovery
        pre = build_transforms(); data = pre({"image": os.path.join(d, "ct.nii.gz")})
        assert tuple(data["image"].shape[1:]) == (H, W, D), "image-only and pair preprocessing disagree on shape"
        p = MetaTensor(torch.sigmoid(torch.from_numpy(logit))[None], meta=data["image"].meta, applied_operations=data["image"].applied_operations)
        po = Invertd(keys="pred", transform=pre, orig_keys="image", nearest_interp=False, to_tensor=True)({"pred": p, "image": data["image"]})["pred"].as_tensor()[0].numpy()
        np.savez_compressed(os.path.join("cache/vp_prob", f"{case}.npz"), p=po.astype(np.float16))
    print(f"[{ci}/{len(cases)}] {case} tumour comps {len(keep_ids)}; tokens (tum/liv) " + " ".join(f"{h}={int(out[h + '_is_tum'].sum())}/{int((~out[h + '_is_tum']).sum())}" for h in HOOKS)
          + f"; interior FN voxels {out['logit_fn_n']} TP {out['logit_tp_n']}; {time.time() - t0:.0f}s", flush=True)
