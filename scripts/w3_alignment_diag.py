"""W3 Stage 1 alignment DIAGNOSTIC (not a result; no tumour metric is computed): reruns model A (swin, shipped pipeline up to the shipped post-processing) on named HCC cases and saves the shipped organ masks (liver, spleen, both kidneys).
Per case: liver Dice, precision, recall and volume ratio vs the GT liver (labels 1+2) on the annotated slices; predicted-vs-GT liver centroid offset in mm (CT affine, array indices); bounding-box IoU; liver Dice under flips and best shifts of the predicted liver
(a misalignment test); input vs output shape and orientation; spleen and kidney sanity checks (no GT exists for them in HCC-TACE-Seg: volumes, and the left/right and inferior/superior relations to the liver centroid in RAS mm); overlay PNGs.
Verdict rule fixed before the run: MISALIGNED if the output shape differs from the input, OR the liver centroid offset > 15 mm, OR a flip/shift of the predicted liver raises Dice by > 0.05 over the unshifted Dice; otherwise ALIGNED-BUT-UNDERSEGMENTED.
Usage (cwd = repo checkout): python scripts/w3_alignment_diag.py <repo_A> <weights> <outdir> <case> [<case> ...]"""
import json
import os
import sys
import types

import matplotlib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Invertd
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

repo_a, wts, outdir = sys.argv[1:4]; cases = sys.argv[4:]
sys.path.insert(0, repo_a); sys.path.insert(0, ".")
from dataset.dataloader import get_loader_without_gt  # noqa: E402
from model.Universal_model import Universal_model  # noqa: E402
from utils.utils import NUM_CLASS, organ_post_process, threshold_organ  # noqa: E402
from glassbox.seg import w3_lib as L  # noqa: E402

os.makedirs(outdir, exist_ok=True); indir = os.path.join(outdir, "in"); os.makedirs(indir, exist_ok=True)
for c in cases:
    d = os.path.join(indir, f"{c}.nii.gz")
    if not os.path.lexists(d):
        os.symlink(os.path.join(L.HCC_ROOT, c, "ct.nii.gz"), d)
args = types.SimpleNamespace(space_x=1.5, space_y=1.5, space_z=1.5, a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, data_root_path=indir, cache_dataset=False, cache_rate=0.6, log_name=os.path.join(outdir, "pp_log"))
loader, tf = get_loader_without_gt(args)
model = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr", encoding="word_embedding"); store = model.state_dict()
for k, v in torch.load(wts, map_location="cpu")["net"].items():
    store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
model.load_state_dict(store); model.cuda().eval()


def invert(batch, mask, key):
    batch[key] = torch.tensor(mask.astype(np.float64))[None, None]
    out = Invertd(keys=[key], transform=tf, orig_keys="image", nearest_interp=True, to_tensor=True)(decollate_batch(batch)[0]); del batch[key]
    return out[key][0].numpy() > 0.5


def centroid_mm(m, affine):
    idx = np.argwhere(m)
    return None if len(idx) == 0 else (affine @ np.append(idx.mean(0), 1.0))[:3]


def bbox_iou(a, b):
    ia, ib = np.argwhere(a), np.argwhere(b)
    lo = np.maximum(ia.min(0), ib.min(0)); hi = np.minimum(ia.max(0), ib.max(0)); inter = np.prod(np.clip(hi - lo + 1, 0, None)); va = np.prod(ia.max(0) - ia.min(0) + 1); vb = np.prod(ib.max(0) - ib.min(0) + 1)
    return float(inter / (va + vb - inter))


R = {}
for batch in loader:
    name = batch["name"][0]
    if name not in cases:
        continue
    with torch.no_grad():
        sig = F.sigmoid(sliding_window_inference(batch["image"].cuda(), (96, 96, 96), 1, model, overlap=0.5, mode="gaussian"))
    hard = threshold_organ(sig).cpu().numpy(); post = organ_post_process(hard.copy(), [1, 2, 3, 6], os.path.join(outdir, "x", name), args)
    nii = nib.load(os.path.join(L.HCC_ROOT, name, "ct.nii.gz")); seg = np.asarray(nib.load(os.path.join(L.HCC_ROOT, name, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); sp = nii.header.get_zooms()[:3]; aff = nii.affine
    org = {k: invert(batch, post[0, i - 1].astype(bool), f"k{i}") for k, i in (("liver", 6), ("spleen", 1), ("right_kidney", 2), ("left_kidney", 3))}
    shapes = {k: list(v.shape) for k, v in org.items()}; gt = L.gt_organ(seg, "hcc"); sh = seg.shape
    liv_full = org["liver"][:sh[0], :sh[1], :sh[2]]; liv = np.zeros(sh, bool); liv[:, :, ann] = liv_full[:, :, ann]; gta = np.zeros(sh, bool); gta[:, :, ann] = gt[:, :, ann]
    vox_ml = float(np.prod(sp)) / 1000; tp = int((liv & gta).sum()); d0 = L.dice(liv, gta)
    flips = {f"flip_{ax}": L.dice(np.flip(liv, ax) & True, gta) for ax in (0, 1, 2)}
    best = {"dice": d0, "shift": (0, 0, 0)}
    for dx in range(-40, 41, 4):
        for dy in range(-40, 41, 4):
            for dz in range(-6, 7, 2):
                dd = L.dice(np.roll(liv, (dx, dy, dz), axis=(0, 1, 2)), gta)
                if dd > best["dice"]:
                    best = {"dice": dd, "shift": (dx, dy, dz)}
    c_p, c_g = centroid_mm(liv, aff), centroid_mm(gta, aff); off = float(np.linalg.norm(c_p - c_g)) if c_p is not None else None
    cl = centroid_mm(liv_full, aff); rel = {}
    for k in ("spleen", "right_kidney", "left_kidney"):
        m = org[k][:sh[0], :sh[1], :sh[2]]; c = centroid_mm(m, aff)
        rel[k] = {"volume_mL": round(float(m.sum()) * vox_ml, 1), "centroid_minus_liver_centroid_RAS_mm": None if c is None else [round(float(x), 1) for x in (c - cl)]}
    ras_ok = {"spleen_left_of_liver(x<0)": rel["spleen"]["centroid_minus_liver_centroid_RAS_mm"][0] < 0 if rel["spleen"]["centroid_minus_liver_centroid_RAS_mm"] else None,
              "right_kidney_right_of_left_kidney": (rel["right_kidney"]["centroid_minus_liver_centroid_RAS_mm"][0] > rel["left_kidney"]["centroid_minus_liver_centroid_RAS_mm"][0]) if rel["right_kidney"]["centroid_minus_liver_centroid_RAS_mm"] and rel["left_kidney"]["centroid_minus_liver_centroid_RAS_mm"] else None,
              "kidneys_inferior_to_liver_centroid(z<0)": all(rel[k]["centroid_minus_liver_centroid_RAS_mm"][2] < 0 for k in ("right_kidney", "left_kidney") if rel[k]["centroid_minus_liver_centroid_RAS_mm"])}
    shape_ok = all(v == list(nii.shape) for v in shapes.values()); gain = max(best["dice"], *flips.values()) - d0
    verdict = "MISALIGNED" if (not shape_ok or (off is not None and off > 15) or gain > 0.05) else "ALIGNED-BUT-UNDERSEGMENTED"
    R[name] = {"verdict": verdict, "liver_dice": round(d0, 4), "precision": round(tp / max(int(liv.sum()), 1), 4), "recall": round(tp / max(int(gta.sum()), 1), 4), "pred_over_gt_liver_volume": round(float(liv.sum() / max(gta.sum(), 1)), 3),
               "liver_mL_pred_annotated_slices": round(float(liv.sum()) * vox_ml, 1), "liver_mL_gt": round(float(gta.sum()) * vox_ml, 1), "liver_mL_pred_whole_volume": round(float(liv_full.sum()) * vox_ml, 1), "n_annotated_slices": int(ann.sum()), "n_slices": int(sh[2]),
               "centroid_offset_mm": None if off is None else round(off, 1), "centroid_offset_RAS_mm": [round(float(x), 1) for x in (c_p - c_g)], "bbox_iou": round(bbox_iou(liv, gta), 4), "dice_under_flips": {k: round(v, 4) for k, v in flips.items()},
               "best_shift_search": {"dice": round(best["dice"], 4), "shift_voxels": best["shift"], "gain_over_unshifted": round(best["dice"] - d0, 4)}, "output_shapes": shapes, "input_shape": list(nii.shape), "axcodes": "".join(nib.aff2axcodes(aff)), "spacing_mm": [round(float(s), 3) for s in sp],
               "shape_orientation_matches_input": shape_ok, "other_organs_sanity": rel, "other_organs_relations": ras_ok}
    np.savez_compressed(os.path.join(outdir, f"organs_{name}.npz"), ann=ann, **{k: v[:sh[0], :sh[1], :sh[2]][:, :, ann] for k, v in org.items()}, gt_liver=gt[:, :, ann])
    ct = np.asarray(nii.dataobj).astype(np.float32); zs = np.flatnonzero(ann); picks = [zs[int(len(zs) * f)] for f in (0.25, 0.5, 0.75)]; fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for a_, z in zip(ax, picks):
        a_.imshow(np.clip(ct[:, :, z].T, -150, 250), cmap="gray", origin="lower", aspect=sp[1] / sp[0])
        a_.contour(gt[:, :, z].T, levels=[0.5], colors="lime", linewidths=1.2); a_.contour(liv_full[:, :, z].T, levels=[0.5], colors="red", linewidths=1.2)
        for k, col in (("spleen", "cyan"), ("right_kidney", "yellow"), ("left_kidney", "orange")):
            a_.contour(org[k][:sh[0], :sh[1], :sh[2]][:, :, z].T, levels=[0.5], colors=col, linewidths=0.9)
        a_.set_title(f"{name} z={z}", fontsize=9); a_.axis("off")
    fig.suptitle(f"{name}: GT liver (green), predicted liver (red), spleen (cyan), R kidney (yellow), L kidney (orange); liver Dice {d0:.3f}, {verdict}", fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(outdir, f"overlay_{name}.png"), dpi=90); plt.close(fig)
    print(name, verdict, json.dumps({k: R[name][k] for k in ("liver_dice", "precision", "recall", "centroid_offset_mm", "bbox_iou")}), flush=True)
json.dump(R, open(os.path.join(outdir, "w3_alignment_diagnostic.json"), "w"), indent=1)
print("DIAG_DONE", flush=True)
