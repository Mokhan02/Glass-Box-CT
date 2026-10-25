"""CPU audit of the W5 augmentation (no GPU, no training): application rate, delta distribution, skips and realised contrast per arm.

Usage: python scripts/w5_aug_audit.py <preprocessed dir (nnUNetPlans_3d_fullres)> <splits_final.json> <out json> [n_patches]
Patches: cases of the fold-0 TRAIN split; 160^3 crops (about nnU-Net's initial patch), centred on a random tumour voxel with probability 1/3 (nnU-Net's foreground oversampling share) and at a random location otherwise. The same patches are used for every arm.
This approximates, and does not replace, the training dataloader (it uses neither its sampler nor its spatial augmentation)."""
import importlib.util
import json
import os
import sys

import blosc2
import numpy as np

here = os.path.dirname(os.path.abspath(__file__)); spec = importlib.util.spec_from_file_location("w5", os.path.join(here, "w5_nnunet_trainer.py")); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
D, splits, out = sys.argv[1:4]; n = int(sys.argv[4]) if len(sys.argv) > 4 else 600
train = json.load(open(splits))[0]["train"]; rng = np.random.default_rng(12345); patches = []
cache = {}
for i in range(n):
    c = train[int(rng.integers(len(train)))]
    if c not in cache:
        cache.clear(); cache[c] = (blosc2.open(D + "/" + c + ".b2nd"), blosc2.open(D + "/" + c + "_seg.b2nd"))
    im_f, sg_f = cache[c]; shp = sg_f.shape[1:]; sg_all = None
    if rng.random() < 1 / 3:
        sg_all = sg_f[:]; tv = np.argwhere(sg_all[0] == W.TUMOUR)
        ctr = tv[int(rng.integers(len(tv)))] if len(tv) else np.array([int(rng.integers(s)) for s in shp])
    else:
        ctr = np.array([int(rng.integers(s)) for s in shp])
    sl = tuple(slice(max(0, int(k) - 80), min(s, int(k) + 80)) for k, s in zip(ctr, shp))
    patches.append((c, im_f[(slice(None),) + sl].astype(np.float32), (sg_all[(slice(None),) + sl] if sg_all is not None else sg_f[(slice(None),) + sl]).astype(np.int16)))
def contrast(hu, seg, comp_mask):
    liver = (seg[0] == W.LIVER) & ~(seg[0] == W.TUMOUR)
    return float(hu[comp_mask].mean())
res = {"n_patches": n, "n_patches_with_tumour": int(sum((p[2] == W.TUMOUR).any() for p in patches)), "cases_in_train_split": len(train), "arms": {}}
base_c = []
for c, im, sg in patches:
    if not (sg == W.TUMOUR).any():
        continue
    from scipy import ndimage as ndi
    hu = im[0] * W.HU_STD + W.HU_MEAN; lab, k = ndi.label(sg[0] == W.TUMOUR); liv = (sg[0] == W.LIVER)
    if liv.sum() < 200:
        continue
    for j in range(1, k + 1):
        m = lab == j
        if m.sum() >= W.MIN_COMPONENT_VOXELS:
            base_c.append(float(hu[m].mean() - hu[liv].mean()))
res["unedited_tumour_minus_patch_liver_HU"] = {"n_components": len(base_c), "mean": float(np.mean(base_c)), "quantiles_5_25_50_75_95": [float(x) for x in np.quantile(base_c, [.05, .25, .5, .75, .95])], "share_brighter_than_liver": float(np.mean(np.array(base_c) > 0))}
for arm, p in (("BRIGHT-50", ("BRIGHT", .5)), ("SHAM-50", ("SHAM", .5)), ("BRIGHT-25", ("BRIGHT", .25))):
    r = np.random.default_rng({"BRIGHT-50": 1, "SHAM-50": 2, "BRIGHT-25": 3}[arm]); tp = tn = 0; applied = 0; deltas = []; skipped = {"no liver": 0}; src = {"local": 0, "patch": 0}; tgt_err = []; sign_ok = []; ncomp = 0
    for c, im, sg in patches:
        has = (sg == W.TUMOUR).any(); tp += int(has)
        new, info = W.shift_tumour_contrast(im, sg, p[0], r, p[1])
        applied += int(info["applied"])
        if info["applied"]:
            hu1 = new[0] * W.HU_STD + W.HU_MEAN; lab, _ = ndi.label(sg[0] == W.TUMOUR)
            for ci in info["components"]:
                if "skipped" in ci:
                    skipped["no liver"] += 1; continue
                ncomp += 1; deltas.append(ci["delta"]); src[ci["liver_source"]] += 1; m = lab == ci["component"]
                tgt_err.append(float(hu1[m].mean() - ci["target"])); sign_ok.append(bool((hu1[m].mean() - ci["liver_mean"]) * np.sign(ci["delta"]) > 0))
    res["arms"][arm] = {"p": p[1], "range_HU": W.ARMS[p[0]], "patches_with_tumour": tp, "patches_edited": applied, "edit_rate_among_tumour_patches": applied / max(tp, 1), "edit_rate_all_patches": applied / len(patches),
                        "edited_components": ncomp, "delta_mean": float(np.mean(deltas)), "delta_min": float(np.min(deltas)), "delta_max": float(np.max(deltas)), "delta_quantiles_5_50_95": [float(x) for x in np.quantile(deltas, [.05, .5, .95])],
                        "liver_source": src, "components_skipped_no_liver": skipped["no liver"], "realised_minus_target_HU_mean": float(np.mean(tgt_err)), "realised_minus_target_HU_abs_p95": float(np.quantile(np.abs(tgt_err), .95)), "share_realised_sign_as_intended": float(np.mean(sign_ok))}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
