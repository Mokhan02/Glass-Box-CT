"""W3 Stage 2, M1 token extraction (results/w3_models/PREREG_stage2.md), research use only for model A.
For every case of the Stage 2 cohort and for the trained network AND a randomly initialised copy (`torch.manual_seed(0)`, same windows, so the token sets are identical), records the activations at the pre-registered layers for every sliding-window token whose centre lies inside the GT liver + tumour
dilated by 10 mm, with the VISTA-PREREG token labels (`glassbox/seg/w3s2.py`), the model's own predicted tumour coverage, the interior distance, 14 local HU features and the head margin of interior GT-tumour voxels. Only windows that contain ROI voxels are run for the recorder (the others cannot produce tokens).
Families: A (Swin UNETR, shipped pipeline), AUNET (the CLIP-Driven U-Net checkpoint, sensitivity), D (nnU-Net fold 0, default inference).
Layer keys written per family (the analysis reads them by name):
  A     mid = swinViT.layers2[0] output (192 ch, stride 8, 12 mm; the tensor the eraser edits) | mid_hs2 = swinViT hidden state 2 (sensitivity) | bottleneck = hidden state 4 (768 ch) | decoder_F = backbone.decoder1 output (48 ch, pooled x8)
  AUNET mid = backbone.down_tr256 output[0] (256 ch, stride 8) | bottleneck = backbone.down_tr512 output[0] (512 ch) | decoder_F = backbone.up_tr64 output (64 ch, pooled x8)
  D     mid = encoder.stages[3] output pooled x2 (256 ch, cell 16 voxels) | mid_s4 = encoder.stages[4] (320 ch, sensitivity) | bottleneck = encoder.stages[5] | decoder_F = decoder.stages[-1] output (32 ch, pooled x16)
Usage (cwd = repo checkout, GPU): python scripts/w3s2_m1_extract.py A|AUNET <outdir> <repo_A> <weights.pth> [--limit N] [--cases c1 c2 ...]
                                  python scripts/w3s2_m1_extract.py D <outdir> <nnU-Net model folder> [--limit N] [--cases ...]"""
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from glassbox.seg import w3_lib as L  # noqa: E402
from glassbox.seg import w3s2 as T  # noqa: E402

fam, outdir = sys.argv[1:3]; rest = [a for a in sys.argv[3:]]
limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None; only = rest[rest.index("--cases") + 1:] if "--cases" in rest else None
pos = [a for a in rest if not a.startswith("--") and a not in (only or []) and a != (str(limit) if limit else None)]
table = L.lesion_table("hcc", include_excluded=True); cases = sorted({k[0] for k in table}); cases = only or cases[:limit]
os.makedirs(os.path.join(outdir, "tokens"), exist_ok=True); os.makedirs(os.path.join(outdir, "tmp_labels"), exist_ok=True); dev = torch.device("cuda")


def mapped_label(case, root):
    """GT label map on the ORIGINAL grid: 0 background, 1 liver (non-tumour), 2 tumour, 3 unknown (unannotated slice); written next to the outputs so both preprocessing pipelines can resample it like the image."""
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); lab = np.zeros(seg.shape, np.uint8); lab[seg == 1] = 1; lab[seg == 2] = 2; lab[:, :, ~ann] = 3
    p = os.path.join(outdir, "tmp_labels", f"{case}.nii.gz"); nib.save(nib.Nifti1Image(lab, nib.load(os.path.join(root, case, "ct.nii.gz")).affine), p); return p


def avg_pool(f, k):
    return f if k == 1 else F.avg_pool3d(f, k)


class Recorder:
    """Forward hooks on one network; `layers` = {name: (source key, pool factor, stride in voxels)}. After a window forward, `tokens(origin, ...)` returns the pooled token features and centre voxels."""

    def __init__(self, net, hooks, layers):
        self.caps, self.layers, self.hs = {}, layers, None
        for key, (module, idx) in hooks.items():
            def h(m, a, o, key=key, idx=idx):
                o = o[idx] if idx is not None and isinstance(o, (tuple, list)) else o; self.caps[key] = o
            module.register_forward_hook(h)

    def tokens(self, name, origin, pad_before, shape, roi):
        src, pool, stride = self.layers[name]; f = avg_pool(self.caps[src].float(), pool)[0]; cs = stride * pool; C, a, b, c = f.shape
        ii = np.stack(np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing="ij"), -1).reshape(-1, 3); xyz = ii * cs + cs // 2 + np.array(origin) - np.array(pad_before)
        ok = np.all((xyz >= 0) & (xyz < np.array(shape)), axis=1); ok[ok] = roi[tuple(xyz[ok].T)]
        return f.reshape(C, -1).T[torch.from_numpy(np.flatnonzero(ok)).to(f.device)].half().cpu().numpy(), xyz[ok], cs


def finish(case, layers_cell, st, recs, xyz_by_layer, pred_mask, margin, L_map):
    """Assemble and save one case: token labels (from the TRAINED model's prediction) for tokens recorded identically for the trained and random-init networks."""
    res = {"gt_tumour_mm3": float((L_map == 2).sum() * float(np.prod(st["spacing"])))}
    for name, cell in layers_cell.items():
        xyz = np.concatenate(xyz_by_layer[name]) if xyz_by_layer[name] else np.zeros((0, 3), np.int64)
        xt = np.concatenate(recs["trained"][name]) if recs["trained"][name] else np.zeros((0, 1), np.float16); xr = np.concatenate(recs["random"][name]) if recs["random"][name] else np.zeros((0, 1), np.float16)
        tok = st["stats"].tokens(xyz, cell) if len(xyz) else {k: np.zeros(0) for k in ("is_tum", "is_liv", "pred_cov", "edt")} | {"hu14": np.zeros((0, 14), np.float32)}
        sel = np.flatnonzero(tok["is_tum"] | tok["is_liv"])
        res[f"{name}_x"] = xt[sel]; res[f"{name}_xr"] = xr[sel]; res[f"{name}_xyz"] = xyz[sel].astype(np.int16); res[f"{name}_is_tum"] = tok["is_tum"][sel]; res[f"{name}_pred_cov"] = tok["pred_cov"][sel].astype(np.float16)
        res[f"{name}_edt"] = tok["edt"][sel].astype(np.float16); res[f"{name}_hu14"] = tok["hu14"][sel].astype(np.float16)
    interior = (L_map == 2) & (st["stats"].edt > T.INTERIOR_MM); rng = np.random.default_rng(0)
    for nm, m in (("logit_fn", interior & ~pred_mask), ("logit_tp", interior & pred_mask)):
        v = margin[m]; res[nm] = (rng.choice(v, 20000, replace=False) if len(v) > 20000 else v).astype(np.float16); res[nm + "_n"] = int(m.sum())
    dst = os.path.join(outdir, "tokens", f"{case}.npz"); np.savez_compressed(dst + ".part.npz", **res); os.replace(dst + ".part.npz", dst)
    print(f"[{case}] tokens (tum/liv) " + " ".join(f"{n}={int(res[n + '_is_tum'].sum())}/{int((~res[n + '_is_tum']).sum())}" for n in layers_cell) + f"; interior FN voxels {res['logit_fn_n']} TP {res['logit_tp_n']}", flush=True)


if fam in ("A", "AUNET"):
    repo_a, wts = pos[0], pos[1]; sys.path.insert(0, repo_a)
    import types

    from monai.inferers import sliding_window_inference
    from monai.transforms import AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord
    from model.Universal_model import Universal_model  # noqa: E402
    from utils.utils import NUM_CLASS, organ_post_process, threshold_organ  # noqa: E402
    bb = "swinunetr" if fam == "A" else "unet"
    tf = Compose([LoadImaged(keys=["image", "label"]), AddChanneld(keys=["image", "label"]), Orientationd(keys=["image", "label"], axcodes="RAS"), Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
                  ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=["image", "label"], source_key="image"), ToTensord(keys=["image", "label"])])

    def build(trained):
        if not trained:
            torch.manual_seed(0)
        m = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone=bb, encoding="word_embedding")
        if trained:
            store = m.state_dict()
            for k, v in torch.load(wts, map_location="cpu")["net"].items():
                store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
            m.load_state_dict(store)
        return m.to(dev).eval()
    nets = {"trained": build(True), "random": build(False)}
    if fam == "A":
        layers = {"mid": ("mid", 1, 8), "mid_hs2": ("hs2", 1, 8), "bottleneck": ("hs4", 1, 32), "decoder_F": ("dec", 8, 1)}
        mk = lambda net: Recorder(net, {"mid": (net.backbone.swinViT.layers2[0], None), "dec": (net.backbone.decoder1, None)}, layers)      # noqa: E731
    else:
        layers = {"mid": ("mid", 1, 8), "bottleneck": ("bott", 1, 8), "decoder_F": ("dec", 8, 1)}
        mk = lambda net: Recorder(net, {"mid": (net.backbone.down_tr256, 0), "bott": (net.backbone.down_tr512, 0), "dec": (net.backbone.up_tr64, None)}, layers)      # noqa: E731
    recs = {k: mk(n) for k, n in nets.items()}
    if fam == "A":
        for k, n in nets.items():                 # hidden states come from the swinViT return value
            n.backbone.swinViT.register_forward_hook(lambda m, a, o, r=recs[k]: (r.caps.__setitem__("hs2", o[2]), r.caps.__setitem__("hs4", o[4])))
    cells = {k: (v[1] * v[2],) * 3 for k, v in layers.items()}
    args = types.SimpleNamespace(log_name=os.path.join(outdir, "pp_log"))
    for ci, case in enumerate(cases, 1):
        if os.path.exists(os.path.join(outdir, "tokens", f"{case}.npz")):
            continue
        t0 = time.time(); root = L.case_root("hcc", case, L.HCC_ROOT); d = tf({"image": os.path.join(root, case, "ct.nii.gz"), "label": mapped_label(case, root)})
        x = d["image"].unsqueeze(0).to(dev); Lmap = d["label"][0].numpy().round().astype(np.int16); hu = (x[0, 0].cpu().numpy() * 425.0 - 175.0).astype(np.float32); sp = (1.5, 1.5, 1.5)
        if not ((Lmap == 2).any() and (Lmap == 1).any()):
            print(f"[{case}] SKIP (no tumour or liver after resampling)", flush=True); continue
        with torch.no_grad():
            logits = sliding_window_inference(x, (96, 96, 96), 1, nets["trained"], overlap=0.5, mode="gaussian")
        sig = torch.sigmoid(logits); post = organ_post_process(threshold_organ(sig).cpu().numpy(), [6, 27], os.path.join(outdir, "x", case), args)
        pred = post[0, 26].astype(bool); margin = logits[0, 26].float().cpu().numpy(); roi = T.roi_mask(Lmap, sp); st = {"stats": T.GridStats(Lmap, pred, hu, sp), "spacing": sp}
        H, W, D = Lmap.shape; diff = [max(96 - s, 0) for s in (H, W, D)]; pb = [dd // 2 for dd in diff]; xp = F.pad(x, (pb[2], diff[2] - pb[2], pb[1], diff[1] - pb[1], pb[0], diff[0] - pb[0]))
        origins = T.window_origins(tuple(xp.shape[2:]), (96, 96, 96), 0.5); xyz_by = {n: [] for n in layers}; tok = {k: {n: [] for n in layers} for k in nets}
        for o in origins:
            lo = np.maximum(np.array(o) - pb, 0); hi = np.minimum(np.array(o) - pb + 96, (H, W, D))
            if not roi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
                continue
            win = xp[:, :, o[0]:o[0] + 96, o[1]:o[1] + 96, o[2]:o[2] + 96]
            for k, net in nets.items():
                with torch.no_grad():
                    net.backbone(win)
                for n in layers:
                    f, xyz, cs = recs[k].tokens(n, o, pb, (H, W, D), roi); tok[k][n].append(f)
                    if k == "trained":
                        xyz_by[n].append(xyz)
        finish(case, cells, st, tok, xyz_by, pred, margin, Lmap)
        print(f"  {ci}/{len(cases)} {time.time() - t0:.0f}s", flush=True)
else:
    mf = pos[0]
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred_ = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True, device=dev, verbose=False, allow_tqdm=False)
    pred_.initialize_from_trained_model_folder(mf, use_folds=(0,), checkpoint_name="checkpoint_final.pth"); cm = pred_.configuration_manager; ps = list(cm.patch_size); spacing = tuple(float(s) for s in cm.spacing)
    arch = pred_.plans_manager.plans["configurations"]["3d_fullres"]["architecture"]; kw = dict(arch["arch_kwargs"])
    for k in arch["_kw_requires_import"]:
        if kw.get(k) is not None:
            mod, name = kw[k].rsplit(".", 1); kw[k] = getattr(__import__(mod, fromlist=[name]), name)
    torch.manual_seed(0); rnd = PlainConvUNet(input_channels=1, num_classes=3, deep_supervision=False, **kw).to(dev).eval(); nets = {"trained": pred_.network.to(dev).eval(), "random": rnd}
    layers = {"mid": ("s3", 2, 8), "mid_s4": ("s4", 1, 16), "bottleneck": ("s5", 1, 32), "decoder_F": ("dec", 16, 1)}
    recs = {k: Recorder(n, {"s3": (n.encoder.stages[3], None), "s4": (n.encoder.stages[4], None), "s5": (n.encoder.stages[5], None), "dec": (n.decoder.stages[-1], None)}, layers) for k, n in nets.items()}
    cells = {k: (v[1] * v[2],) * 3 for k, v in layers.items()}
    fg = pred_.plans_manager.plans["foreground_intensity_properties_per_channel"]["0"] if "foreground_intensity_properties_per_channel" in pred_.plans_manager.plans else None
    for ci, case in enumerate(cases, 1):
        if os.path.exists(os.path.join(outdir, "tokens", f"{case}.npz")):
            continue
        t0 = time.time(); root = L.case_root("hcc", case, L.HCC_ROOT)
        data, seg, props = cm.preprocessor_class(verbose=False).run_case([os.path.join(root, case, "ct.nii.gz")], mapped_label(case, root), pred_.plans_manager, cm, pred_.dataset_json)
        Lmap = np.where(seg[0] < 0, 3, seg[0]).astype(np.int16); img = data[0].astype(np.float32)
        mean, std = (float(fg["mean"]), float(fg["std"])) if fg else (99.48, 37.14); hu = np.clip(img * std + mean, -15.0, 197.0).astype(np.float32); Z = Lmap.shape
        if not ((Lmap == 2).any() and (Lmap == 1).any()):
            print(f"[{case}] SKIP (no tumour or liver after resampling)", flush=True); continue
        with torch.no_grad():
            lg = pred_.predict_logits_from_preprocessed_data(torch.from_numpy(data)).float().numpy()
        pred = lg.argmax(0) == 2; margin = lg[2] - np.maximum(lg[0], lg[1]); roi = T.roi_mask(Lmap, spacing); st = {"stats": T.GridStats(Lmap, pred, hu, spacing), "spacing": spacing}
        xp, rev = pad_nd_image(torch.from_numpy(data), ps, "constant", {"value": 0}, True, None); pb = [s.start for s in rev[1:]]; slicers = pred_._internal_get_sliding_window_slicer(tuple(xp.shape[1:]))
        xyz_by = {n: [] for n in layers}; tok = {k: {n: [] for n in layers} for k in nets}
        for sl in slicers:
            o = [s.start for s in sl[1:]]; lo = np.maximum(np.array(o) - pb, 0); hi = np.minimum(np.array(o) - pb + ps, Z)
            if not roi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
                continue
            win = xp[sl][None].to(dev).float()
            for k, net in nets.items():
                with torch.no_grad():
                    net(win)
                for n in layers:
                    f, xyz, cs = recs[k].tokens(n, o, pb, Z, roi); tok[k][n].append(f)
                    if k == "trained":
                        xyz_by[n].append(xyz)
        finish(case, cells, st, tok, xyz_by, pred, margin, Lmap)
        print(f"  {ci}/{len(cases)} {time.time() - t0:.0f}s", flush=True)
print("EXTRACT_DONE", flush=True)
