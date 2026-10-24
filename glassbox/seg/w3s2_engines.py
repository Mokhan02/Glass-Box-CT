"""W3 Stage 2 inference engines for M2 (erasure), M3 (counterfactual) and the M4 baseline maps (research use only for model A; torch / monai / nnunetv2 imported lazily so the pure helpers are testable anywhere).
One interface for both models: `prepare(ct_path, label_path)` -> ctx; `predict(ctx, eraser=None)` -> tumour / liver masks on the ORIGINAL grid (the Stage 1 pipelines: A = the authors' shipped pipeline with its post-processing, D = nnUNetv2 defaults with mirroring) and the tumour decision margin on the model grid;
`record_tokens(ctx, roi)` -> mean feature of the pre-registered "mid" tensor over tokens inside `roi` (M3 exploratory projection). The eraser is the affine map x - d (a.(x - mu)) / c applied to the intervention tensor in every window (A: output of backbone.swinViT.layers2[0]; D: output of encoder.stages[3]),
which feeds both the skip connection and the deeper layers (verified by scripts/w3s2_hook_check.py). Eraser dict: numpy mu [C], d [C], a [C], c scalar."""
from __future__ import annotations

import os
import sys

import numpy as np

from glassbox.seg import w3s2 as T


def apply_eraser_torch(out, p):
    """x' = x - d (a.(x - mu)) / c on a (B, C, X, Y, Z) tensor (the same formula as glassbox.seg.w2_lib.eraser_apply_np); `p` holds torch tensors mu, d, a and a float c."""
    import torch
    t = torch.einsum("bcxyz,c->bxyz", out - p["mu"][None, :, None, None, None], p["a"]) / p["c"]
    return out - p["d"][None, :, None, None, None] * t[:, None]


def identity_eraser(dim):
    return {"mu": np.zeros(dim, np.float32), "d": np.zeros(dim, np.float32), "a": np.zeros(dim, np.float32), "c": 1.0}


class Site:
    """Forward hook on the intervention module: applies the eraser (if set) and returns the modified tensor; returns nothing otherwise (a hook that returns a value REPLACES the output)."""

    def __init__(self, module, dev):
        import torch
        self.torch, self.dev, self.eraser, self.last = torch, dev, None, None
        module.register_forward_hook(self._hook)

    def set_eraser(self, e):
        torch = self.torch
        self.eraser = None if e is None else {"mu": torch.as_tensor(np.asarray(e["mu"], np.float32), device=self.dev), "d": torch.as_tensor(np.asarray(e["d"], np.float32), device=self.dev),
                                              "a": torch.as_tensor(np.asarray(e["a"], np.float32), device=self.dev), "c": float(e["c"])}

    def _hook(self, module, args, out):
        self.last = out.detach()
        return None if self.eraser is None else apply_eraser_torch(out, self.eraser)


def mapped_label(case, root, outdir, lesion_ids=None):
    """GT label map on the ORIGINAL grid written as NIfTI: 0 background, 1 liver (non-tumour), 2 tumour, 3 unknown (unannotated slice); `lesion_ids` {rank: bool mask of the full-volume lesion} are coded 10 + rank (M3 per-lesion token membership)."""
    import nibabel as nib
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); lab = np.zeros(seg.shape, np.uint8); lab[seg == 1] = 1; lab[seg == 2] = 2; lab[:, :, ~ann] = 3
    for r, m in (lesion_ids or {}).items():
        lab[m] = 10 + r
    os.makedirs(os.path.join(outdir, "tmp_labels"), exist_ok=True); p = os.path.join(outdir, "tmp_labels", f"{case}.nii.gz")
    nib.save(nib.Nifti1Image(lab, nib.load(os.path.join(root, case, "ct.nii.gz")).affine), p); return p


class Ctx:
    pass


class EngineA:
    """CLIP-Driven Universal Model, Swin UNETR (or the U-Net checkpoint), the authors' shipped pipeline up to `organ_post_process` for the liver and Liver Tumor classes."""
    name, spacing, order = "A", (1.5, 1.5, 1.5), "xyz"

    def __init__(self, repo_a, weights, scratch, dev):
        import torch
        sys.path.insert(0, repo_a)
        from model.Universal_model import Universal_model
        from monai.transforms import AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd, ToTensord
        from utils.utils import NUM_CLASS
        self.torch, self.dev, self.scratch = torch, dev, scratch; m = Universal_model(img_size=(96, 96, 96), in_channels=1, out_channels=NUM_CLASS, backbone="swinunetr", encoding="word_embedding"); store = m.state_dict()
        for k, v in torch.load(weights, map_location="cpu")["net"].items():
            store[("backbone." + ".".join(k.split(".")[1:])) if ("swinViT" in k or "encoder" in k or "decoder" in k) else ".".join(k.split(".")[1:])] = v
        m.load_state_dict(store); self.model = m.to(dev).eval(); self.site = Site(self.model.backbone.swinViT.layers2[0], dev); self.dim = 192
        self.tf_img = Compose([LoadImaged(keys=["image"]), AddChanneld(keys=["image"]), Orientationd(keys=["image"], axcodes="RAS"), Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
                               ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=["image"], source_key="image"), ToTensord(keys=["image"])])
        self.tf_lab = Compose([LoadImaged(keys=["image", "label"]), AddChanneld(keys=["image", "label"]), Orientationd(keys=["image", "label"], axcodes="RAS"), Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
                               ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True), CropForegroundd(keys=["image", "label"], source_key="image"), ToTensord(keys=["image", "label"])])
        self.args = type("A", (), {"log_name": os.path.join(scratch, "pp_log")})()

    def prepare(self, ct_path, label_path=None):
        c = Ctx(); d = self.tf_img({"image": ct_path}); c.d = d; c.x = d["image"].unsqueeze(0).to(self.dev); c.Lmap = None
        if label_path is not None:
            dl = self.tf_lab({"image": ct_path, "label": label_path}); assert tuple(dl["image"].shape) == tuple(d["image"].shape), "image-only and image+label preprocessing disagree"; c.Lmap = dl["label"][0].numpy().round().astype(np.int16)
        c.grid_spacing = self.spacing; return c

    def _orig(self, ctx, mask):
        """Nearest-neighbour inversion to the original grid exactly as the shipped save_results (MONAI 0.9.0 dictionary-style meta: `image_meta_dict` and `image_transforms` travel in the dict; images here are plain tensors, not MetaTensors)."""
        from monai.transforms import Invertd
        dd = dict(ctx.d); dd["k"] = self.torch.from_numpy(mask.astype(np.float32))[None]
        return Invertd(keys=["k"], transform=self.tf_img, orig_keys="image", nearest_interp=True, to_tensor=True)(dd)["k"][0].numpy() > 0.5

    def predict(self, ctx, eraser=None):
        from monai.inferers import sliding_window_inference
        from utils.utils import organ_post_process, threshold_organ
        torch = self.torch; self.site.set_eraser(eraser)
        try:
            with torch.no_grad():
                logits = sliding_window_inference(ctx.x, (96, 96, 96), 1, self.model, overlap=0.5, mode="gaussian")
        finally:
            self.site.set_eraser(None)
        post = organ_post_process(threshold_organ(torch.sigmoid(logits)).cpu().numpy(), [6, 27], os.path.join(self.scratch, "x", "p"), self.args)
        return {"tumour": self._orig(ctx, post[0, 26] > 0), "liver": self._orig(ctx, post[0, 5] > 0), "margin_grid": logits[0, 26].float().cpu().numpy()}

    def record_tokens(self, ctx, roi):
        """Mean mid feature (192-d) over the native 12 mm tokens whose centre lies inside `roi` (grid mask), same window grid as the M1 extraction (96^3, overlap 0.5)."""
        import torch.nn.functional as F
        torch = self.torch; H, W, D = roi.shape; diff = [max(96 - s, 0) for s in (H, W, D)]; pb = [dd // 2 for dd in diff]
        xp = F.pad(ctx.x, (pb[2], diff[2] - pb[2], pb[1], diff[1] - pb[1], pb[0], diff[0] - pb[0])); tot, n = np.zeros(self.dim, np.float64), 0
        for o in T.window_origins(tuple(xp.shape[2:]), (96, 96, 96), 0.5):
            lo = np.maximum(np.array(o) - pb, 0); hi = np.minimum(np.array(o) - pb + 96, (H, W, D))
            if not roi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
                continue
            with torch.no_grad():
                self.model.backbone(xp[:, :, o[0]:o[0] + 96, o[1]:o[1] + 96, o[2]:o[2] + 96])
            f = self.site.last[0].float(); C, a, b, c = f.shape; ii = np.stack(np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing="ij"), -1).reshape(-1, 3); xyz = ii * 8 + 4 + np.array(o) - np.array(pb)
            ok = np.all((xyz >= 0) & (xyz < np.array([H, W, D])), axis=1); ok[ok] = roi[tuple(xyz[ok].T)]
            if ok.any():
                sel = f.reshape(C, -1).T[torch.from_numpy(np.flatnonzero(ok)).to(f.device)]; tot += sel.sum(0).double().cpu().numpy(); n += int(ok.sum())
        return (tot / max(n, 1)).astype(np.float32), n


class EngineD:
    """nnU-Net v2 fold 0 (checkpoint_final), `predict_logits_from_preprocessed_data` (mirroring on, step 0.5) and nnU-Net's own export to the original geometry."""
    name, order = "D", "zyx"

    def __init__(self, model_folder, scratch, dev):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.torch, self.dev, self.scratch = torch, dev, scratch
        self.pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True, device=dev, verbose=False, allow_tqdm=False)
        self.pred.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name="checkpoint_final.pth"); self.net = self.pred.network.to(dev).eval(); self.cm = self.pred.configuration_manager
        self.spacing = tuple(float(s) for s in self.cm.spacing); self.site = Site(self.net.encoder.stages[3], dev); self.dim = 256; os.makedirs(scratch, exist_ok=True)

    def prepare(self, ct_path, label_path=None):
        c = Ctx(); c.data, seg, c.props = self.cm.preprocessor_class(verbose=False).run_case([ct_path], label_path, self.pred.plans_manager, self.cm, self.pred.dataset_json)
        c.Lmap = None if seg is None else np.where(seg[0] < 0, 3, seg[0]).astype(np.int16); c.grid_spacing = self.spacing; return c

    def predict(self, ctx, eraser=None):
        import nibabel as nib
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        torch = self.torch; self.site.set_eraser(eraser)
        try:
            with torch.no_grad():
                lg = self.pred.predict_logits_from_preprocessed_data(torch.from_numpy(ctx.data))
        finally:
            self.site.set_eraser(None)
        lg = lg.float(); margin = (lg[2] - torch.maximum(lg[0], lg[1])).numpy(); tmp = os.path.join(self.scratch, "pred"); export_prediction_from_logits(lg, ctx.props, self.cm, self.pred.plans_manager, self.pred.dataset_json, tmp, False)
        seg = np.asarray(nib.load(tmp + ".nii.gz").dataobj)
        return {"tumour": seg == 2, "liver": seg >= 1, "margin_grid": margin}

    def record_tokens(self, ctx, roi):
        """Mean stage-3 feature (256-d) over the 2x-pooled tokens (cell 16 voxels) whose centre lies inside `roi`; windows as the M1 extraction (nnU-Net sliding-window grid, step 0.5)."""
        import torch.nn.functional as F
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        torch = self.torch; ps = list(self.cm.patch_size); Z = roi.shape; xp, rev = pad_nd_image(torch.from_numpy(ctx.data), ps, "constant", {"value": 0}, True, None); pb = [s.start for s in rev[1:]]
        tot, n = np.zeros(self.dim, np.float64), 0
        for sl in self.pred._internal_get_sliding_window_slicers(tuple(xp.shape[1:])):
            o = [s.start for s in sl[1:]]; lo = np.maximum(np.array(o) - pb, 0); hi = np.minimum(np.array(o) - pb + ps, Z)
            if not roi[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any():
                continue
            with torch.no_grad():
                self.net(xp[sl][None].to(self.dev).float())
            f = F.avg_pool3d(self.site.last.float(), 2)[0]; C, a, b, c = f.shape; ii = np.stack(np.meshgrid(np.arange(a), np.arange(b), np.arange(c), indexing="ij"), -1).reshape(-1, 3); xyz = ii * 16 + 8 + np.array(o) - np.array(pb)
            ok = np.all((xyz >= 0) & (xyz < np.array(Z)), axis=1); ok[ok] = roi[tuple(xyz[ok].T)]
            if ok.any():
                sel = f.reshape(C, -1).T[torch.from_numpy(np.flatnonzero(ok)).to(f.device)]; tot += sel.sum(0).double().cpu().numpy(); n += int(ok.sum())
        return (tot / max(n, 1)).astype(np.float32), n
