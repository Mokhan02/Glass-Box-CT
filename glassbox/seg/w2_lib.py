"""Shared helpers for the W2-M GPU runners (M2 erasure, M3 counterfactual): lesion geometry (copied verbatim from scripts/w2v2_run.py), a point-prompt Runner with an optional mid-encoder hook
(eraser or token recorder), and the metrics used since W2 v3. torch / monai are imported lazily so the geometry and metric helpers can be tested anywhere."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

MID_STRIDE = 8


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def axial_diam(mask, sx, sy):
    return max((feret(mask[:, :, z], sx, sy) for z in np.flatnonzero(mask.any(axis=(0, 1)))), default=0.0)


def bbox(mask, pad, shape):
    idx = np.nonzero(mask); lo = [max(int(i.min()) - pad, 0) for i in idx]; hi = [min(int(i.max()) + pad + 1, n) for i, n in zip(idx, shape)]
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def interior_point(comp, sp):
    """Voxel of maximum distance to the boundary, ties to the lowest flat index (bounding-box computation, identical to the full volume)."""
    bb = bbox(comp, 1, comp.shape); c = comp[bb]
    d = ndimage.distance_transform_edt(np.pad(c, 1), sampling=sp)[1:-1, 1:-1, 1:-1]
    loc = np.unravel_index(np.argmax(np.where(c, d, -1)), c.shape)
    return tuple(int(l + b.start) for l, b in zip(loc, bb))


def case_geometry(root, case):
    """CT/GT geometry of a converted C4KC-KiTS case: (nifti, spacing, seg array, annotated-slice mask, their indices, lesion label volume on annotated slices, lesion list [(diameter, id)])."""
    import nibabel as nib
    from glassbox.audit.contrast import select_lesions
    nii = nib.load(os.path.join(root, case, "ct.nii.gz")); sp = tuple(float(z) for z in nii.header.get_zooms()[:3])
    seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann)
    lab, sel = select_lesions((seg == 2)[:, :, ann], sp)
    return nii, sp, seg, ann, zidx, lab, sel


def lesion_metrics(pred_a, comp_c, sp):
    """Metrics of one prediction (annotated slices) for one lesion, as in W2 v3: coverage, dice_full, volume ratio, inside share, largest overlapping predicted diameter, and the failure mode."""
    n_p, n_l = int(pred_a.sum()), int(comp_c.sum()); inter = int((pred_a & comp_c).sum()); pdiam = 0.0
    if n_p and inter:
        plab, _ = ndimage.label(pred_a, structure=np.ones((3, 3, 3))); ov = [i for i in np.unique(plab[comp_c]) if i > 0]
        pdiam = max((axial_diam(plab == i, sp[0], sp[1]) for i in ov), default=0.0)
    vol, inside, cov = n_p / n_l, inter / max(n_p, 1), inter / n_l
    return {"coverage": cov, "dice_full": 2 * inter / max(n_p + n_l, 1), "pred_over_lesion_volume": vol, "pred_frac_in_lesion": inside, "pred_diam_mm": pdiam, "mode": failure_mode(vol, inside, cov)}


def failure_mode(vol, inside, cov):
    """LEAK = predicted/lesion volume >= 3x or inside share <= 0.3; else UNDER = coverage < 0.5; else OK (W2 v3 definition, mutually exclusive in this order)."""
    if vol >= 3.0 or inside <= 0.3:
        return "LEAK"
    return "UNDER" if cov < 0.5 else "OK"


def eraser_apply_np(x, mu, d, a, c):
    """Rank-1 affine eraser x' = x - d (a.(x - mu)) / c on feature rows (numpy; the same formula as the network hook)."""
    return x - np.outer((x - mu) @ a, d) / c


class Runner:
    """Frozen VISTA-3D with the official point-window inferer. `set_eraser(None | (mu, d, a, c))` applies the affine eraser at the mid-encoder stage-2 output of every window;
    `record(True)` stores, per window, the avg-pooled 12 mm mid tokens (192-d) with their voxel centres in the preprocessed grid."""

    def __init__(self, inferer_dir, weights):
        import torch
        from glassbox.data.preprocess import build_transforms
        from glassbox.seg.infer import load_network
        sys.path.insert(0, inferer_dir)
        from inferer import Vista3dInferer
        self.torch = torch; self.dev = torch.device("cuda"); self.net = load_network(weights, self.dev)
        self.inferer = Vista3dInferer(roi_size=(128, 128, 128), overlap=0.3, use_point_window=True, sw_batch_size=1)
        self.pre = build_transforms("image"); self.eraser = None; self.recording = False; self.tokens = []; self._seen = []; self._coords = None; self._shape = None
        self.net.point_head.register_forward_pre_hook(self._point_hook, with_kwargs=True)
        self.net.register_forward_pre_hook(self._win_hook, with_kwargs=True)
        self.net.image_encoder.encoder.layers[2]["blocks"].register_forward_hook(self._mid_hook)

    def _point_hook(self, module, args, kwargs):
        pl = args[2] if len(args) > 2 else kwargs["point_labels"]; self._seen.append(sorted({float(v) for v in self.torch.unique(pl).tolist() if v != -1}))

    def _win_hook(self, module, args, kwargs):
        sl = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]; self._coords = tuple(s.start for s in sl[-3:])

    def _mid_hook(self, module, args, out):
        torch = self.torch
        if self.recording and self._coords is not None:
            import torch.nn.functional as F
            from glassbox.extract.hooks import window_pads
            tok = F.avg_pool3d(out.detach(), 2)[0]; g = tok.shape[-1]; before = window_pads(self._shape)[0]
            ii = torch.stack(torch.meshgrid(*[torch.arange(g)] * 3, indexing="ij"), -1)
            xyz = ii * MID_STRIDE + MID_STRIDE // 2 + torch.tensor(self._coords) - torch.tensor(before)
            self.tokens.append((tok.flatten(1).T.float().cpu().numpy(), xyz.reshape(-1, 3).numpy()))
        if self.eraser is None:
            return None
        mu, d, aa, c = self.eraser
        t = torch.einsum("bcxyz,c->bxyz", out - mu[None, :, None, None, None], aa) / c
        return out - d[None, :, None, None, None] * t[:, None]

    def set_eraser(self, params):
        torch = self.torch
        self.eraser = None if params is None else tuple(torch.as_tensor(np.asarray(p, dtype=np.float32), device=self.dev) if i < 3 else float(p) for i, p in enumerate(params))

    def record(self, on):
        self.recording = on; self.tokens = []

    def prep(self, path):
        data = self.pre({"image": path}); img = data["image"].unsqueeze(0).to(self.dev); self._shape = tuple(img.shape[2:])
        return data, img, np.linalg.inv(np.array(data["image"].affine, float)) @ np.array(data["image"].meta["original_affine"], float)

    def predict(self, data, img, M, label, point_label, point_vox):
        """Binary prediction on the ORIGINAL grid (prob > 0.5) for one label prompt and one positive point carrying `point_label` (original voxel indices)."""
        torch = self.torch
        from monai.data import MetaTensor
        from monai.transforms import Invertd
        p = M @ np.array([*point_vox, 1.0]); pts = torch.tensor(p[:3][None, None, :], dtype=torch.float32, device=self.dev)
        self._seen = []; self.tokens = [] if self.recording else self.tokens
        with torch.no_grad():
            logits = self.inferer(inputs=img, network=self.net, point_coords=pts, point_labels=torch.full((1, 1), float(point_label), device=self.dev), class_vector=torch.tensor([[label]], device=self.dev))
        seen = {v for s in self._seen for v in s}
        assert seen == {float(point_label)}, f"point label at the point head was {seen}, expected {point_label}"
        probs = torch.sigmoid(logits.reshape(1, *img.shape[2:]).float().cpu())
        pred = MetaTensor(probs, meta=data["image"].meta, applied_operations=data["image"].applied_operations)
        out = Invertd(keys="pred", transform=self.pre, orig_keys="image", nearest_interp=False, to_tensor=True)({"pred": pred, "image": data["image"]})
        return out["pred"].as_tensor()[0].numpy() > 0.5
