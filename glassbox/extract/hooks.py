"""Activation extraction for VISTA3D (bundle `vista3d132`).

Hook points (shapes per 128^3 window, C x D x H x W):
  bottleneck  image_encoder.encoder stage 5     768 x 8^3    token stride 16 vox (24 mm)
  mid         image_encoder.encoder stage 3     192 x 32^3 -> avg-pooled x2 -> 16^3, stride 8 vox (12 mm)
  decoder_F   class_head.image_post_mapping      48 x 128^3 -> avg-pooled x8 -> 16^3, stride 8 vox (12 mm)
The encoder is class-agnostic (trained on clicks, frozen for the auto head), so one forward pass serves every class.
Every token carries: voxel centre in the preprocessed (1.5 mm, RAS, cropped) volume, the majority ground-truth
label in its cell, mean HU in its cell, and the window it came from. Windows overlap (0.3), so a voxel can
appear in several windows; `patch` identifies the window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from glassbox.data.preprocess import HU_MAX, HU_MIN, PATCH_SIZE

# name -> (token stride in voxels, pool kernel applied to the raw feature map)
HOOKS = {"bottleneck": (16, 1), "mid": (8, 2), "decoder_F": (8, 8)}
AIR_HU = -800.0
AIR_KEEP = 0.10


def window_pads(spatial_shape):
    """Symmetric pad (before) that MONAI's sliding_window_inference applies when the image is smaller than the ROI."""
    return [max(PATCH_SIZE[k] - spatial_shape[k], 0) // 2 for k in range(3)], [
        max(PATCH_SIZE[k] - spatial_shape[k], 0) - max(PATCH_SIZE[k] - spatial_shape[k], 0) // 2 for k in range(3)
    ]


class ActivationRecorder:
    def __init__(self, network, label_volume, n_labels: int, seed: int = 0):
        """label_volume: LongTensor [H,W,D] in the preprocessed space, values 0..n_labels."""
        self.net = network
        self.n_labels = n_labels
        self.shape = tuple(label_volume.shape)
        before, after = window_pads(self.shape)
        self.before = before
        pad = [p for k in (2, 1, 0) for p in (before[k], after[k])]
        # -1 marks voxels outside the real image (replicate-padding region)
        self.label = F.pad(label_volume.long() + 1, pad, value=0)  # 0 = outside, 1 = background, 2.. = classes
        self.rng = np.random.default_rng(seed)
        self.coords: list = []  # window slices only (cheap); activations are reduced to tokens as each window finishes
        self._cur: dict = {}
        self._out = {k: {"x": [], "xyz": [], "label": [], "hu": [], "patch": []} for k in HOOKS}
        self._handles = [
            network.register_forward_pre_hook(self._pre, with_kwargs=True),
            network.image_encoder.encoder.register_forward_hook(self._enc),
            network.class_head.image_post_mapping.register_forward_hook(self._f),
        ]

    def _pre(self, module, args, kwargs):
        slices = args[1][0] if len(args) > 1 else kwargs["patch_coords"][0]
        self.coords.append(tuple(slices[-3:]))
        self._cur = {"win": args[0].detach()}
        return None

    def _enc(self, module, args, out):
        self._cur["mid"], self._cur["bottleneck"] = out[2].detach(), out[4].detach()

    def _f(self, module, args, out):
        # The decoder output F is the last hook to fire in a window: reduce everything now, keep only CPU tokens.
        self._cur["decoder_F"] = out.detach()
        self._ingest(len(self.coords) - 1)
        self._cur = {}

    def close(self):
        for h in self._handles:
            h.remove()

    def _ingest(self, p: int) -> None:
        sl, win = self.coords[p], self._cur["win"]
        starts = [s.start for s in sl]
        lab = self.label[tuple(slice(s.start, s.stop) for s in sl)]  # [128,128,128]
        onehot = F.one_hot(lab.to(win.device), self.n_labels + 2).permute(3, 0, 1, 2).float()[None]
        maps = {k: self._cur[k] for k in HOOKS}
        for name, (stride, pool) in HOOKS.items():
            x = maps[name]
            if pool > 1:
                x = F.avg_pool3d(x, pool)
            g = x.shape[-1]
            assert g * stride == PATCH_SIZE[0], (name, tuple(x.shape))
            maj = F.avg_pool3d(onehot, stride)[0].argmax(0)  # [g,g,g]  0 = outside
            hu = F.avg_pool3d(win, stride)[0, 0] * (HU_MAX - HU_MIN) + HU_MIN
            ii = torch.stack(torch.meshgrid(*[torch.arange(g)] * 3, indexing="ij"), -1).to(x.device)
            centre = ii * stride + stride // 2
            xyz = centre + torch.tensor(starts, device=x.device) - torch.tensor(self.before, device=x.device)
            keep = (maj > 0).flatten()
            air = (hu.flatten() < AIR_HU) & keep
            if air.any():
                drop = torch.from_numpy(self.rng.random(int(air.sum())) > AIR_KEEP).to(x.device)
                idx = air.nonzero()[:, 0]
                keep[idx[drop]] = False
            feats = x[0].flatten(1).T  # [g^3, C]
            o = self._out[name]
            o["x"].append(feats[keep].half().cpu())
            o["xyz"].append(xyz.reshape(-1, 3)[keep].short().cpu())
            o["label"].append((maj.flatten()[keep] - 1).to(torch.int8).cpu())  # 0 = background, 1.. = classes
            o["hu"].append(hu.flatten()[keep].half().cpu())
            o["patch"].append(torch.full((int(keep.sum()),), p, dtype=torch.int16))

    def collect(self) -> dict:
        """Token tables for the whole scan. Call after inference, then close()."""
        assert len(self.coords) > 0, "no windows were recorded"
        return {
            f"{name}_{field}": torch.cat(v).numpy() for name, d in self._out.items() for field, v in d.items()
        }
