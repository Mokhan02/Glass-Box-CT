"""W6: the W5 tumour-contrast augmentation ported to VISTA-3D's training pipeline (results/w6_vista_finetune/PREREG.md, section 5).
`shift_tumour_contrast_hu` is the W5 function (scripts/w5_nnunet_trainer.py, `shift_tumour_contrast`) on an HU patch: with probability p PER PATCH (one draw), every tumour component of at least MIN_COMPONENT_VOXELS voxels is shifted by an additive, Gaussian-feathered offset so that its mean HU equals the mean of the local liver plus
delta, delta ~ U(arm range); the local liver = liver voxels (all tumour voxels excluded) within SHELL_VOXELS voxels of the component, else the patch's liver (at least MIN_LIVER_VOXELS), else the component is left unchanged. Differences from W5, all physical-unit adaptations: SHELL_VOXELS 8 (12 mm at 1.5 mm, W5: 12 voxels of about 1 mm)
and the clip window is VISTA's (-963.8 to 1053.7 HU, wide) instead of nnU-Net's (-15 to 197 HU). `BrightTumourShiftd` is the MONAI dictionary transform: applied right after the random crop, on the scaled patch, converting back to HU and rescaling; the labels are local ones (1 liver, 2 tumour) and are never modified.
Pure numpy / scipy (torch only in the dictionary transform)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

A_MIN, A_MAX = -963.8247715525971, 1053.678477684517      # NV-Segment-CT `ScaleIntensityRanged` window (configs/train.json), scaled to [0, 1]
LIVER, TUMOUR = 1, 2
MIN_COMPONENT_VOXELS = 30
SHELL_VOXELS = 8              # 12 mm at 1.5 mm isotropic
MIN_LIVER_VOXELS = 200
FEATHER_SIGMA = 1.0
ARMS = {"BRIGHT": (10.0, 60.0), "SHAM": (-60.0, -10.0)}      # delta (HU) of the tumour mean relative to the local liver mean, uniform


def _local_liver_mean(hu, liver, comp, any_tumour, shell=SHELL_VOXELS):
    sl = ndi.find_objects(comp.astype(np.uint8))[0]
    lo = [max(0, s.start - shell) for s in sl]; hi = [min(n, s.stop + shell) for s, n in zip(sl, comp.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    near = ndi.binary_dilation(comp[box], iterations=shell) & liver[box] & ~any_tumour[box]
    if near.sum() >= MIN_LIVER_VOXELS:
        return float(hu[box][near].mean()), "local"
    rest = liver & ~any_tumour
    if rest.sum() >= MIN_LIVER_VOXELS:
        return float(hu[rest].mean()), "patch"
    return None, "none"


def shift_tumour_contrast_hu(hu, seg, arm, rng, p, shell=SHELL_VOXELS, a_min=A_MIN, a_max=A_MAX):
    """hu: float array (X, Y, Z) in HU; seg: int array (X, Y, Z), 1 liver, 2 tumour (anything else ignored). Returns (new_hu, info); `rng` is a numpy Generator; the first draw is the per-patch Bernoulli, as in W5."""
    info = {"applied": False, "components": []}
    if p <= 0 or rng.random() >= p:
        return hu, info
    lo, hi = ARMS[arm]; tum = seg == TUMOUR; liver = seg == LIVER
    if not tum.any():
        return hu, info
    lab, n = ndi.label(tum, structure=np.ones((3, 3, 3), int)); out = hu.astype(np.float64).copy(); h = hu.astype(np.float64); applied = False
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() < MIN_COMPONENT_VOXELS:
            continue
        liv, src = _local_liver_mean(h, liver, comp, tum, shell)
        if liv is None:
            info["components"].append({"component": k, "skipped": "no liver"}); continue
        delta = float(rng.uniform(lo, hi)); target = liv + delta; w = ndi.gaussian_filter(comp.astype(np.float64), FEATHER_SIGMA)
        d = (target - h[comp].mean()) / w[comp].mean()
        out += d * w; applied = True
        info["components"].append({"component": k, "voxels": int(comp.sum()), "liver_mean": liv, "liver_source": src, "delta": delta, "target": target, "offset": float(d)})
    info["applied"] = applied
    return np.clip(out, a_min, a_max).astype(hu.dtype), info


try:
    import torch
    from monai.transforms import MapTransform, Randomizable
except ImportError:      # unit tests of the pure function do not need torch / MONAI
    MapTransform = None

if MapTransform is not None:
    class BrightTumourShiftd(Randomizable, MapTransform):
        """MONAI dictionary transform: image (1, X, Y, Z) scaled to [0, 1] by `ScaleIntensityRanged`, label (1, X, Y, Z) local labels (1 liver, 2 tumour)."""

        def __init__(self, keys=("image", "label"), arm="BRIGHT", p=0.25, shell=SHELL_VOXELS, a_min=A_MIN, a_max=A_MAX, seed_offset=0):
            MapTransform.__init__(self, keys); self.arm, self.p, self.shell, self.a_min, self.a_max = arm, p, shell, a_min, a_max
            if arm not in ARMS:
                raise ValueError(arm)

        def __call__(self, data):
            d = dict(data); img = d[self.keys[0]]; lab = d[self.keys[1]]
            x = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img); s = lab.detach().cpu().numpy() if hasattr(lab, "detach") else np.asarray(lab)
            hu = x[0].astype(np.float64) * (self.a_max - self.a_min) + self.a_min
            rng = np.random.default_rng(int(self.R.randint(0, 2**31 - 1)))      # Randomizable.R: seeded per worker by MONAI's set_determinism / worker init
            new, info = shift_tumour_contrast_hu(hu, s[0], self.arm, rng, self.p, self.shell, self.a_min, self.a_max); d["bright_applied"] = int(info["applied"])
            out = ((new - self.a_min) / (self.a_max - self.a_min)).astype(x.dtype)[None]
            d[self.keys[0]] = type(img)(out) if hasattr(img, "detach") and not hasattr(img, "meta") else (img.new_tensor(out) if hasattr(img, "new_tensor") else out)
            return d
