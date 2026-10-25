"""W5 trainers: model D's recipe (nnU-Net v2 `nnUNetTrainer`, default plans, fold 0, 1000 epochs, seed 20261020) with ONE change, an online tumour-contrast augmentation applied first in the training transform chain.

Installed by copying this file to `<venv_nnunet>/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/nnUNetTrainerW5.py`; nnU-Net finds trainers by class name inside that package, so the three trainer classes below are
`nnUNetTrainerW5Bright50`, `nnUNetTrainerW5Sham50`, `nnUNetTrainerW5Bright25` and `nnUNetTrainerW5D0Seed2` (model D's recipe with no augmentation and a different fixed seed, used only as a training-variance estimate). The file's hash is recorded before any run (results/w5_training/code_hashes.txt).

The augmentation (`shift_tumour_contrast`) works on the PREPROCESSED patch: nnU-Net's CT normalisation has already clipped to [-15, 197] HU (0.5 / 99.5 percentiles of the training foreground) and z-scored with mean 99.48, std 37.14. The
function converts back to HU, edits, clips to the same window and converts back, so the model sees values in the same window it sees at inference. Ranges are fixed from physiology (see PREREG), not from the HCC cohort.
Seeding does not make training bitwise deterministic (same caveat as W3 model D)."""
import random

import numpy as np
from scipy import ndimage as ndi

# nnU-Net CTNormalization constants of the W3 plan (Dataset003_Liver, nnUNetPlans.json); asserted against the plans at trainer start.
HU_MEAN, HU_STD, HU_LO, HU_HI = 99.48007202148438, 37.13840103149414, -15.0, 197.0
LIVER, TUMOUR = 1, 2
MIN_COMPONENT_VOXELS = 30     # smaller tumour components are left unchanged
SHELL_VOXELS = 12             # local liver = liver voxels within this many voxels of the component (about 9 to 12 mm)
MIN_LIVER_VOXELS = 200        # fewer local liver voxels -> fall back to all liver in the patch; fewer still -> component unchanged
FEATHER_SIGMA = 1.0           # voxels; Gaussian feathering of the lesion mask
SEED = 20261020
SEED2 = 20261121    # D0-seed2 only; chosen at the PREREG approval, before any W5 run
ARMS = {"BRIGHT": (10.0, 60.0), "SHAM": (-60.0, -10.0)}   # delta (HU) of the tumour mean relative to the local liver mean, uniform


def _local_liver_mean(hu, liver, comp, any_tumour):
    """Mean HU of liver voxels near `comp` (within SHELL_VOXELS, excluding every tumour voxel); fall back to the whole patch's liver; None if too few."""
    sl = ndi.find_objects(comp.astype(np.uint8))[0]
    lo = [max(0, s.start - SHELL_VOXELS) for s in sl]; hi = [min(n, s.stop + SHELL_VOXELS) for s, n in zip(sl, comp.shape)]
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    near = ndi.binary_dilation(comp[box], iterations=SHELL_VOXELS) & liver[box] & ~any_tumour[box]
    if near.sum() >= MIN_LIVER_VOXELS:
        return float(hu[box][near].mean()), "local"
    rest = liver & ~any_tumour
    if rest.sum() >= MIN_LIVER_VOXELS:
        return float(hu[rest].mean()), "patch"
    return None, "none"


def shift_tumour_contrast(image, seg, arm, rng, p):
    """Additive, feathered shift of each tumour component so its mean HU = local liver mean + delta, delta ~ U(arm range); applied to a sample with probability `p`.

    image: float array (1, X, Y, Z), nnU-Net-normalised; seg: int array (1, X, Y, Z) with 1 = liver, 2 = tumour (anything else, incl. -1 padding, is ignored).
    Returns (new_image, info) with the segmentation untouched (it is never modified). Texture is preserved (offset only); the result is clipped to the CT window.
    """
    info = {"applied": False, "components": []}
    if p <= 0 or rng.random() >= p:
        return image, info
    lo, hi = ARMS[arm]
    s = seg[0]; hu = image[0].astype(np.float64) * HU_STD + HU_MEAN
    tum = s == TUMOUR; liver = s == LIVER
    if not tum.any():
        return image, info
    lab, n = ndi.label(tum)
    out = hu.copy(); applied = False
    for k in range(1, n + 1):
        comp = lab == k
        if comp.sum() < MIN_COMPONENT_VOXELS:
            continue
        liv, src = _local_liver_mean(hu, liver, comp, tum)
        if liv is None:
            info["components"].append({"component": k, "skipped": "no liver"}); continue
        delta = float(rng.uniform(lo, hi)); target = liv + delta
        w = ndi.gaussian_filter(comp.astype(np.float64), FEATHER_SIGMA)
        d = (target - hu[comp].mean()) / w[comp].mean()      # so the mean over the component of (hu + d*w) equals target (before clipping)
        out += d * w
        applied = True
        info["components"].append({"component": k, "voxels": int(comp.sum()), "liver_mean": liv, "liver_source": src, "delta": delta, "target": target, "offset": float(d)})
    info["applied"] = applied
    out = np.clip(out, HU_LO, HU_HI)
    return ((out - HU_MEAN) / HU_STD).astype(image.dtype)[None], info


try:
    import torch
    from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
    from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
except ImportError:      # unit tests of the pure function run without nnU-Net / torch
    nnUNetTrainer = None

if nnUNetTrainer is not None:
    class TumourContrastTransform(BasicTransform):
        """Prepended to nnU-Net's training transforms (acts on the un-augmented patch with clean labels, so the later spatial and intensity augmentations transform image and label together)."""

        def __init__(self, arm, p):
            super().__init__(); self.arm, self.p = arm, p

        def apply(self, data_dict, **params):
            if data_dict.get("image") is not None and data_dict.get("segmentation") is not None:
                img = data_dict["image"]; rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))   # torch's per-worker stream seeds numpy
                new, _ = shift_tumour_contrast(img.numpy(), data_dict["segmentation"].numpy(), self.arm, rng, self.p)
                data_dict["image"] = torch.from_numpy(new)
            return data_dict

    class _W5Base(nnUNetTrainer):
        ARM, P, SEED = None, 0.0, SEED

        def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
            random.seed(self.SEED); np.random.seed(self.SEED); torch.manual_seed(self.SEED); torch.cuda.manual_seed_all(self.SEED)
            super().__init__(plans, configuration, fold, dataset_json, device)
            assert self.num_epochs == 1000, "W5 schedule is 1000 epochs (as W3 model D)"
            fp = plans["foreground_intensity_properties_per_channel"]["0"]
            assert abs(fp["mean"] - HU_MEAN) < 1e-6 and abs(fp["std"] - HU_STD) < 1e-6 and fp["percentile_00_5"] == HU_LO and fp["percentile_99_5"] == HU_HI, "plan CT normalisation differs from the constants in this file"
            self.print_to_log_file(f"W5: seed {self.SEED}; augmentation {self.ARM} p={self.P}, delta range {ARMS.get(self.ARM)} HU relative to local liver mean; epochs {self.num_epochs}")

        def get_training_transforms(self, *args, **kwargs):      # instance method override of the base staticmethod (called as self.get_training_transforms(...))
            base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
            if self.ARM is None:
                return base
            return ComposeTransforms([TumourContrastTransform(self.ARM, self.P), base])

    class nnUNetTrainerW5Bright50(_W5Base):
        ARM, P = "BRIGHT", 0.5

    class nnUNetTrainerW5Sham50(_W5Base):
        ARM, P = "SHAM", 0.5

    class nnUNetTrainerW5Bright25(_W5Base):
        ARM, P = "BRIGHT", 0.25

    class nnUNetTrainerW5D0Seed2(_W5Base):
        ARM, P, SEED = None, 0.0, SEED2      # no augmentation: model D's recipe with seed 20261121
