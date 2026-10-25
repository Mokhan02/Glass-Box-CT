"""Unit tests of the W5 tumour-contrast augmentation (synthetic volume; no nnU-Net or GPU needed)."""
import importlib.util
import os

import numpy as np
import pytest
from scipy import ndimage as ndi

spec = importlib.util.spec_from_file_location("w5", os.path.join(os.path.dirname(__file__), "..", "scripts", "w5_nnunet_trainer.py")); W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)


def volume(seed=0, two=False):
    rng = np.random.default_rng(seed); n = 64; g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"))
    seg = np.zeros((n, n, n), np.int16); seg[(g[0] - 32) ** 2 + (g[1] - 32) ** 2 + (g[2] - 32) ** 2 < 28 ** 2] = W.LIVER
    c1 = (g[0] - 22) ** 2 + (g[1] - 32) ** 2 + (g[2] - 32) ** 2 < 5 ** 2; seg[c1] = W.TUMOUR
    if two:
        c2 = (g[0] - 42) ** 2 + (g[1] - 32) ** 2 + (g[2] - 32) ** 2 < 5 ** 2; seg[c2] = W.TUMOUR
    hu = np.where(seg == W.TUMOUR, 60.0, np.where(seg == W.LIVER, 100.0, 20.0)) + rng.normal(0, 6, seg.shape)
    hu = np.clip(hu, W.HU_LO, W.HU_HI)
    return ((hu - W.HU_MEAN) / W.HU_STD).astype(np.float32)[None], seg[None], hu


def tum_mean(img, seg):
    return float((img[0][seg[0] == W.TUMOUR] * W.HU_STD + W.HU_MEAN).mean())


def test_p_zero_is_identity():
    img, seg, _ = volume(); out, info = W.shift_tumour_contrast(img, seg, "BRIGHT", np.random.default_rng(0), 0.0)
    assert out is img and not info["applied"]


@pytest.mark.parametrize("arm,lo,hi", [("BRIGHT", 10, 60), ("SHAM", -60, -10)])
def test_tumour_mean_lands_at_target_and_sign(arm, lo, hi):
    for s in range(15):
        img, seg, _ = volume(s); seg0 = seg.copy(); out, info = W.shift_tumour_contrast(img, seg, arm, np.random.default_rng(s), 1.0)
        c = info["components"][0]; assert info["applied"] and lo <= c["delta"] <= hi
        assert abs(tum_mean(out, seg) - c["target"]) < 0.5            # exact up to float32 and (here absent) clipping
        assert (tum_mean(out, seg) - c["liver_mean"]) * np.sign(lo) > 0   # bright: above liver; sham: below liver
        assert np.array_equal(seg, seg0)                               # segmentation untouched
    # the contrast of the sham stays dark and the bright becomes bright whatever the starting contrast (start 70 vs liver ~110)


def test_texture_preserved_and_locality_and_window():
    img, seg, hu = volume(3); out, info = W.shift_tumour_contrast(img, seg, "BRIGHT", np.random.default_rng(3), 1.0)
    new = out[0] * W.HU_STD + W.HU_MEAN; diff = (new - hu); tum = seg[0] == W.TUMOUR
    inner = ndi.binary_erosion(tum, iterations=3)
    assert diff[inner].std() < 1.0                                     # additive offset: interior texture is unchanged (only the feathered border weight varies)
    assert np.corrcoef(new[inner], hu[inner])[0, 1] > 0.95
    g = np.stack(np.meshgrid(*[np.arange(64)] * 3, indexing="ij")); far = ((g[0] - 22) ** 2 + (g[1] - 32) ** 2 + (g[2] - 32) ** 2) > 13 ** 2
    assert np.allclose(diff[far], 0, atol=1e-3)                        # feathering is local
    assert new.min() >= W.HU_LO - 1e-3 and new.max() <= W.HU_HI + 1e-3


def test_clip_to_window():
    img, seg, hu = volume(5); hu2 = np.where(seg[0] == W.TUMOUR, 190.0, hu); img2 = ((hu2 - W.HU_MEAN) / W.HU_STD).astype(np.float32)[None]
    out, _ = W.shift_tumour_contrast(img2, seg, "BRIGHT", np.random.default_rng(1), 1.0); new = out[0] * W.HU_STD + W.HU_MEAN
    assert new.max() <= W.HU_HI + 1e-3


def test_components_independent_and_small_or_no_liver_untouched():
    img, seg, _ = volume(2, two=True); out, info = W.shift_tumour_contrast(img, seg, "BRIGHT", np.random.default_rng(2), 1.0)
    d = [c["delta"] for c in info["components"]]; assert len(d) == 2 and d[0] != d[1]
    seg2 = seg.copy(); seg2[0, :, :, :] = np.where(seg2[0] == W.LIVER, 0, seg2[0])        # no liver in the patch
    out2, info2 = W.shift_tumour_contrast(img, seg2, "BRIGHT", np.random.default_rng(2), 1.0); assert not info2["applied"] and np.array_equal(out2, img)
    seg3 = seg.copy(); seg3[0][seg3[0] == W.TUMOUR] = W.LIVER; seg3[0, 30:32, 30:32, 30:32] = W.TUMOUR    # 8 voxels < MIN_COMPONENT_VOXELS
    out3, info3 = W.shift_tumour_contrast(img, seg3, "BRIGHT", np.random.default_rng(2), 1.0); assert not info3["applied"] and np.array_equal(out3, img)


def test_application_probability():
    img, seg, _ = volume(0); rng = np.random.default_rng(0); k = sum(W.shift_tumour_contrast(img, seg, "BRIGHT", rng, 0.5)[1]["applied"] for _ in range(300))
    assert 120 < k < 180


def test_arm_ranges_fixed_from_physiology():
    assert W.ARMS == {"BRIGHT": (10.0, 60.0), "SHAM": (-60.0, -10.0)}


def test_seeds_distinct_and_fixed():
    assert W.SEED == 20261020 and W.SEED2 == 20261121 and W.SEED != W.SEED2
