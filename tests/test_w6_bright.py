"""W6 augmentation tests (PREREG section 5: phantom tests (i) to (vii))."""
import importlib.util
import os

import numpy as np
import pytest

from glassbox.seg import w6_bright as B

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def phantom(shape=(60, 60, 60), liver_hu=100.0, tum_hu=55.0, seed=0):
    rng = np.random.default_rng(seed); hu = np.full(shape, -800.0); seg = np.zeros(shape, np.int8)
    seg[5:55, 5:55, 5:55] = B.LIVER; hu[5:55, 5:55, 5:55] = liver_hu + rng.normal(0, 3, (50, 50, 50))
    for (x, y, z, r) in ((20, 20, 30, 5), (40, 38, 30, 4)):      # two tumour components
        g = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")); m = ((g[0] - x) ** 2 + (g[1] - y) ** 2 + (g[2] - z) ** 2) <= r * r
        seg[m] = B.TUMOUR; hu[m] = tum_hu + rng.normal(0, 3, m.sum())
    return hu, seg


def comps(seg):
    from scipy import ndimage as ndi
    lab, n = ndi.label(seg == B.TUMOUR, structure=np.ones((3, 3, 3), int)); return [lab == k for k in range(1, n + 1)]


def test_mean_lands_at_local_liver_plus_delta_and_the_outside_is_unchanged():
    hu, seg = phantom(); new, info = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(1), 1.0)
    assert info["applied"] and len(info["components"]) == 2
    for c, ci in zip(comps(seg), info["components"]):
        assert abs(new[c].mean() - ci["target"]) < 2.0 and 10 <= ci["delta"] <= 60          # (i) and (iii)
        assert new[c].mean() - ci["liver_mean"] > 8
    from scipy import ndimage as ndi
    far = ndi.distance_transform_edt(seg != B.TUMOUR) > 8; assert np.array_equal(new[far], hu[far])      # (ii) the Gaussian kernel (truncated at 4 sigma per axis) reaches at most 4*sqrt(3) = 7 voxels; the liver and the air are untouched
    assert np.array_equal(seg, phantom()[1])      # labels are never touched


def test_sham_darkens_and_bright_brightens():
    hu, seg = phantom(tum_hu=70.0)
    for arm, sign in (("BRIGHT", 1), ("SHAM", -1)):
        new, info = B.shift_tumour_contrast_hu(hu, seg, arm, np.random.default_rng(2), 1.0)
        for c, ci in zip(comps(seg), info["components"]):
            assert sign * (ci["target"] - ci["liver_mean"]) >= 9.9 and sign * (new[c].mean() - ci["liver_mean"]) > 5       # (iii)


def test_clipping_dtype_and_determinism():
    hu, seg = phantom(liver_hu=1040.0, tum_hu=1000.0); hu = hu.astype(np.float32)
    n1, _ = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(3), 1.0); n2, _ = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(3), 1.0)
    assert n1.dtype == np.float32 and n1.max() <= B.A_MAX and n1.min() >= B.A_MIN and np.array_equal(n1, n2)      # (iv) and (v)
    n3, _ = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(4), 1.0); assert not np.array_equal(n1, n3)


def test_edit_frequency_is_per_patch_and_small_components_are_skipped():
    hu, seg = phantom(); rng = np.random.default_rng(5); k = 0; both = True
    for _ in range(2000):
        _, info = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", rng, 0.25); k += info["applied"]
        if info["applied"]:
            both &= len(info["components"]) == 2      # (vi): an edited patch edits ALL its qualifying components
    assert abs(k / 2000 - 0.25) < 0.03 and both
    seg2 = seg.copy(); seg2[comps(seg)[1]] = B.LIVER; seg2[10, 10, 10] = B.TUMOUR      # one 1-voxel "tumour" below the 30-voxel floor
    _, info = B.shift_tumour_contrast_hu(hu, seg2, "BRIGHT", np.random.default_rng(6), 1.0); assert len(info["components"]) == 1 and info["components"][0]["voxels"] > 30
    assert B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(7), 0.0)[0] is hu      # p = 0 is the identity


def test_parity_with_the_w5_function():
    spec = importlib.util.spec_from_file_location("w5", os.path.join(HERE, "scripts", "w5_nnunet_trainer.py")); W5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(W5)
    W5.SHELL_VOXELS = B.SHELL_VOXELS      # the one documented physical-unit difference, made equal for the parity check
    hu, seg = phantom(liver_hu=100.0, tum_hu=80.0)      # inside W5's CT window (-15 to 197 HU), so its clip is inactive
    norm = ((hu - W5.HU_MEAN) / W5.HU_STD).astype(np.float64)[None]; sg = seg[None]
    out5, i5 = W5.shift_tumour_contrast(norm, sg, "BRIGHT", np.random.default_rng(11), 1.0); out6, i6 = B.shift_tumour_contrast_hu(hu, seg, "BRIGHT", np.random.default_rng(11), 1.0)
    hu5 = out5[0] * W5.HU_STD + W5.HU_MEAN
    inside = (seg > 0) & (hu < 190)
    assert [round(c["delta"], 9) for c in i5["components"]] == [round(c["delta"], 9) for c in i6["components"]]
    assert np.allclose(hu5[inside], out6[inside], atol=1e-3)      # the same HU edit (the CT window clip of W5 does not act here)


@pytest.mark.skipif(B.MapTransform is None, reason="MONAI/torch not installed")
def test_dictionary_transform_roundtrip_on_the_scaled_patch():
    import torch
    hu, seg = phantom(tum_hu=70.0); scaled = ((np.clip(hu, B.A_MIN, B.A_MAX) - B.A_MIN) / (B.A_MAX - B.A_MIN)).astype(np.float32)
    t = B.BrightTumourShiftd(arm="BRIGHT", p=1.0, scaled=True); t.set_random_state(seed=3)
    d = t({"image": torch.from_numpy(scaled[None]), "label": torch.from_numpy(seg[None].astype(np.int64))})
    new_hu = d["image"].numpy()[0].astype(np.float64) * (B.A_MAX - B.A_MIN) + B.A_MIN
    assert d["image"].shape == (1, 60, 60, 60) and d["image"].dtype == torch.float32 and torch.equal(d["label"], torch.from_numpy(seg[None].astype(np.int64)))
    for c in comps(seg):
        assert new_hu[c].mean() - hu[c].mean() > 5      # the tumours are brighter after the edit (scaled-patch round trip)


@pytest.mark.skipif(B.MapTransform is None, reason="MONAI/torch not installed")
def test_dictionary_transform_in_hu_mode_does_not_clip_and_leaves_the_clip_to_the_bundle():
    import torch
    hu, seg = phantom(liver_hu=1040.0, tum_hu=1000.0); t = B.BrightTumourShiftd(arm="BRIGHT", p=1.0); t.set_random_state(seed=4)
    d = t({"image": torch.from_numpy(hu.astype(np.float32)[None]), "label": torch.from_numpy(seg[None].astype(np.int64))}); new = d["image"].numpy()[0]
    assert d["bright_applied"] == 1 and new.max() > B.A_MAX      # an edited value above the bundle window is NOT clipped here: ScaleIntensityRanged(clip=True) does that next
    from monai.transforms import ScaleIntensityRanged
    sc = ScaleIntensityRanged(keys="image", a_min=B.A_MIN, a_max=B.A_MAX, b_min=0.0, b_max=1.0, clip=True)(d)["image"]
    assert float(sc.max()) <= 1.0 and float(sc.min()) >= 0.0


def test_parity_gate_pure_parts_choose_seeded_nontrivial_cases_and_compare():
    spec = importlib.util.spec_from_file_location("pg", os.path.join(HERE, "scripts", "w6_parity_gate.py")); PG = importlib.util.module_from_spec(spec); spec.loader.exec_module(PG)
    cands = [("half0", f"{i}_1", 500 if i % 3 == 0 else 5000 + i) for i in range(1, 40)]; a = PG.choose_cases(cands, 3); assert a == PG.choose_cases(cands, 3) and len(a) == 3 and all(c[2] >= 1000 for c in a)
    m = np.zeros((20, 20, 20), bool); m[5:10, 5:10, 5:10] = True; r = PG.compare(m, m); assert r["byte_identical"] and r["pass"] and r["dice"] == 1.0
    m2 = m.copy(); m2[5, 5, 5] = False; r2 = PG.compare(m2, m); assert not r2["byte_identical"] and r2["pass"] and r2["dice"] > 0.99      # one voxel of 125: Dice 0.996
    m3 = np.zeros_like(m); m3[5:10, 5:10, 5:7] = True; assert not PG.compare(m3, m)["pass"]      # a real difference fails
    assert PG.dice(np.zeros(5), np.zeros(5)) == 1.0


def test_liver_rule_pooled_fraction_and_decision():
    spec = importlib.util.spec_from_file_location("lr", os.path.join(HERE, "scripts", "w6_liver_rule.py")); LR = importlib.util.module_from_spec(spec); spec.loader.exec_module(LR)
    t = np.zeros((10, 10, 10), bool); t[2:6, 2:6, 2:6] = True; full = np.ones_like(t); none = np.zeros_like(t); half = np.zeros_like(t); half[2:4, 2:6, 2:6] = True
    f, per = LR.pooled_fraction([(full, t), (none, t), (half, np.zeros_like(t))]); assert abs(f - 0.5) < 1e-9 and len(per) == 2      # a case without tumour voxels is skipped
    assert LR.decide(0.5) == "[[2,26]]" and LR.decide(0.49) == "[[1,1],[2,26]]"
