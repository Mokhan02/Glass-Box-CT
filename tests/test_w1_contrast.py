import numpy as np

from glassbox.audit.contrast import case_lesions, select_lesions


def _phantom(lesion_hu, liver_hu=100.0, shape=(80, 80, 60), sp=(1.0, 1.0, 1.0), r=8):
    """r is the lesion radius in mm (sphere in physical space); the liver is a block around it."""
    ct = np.full(shape, -1000.0); liver = np.zeros(shape, bool)
    liver[8:-8, 8:-8, 3:-3] = True; ct[liver] = liver_hu
    g = np.indices(shape).astype(float); c = [(n - 1) / 2 for n in shape]
    lesion = sum(((g[i] - c[i]) * sp[i]) ** 2 for i in range(3)) <= r * r
    ct[lesion] = lesion_hu
    return ct, lesion, liver | lesion, sp   # the liver label may or may not include the lesion; the function removes it either way


def test_bright_lesion_has_positive_contrast():
    ct, les, liv, sp = _phantom(130.0)
    (row,) = case_lesions(ct, les, liv, sp)
    assert abs(row["contrast_hu"] - 30.0) < 1e-6 and not row["fallback"] and abs(row["diam_mm"] - 16) < 1.5


def test_dark_lesion_has_negative_contrast():
    ct, les, liv, sp = _phantom(60.0)
    (row,) = case_lesions(ct, les, liv, sp)
    assert abs(row["contrast_hu"] + 40.0) < 1e-6


def test_liver_label_excluding_lesion_gives_same_answer():
    ct, les, liv, sp = _phantom(130.0)
    (a,) = case_lesions(ct, les, liv, sp); (b,) = case_lesions(ct, les, liv & ~les, sp)
    assert a["contrast_hu"] == b["contrast_hu"]


def test_shell_excludes_the_inner_5mm_and_anisotropic_spacing_is_respected():
    ct, les, liv, sp = _phantom(130.0, shape=(200, 200, 50), sp=(0.5, 0.5, 2.0), r=8)
    ct[(~les) & liv & (ndist(les, sp) < 5)] = 1000.0      # corrupt everything within 5 mm: must not enter L
    (row,) = case_lesions(ct, les, liv, sp)
    assert abs(row["liver_hu"] - 100.0) < 1e-6


def ndist(les, sp):
    from scipy import ndimage
    return ndimage.distance_transform_edt(~les, sampling=sp)


def test_tiny_lesion_below_10mm_is_not_selected():
    ct, les, liv, sp = _phantom(130.0, r=4)
    assert case_lesions(ct, les, liv, sp) == []


def test_liver_mask_all_tumour_gives_nan_not_a_number():
    ct, les, liv, sp = _phantom(130.0)
    (row,) = case_lesions(ct, les, les.copy(), sp)       # liver label is just the tumour
    assert np.isnan(row["contrast_hu"]) and row["n_liver_vox"] == 0


def test_fallback_when_shell_is_small():
    ct, les, liv, sp = _phantom(130.0, r=8)
    liv = np.zeros_like(liv); liv[36:44, 36:44, 28:32] = True; liv |= les     # tiny liver sliver: shell < 200 voxels
    (row,) = case_lesions(ct, les, liv, sp)
    assert row["fallback"]


def test_two_largest_per_case_and_all_lesions_option():
    ct, les, liv, sp = _phantom(130.0, r=8)
    g = np.indices(ct.shape); small = ((g[0] - 20) ** 2 + (g[1] - 20) ** 2 + (g[2] - 30) ** 2) <= 36; third = ((g[0] - 60) ** 2 + (g[1] - 60) ** 2 + (g[2] - 30) ** 2) <= 25
    t = les | small | third; ct[t] = 130.0
    assert len(case_lesions(ct, t, liv | t, sp)) == 2
    assert len(case_lesions(ct, t, liv | t, sp, n_lesions=None)) == 3
    lab, sel = select_lesions(t, sp); assert sel[0][0] > sel[1][0]
