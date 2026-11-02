import csv
import importlib.util
import os

import pytest

sitk = pytest.importorskip("SimpleITK")
np = pytest.importorskip("numpy")
HERE = os.path.dirname(os.path.dirname(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "scripts", name + ".py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def write_meta(path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "ct_phase", "ct_file_name", "tumor_count", "slice_thickness"])
        for p in range(1, 41):
            for ph in (1, 2):
                w.writerow([p, ph, f"{p}_{ph}", 1 if (ph == 1 and p <= 20) or (ph == 2 and p > 20) else 0, 1.0])


def test_registration_sample_is_deterministic_and_stratified(tmp_path):
    R = load("w4_waw_register"); write_meta(tmp_path / "m.csv")
    a, b = R.pick_patients(str(tmp_path / "m.csv"), ""), R.pick_patients(str(tmp_path / "m.csv"), "")
    assert a == b and len(a) == 5 and [ph for _, ph in a] == [1, 1, 1, 2, 2] and len({p for p, _ in a}) == 5
    assert all(int(p) <= 20 for p, ph in a if ph == 1) and all(int(p) > 20 for p, ph in a if ph == 2)


def test_dice_and_centroid_helpers():
    R = load("w4_waw_register"); arr = np.zeros((10, 10, 10), np.uint8); arr[2:5, 2:5, 2:5] = 1; a = sitk.GetImageFromArray(arr); a.SetSpacing((2.0, 2.0, 2.0))
    arr2 = np.zeros_like(arr); arr2[2:5, 2:5, 3:6] = 1; b = sitk.GetImageFromArray(arr2); b.SetSpacing((2.0, 2.0, 2.0))
    assert R.dice(a, a) == 1.0 and abs(R.dice(a, b) - 2 * 18 / 54) < 1e-9
    assert np.allclose(R.centroid_mm(b) - R.centroid_mm(a), [2.0, 0, 0])      # one voxel along the last array axis (x) at 2 mm


def test_rigid_registration_recovers_a_known_shift_on_a_synthetic_phantom():
    R = load("w4_waw_register"); rng = np.random.default_rng(0)
    vol = np.full((40, 80, 80), -50.0, np.float32); zz, yy, xx = np.mgrid[:40, :80, :80]
    liver = (((zz - 20) / 14.0) ** 2 + ((yy - 40) / 28.0) ** 2 + ((xx - 40) / 30.0) ** 2) < 1; vol[liver] = 100.0; vol += rng.normal(0, 3, vol.shape).astype(np.float32)
    vol[(((zz - 20) / 4) ** 2 + ((yy - 34) / 7.0) ** 2 + ((xx - 48) / 7.0) ** 2) < 1] = 130.0
    fixed = sitk.GetImageFromArray(vol); fixed.SetSpacing((3.0, 3.0, 3.0))
    moved_arr = np.roll(vol, (1, 3, -4), axis=(0, 1, 2)); moving = sitk.GetImageFromArray(moved_arr); moving.SetSpacing((3.0, 3.0, 3.0))
    fl = sitk.GetImageFromArray(liver.astype(np.uint8)); fl.CopyInformation(fixed); ml = sitk.GetImageFromArray(np.roll(liver, (1, 3, -4), axis=(0, 1, 2)).astype(np.uint8)); ml.CopyInformation(moving)
    final, qa = R.register(fixed, moving, fl, ml, deform_below=0.0)
    assert qa["liver_dice"]["affine"] > qa["liver_dice"]["none"] and qa["liver_dice"]["affine"] > 0.95


def test_deformable_stage_runs_when_requested():
    R = load("w4_waw_register"); rng = np.random.default_rng(1)
    vol = np.full((40, 80, 80), -50.0, np.float32); zz, yy, xx = np.mgrid[:40, :80, :80]
    liver = (((zz - 20) / 14.0) ** 2 + ((yy - 40) / 28.0) ** 2 + ((xx - 40) / 30.0) ** 2) < 1; vol[liver] = 100.0; vol += rng.normal(0, 3, vol.shape).astype(np.float32)
    fixed = sitk.GetImageFromArray(vol); fixed.SetSpacing((3.0, 3.0, 3.0)); moving = sitk.GetImageFromArray(np.roll(vol, (1, 2, 2), axis=(0, 1, 2))); moving.SetSpacing((3.0, 3.0, 3.0))
    fl = sitk.GetImageFromArray(liver.astype(np.uint8)); fl.CopyInformation(fixed); ml = sitk.GetImageFromArray(np.roll(liver, (1, 2, 2), axis=(0, 1, 2)).astype(np.uint8)); ml.CopyInformation(moving)
    final, qa = R.register(fixed, moving, fl, ml, deform_below=1.1)
    assert qa["final_transform"] == "deformable" and "deformable" in qa["liver_dice"]


def test_reference_parenchyma_phantom_two_tumours_and_a_bright_vessel():
    """A liver (label 5) holding two tumours (+30 HU and -40 HU against 100 HU parenchyma) and a bright (300 HU) portal vein (label 64) 8 mm from tumour 1, i.e. inside its 5 to 20 mm shell, plus a bright hepatic vein
    (left inside label 5, as in exp8) far from both. Excluding label 64 gives the exact parenchyma reference; including the vessel as liver biases the contrast."""
    from glassbox.audit.contrast import case_lesions
    from glassbox.audit.waw import LIVER, PORTAL_SPLENIC_VEIN, reference_parenchyma
    n = 90; ct = np.full((n, n, n), 100.0, np.float32); organ = np.zeros((n, n, n), np.uint8); organ[5:85, 5:85, 5:85] = LIVER
    zz, yy, xx = np.mgrid[:n, :n, :n]
    t1 = ((xx - 30) ** 2 + (yy - 30) ** 2 + (zz - 45) ** 2) <= 7 ** 2; t2 = ((xx - 65) ** 2 + (yy - 60) ** 2 + (zz - 45) ** 2) <= 7 ** 2; ct[t1] = 130.0; ct[t2] = 60.0
    pv = (np.abs(yy - 30) <= 5) & (np.abs(zz - 45) <= 5) & (xx >= 45) & (xx <= 58); ct[pv] = 300.0; organ[pv] = PORTAL_SPLENIC_VEIN          # a thick vessel starting 8 mm from tumour 1's surface (inside its 5 to 20 mm shell), running along x
    hv = (np.abs(yy - 78) <= 1) & (np.abs(zz - 78) <= 1) & (xx >= 20) & (xx <= 60); ct[hv] = 300.0                                         # hepatic vein, labelled liver (stays in the reference)
    tumour = t1 | t2; ref = reference_parenchyma(organ, tumour)
    assert not (ref & tumour).any() and not (ref & pv).any() and ref[hv].all() and not (ref & ~(organ > 0)).any()                  # tumours and the portal vein are out, the hepatic vein is in
    rows = case_lesions(ct, tumour, organ == LIVER, (1.0, 1.0, 1.0), n_lesions=None, min_diam_mm=10.0, exclude=(organ == PORTAL_SPLENIC_VEIN))
    c = sorted(r["contrast_hu"] for r in rows); assert len(rows) == 2 and c[0] == pytest.approx(60.0 - 100.0, abs=1.5) and c[1] == pytest.approx(130.0 - 100.0, abs=1.5)          # only the thin hepatic vein (far outside both shells) perturbs the mean slightly
    organ_as_liver = organ.copy(); organ_as_liver[pv] = LIVER                                                                                # the failure the exclusion prevents: the vessel counted as parenchyma
    biased = case_lesions(ct, tumour, organ_as_liver == LIVER, (1.0, 1.0, 1.0), n_lesions=None, min_diam_mm=10.0)
    near = [r for r in biased if r["tumour_hu"] > 120][0]; assert near["contrast_hu"] < 30.0 - 3.0      # the bright vessel in tumour 1's shell raises the reference and lowers the contrast


def test_contrast_scripts_use_the_shared_reference():
    for name, needle in (("w4_waw_contrast.py", "exclude=(liver_lab == PORTAL_SPLENIC_VEIN)"), ("w4_waw_paired_contrast.py", "reference_parenchyma(org_m, lab > 0)")):
        assert needle in open(os.path.join(HERE, "scripts", name)).read(), name


def test_field_of_view_fraction_identity_and_shift():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pc", os.path.join(HERE, "scripts", "w4_waw_paired_contrast.py")); PC = importlib.util.module_from_spec(spec); spec.loader.exec_module(PC)
    mov = sitk.GetImageFromArray(np.zeros((20, 30, 40), np.float32)); mov.SetSpacing((1.0, 1.0, 2.0)); fix = sitk.GetImageFromArray(np.zeros((10, 30, 40), np.float32)); fix.SetSpacing((1.0, 1.0, 2.0))      # the target scan covers z index 0..9 only (20 mm of 40 mm)
    pts_in = np.argwhere(np.ones((5, 5, 3), bool)) + np.array([5, 5, 2]); pts_out = np.argwhere(np.ones((5, 5, 3), bool)) + np.array([5, 5, 14])      # [x, y, z] indices on the moving grid
    ident = sitk.Transform(3, sitk.sitkIdentity)
    assert PC.fov_inside_fraction(pts_in, mov, fix, ident) == 1.0 and PC.fov_inside_fraction(pts_out, mov, fix, ident) == 0.0
    half = np.concatenate([pts_in, pts_out]); assert 0.4 < PC.fov_inside_fraction(half, mov, fix, ident) < 0.6
