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
