import json
import csv
import importlib.util
import os
import sys

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


def test_sign_count_rules_and_categories(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "scripts", "w4_waw_sign_counts.py")); SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
    assert SC.flip(10, -8, 5) and not SC.flip(10, -3, 5) and SC.flip(10, -3, 0) and not SC.flip(10, 8, 0) and not SC.flip(None, -8, 0)
    assert SC.comparable(10, -6, 5, 0.5) and not SC.comparable(10, -4, 5, 0.5) and not SC.comparable(20, 8, 5, 0.5) and SC.comparable(20, 8, 5, 0.33) and not SC.comparable(0, 0, 0, 0.5)
    fields = ["patient", "mask_phase", "target_phase", "rank", "c_arterial", "c_portal", "c_arterial_eroded", "c_portal_eroded", "c_arterial_median", "c_portal_median", "category"]
    rows = [dict(zip(fields, ["1", "1", "2", "1", "20", "-10", "18", "-8", "20", "-9", "K"])), dict(zip(fields, ["2", "1", "2", "1", "20", "5", "18", "4", "20", "5", "K"])), dict(zip(fields, ["3", "2", "1", "1", "", "", "", "", "", "", "F"])), dict(zip(fields, ["4", "2", "1", "1", "15", "-14", "15", "-14", "15", "-14", "Q"]))]
    pd = tmp_path / "p"; pd.mkdir()
    with open(pd / "tumours_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    qf = ["patient", "mask_phase", "liver_dice_final", "propagated_inside_target_liver", "volume_ratio", "qa_pass", "n_lesions_fov_clipped", "error"]
    with open(pd / "patient_qa.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=qf); w.writeheader()
        for p, d in (("1", 0.96), ("2", 0.97), ("3", 0.96), ("4", 0.9), ("5", "")):
            w.writerow({"patient": p, "mask_phase": 1, "liver_dice_final": d, "propagated_inside_target_liver": 0.99, "volume_ratio": 1.0, "qa_pass": int(p in "12"), "n_lesions_fov_clipped": int(p == "3"), "error": "RuntimeError: empty" if p == "5" else ""})
    mc = tmp_path / "m.csv"; mc.write_text("patient,phase,rank\n1,1,1\n2,1,1\n3,2,1\n4,2,1\n5,2,1\n")
    r = SC.main(str(pd), str(mc), str(tmp_path / "o"))
    assert r["categories"] == {"K": 2, "F": 1, "Q": 1, "E": 1} and r["category_check_sum"] == 5 and r["patients_E_error"] == ["5"]
    g = r["grid_all_kept"]["delta=5,rho=0.5"]; assert g == {"sign_flips": 1, "comparable_abs_contrast": 1, "flip_and_comparable (effective N)": 1}      # patient 1: 20 vs -10 flips, ratio 0.5; patient 2: 20 vs 5 does not
    assert r["Q_first_failed_criterion_per_tumour"] == {"liver_dice<0.92": 1}


def _write_series(root, case, shape=(40, 40, 30), tumour_boxes=((10, 20, 10, 20, 10, 20),), propagated=False, dil=False):
    """A tiny ground-truth series in the layout of scripts/w4_waw_paired_contrast.py --save-dir."""
    nib = pytest.importorskip("nibabel"); d = root / case; d.mkdir(parents=True)
    seg = np.zeros(shape, np.uint8); seg[2:38, 2:38, 3:27] = 1; lab = np.zeros(shape, np.uint16)
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(tumour_boxes, 1):
        seg[x0:x1, y0:y1, z0:z1] = 2; lab[x0:x1, y0:y1, z0:z1] = i
    aff = np.diag([1.0, 1.0, 2.0, 1.0]); nib.save(nib.Nifti1Image(np.zeros(shape, np.int16), aff), d / "ct.nii.gz"); nib.save(nib.Nifti1Image(seg, aff), d / "seg.nii.gz"); nib.save(nib.Nifti1Image(lab, aff), d / "tumour_labels.nii.gz")
    if dil:
        from scipy import ndimage
        nib.save(nib.Nifti1Image(np.where(ndimage.binary_dilation(lab > 0, iterations=2), 1, 0).astype(np.uint16) * np.where(lab > 0, lab, 1), aff), d / "tumour_labels_dil2.nii.gz")
    with open(d / "tumours.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient", "series_phase", "drawn_phase", "is_propagated", "rank_drawn_phase", "label_id", "diam_mm_drawn_phase", "fov_inside_fraction"])
        for i in range(1, len(tumour_boxes) + 1):
            w.writerow(["7", case.split("_")[1], "1", int(propagated), i, i, 12.0, 1.0])
    return seg, lab


def test_detection_is_the_w3_coverage_rule_and_the_dilation_changes_only_the_propagated_series(tmp_path):
    from glassbox.seg import w4_waw as W
    seg, lab = _write_series(tmp_path, "7_1")
    ser = W.read_series(str(tmp_path), "7_1"); pred = np.zeros(seg.shape, bool); pred[10:20, 10:20, 10:15] = True      # half of the tumour's slices: coverage 0.5
    r = W.score_arrays("m", ser["seg"], ser["labels"], ser["labels_dil2"], ser["tumours"], pred, ser["spacing"])[0]
    assert r["coverage"] == 0.5 and r["detected"] == 1 and r["coverage_dil2"] == r["coverage"] and r["is_propagated"] == 0       # a drawn mask is not dilated
    pred2 = np.zeros(seg.shape, bool); pred2[10:20, 10:20, 10:11] = True      # one slice of ten: coverage exactly 0.10 -> detected (>= 0.10)
    assert W.score_arrays("m", ser["seg"], ser["labels"], None, ser["tumours"], pred2, ser["spacing"])[0]["detected"] == 1
    pred3 = np.zeros(seg.shape, bool); pred3[10:20, 10:20, 10:10] = True; pred3[19, 19, 19] = True      # one voxel of 1000: coverage 0.001 -> missed
    assert W.score_arrays("m", ser["seg"], ser["labels"], None, ser["tumours"], pred3, ser["spacing"])[0]["detected"] == 0
    seg2, lab2 = _write_series(tmp_path, "7_2", propagated=True, dil=True); s2 = W.read_series(str(tmp_path), "7_2")
    r2 = W.score_arrays("m", s2["seg"], s2["labels"], s2["labels_dil2"], s2["tumours"], pred, s2["spacing"])[0]
    assert r2["is_propagated"] == 1 and r2["coverage_dil2"] < r2["coverage"] and r2["diam_mm_series"] > 8      # the dilated reference has more voxels, so the same prediction covers a smaller share of it


def test_score_script_writes_scores_and_the_alignment_gate(tmp_path):
    import importlib.util
    nib = pytest.importorskip("nibabel"); spec = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "scripts", "w4_waw_score.py")); SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
    gt = tmp_path / "gt"; gt.mkdir(); seg, lab = _write_series(gt, "7_1", shape=(80, 40, 30)); pd = tmp_path / "pred"; pd.mkdir()
    tum = np.zeros(seg.shape, bool); tum[10:20, 10:20, 10:20] = True; np.savez_compressed(pd / "7_1.npz", tumour=tum, liver=(seg > 0))
    rows, align = SC.main([str(gt), str(pd), "m", str(tmp_path / "out")])
    assert rows[0]["detected"] == 1 and align[0]["gate_pass"] and (tmp_path / "out" / "scores_m.csv").exists()
    np.savez_compressed(pd / "7_1.npz", tumour=tum, liver=np.roll(seg > 0, 12, axis=0))      # a liver prediction shifted 12 voxels (12 mm) along x
    rows, align = SC.main([str(gt), str(pd), "m", str(tmp_path / "out2")]); assert not align[0]["gate_pass"]


def test_w4_stats_recover_known_effects_and_labels():
    from glassbox.seg import w4_stats as S
    rng = np.random.default_rng(3); n = 400; pat = np.repeat(np.arange(200), 2); sa = rng.random(n) < 0.6; sp = rng.random(n) < 0.4
    al = rng.normal(0.5, 1.0, n); pa = 1 / (1 + np.exp(-(al + 0.0 + 1.5 * sa))); pp = 1 / (1 + np.exp(-(al + 1.5 * sp))); da = (rng.random(n) < pa).astype(int); dp = (rng.random(n) < pp).astype(int)
    D = np.column_stack([sa.astype(float) - sp, rng.normal(size=n)])
    r = S.cond_logit(pat, da, dp, D, ["sign", "mag"], "t", b=300); c = r["coef"]["sign"]
    assert r["status"].startswith("OK") and c["boot_OR_lo"] < np.exp(1.5) < c["boot_OR_hi"] and S.sign_verdict(r) == "REAL-PHASE BRIGHTNESS EFFECT" and r["discordant"] > 50
    r0 = S.cond_logit(pat, da, dp, np.column_stack([np.zeros(n), D[:, 1]]), ["sign", "mag"], "t0", b=100); assert r0["status"] == "NOT ESTIMABLE"      # a constant column is rank-deficient
    X = np.column_stack([np.ones(8), np.r_[np.ones(4), np.zeros(4)]]); z = np.r_[np.ones(4), np.zeros(4)]; b, f = S.fit_robust(X, z); assert f and np.all(np.isfinite(b)) and abs(b[1]) < 12      # perfect separation -> Jeffreys
    assert S.closes_verdict(-0.3, -0.5, -0.1) == "CLOSES" and S.closes_verdict(-0.05, -0.08, 0.0) == "DOES NOT CLOSE" and S.closes_verdict(-0.2, -0.4, 0.05) == "INCONCLUSIVE"
    assert S.gap_label(0.3, 0.1, 0.5) == "GAP" and S.gap_label(0.02, -0.05, 0.08) == "NO GAP" and S.model_gap_label("GAP", "INCONCLUSIVE") == "GAP (detection only)"
    pt = list(np.repeat(np.arange(100), 4)); br = [i % 2 == 0 for i in range(400)]; v1 = [1.0 if not b_ else 0.2 for b_ in br]; v0 = [1.0 if not b_ else 0.8 for b_ in br]
    d = S.paired_delta(pt, br, {"a1": v1, "a0": v0}, "a1", "a0", "x", b=200); assert abs(d["delta"] - 0.6) < 1e-9 and d["gap_arm1"] > d["gap_arm0"]
    e = S.effective_n_design(np.arange(20), np.ones(20), "e", b=200); assert e["label"] == "SIGN EFFECT" and S.effective_n_design([], [], "e")["n"] == 0


def test_analysis_runs_end_to_end_on_synthetic_scores(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("an", os.path.join(HERE, "scripts", "w4_waw_analyze.py")); AN = importlib.util.module_from_spec(spec); spec.loader.exec_module(AN)
    from glassbox.seg import w4_waw as W
    rng = np.random.default_rng(5); paired = tmp_path / "paired"; paired.mkdir(); tr = []; arms = {a: [] for a in AN.ARMS}
    for p in range(1, 61):
        for rk in (1, 2):
            ca, cp = rng.normal(15, 20), rng.normal(5, 20); drawn = int(rng.integers(1, 3)); tr.append({"patient": p, "mask_phase": drawn, "target_phase": 3 - drawn, "rank": rk, "diam_mm": 20.0, "c_arterial": ca, "c_portal": cp, "c_arterial_eroded": ca, "c_portal_eroded": cp,
                                                                                                         "c_arterial_median": ca, "c_portal_median": cp, "category": "K"})
            for sp in (1, 2):
                c = ca if sp == 1 else cp
                for a in AN.ARMS:
                    det = int(rng.random() < (0.9 if c <= 0 else 0.45)); arms[a].append({"arm": a, "patient": p, "series_phase": sp, "drawn_phase": drawn, "is_propagated": int(sp != drawn), "rank_drawn_phase": rk, "label_id": rk, "diam_mm_drawn_phase": 20.0, "diam_mm_series": 20.0,
                                                                                         "fov_inside_fraction": 1.0, "n_voxels": 100, "coverage": 0.5 * det, "dice_full": 0.6 * det, "dice_within_10mm": 0.6 * det, "coverage_dil2": 0.5 * det, "detected": det, "detected_dil2": det, "pred_diam_mm": 1, "mode": "OK"})
    with open(paired / "tumours_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr[0])); w.writeheader(); w.writerows(tr)
    sd = tmp_path / "s"; sd.mkdir()
    for a, rows in arms.items():
        with open(sd / f"scores_{a}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=W.FIELDS); w.writeheader(); w.writerows(rows)
        json.dump({"series_gate_fail": [], "per_series": []}, open(sd / f"alignment_{a}.json", "w"))
    AN.main([str(paired), str(tmp_path / "out"), str(sd), "--b-scale", "0.05", "--host1-paired", str(paired)]); pr = json.load(open(tmp_path / "out" / "primary.json"))
    assert pr["a"]["vista"]["all"]["label"].startswith("GAP")
    assert pr["b"]["vista"]["coef"]["sign"]["OR"] < 1 and (tmp_path / "out" / "sensitivity_table.json").exists() and "host1_scouting_contrasts" in json.load(open(tmp_path / "out" / "sensitivity_table.json"))["sensitivities"] and "dilation_2mm_propagated" in json.load(open(tmp_path / "out" / "sensitivity_table.json"))["sensitivities"]


def test_split_is_a_seeded_balanced_partition():
    spec = importlib.util.spec_from_file_location("sp", os.path.join(HERE, "scripts", "w4_waw_split.py")); SP = importlib.util.module_from_spec(spec); spec.loader.exec_module(SP)
    ids = [str(i) for i in range(1, 200)]; a, b = SP.split(ids); assert len(a) == 100 and len(b) == 99 and not set(a) & set(b) and set(a) | set(b) == set(ids) and SP.split(ids) == (a, b) and SP.split(ids[::-1]) == (a, b)


def test_gt_compare_fails_only_on_structure_and_reports_numeric_differences(tmp_path):
    spec = importlib.util.spec_from_file_location("gc", os.path.join(HERE, "scripts", "w4_waw_gt_compare.py")); GC = importlib.util.module_from_spec(spec); spec.loader.exec_module(GC)
    def write(d, c_portal, cat="K", qa="1", transform="affine"):
        d.mkdir(parents=True, exist_ok=True); cols = ["patient", "rank", "category"] + GC.COLS
        with open(d / "tumours_paired.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerow({**{c: "5" for c in GC.COLS}, "patient": "1", "rank": "1", "category": cat, "c_portal": c_portal})
        with open(d / "patient_qa.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["patient", "qa_pass", "transform"] + GC.QCOLS); w.writeheader(); w.writerow({**{c: "0.9" for c in GC.QCOLS}, "patient": "1", "qa_pass": qa, "transform": transform})
    write(tmp_path / "ref", "5"); write(tmp_path / "new", "-3", transform="deformable"); r = GC.compare(str(tmp_path / "ref"), str(tmp_path / "new"), str(tmp_path / "o1"))
    assert r["structural_ok"] and len(r["sign_changes"]) == 1 and r["transform_changes"] and r["contrast_difference_HU"]["c_portal"]["max"] == 8.0
    write(tmp_path / "new2", "5", cat="Q"); assert not GC.compare(str(tmp_path / "ref"), str(tmp_path / "new2"), str(tmp_path / "o2"))["structural_ok"]
    write(tmp_path / "new3", "5", qa="0"); assert not GC.compare(str(tmp_path / "ref"), str(tmp_path / "new3"), str(tmp_path / "o3"))["structural_ok"]


def test_parallel_scoring_is_byte_identical_to_serial(tmp_path):
    spec = importlib.util.spec_from_file_location("sc2", os.path.join(HERE, "scripts", "w4_waw_score.py")); SC = importlib.util.module_from_spec(spec); sys.modules["sc2"] = SC; spec.loader.exec_module(SC)      # registered so that the forked workers can unpickle score_case
    gt = tmp_path / "gt"; gt.mkdir(); pd = tmp_path / "pred"; pd.mkdir(); rng = np.random.default_rng(2)
    for pid in (3, 4, 5):
        for sp in (1, 2):
            seg, lab = _write_series(gt, f"{pid}_{sp}", shape=(60, 40, 30)); tum = np.zeros(seg.shape, bool); tum[10:20, 10:20, 10:10 + int(rng.integers(1, 10))] = True
            np.savez_compressed(pd / f"{pid}_{sp}.npz", tumour=tum, liver=np.roll(seg > 0, int(rng.integers(0, 3)), axis=0))
    SC.main([str(gt), str(pd), "m", str(tmp_path / "o1")]); SC.main([str(gt), str(pd), "m", str(tmp_path / "o3"), "--workers", "3"])
    for f in ("scores_m.csv", "alignment_m.json"):
        assert (tmp_path / "o1" / f).read_bytes() == (tmp_path / "o3" / f).read_bytes()


def test_reorient_round_trip_restores_the_original_grid_exactly(tmp_path):
    nib = pytest.importorskip("nibabel"); spec = importlib.util.spec_from_file_location("ro", os.path.join(HERE, "scripts", "w4_waw_reorient.py")); RO = importlib.util.module_from_spec(spec); spec.loader.exec_module(RO)
    rng = np.random.default_rng(4); data = rng.integers(0, 5, (12, 9, 7)).astype(np.int16)
    # an LSP-like affine: array axis0 -> L, axis1 -> S, axis2 -> P
    aff = np.array([[-0.8, 0, 0, 100.0], [0, 0, -3.0, 50.0], [0, 0.0, 0, 0], [0, 0, 0, 1]]); aff = np.array([[-0.8, 0, 0, 100.0], [0, 0, 3.0, -20.0], [0, -0.7, 0, 30.0], [0, 0, 0, 1]])      # x -> -x (L), axis2 -> +y... a permuted, non-LAS orientation
    img = nib.Nifti1Image(data, aff); assert "".join(nib.aff2axcodes(aff)) != "LAS"
    arr, aff2, _ = RO.to_target(img); assert "".join(nib.aff2axcodes(aff2)) == "LAS" and sorted(arr.shape) == sorted(data.shape) and np.sort(arr.ravel()).tolist() == np.sort(data.ravel()).tolist()
    # voxel (i, j, k) of the original and its image in the new array have the same world position
    for ijk in ((0, 0, 0), (3, 5, 2), (11, 8, 6)):
        w = (aff @ np.r_[ijk, 1])[:3]; inv = np.linalg.inv(aff2) @ np.r_[w, 1]; assert arr[tuple(int(round(v)) for v in inv[:3])] == data[ijk]
    back = RO.back_to_original(arr, img); assert back.shape == data.shape and np.array_equal(back, data)


def test_paired_mean_diff_matches_the_mean_and_covers_it():
    from glassbox.seg import w4_stats as S
    pat = np.repeat(np.arange(40), 2); v1 = np.ones(80); v0 = np.zeros(80); r = S.paired_mean_diff(pat, v1, v0, "t", b=200); assert r["delta"] == 1.0 and r["lo"] == 1.0 and r["hi"] == 1.0 and r["n_patients"] == 40
    rng = np.random.default_rng(1); v1 = (rng.random(80) < 0.9).astype(float); v0 = (rng.random(80) < 0.1).astype(float); r = S.paired_mean_diff(pat, v1, v0, "t", b=500); assert r["lo"] < r["delta"] < r["hi"] and 0.6 < r["delta"] < 0.95
