import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w3_analyze as A  # noqa: E402
from glassbox.seg import w3_lib as L  # noqa: E402


def _r(bright, cov, dice, case, diam=40.0, pdiam=None, arm="x"):
    return {"arm": arm, "organ": "hcc", "case": case, "rank": 1, "bright": bright, "diam": diam, "c": 5.0 if bright else -5.0, "cov": cov, "dice": dice, "dice10": dice, "pdiam": diam * min(cov * 2, 1.0) if pdiam is None else pdiam, "vol": 1.0, "inside": 0.9}


def test_model_label_table():
    G, N, I = "GAP", "NO MEANINGFUL GAP", "INCONCLUSIVE"
    assert A.model_label(G, G) == "GAP (detection and Dice)" and A.model_label(I, G) == "GAP (Dice only)" and A.model_label(G, N) == "GAP (detection only)"
    assert A.model_label(N, N) == "NO MEANINGFUL GAP (both)" and A.model_label(N, I) == "INCONCLUSIVE" and A.model_label(I, I) == "INCONCLUSIVE"


def test_detection_threshold_is_coverage_ge_010():
    assert A.det(_r(True, 0.10, 0.1, "a")) == 1.0 and A.det(_r(True, 0.0999, 0.1, "a")) == 0.0


def test_logit_recovers_a_bright_penalty_and_flags_separation():
    rng = np.random.default_rng(1); rs = []
    for i in range(400):
        br = i % 2 == 0; d = float(rng.uniform(15, 80)); p = 1 / (1 + np.exp(-(0.5 + 0.03 * (d - 40) - 1.5 * br))); rs.append(_r(br, 0.5 if rng.random() < p else 0.0, 0.5, f"c{i}", diam=d))
    assert -2.2 < A.logit_coef(rs) < -0.8
    sep = [_r(True, 0.0, 0.0, f"s{i}", diam=30.0 + i) for i in range(20)] + [_r(False, 0.9, 0.9, f"t{i}", diam=30.0 + i) for i in range(20)]
    assert not np.isfinite(A.logit_coef(sep))


def test_synthetic_gap_and_null_through_analyse():
    rng = np.random.default_rng(2); rs = []
    for i in range(160):
        br = i % 3 == 0; cov = float(np.clip(rng.normal(0.35 if br else 0.75, 0.15), 0, 1)); rs.append(_r(br, cov, cov * 0.9, f"c{i}"))
    r = A.analyse(rs, np.random.default_rng(0)); assert r["detection_gap"]["gap"] > 0 and r["dice_verdict"] == "GAP" and r["label"].startswith("GAP")
    null = [_r(i % 3 == 0, 0.7, 0.6, f"n{i}") for i in range(160)]; r0 = A.analyse(null, np.random.default_rng(0)); assert r0["label"] == "NO MEANINGFUL GAP (both)"


def test_recist_misses_count_zero_and_detected_error():
    rs = [_r(False, 0.8, 0.7, "a", diam=40.0, pdiam=30.0), _r(False, 0.0, 0.0, "a", diam=20.0, pdiam=0.0), _r(True, 0.5, 0.5, "b", diam=50.0, pdiam=40.0)]
    c = A.recist(rs, np.random.default_rng(0))
    assert abs(c["sld_median_rel_error"] - np.median([(30 - 60) / 60, (40 - 50) / 50])) < 1e-9 and c["n_detected"] == 2 and abs(c["detected_median_error_mm"] + 10.0) < 1e-9


def test_step_attribution_names_largest_change():
    def g(det_gap, dice_gap):
        return {"detection_gap": {"gap": det_gap}, "dice_gap": {"gap": dice_gap}}
    R = {"A_swin_raw": g(0.30, 0.20), "A_swin_step_a": g(0.29, 0.10), "A_swin_step_ab": g(0.10, 0.09), "A_swin_shipped": g(0.09, 0.085)}
    s = A.step_attribution(R); assert s["largest_detection"].startswith("(b)") and s["largest_dice"].startswith("(a)")
    assert A.step_attribution({"A_swin_raw": g(0, 0)})["note"]


def test_disagree_detects_label_or_verdict_change():
    base = {"label": "GAP (detection and Dice)", "detection_verdict": "GAP", "dice_verdict": "GAP"}
    assert not A.disagree({"A_swin_shipped": base, "A_swin_raw": dict(base)}) and A.disagree({"A_swin_shipped": base, "A_swin_raw": {**base, "dice_verdict": "INCONCLUSIVE", "label": "GAP (detection only)"}})


def test_reference_check_reproduces_and_detects_mismatch():
    d = tempfile.mkdtemp(); uid = os.path.join(d, "uid.csv"); ref = os.path.join(d, "ref.csv"); new = []
    with open(uid, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "route", "rank", "diam_mm"]); [w.writerow([f"c{i}", "uid", 1, 30 + i]) for i in range(43)]; [w.writerow([f"c{i}", "uid", 2, 20 + i]) for i in range(43)]
    with open(ref, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "gt", "pred", "detected", "detected_any", "cover"])
        for i in range(43):
            for rk, g in ((1, 30 + i), (2, 20 + i)):
                det_ = (i + rk) % 5 != 0; pred = g * 0.7 if det_ else 0.0; w.writerow([f"c{i}", g, pred, det_, det_, 0.5 if det_ else 0.0]); new.append({**_r(False, 0.5 if det_ else 0.0, 0.5, f"c{i}", diam=float(g), pdiam=pred), "rank": rk})
    out = A.reference_check(new, ref, uid, np.random.default_rng(0)); assert out["n_lesions_matched"] == 86 and out["reproduced"] and out["n_flag_disagreements"] == 0
    for r in new[:4]:
        r["cov"] = 0.0 if r["cov"] else 0.5
    assert not A.reference_check(new, ref, uid, np.random.default_rng(0))["reproduced"]


def test_cohort_tables_and_alignment_rule():
    h = L.lesion_table("hcc"); k = L.lesion_table("kidney"); h0 = L.lesion_table("hcc", as_converted=True)
    hi = L.lesion_table("hcc", include_excluded=True)
    assert len(hi) == 86 and sum(float(r["contrast_hu"]) > 0 for r in hi.values()) == 29 and len({c for c, _ in hi}) == 75          # HCC_019_773068 corrected: dark -> bright
    assert "HCC_027_967743" in {c for c, _ in hi} and "HCC_027_967743" not in {c for c, _ in h} and len({c for c, _ in h}) == 74 and len(h) == 85          # HCC_027_967743 excluded from the primary (data-quality rule)
    assert sum(float(r["contrast_hu"]) > 0 for r in h.values()) == 28
    assert sum(float(r["contrast_hu"]) > 0 for r in L.lesion_table("hcc", as_converted=True, include_excluded=True).values()) == 28 and float(h0[("HCC_019_773068", 1)]["contrast_hu"]) < 0 < float(h[("HCC_019_773068", 1)]["contrast_hu"])
    assert L.case_root("hcc", "HCC_019_773068", "base").endswith("heldout_nii_fixed/hcc_tace_seg") and L.case_root("hcc", "HCC_019_773068", "base", as_converted=True) == "base" and L.case_root("hcc", "HCC_004_414785", "base") == "base"
    assert len(k) == 219 and sum(float(r["contrast_hu"]) > 0 for r in k.values()) == 53 and len({c for c, _ in k}) == 210
    seg = np.zeros((10, 10, 3), np.uint8); seg[2:8, 2:8] = 1; seg[4:6, 4:6] = 2; pred = (seg > 0)
    assert L.alignment(pred, seg, "hcc") == 1.0 and L.alignment(np.roll(pred, 4, axis=0), seg, "hcc") < L.ALIGN_MIN_DICE


def test_as_converted_rows_are_swapped_only_in_the_sensitivity():
    rows = [_r(True, 0.5, 0.5, "good", arm="m"), _r(True, 0.0, 0.0, "fixed", arm="m"), _r(False, 0.9, 0.9, "fixed", arm="asconv__m")]
    prim = A.split_asconv(rows, False); sens = A.split_asconv(rows, True)
    assert [(r["arm"], r["case"], r["bright"]) for r in prim] == [("m", "good", True), ("m", "fixed", True)]
    assert sorted((r["arm"], r["case"], r["bright"]) for r in sens) == [("m", "fixed", False), ("m", "good", True)]


def test_alignment_report_gate():
    seg = np.zeros((60, 120, 6), np.uint8); seg[10:40, 15:45, :] = 1; seg[20:30, 25:35, :] = 2; ann = np.ones(6, bool); sp = (1.0, 1.0, 2.5); gt = seg > 0
    ok = L.alignment_report(gt.copy(), seg, ann, sp, seg.shape); assert ok["pass"] and ok["organ_dice"] == 1.0 and ok["centroid_offset_mm"] == 0.0
    assert not L.alignment_report(np.roll(gt, 20, axis=1), seg, ann, sp, seg.shape)["pass"]                       # 20 mm displacement fails
    assert L.alignment_report(np.roll(gt, 10, axis=1), seg, ann, sp, seg.shape)["pass"]                           # 10 mm is inside the 15 mm limit
    under = gt.copy(); under[:, :, :] = False; under[10:40, 15:30, :] = True                                      # under-segmented but placed correctly: low Dice, still passes (offset 7.5 mm)
    r = L.alignment_report(under, seg, ann, sp, seg.shape); assert r["low_dice_lt_0.85"] and r["pass"]
    assert not L.alignment_report(gt.copy(), seg, ann, sp, (60, 120, 7))["pass"]                                   # shape mismatch fails
    flipped = np.zeros_like(gt); flipped[10:40, 15:45, :] = True; seg2 = np.zeros_like(seg); seg2[10:40, 15:45, :] = 1; seg2[10:30, 15:45, :] = 0     # GT in the lower part only; prediction in the upper part: flip along x fixes it
    assert L.alignment_report(np.flip(seg2 > 0, 0), seg2, ann, sp, seg2.shape)["flip_improves_dice"]


def test_per_arm_seed_makes_an_arms_cis_independent_of_the_other_arms(tmp_path):
    import json
    assert A.arm_seed("A_swin_shipped") == A.arm_seed("A_swin_shipped") != A.arm_seed("A_swin_raw") and A.arm_seed("x") == int("2d711642", 16)       # sha256("x") = 2d711642b726...
    rng = np.random.default_rng(5); rows = []
    for arm in ("a", "b"):
        for i in range(60):
            br = i % 3 == 0; cov = float(np.clip(rng.normal(0.3 if br else 0.7, 0.2), 0, 1)); rows.append({"arm": arm, "organ": "hcc", "case": f"c{i}", "rank": 1, "group": "bright" if br else "dark", "stratum": "30-60 mm", "diam_mm": 40.0 + i % 7, "contrast_hu": 5.0 if br else -5.0,
                                                                                                               "coverage": cov, "dice_full": cov * 0.8, "dice_within_10mm": cov * 0.8, "pred_diam_mm": 30.0, "pred_over_lesion_volume": 1.0, "pred_frac_in_lesion": 0.9, "mode": "OK"})
    import csv
    for name, arms in (("one", ("a",)), ("two", ("a", "b"))):
        d = tmp_path / name; d.mkdir(); f = d / "l.csv"
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows([r for r in rows if r["arm"] in arms])
        A.main(str(d), [str(f)])
    a1 = json.load(open(tmp_path / "one" / "w3_results.json"))["a"]; a2 = json.load(open(tmp_path / "two" / "w3_results.json"))["a"]
    assert a1["detection_gap"] == a2["detection_gap"] and a1["dice_gap"] == a2["dice_gap"] and a1["size_adjusted_dice"] == a2["size_adjusted_dice"]


def test_shift_test_classifies_misaligned_aligned_and_model_side():
    seg = np.zeros((80, 160, 8), np.uint8); seg[10:50, 40:100, :] = 1; ann = np.ones(8, bool); sp = (1.0, 1.0, 2.5); gt = seg > 0
    a = L.shift_test(gt.copy(), seg, ann, sp); assert a["verdict"] == "ALIGNED" and a["best_shift_mm"] == [0.0, 0.0, 0.0] and a["dice_unshifted"] == 1.0
    m = L.shift_test(np.roll(gt, 25, axis=1), seg, ann, sp); assert m["verdict"] == "MISALIGNED" and m["best_shift_mm"][1] == -25.0 and m["best_dice"] == 1.0 and m["gain"] > 0.3      # prediction 25 mm off: best translation brings it back
    assert L.shift_test(np.roll(gt, 4, axis=1), seg, ann, sp)["verdict"] == "ALIGNED"                          # 4 mm is inside the 5 mm limit
    under = np.zeros_like(gt); under[10:50, 40:60, :] = True; u = L.shift_test(under, seg, ann, sp)          # placed correctly but small: the best translation may move it, with a large gain only if it fits better
    stray = gt.copy(); stray[60:78, 120:150, :] = True; s = L.shift_test(stray, seg, ann, sp); assert s["verdict"] in ("ALIGNED", "MODEL-SIDE") and s["gain"] < 0.05 and u["gain"] >= 0.0       # stray blob: no translation helps
    other = np.zeros_like(gt); other[60:78, 120:150, :] = True; assert L.shift_test(other, seg, ann, sp)["verdict"] in ("MISALIGNED", "MODEL-SIDE")
