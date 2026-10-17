import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w2v3_analyze as A  # noqa: E402


def _r(bright, dice, case, diam=40.0, vol=1.0, inside=0.9, cov=0.8):
    return {"arm": "x", "case": case, "rank": 1, "bright": bright, "diam": diam, "c": 5.0 if bright else -5.0, "cov": cov, "dice": dice, "dice10": dice, "pdiam": diam, "vol": vol, "inside": inside}


def test_failure_modes_are_exclusive_and_ordered():
    assert A.mode(_r(True, 0.5, "a", vol=3.0)) == "LEAK" and A.mode(_r(True, 0.5, "a", inside=0.3)) == "LEAK"
    assert A.mode(_r(True, 0.5, "a", vol=1.0, inside=0.9, cov=0.49)) == "UNDER" and A.mode(_r(True, 0.5, "a", vol=5.0, cov=0.1)) == "LEAK"      # LEAK takes precedence over UNDER
    assert A.mode(_r(True, 0.5, "a")) == "OK"


def test_verdict_rule_and_order():
    assert A.verdict({"gap": 0.15, "lo": 0.05, "hi": 0.25}) == "GAP"
    assert A.verdict({"gap": 0.05, "lo": 0.01, "hi": 0.09}) == "NO MEANINGFUL GAP"
    assert A.verdict({"gap": 0.15, "lo": -0.02, "hi": 0.30}) == "INCONCLUSIVE"
    assert A.verdict({"gap": 0.12, "lo": 0.02, "hi": 0.09}) == "GAP"          # GAP is checked before NO MEANINGFUL GAP


def test_dice_gap_and_size_adjustment_on_synthetic_data():
    rng = np.random.default_rng(3); rs = []
    for i in range(150):
        bright = i % 3 == 0; rs.append(_r(bright, float(np.clip(0.75 - (0.25 if bright else 0.0) + rng.normal(0, 0.05), 0, 1)), f"c{i}", diam=float(rng.uniform(12, 90))))
    res, _ = A.analyse(rs, np.random.default_rng(0)); g = res["primary_dice_gap"]
    assert A.verdict(g) == "GAP" and 0.2 < g["gap"] < 0.3 and not res["size_adjusted"]["adjusted_effect_gone"]
    null = [_r(i % 3 == 0, float(np.clip(0.7 + rng.normal(0, 0.05), 0, 1)), f"n{i}") for i in range(150)]
    assert A.verdict(A.analyse(null, np.random.default_rng(0))[0]["primary_dice_gap"]) == "NO MEANINGFUL GAP"


def test_negative_headline_with_same_sign_adjusted_effect_is_not_flagged_gone():
    rs = [_r(i % 2 == 1, 0.8 if i % 2 == 1 else 0.5, f"s{i}", diam=40.0 + (i % 7)) for i in range(160)]      # bright better: negative gap
    res, _ = A.analyse(rs, np.random.default_rng(0)); assert res["primary_dice_gap"]["gap"] < 0 and not res["size_adjusted"]["adjusted_effect_gone"]


def test_end_to_end_on_small_synthetic_csvs():
    d = tempfile.mkdtemp(); rng = np.random.default_rng(5)
    cols = ["stage", "arm", "organ", "label_prompt", "point_label", "case", "rank", "group", "stratum", "diam_mm", "contrast_hu", "coverage", "dice_full", "dice_within_10mm", "pred_diam_mm", "point_comp_diam_mm",
            "pred_over_lesion_volume", "pred_frac_in_lesion", "point_x", "point_y", "point_z", "t_inference_s"]
    for stage, n in (("main", 60), ("pilot", 10)):
        for organ, arms in (("kidney", ["kidney_133_1"]), ("hcc", ["hcc_26_3", "hcc_133_1"])):
            with open(f"{d}/w2v2_{stage}_{organ}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(cols)
                for arm in arms:
                    for i in range(n):
                        bright = i % 2 == 0; dice = float(np.clip(0.7 - (0.3 if bright and arm != "hcc_26_3" else 0) + rng.normal(0, 0.05), 0, 1)); diam = float(rng.uniform(12, 90))
                        w.writerow([stage, arm, organ, 133, 1, f"{stage}{i}", 1, "bright" if bright else "dark", "x", diam, 5 if bright else -5, 0.8, dice, dice, diam, diam, 1.0, 0.9, 1, 1, 1, 1])
    A.main(d, d, False); assert os.path.exists(f"{d}/w2v3_results.json") and os.path.exists(f"{d}/w2v3_results.md")
    import json; R = json.load(open(f"{d}/w2v3_results.json"))
    assert R["kidney_133_1"]["verdict"] == "GAP" and R["hcc_133_1"]["verdict"] == "GAP" and R["hcc_26_3"]["verdict"] == "NO MEANINGFUL GAP" and R["replication_like_for_like"]["replicates"]
