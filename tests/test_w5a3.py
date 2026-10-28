import csv
import os

import numpy as np
import pytest

from glassbox.seg import w5a3 as W

HERE = os.path.dirname(os.path.dirname(__file__))
FIELDS = ["case", "rank", "arm", "condition", "T", "L", "c", "target", "realised_mean", "flagged", "coverage", "dice_full", "dice_within_10mm", "pred_diam_mm", "n_tokens"]


def write(path, lesions):
    """lesions: list of (case, rank, arm, base_cov, edit_cov, ctrl_cov[, flagged])"""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader()
        for case, rank, arm, b, e, c, *fl in lesions:
            for cond, cov in (("base", b), ("edit", e), ("control", c)):
                w.writerow({"case": case, "rank": rank, "arm": arm, "condition": cond, "coverage": cov, "dice_within_10mm": cov / 2, "flagged": str(bool(fl and fl[0] and cond != "base"))})


def pair(tmp_path, n=30, d0_edit=0.0, b50_edit=0.6, ctrl=0.9, arm="A", jitter=0.02):
    rng = np.random.default_rng(0); L0, L1 = [], []
    for i in range(n):
        j = rng.normal(0, jitter)
        L0.append((f"C{i:03d}", 1, arm, 0.9, d0_edit + j, ctrl)); L1.append((f"C{i:03d}", 1, arm, 0.95, b50_edit + j, ctrl))
    write(tmp_path / "d0.csv", L0); write(tmp_path / "b50.csv", L1)
    return W.load_m3(str(tmp_path / "d0.csv")), W.load_m3(str(tmp_path / "b50.csv"))


def test_verdict_rule_order_and_margins():
    assert W.verdict(0.30, 0.20, 0.40, 49) == "REDUCED"
    assert W.verdict(0.30, 0.20, 0.40, 9) == "INCONCLUSIVE"            # fewer than 10 lesions
    assert W.verdict(0.12, -0.01, 0.25, 49) == "INCONCLUSIVE"          # CI includes 0
    assert W.verdict(0.08, 0.02, 0.15, 49) == "INCONCLUSIVE"           # CI above 0 but the point is under the 0.10 margin and the upper bound is not below it
    assert W.verdict(0.05, 0.01, 0.09, 49) == "NOT REDUCED"            # upper bound below the margin
    assert W.verdict(0.0, 0.0, 0.0, 49) == "NOT REDUCED"
    assert W.verdict(0.10, 0.001, 0.2, 49) == "REDUCED"                # point exactly at the margin


def test_large_reduction_is_called_reduced(tmp_path):
    d0, b50 = pair(tmp_path, d0_edit=0.0, b50_edit=0.6)                # D_A: -0.9 in D0, -0.3 in BRIGHT-50 -> Delta +0.6
    r = W.contrast(d0, b50, "A")
    assert r["n_common"] == 30 and r["verdict"] == "REDUCED" and r["delta"]["point"] == pytest.approx(0.6, abs=0.03) and r["delta"]["ci"][0] > 0.5
    assert r["bright50_own"]["residual_harm"] == "PRESENT"             # still below zero
    assert r["relative_reduction_vs_d0_harm"] == pytest.approx(0.6 / 0.9, abs=0.05)


def test_identical_models_give_zero_and_not_reduced(tmp_path):
    d0, _ = pair(tmp_path)
    r = W.contrast(d0, d0, "A")
    assert r["delta"]["point"] == 0 and r["delta"]["ci"] == [0, 0] and r["verdict"] == "NOT REDUCED"


def test_minimum_lesion_rule_and_matching(tmp_path):
    d0, b50 = pair(tmp_path, n=8)
    assert W.contrast(d0, b50, "A")["verdict"] == "INCONCLUSIVE"       # 8 common lesions
    d0, b50 = pair(tmp_path, n=30)
    for k in list(b50)[:5]:
        del b50[k]                                                      # 5 of D0's lesions are not eligible in BRIGHT-50
    r = W.contrast(d0, b50, "A")
    assert r["n_common"] == 25 and r["n_eligible_d0"] == 30 and len(r["d0_only"]) == 5


def test_only_the_requested_arm_is_used_and_arm_b_is_listed_without_a_verdict(tmp_path):
    L0 = [(f"C{i:03d}", 1, "B", 0.0, 0.5, 0.0) for i in range(22)]; L1 = [(f"C{i:03d}", 1, "B", 0.0, 0.3, 0.0) for i in range(3)]
    write(tmp_path / "d0.csv", L0); write(tmp_path / "b50.csv", L1); d0, b50 = W.load_m3(str(tmp_path / "d0.csv")), W.load_m3(str(tmp_path / "b50.csv"))
    assert W.contrast(d0, b50, "A")["n_common"] == 0
    b = W.arm_b_listing(d0, b50)
    assert b["n_bright50"] == 3 and b["n_d0"] == 22 and b["verdict"].startswith("NONE") and len(b["lesions"]) == 3 and b["lesions"][0]["d0"]["dcov_edit"] == 0.5


def test_cohort_restriction(tmp_path):
    d0, b50 = pair(tmp_path, n=12)
    write(tmp_path / "x.csv", [("HCC_027_x", 1, "A", 0.9, 0.0, 0.9)]); assert not W.load_m3(str(tmp_path / "x.csv"), cases={"C000"})
    assert len(W.load_m3(str(tmp_path / "d0.csv"), cases={"C000", "C001"})) == 2


def test_stage2_scripts_keep_their_defaults_and_expose_only_the_two_overrides():
    sc = open(os.path.join(HERE, "scripts", "w3s2_m3_smoke_check.py")).read(); assert 'os.environ.get("W5A3_STAGE1_D_TABLE", "results/w3_models/stage1/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv")' in sc
    hc = open(os.path.join(HERE, "scripts", "w3s2_hook_check.py")).read(); m2 = open(os.path.join(HERE, "scripts", "w3s2_m2_run.py")).read()
    assert 'os.environ.get("W5A3_D_TRAINER", "nnUNetTrainerW3Seed")' in hc and 'os.environ.get("W5A3_STAGE1_D_TABLE", f"{stage1}/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv")' in hc
    assert 'os.environ.get("W5A3_STAGE1_D_TABLE", "results/w3_models/stage1/D_nnunet_hcc/w3_lesions_D_nnunet_hcc.csv")' in m2


def test_m2_by_group_splits_and_bootstraps():
    by = {}; group = {}
    for i in range(20):
        k = (f"C{i:03d}", "1"); g = "bright" if i % 2 else "dark"; group[k] = g
        row = lambda cov, dice: {"coverage": cov, "dice_within_10mm": dice}      # noqa: E731
        by[k] = {"base": row(0.9, 0.8), "leace": row(0.5, 0.8 - (0.4 if g == "bright" else 0.1)), "random": row(0.9, 0.8), "hu": row(0.9, 0.8)}
    r = W.m2_by_group(by, group)
    assert r["bright"]["n_lesions"] == 10 and r["dark"]["n_lesions"] == 10
    assert r["bright"]["leace"]["mean_dice_change"]["point"] == pytest.approx(-0.4) and r["dark"]["leace"]["mean_dice_change"]["point"] == pytest.approx(-0.1)
    assert r["bright"]["random"]["mean_dice_change"]["point"] == 0 and r["bright"]["leace"]["still_detected"] == 1.0


def test_merge_m2_concatenates_shards_and_rejects_overlap(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mm", os.path.join(HERE, "scripts", "w5a3_merge_m2.py")); mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    for i, cases in enumerate((["C1", "C2"], ["C3"])):
        d = tmp_path / f"s{i}"; (d / "maps").mkdir(parents=True)
        with open(d / "lesions.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "rank", "arm", "coverage"]); w.writeheader(); [w.writerow({"case": c, "rank": 1, "arm": "D_base", "coverage": 0.5}) for c in cases]
        with open(d / "cases.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["case", "cond", "organ_dice"]); w.writeheader(); [w.writerow({"case": c, "cond": "base", "organ_dice": 0.9}) for c in cases]
        (d / "done_cases.txt").write_text("\n".join(cases) + "\n"); (d / "maps" / f"{cases[0]}.npz").write_bytes(b"x")
        import json
        json.dump({"base_vs_stage1": [{"case": c, "rank": 1, "abs_cov_diff": 0.0001 * (i + 1), "flag_same": True} for c in cases], "identity_vs_base": [{"case": cases[0], "rank": 1, "abs_cov_diff": 0.0}]}, open(d / "determinism.json", "w"))
    r = mm.merge(str(tmp_path / "out"), [str(tmp_path / "s0"), str(tmp_path / "s1")])
    assert r["cases"] == 3 and r["summary"]["detection_flag_mismatches"] == 0 and r["summary"]["max_base_vs_stage1_cov_diff"] == pytest.approx(0.0002)
    assert len(list(csv.DictReader(open(tmp_path / "out" / "lesions.csv")))) == 3 and sorted(os.listdir(tmp_path / "out" / "maps")) == ["C1.npz", "C3.npz"]
    with pytest.raises(SystemExit):
        mm.merge(str(tmp_path / "out2"), [str(tmp_path / "s0"), str(tmp_path / "s0")])           # the same folder twice
    (tmp_path / "s2").mkdir(); import shutil
    for fn in ("lesions.csv", "cases.csv"):
        shutil.copy(tmp_path / "s0" / fn, tmp_path / "s2" / fn)
    with pytest.raises(SystemExit):
        mm.merge(str(tmp_path / "out3"), [str(tmp_path / "s0"), str(tmp_path / "s2")])           # the same case in two shards
