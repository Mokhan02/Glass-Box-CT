"""Amendment 2 tests: the Seed2 trainer differs from Bright50 only in the seed; the early check and the replication report behave."""
import ast
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__); S = os.path.join(HERE, "..", "scripts")


def classes(path):
    return {n.name: n for n in ast.walk(ast.parse(open(os.path.join(S, path)).read())) if isinstance(n, ast.ClassDef)}      # the W5 trainer classes sit inside an `if nnUNetTrainer is not None:` block


def test_seed2_is_bright50_with_only_the_seed_changed():
    c = classes("w5_nnunet_trainer_seed2.py"); assert list(c) == ["nnUNetTrainerW5Bright50Seed2"]
    cls = c["nnUNetTrainerW5Bright50Seed2"]; assert [ast.unparse(b) for b in cls.bases] == ["nnUNetTrainerW5Bright50"]
    body = [n for n in cls.body if not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))]      # ignore a docstring
    assert len(body) == 1 and isinstance(body[0], ast.Assign) and [t.id for t in body[0].targets] == ["SEED"] and ast.unparse(body[0].value) == "SEED_B50_S2"      # no other attribute or method is overridden
    src = open(os.path.join(S, "w5_nnunet_trainer_seed2.py")).read(); assert "SEED_B50_S2 = 20261028" in src
    base = classes("w5_nnunet_trainer.py"); assert "nnUNetTrainerW5Bright50" in base      # the parent exists unchanged in the committed trainer file
    b50 = base["nnUNetTrainerW5Bright50"]; assert [ast.unparse(n) for n in b50.body if isinstance(n, ast.Assign)] == ["(ARM, P) = ('BRIGHT', 0.5)"]


def test_seed_values_distinct_from_the_other_fixed_seeds():
    assert 20261028 not in (20261020, 20261121)


def test_instance_level_when_nnunet_importable():
    pytest.importorskip("nnunetv2")
    sys.path.insert(0, S)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5 import nnUNetTrainerW5Bright50 as A
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerW5Seed2 import nnUNetTrainerW5Bright50Seed2 as B
    assert (A.ARM, A.P) == (B.ARM, B.P) and A.SEED == 20261020 and B.SEED == 20261028 and A.get_training_transforms is B.get_training_transforms


def run_early(tmp_path, seed_line, times, losses, edits):
    tl = tmp_path / "tl.txt"; tl.write_text(seed_line + "\n" + "".join(f"x: train_loss {a}\nx: val_loss {b}\nx: Epoch time: {t} s\n" for t, (a, b) in zip(times, losses)))
    el = tmp_path / "el.txt"; el.write_text("".join(f"BRIGHT {t} {a} {d}\n" for t, a, d in edits)); out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, os.path.join(S, "w5_early_check.py"), "20261028", str(tl), str(el), str(out)], capture_output=True, text=True); return r.returncode, json.load(open(out))


GOOD_SEED = "2026: W5: seed 20261028; augmentation BRIGHT p=0.5, delta range (10.0, 60.0) HU relative to local liver mean; epochs 1000"
GOOD_E = [(1, 1, "20.5,31.0")] * 50 + [(1, 0, "")] * 50 + [(0, 0, "")] * 20


def test_early_check_pass_and_failures(tmp_path):
    rc, r = run_early(tmp_path, GOOD_SEED, [150, 61], [(-0.1, -0.2), (-0.2, -0.3)], GOOD_E); assert rc == 0 and r["pass"]
    rc, r = run_early(tmp_path, GOOD_SEED.replace("20261028", "20261020"), [150, 61], [(-0.1, -0.2)] * 2, GOOD_E); assert rc == 1 and not r["checks"]["seed_in_log"]
    rc, r = run_early(tmp_path, GOOD_SEED, [150, 80], [(-0.1, -0.2)] * 2, GOOD_E); assert rc == 1 and not r["checks"]["epoch2_time_within_10pct"]
    rc, r = run_early(tmp_path, GOOD_SEED, [150, 61], [(-0.1, -0.2), (float("nan"), -0.3)], GOOD_E); assert rc == 1 and not r["checks"]["losses_finite"]
    rc, r = run_early(tmp_path, GOOD_SEED, [150, 61], [(-0.1, -0.2)] * 2, [(1, 0, "")] * 100); assert rc == 1 and not r["checks"]["transform_active"]
    rc, r = run_early(tmp_path, GOOD_SEED, [150, 61], [(-0.1, -0.2)] * 2, [(1, 1, "-20.0")] * 50 + [(1, 0, "")] * 50); assert rc == 1 and not r["checks"]["transform_active"]


def arms(det, dice):
    base = {"n_dark": 57, "n_bright": 28, "detection_dark": 0.9, "detection_bright": 0.5, "dice_dark": 0.6, "dice_bright": 0.3, "mean_dice_all": 0.5, "median_signed_sld_error": -0.2}
    return {"D0": {**base, "detection_gap": 0.645, "dice_gap": 0.419}, "S50": {**base, "detection_gap": 0.52, "dice_gap": 0.431}, "B50": {**base, "detection_gap": det, "dice_gap": dice}}


def test_replication_report(tmp_path):
    a = tmp_path / "a.json"; b = tmp_path / "b.json"; c = tmp_path / "c.json"
    a.write_text(json.dumps({"arms": arms(0.09, 0.185), "prediction": "CONFIRMED (detection and Dice)"}))
    b.write_text(json.dumps({"arms": arms(0.15, 0.20), "prediction": "CONFIRMED (detection and Dice)", "verdicts": {}, "contrasts": {}}))
    c.write_text(json.dumps({"arms": {**arms(0.15, 0.20), "S50": {**arms(0.15, 0.2)["S50"], "dice_gap": 0.9}}, "prediction": "NOT CONFIRMED"}))
    r = subprocess.run([sys.executable, os.path.join(S, "w5_replication_report.py"), str(a), str(b), str(tmp_path / "o")], capture_output=True, text=True); assert r.returncode == 0
    res = json.load(open(tmp_path / "o" / "w5_replication.json")); assert res["replicates"] and abs(res["seed_variance_descriptive"]["abs_detection_gap_difference"] - 0.06) < 1e-9 and abs(res["seed_variance_descriptive"]["abs_dice_gap_difference"] - 0.015) < 1e-9
    r = subprocess.run([sys.executable, os.path.join(S, "w5_replication_report.py"), str(a), str(c), str(tmp_path / "o2")], capture_output=True, text=True); assert r.returncode != 0      # D0/S50 differ: refuses
    b.write_text(json.dumps({"arms": arms(0.15, 0.20), "prediction": "INCONCLUSIVE", "verdicts": {}, "contrasts": {}}))
    subprocess.run([sys.executable, os.path.join(S, "w5_replication_report.py"), str(a), str(b), str(tmp_path / "o3")], capture_output=True, text=True); assert not json.load(open(tmp_path / "o3" / "w5_replication.json"))["replicates"]
