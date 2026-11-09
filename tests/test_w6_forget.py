import importlib.util
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "scripts", name + ".py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_selection_dice_and_drift_classes():
    E = load("w6_forget_eval"); items = [f"c{i}" for i in range(100)]; a = E.seeded_sample(items, 20, "w6_drift"); assert a == E.seeded_sample(items[::-1], 20, "w6_drift") and len(set(a)) == 20 and a != E.seeded_sample(items, 20, "w6_kits")
    assert E.dice(np.zeros(4), np.zeros(4)) == 1.0 and abs(E.dice(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])) - 2 / 3) < 1e-9
    assert E.drift_classes({"0": "bg", "1": "liver", "26": "hepatic tumor", "132": "airway"}) == [1, 132]


def test_organ_change_rule():
    F = load("w6_forget_analyze"); rng = np.random.default_rng(0); base = {f"c{i}": 0.9 + rng.normal(0, 0.02) for i in range(60)}
    same = {k: v + rng.normal(0, 0.003) for k, v in base.items()}; drop = {k: v - 0.05 for k, v in base.items()}
    assert F.paired_change(base, same)["pass"] and not F.paired_change(base, drop)["pass"]


def test_drift_floor_and_pass_and_fp():
    F = load("w6_forget_analyze"); big = np.zeros((20, 20, 20), bool); big[2:18, 2:18, 2:18] = True; small = np.zeros((20, 20, 20), bool); small[5:7, 5:7, 5:7] = True
    base = {"1": [big, big], "2": [small, small]}; vol = {"1": [50.0, 60.0], "2": [1.0, 2.0]}
    same = {"1": [big, big], "2": [small, np.zeros_like(small)]}; r = F.drift_summary(base, same, vol); assert r["pass"] and r["classes"]["2"]["above_floor"] is False and r["classes"]["2"]["mean_dice"] == 0.5      # a small class may drift; reported only
    shifted = np.roll(big, 4, axis=0); r2 = F.drift_summary(base, {"1": [shifted, big], "2": base["2"]}, vol); assert not r2["pass"] and r2["failing_above_floor"] == ["1"]
    gt = np.zeros((40, 40, 40), bool); gt[10:14, 10:14, 10:14] = True; pred = np.zeros_like(gt); pred[10:14, 10:14, 10:14] = True; pred[30:32, 30:32, 30:32] = True
    assert abs(F.fp_ml(pred, gt, (1.0, 1.0, 1.0)) - 8 / 1000.0) < 1e-9 and F.fp_ml(np.zeros_like(gt), gt, (1.0, 1.0, 1.0)) == 0.0
