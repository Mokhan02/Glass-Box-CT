"""Tests for scripts/w6b_secondary.py on a small synthetic guard folder (3 classes, 4 series)."""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("w6b_secondary", os.path.join(HERE, "scripts", "w6b_secondary.py")); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def make(root, shift, series=("half0/1_1", "half0/2_1", "half1/3_1", "half1/4_1"), classes=(1, 3, 5), kidney=0.9):
    os.makedirs(os.path.join(root, "drift"), exist_ok=True); json.dump({"series": list(series), "classes": list(classes)}, open(os.path.join(root, "drift_meta.json"), "w"))
    for s in series:
        m = {}
        for c in classes:
            a = np.zeros((20, 20, 20), bool); a[2:12, 2:12, 2:12] = True      # 1000 voxels
            if shift.get(c):
                a = np.roll(a, shift[c], axis=0)
            m[str(c)] = a
        np.savez(os.path.join(root, "drift", s.replace("/", "_") + ".npz"), spacing=np.array([10.0, 10.0, 10.0]), **m)      # 10 mm voxels: 1 mL each, 1000 mL class volume
    with open(os.path.join(root, "kidney.csv"), "w") as f:
        f.write("case,dice\n" + "".join(f"k{i},{kidney - 0.01 * i}\n" for i in range(6)))


def test_counts_floor_and_differences(tmp_path):
    b, ref, arm = (str(tmp_path / x) for x in ("b", "ref", "arm")); make(b, {}); make(ref, {1: 8, 3: 8, 5: 8}, kidney=0.8); make(arm, {1: 8}, kidney=0.85)      # shifting by 8 voxels leaves 2 of 10 overlapping: Dice 0.2
    out = str(tmp_path / "o.json"); S.main([out, "--base", b, "--ref", f"REF={ref}", "--arm", f"ARM={arm}", "--b", "200"]); r = json.load(open(out))
    assert r["n_above_floor"] == 3 and r["arms"]["REF"]["failing_classes"] == 3 and r["arms"]["ARM"]["failing_classes"] == 1
    assert r["arms"]["ARM"]["failing_minus_reference"] == -2 and r["arms"]["ARM"]["failing_minus_reference_ci"] == [-2.0, -2.0]
    assert abs(r["arms"]["ARM"]["kidney_change_vs_baseline"]["mean"] - (-0.05)) < 1e-9 and abs(r["arms"]["ARM"]["kidney_change_vs_reference"]["mean"] - 0.05) < 1e-9
    assert r["arms"]["ARM"]["lowest_class_mean_dice_above_floor"] < 0.3 and r["arms"]["REF"]["failing_ci"][0] == 3
