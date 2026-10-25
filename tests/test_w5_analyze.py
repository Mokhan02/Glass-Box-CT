import csv
import importlib.util
import json
import os

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("w5a", os.path.join(os.path.dirname(__file__), "..", "scripts", "w5_analyze.py")); A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
FIELDS = ["arm", "case", "rank", "group", "diam_mm", "coverage", "dice_full", "pred_diam_mm"]


def write(path, det_dark, det_bright, dice_dark, dice_bright, n_cases=30, extra_arm=False):
    rng = np.random.default_rng(1); rows = []
    for i in range(n_cases):
        for rank, (g, pd, pdice) in enumerate((("dark", det_dark, dice_dark), ("bright", det_bright, dice_bright)), 1):
            hit = (i / n_cases) < pd
            rows.append({"arm": "D_nnunet", "case": f"C{i:03d}", "rank": rank, "group": g, "diam_mm": 20.0, "coverage": 0.8 if hit else 0.0, "dice_full": pdice if hit else 0.0, "pred_diam_mm": 18.0 if hit else 0.0})
            if extra_arm:
                rows.append({**rows[-1], "arm": "asconv__D_nnunet", "coverage": 0.0})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["organ"] if False else FIELDS); w.writeheader(); w.writerows(rows)


def setup(tmp_path, b50_bright=0.9, s50_bright=0.2, d0_bright=0.2, **kw):
    for k, v in (("D0", d0_bright), ("B50", b50_bright), ("S50", s50_bright)):
        write(tmp_path / f"{k}.csv", 0.9, v, 0.7, 0.7 * v / 0.9 if v else 0.0, extra_arm=(k == "D0"), **kw)
    return {k: str(tmp_path / f"{k}.csv") for k in ("D0", "B50", "S50")}


def test_closes_and_confirmed(tmp_path):
    r = A.main(str(tmp_path / "o"), setup(tmp_path), {})
    assert r["arms"]["D0"]["detection_gap"] > 0.5 and r["arms"]["B50"]["detection_gap"] < 0.1
    assert r["contrasts"]["B50_minus_D0"]["det"]["delta"] < -0.5
    assert r["prediction"].startswith("CONFIRMED") and r["verdicts"]["D0_det"] == "CLOSES" and r["verdicts"]["S50_det"] == "CLOSES"
    assert r["n_lesions"] == 60 and "SINGLE-SEED" in r["label"]


def test_not_confirmed_when_no_change(tmp_path):
    r = A.main(str(tmp_path / "o"), setup(tmp_path, b50_bright=0.2), {})
    assert r["prediction"] == "NOT CONFIRMED" and r["verdicts"]["D0_det"] == "DOES NOT CLOSE"


def test_detection_only_when_dice_unchanged(tmp_path):
    p = setup(tmp_path, b50_bright=0.9); write(tmp_path / "B50.csv", 0.9, 0.9, 0.9, 0.0)      # bright found but with Dice 0: Dice gap does not close
    r = A.main(str(tmp_path / "o"), p, {}); assert r["prediction"] == "CONFIRMED (detection only)" and r["verdicts"]["D0_dice"] == "DOES NOT CLOSE"
    assert r["verdicts"]["D0_det"] == "CLOSES"


def test_failed_gate_not_evaluable(tmp_path):
    p = setup(tmp_path); g = tmp_path / "g.json"; g.write_text(json.dumps({"liver_dice": 0.9, "tumour_dice": 0.5, "pass": False}))
    r = A.main(str(tmp_path / "o"), p, {"B50": str(g)}); assert r["prediction"].startswith("NOT EVALUABLE")


def test_mismatched_lesions_rejected(tmp_path):
    p = setup(tmp_path); write(tmp_path / "S50.csv", 0.9, 0.2, 0.7, 0.1, n_cases=29)
    with pytest.raises(SystemExit):
        A.main(str(tmp_path / "o"), p, {})


def test_seed_stable_and_deterministic(tmp_path):
    assert A.seed("W5_B50_vs_D0_det") == A.seed("W5_B50_vs_D0_det")
    p = setup(tmp_path); r1 = A.main(str(tmp_path / "o1"), p, {}); r2 = A.main(str(tmp_path / "o2"), p, {}); assert r1["contrasts"] == r2["contrasts"]
