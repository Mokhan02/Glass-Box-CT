import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w2m_m2_analyze as A  # noqa: E402


def _write(d, drops, organ_drop, n_cases=60):
    rng = np.random.default_rng(0); base, les, org = [], [], []
    for i in range(n_cases):
        for rank in (1, 2) if i % 3 == 0 else (1,):
            b = float(np.clip(0.6 + rng.normal(0, 0.05), 0.05, 1)); key = {"case": f"k{i}", "rank": rank}
            mk = lambda dice: {"pred_over_lesion_volume": 1.0, "pred_frac_in_lesion": 0.8, "coverage": 0.7, "dice_full": dice, "contrast_hu": -5.0 if i % 2 else 5.0}
            base.append({"arm": "kidney_133_1", **key, **mk(b), "point_x": 1, "point_y": 1, "point_z": 1})
            for c in A.COND:
                les.append({"condition": c, **key, **mk(float(np.clip(b + drops[c] + rng.normal(0, 0.02), 0, 1)))})
    for i in range(0, n_cases, 2):
        org.append({"case": f"k{i}", "dice_unhooked": 0.9, **{f"dice_{c}": 0.9 + (organ_drop if c == "leace_native" else 0.0) + float(rng.normal(0, 0.01)) for c in A.COND}})
    for name, rows in (("base.csv", base), ("les.csv", les), ("org.csv", org)):
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def test_used_when_leace_drops_dice_beyond_the_controls_and_the_organ_holds():
    d = tempfile.mkdtemp(); _write(d, {"leace_native": -0.20, "random_native": 0.0, "hu_native": -0.01, "leace_hcc": 0.0}, -0.01)
    A.main(f"{d}/base.csv", f"{d}/les.csv", f"{d}/org.csv", d); import json; R = json.load(open(f"{d}/m2_results.json"))["decision_primary"]
    assert R["verdict"] == "USED" and not R["erasure_too_blunt"]


def test_too_blunt_when_the_organ_dice_also_drops():
    d = tempfile.mkdtemp(); _write(d, {"leace_native": -0.20, "random_native": 0.0, "hu_native": 0.0, "leace_hcc": 0.0}, -0.20)
    A.main(f"{d}/base.csv", f"{d}/les.csv", f"{d}/org.csv", d); import json; R = json.load(open(f"{d}/m2_results.json"))["decision_primary"]
    assert R["verdict"] == "NOT NECESSARY" and R["erasure_too_blunt"] and R["cond1_lesion_dice_drop"] and R["cond2_beyond_both_controls"]


def test_not_necessary_when_the_controls_drop_as_much():
    d = tempfile.mkdtemp(); _write(d, {"leace_native": -0.10, "random_native": -0.09, "hu_native": -0.10, "leace_hcc": 0.0}, 0.0)
    A.main(f"{d}/base.csv", f"{d}/les.csv", f"{d}/org.csv", d); import json; R = json.load(open(f"{d}/m2_results.json"))["decision_primary"]
    assert R["verdict"] == "NOT NECESSARY" and not R["cond2_beyond_both_controls"]


def test_improves_flag():
    d = tempfile.mkdtemp(); _write(d, {"leace_native": 0.10, "random_native": 0.0, "hu_native": 0.0, "leace_hcc": 0.0}, 0.0)
    A.main(f"{d}/base.csv", f"{d}/les.csv", f"{d}/org.csv", d); import json; R = json.load(open(f"{d}/m2_results.json"))["decision_primary"]
    assert R["erasure_improves"] and R["verdict"] == "NOT NECESSARY"
