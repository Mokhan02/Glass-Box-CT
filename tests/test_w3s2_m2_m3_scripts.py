import csv
import json
import os
import subprocess
import sys

import numpy as np

PRIMARY = [l.strip() for l in open("results/w3_models/stage2/m1/cases_primary74.txt") if l.strip()]


def _run(args):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, check=True)


def _tokens(path, dim=8, seed=0):
    rng = np.random.default_rng(seed); os.makedirs(os.path.join(path, "tokens"), exist_ok=True)
    for c in PRIMARY:
        n = 80; tum = np.r_[np.ones(30, bool), np.zeros(50, bool)]; x = rng.normal(0, 1, (n, dim)).astype(np.float32); x[tum, 0] += 2.5; hu = rng.normal(60, 10, (n, 14)).astype(np.float16)
        np.savez_compressed(os.path.join(path, "tokens", c + ".npz"), mid_x=x.astype(np.float16), mid_is_tum=tum, mid_hu14=hu)


def test_m2_fit_and_analyze_end_to_end_synthetic(tmp_path):
    _tokens(str(tmp_path / "tok")); _run(["scripts/w3s2_m2_fit.py", str(tmp_path / "tok"), "results/w3_models/stage2/m1/cases_primary74.txt", str(tmp_path / "er.npz")])
    run = tmp_path / "run"; run.mkdir(); rng = np.random.default_rng(1); lrows, crows = [], []
    for c in PRIMARY:
        for cond, dsh in (("base", 0.0), ("leace", -0.3), ("random", -0.01), ("hu", -0.02)):
            crows.append({"case": c, "cond": cond, "fold": 0, "organ_dice": 0.95 + (0.0 if cond != "leace" else -0.005) + rng.normal(0, 0.002), "centroid_offset_mm": 1.0, "tumour_voxels": 100})
            for rank in (1, 2):
                lrows.append({"arm": f"A_{cond}", "organ": "hcc", "case": c, "rank": rank, "group": "dark", "stratum": "30-60 mm", "diam_mm": 40.0, "contrast_hu": -5.0, "coverage": 0.8 + dsh, "dice_full": 0.6 + dsh, "dice_within_10mm": 0.6 + dsh + rng.normal(0, 0.03),
                              "pred_diam_mm": 30.0, "pred_over_lesion_volume": 1.0, "pred_frac_in_lesion": 0.9, "mode": "OK"})
    for name, rows in (("lesions.csv", lrows), ("cases.csv", crows)):
        with open(run / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    _run(["scripts/w3s2_m2_analyze.py", "A", str(run), str(tmp_path / "tok"), str(tmp_path / "er.npz"), "results/w3_models/stage2/m1/cases_primary74.txt", str(tmp_path / "out")])
    r = json.load(open(tmp_path / "out" / "m2_results.json")); assert r["verdict"] == "USED" and r["meets_M2"] and r["probe_check"]["post_erasure_auroc"][2] <= 0.60 and r["probe_check"]["pre_erasure_auroc"][0] > 0.9
    assert r["mean_change_lesion_dice"]["leace"][2] < -0.2 and abs(r["mean_change_liver_dice_under_leace"][0]) < 0.05


def test_m3_analyze_end_to_end_synthetic(tmp_path):
    _tokens(str(tmp_path / "tok")); run = tmp_path / "run"; (run / "tokens").mkdir(parents=True); rng = np.random.default_rng(2); rows = []
    for i, c in enumerate(PRIMARY[:42]):
        arm = "A" if i < 30 else "B"; base = 0.7 if arm == "A" else 0.03; edit = 0.2 if arm == "A" else 0.5; ctrl = base
        for cond, cov in (("base", base), ("edit", edit), ("control", ctrl)):
            rows.append({"case": c, "rank": 1, "arm": arm, "condition": cond, "T": 80.0, "L": 90.0, "c": -10.0 if arm == "A" else 10.0, "target": 0, "realised_mean": 0, "flagged": False, "coverage": cov + rng.normal(0, 0.03), "dice_full": 0.5, "dice_within_10mm": cov * 0.8, "pred_diam_mm": 20.0, "n_tokens": 10})
        base_v = rng.normal(0, 1, 8); np.savez_compressed(run / "tokens" / f"{c}_r1.npz", base=base_v, edit=base_v + (np.eye(8)[0] * (-3.0 if arm == "A" else 3.0)), n=10)
    with open(run / "m3_lesions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    _run(["scripts/w3s2_m3_analyze.py", "A", str(run), str(tmp_path / "tok"), "results/w3_models/stage2/m1/cases_primary74.txt", str(tmp_path / "out")])
    r = json.load(open(tmp_path / "out" / "m3_results.json")); assert r["verdict"] == "SUPPORTED" and r["arm_A"]["n"] == 30 and r["arm_B"]["n"] == 12 and r["arm_A"]["median_D"][2] < 0 and r["arm_B"]["median_D"][1] > 0
    assert r["exploratory_projection"]["A"]["label"] == "EXPLORATORY"
