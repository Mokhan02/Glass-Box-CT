import csv
import importlib.util
import json
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(__file__))
spec = importlib.util.spec_from_file_location("w5a4", os.path.join(HERE, "scripts", "w5a4_dose_response.py")); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
FIELDS = ["arm", "case", "rank", "group", "diam_mm", "coverage", "dice_full", "pred_diam_mm"]


def write(path, bright_rate, dark_rate=0.9, n=74, dice_dark=0.7, dice_bright=0.6):
    """One dark and one bright lesion per case; the first round(rate * n) cases detect (deterministic, so the per-arm gaps are exact)."""
    rows = []
    for i in range(n):
        for rank, (g, rate, dc) in enumerate((("dark", dark_rate, dice_dark), ("bright", bright_rate, dice_bright)), 1):
            hit = ((i * 7919) % n) < round(rate * n)       # a fixed pseudo-random order, so different rates nest and cases are mixed
            rows.append({"arm": "D_nnunet", "case": f"C{i:03d}", "rank": rank, "group": g, "diam_mm": 20.0, "coverage": 0.8 if hit else 0.0, "dice_full": dc if hit else 0.0, "pred_diam_mm": 18.0 if hit else 0.0})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return str(path)


def arms(tmp_path, rates):
    return {k: write(tmp_path / f"{k}.csv", r) for k, r in rates.items()}


GOOD = {"D0": 0.20, "D0S2": 0.24, "B25": 0.55, "B50": 0.85, "B50S2": 0.88}


def test_label_rules_in_the_order_stated():
    L = D.label_metric
    assert L(0.6, 0.35, 0.1, (-0.4, -0.1), (0.1, 0.4)) == ("DOSE-RESPONSE", "graded")
    assert L(0.6, 0.15, 0.1, (-0.6, -0.3), (-0.1, 0.2)) == ("DOSE-RESPONSE", "saturating")      # C_high includes 0: saturating, label unchanged (C_high is descriptive)
    assert L(0.6, 0.35, 0.1, (-0.5, 0.05), (0.1, 0.4))[0] == "PARTIAL"                          # between, but C_low includes 0
    assert L(0.6, 0.7, 0.1, (-0.1, 0.3), (0.5, 0.8)) == ("NONE", None)                          # not between
    assert L(0.6, 0.05, 0.1, (-0.8, -0.4), (-0.2, 0.0)) == ("NONE", None)
    assert L(0.30, 0.2, 0.15, (-0.2, -0.01), (0.0, 0.2))[0] == "PARTIAL (not resolvable at this separation)"      # g0 - g5 = 0.15 < 0.20


def test_overall_label():
    assert D.overall({"det": "DOSE-RESPONSE", "dice": "DOSE-RESPONSE"}) == "DOSE-RESPONSE (detection and Dice)"
    assert D.overall({"det": "DOSE-RESPONSE", "dice": "PARTIAL"}) == "DOSE-RESPONSE (detection only)"
    assert D.overall({"det": "NONE", "dice": "DOSE-RESPONSE"}) == "DOSE-RESPONSE (Dice only)"
    assert D.overall({"det": "PARTIAL", "dice": "NONE"}) == "PARTIAL" and D.overall({"det": "NONE", "dice": "NONE"}) == "NONE"


def run(tmp_path, rates, gates=None):
    out = tmp_path / "out"; D.main(str(out), arms(tmp_path, rates), gates or {}); return json.load(open(out / "w5a4_results.json"))


def test_dose_response_end_to_end_on_synthetic_gaps(tmp_path):
    r = run(tmp_path, GOOD)
    assert r["labels"]["overall"] == "DOSE-RESPONSE (detection and Dice)" and r["contrasts"]["det"]["shape"] == "graded"
    c = r["contrasts"]["det"]; assert c["g5"] < c["g25"] < c["g0"] and c["C_low"]["hi"] < 0 and c["C_high"]["lo"] > 0 and c["separation_g0_minus_g5"] > 0.5
    assert r["trend"]["det"]["slope"] < 0 and r["trend"]["det"]["slope_ci"][1] < 0 and abs(r["trend"]["det"]["B25_residual"]) < 0.2
    assert r["seed_variance"]["det"]["abs_gap_D0_minus_D0S2"] == pytest.approx(abs(r["arms"]["D0"]["detection_gap"] - r["arms"]["D0S2"]["detection_gap"]))


def test_saturating_curve_when_b25_matches_full_dose(tmp_path):
    r = run(tmp_path, {"D0": 0.20, "D0S2": 0.24, "B25": 0.85, "B50": 0.85, "B50S2": 0.88})
    assert r["contrasts"]["det"]["label"] == "DOSE-RESPONSE" and r["contrasts"]["det"]["shape"] == "saturating"


def test_none_when_b25_looks_like_the_baseline(tmp_path):
    r = run(tmp_path, {"D0": 0.20, "D0S2": 0.24, "B25": 0.15, "B50": 0.85, "B50S2": 0.88})
    assert r["labels"]["overall"] == "NONE"


def test_failed_gate_makes_the_label_not_evaluable(tmp_path):
    g = tmp_path / "gate.json"; g.write_text(json.dumps({"liver_dice": 0.96, "tumour_dice": 0.5, "pass": False}))
    r = run(tmp_path, GOOD, {"B25": str(g)})
    assert r["labels"]["overall"].startswith("NOT EVALUABLE") and "B25" in r["labels"]["overall"]


def test_arms_must_cover_the_same_lesions(tmp_path):
    a = arms(tmp_path, GOOD); rows = list(csv.DictReader(open(a["B25"]))); rows = rows[:-2]
    with open(a["B25"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with pytest.raises(SystemExit):
        D.main(str(tmp_path / "o"), a, {})


def _log(tmp_path, seed, arm, p, e2=61.5, loss="-0.25"):
    t = tmp_path / "train.log"
    t.write_text(f"W5: seed {seed}; augmentation {arm} p={p}, delta range (10.0, 60.0) HU relative to local liver mean; epochs 1000\n"
                 f"Epoch 0\ntrain_loss {loss}\nval_loss -0.1\nEpoch time: 80.2 s\nEpoch 1\ntrain_loss -0.3\nval_loss -0.2\nEpoch time: {e2} s\n"); return str(t)


def _edits(tmp_path, n_tum, n_edited, delta="35.0"):
    lines = []
    for i in range(n_tum):
        lines.append(f"BRIGHT 1 {1 if i < n_edited else 0} {delta if i < n_edited else ''}\n")
    lines += ["BRIGHT 0 0 \n"] * 5
    p = tmp_path / "edits.log"; p.write_text("".join(lines)); return str(p)


def _check(tmp_path, tag, log, edits):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "w5a4_early_check.py"), tag, log, edits, str(tmp_path / "o.json")], capture_output=True, text=True); return r.returncode, json.load(open(tmp_path / "o.json"))


def test_early_check_bright25_pass_and_the_failure_modes(tmp_path):
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.25), _edits(tmp_path, 100, 23)); assert rc == 0 and j["pass"]
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.25), _edits(tmp_path, 100, 48)); assert rc == 1 and not j["checks"]["transform_active"]         # p = 0.5-like share
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.5), _edits(tmp_path, 100, 23)); assert rc == 1 and not j["checks"]["arm_line_as_expected"]
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261028, "BRIGHT", 0.25), _edits(tmp_path, 100, 23)); assert rc == 1 and not j["checks"]["seed_in_log"]
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.25, e2=95.0), _edits(tmp_path, 100, 23)); assert rc == 1 and not j["checks"]["epoch2_time_within_10pct"]
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.25, loss="nan"), _edits(tmp_path, 100, 23)); assert rc == 1 and not j["checks"]["losses_finite"]
    rc, j = _check(tmp_path, "B25", _log(tmp_path, 20261020, "BRIGHT", 0.25), _edits(tmp_path, 100, 23, delta="75.0")); assert rc == 1 and not j["checks"]["transform_active"]      # a delta outside +10..+60 HU


def test_early_check_d0seed2_needs_no_augmentation_and_the_reserved_seed(tmp_path):
    empty = tmp_path / "empty.log"; empty.write_text("")
    rc, j = _check(tmp_path, "D0S2", _log(tmp_path, 20261121, "None", 0.0), str(empty)); assert rc == 0 and j["pass"]
    rc, j = _check(tmp_path, "D0S2", _log(tmp_path, 20261121, "None", 0.0), str(tmp_path / "missing.log")); assert rc == 0                                                      # a missing edit log means no augmentation ran
    rc, j = _check(tmp_path, "D0S2", _log(tmp_path, 20261121, "None", 0.0), _edits(tmp_path, 10, 3)); assert rc == 1 and not j["checks"]["no_augmentation"]
    rc, j = _check(tmp_path, "D0S2", _log(tmp_path, 20261020, "None", 0.0), str(empty)); assert rc == 1 and not j["checks"]["seed_in_log"]
    rc, j = _check(tmp_path, "D0S2", _log(tmp_path, 20261121, "BRIGHT", 0.5), str(empty)); assert rc == 1 and not j["checks"]["arm_line_as_expected"]


def test_dose_response_without_d0seed2_uses_d0_alone(tmp_path):
    a = arms(tmp_path, GOOD); a.pop("D0S2")
    out = tmp_path / "o4"; D.main(str(out), a, {}); r = json.load(open(out / "w5a4_results.json"))
    assert r["p0_runs"] == ["D0"] and r["labels"]["overall"] == "DOSE-RESPONSE (detection and Dice)"
    c = r["contrasts"]["det"]; assert c["g0"] == pytest.approx(r["arms"]["D0"]["detection_gap"]) and "baseline" in r["seed_variance"]["det"] and "abs_gap_D0_minus_D0S2" not in r["seed_variance"]["det"]
    assert "B50_minus_D0" in r["w5_contrasts_vs_baselines"]["det"] and not any("D0S2" in k for k in r["w5_contrasts_vs_baselines"]["det"])
    assert r["trend"]["det"]["slope"] < 0


def test_dose_response_still_requires_the_other_four_arms(tmp_path):
    a = arms(tmp_path, GOOD); a.pop("D0S2"); a.pop("B50S2")
    with pytest.raises(SystemExit):
        D.main(str(tmp_path / "o5"), a, {})
