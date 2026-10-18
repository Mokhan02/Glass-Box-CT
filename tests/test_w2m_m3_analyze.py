import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w2m_m3_analyze as A  # noqa: E402


def _runs(d, a_leak_edit, a_leak_ctrl, b_leak_edit, b_leak_ctrl, nb=25, na=60, seed=0):
    rng = np.random.default_rng(seed); rows = []
    def add(arm, i, p_edit, p_ctrl):
        for cond, p in (("baseline", 0.0 if arm == "A" else 1.0), ("control", p_ctrl), ("edit", p_edit)):
            leak = rng.random() < p
            rows.append({"arm": arm, "case": f"{arm}{i}", "rank": 1, "condition": cond, "diam_mm": 30, "contrast_hu": -5 if arm == "A" else 5, "T": 1, "L": 1, "target_mean": 1, "realised_mean": 1, "flagged": False, "mode": "LEAK" if leak else "OK",
                         "coverage": 0.8, "dice_full": 0.5 + (0.0 if leak else 0.2), "pred_over_lesion_volume": 1, "pred_frac_in_lesion": 1, "pred_diam_mm": 1, "n_lesion_tokens": 3, "projection_native_probe": float(rng.normal(0, 1))})
    [add("A", i, a_leak_edit, a_leak_ctrl) for i in range(na)]; [add("B", i, b_leak_edit, b_leak_ctrl) for i in range(nb)]
    with open(os.path.join(d, "runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _run(*a, **k):
    d = tempfile.mkdtemp(); _runs(d, *a, **k); A.main(f"{d}/runs.csv", d); import json; return json.load(open(f"{d}/m3_results.json"))


def test_supported_when_both_arms_move_in_the_hypothesised_direction():
    R = _run(0.9, 0.05, 0.05, 0.9)
    assert R["verdict"]["verdict"] == "SUPPORTED"


def test_partial_when_only_arm_a_moves():
    assert _run(0.9, 0.05, 0.9, 0.9)["verdict"]["verdict"] == "PARTIAL"


def test_not_supported_when_edits_do_nothing_beyond_the_controls():
    R = _run(0.1, 0.1, 0.9, 0.9, na=120, nb=60)
    assert R["verdict"]["verdict"] in ("NOT SUPPORTED", "INCONCLUSIVE") and not R["verdict"]["arm_A_meets"] and not R["verdict"]["arm_B_meets"]


def test_inconclusive_with_a_small_arm():
    assert _run(0.9, 0.05, 0.05, 0.9, nb=6)["verdict"]["verdict"].startswith("INCONCLUSIVE")
