import csv
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w2m_route_analyze as A  # noqa: E402


def _gen(d, d133, dpc, organ_drop, n=76, seed=0):
    """Synthetic baselines (W2 v3 style), route lesion rows for two arms, automatic-route rows with an unhooked baseline, and an organ table."""
    rng = np.random.default_rng(seed); base, les, auto, org = [], [], [], []
    for i in range(n):
        case = f"h{i // 1}"; key = {"case": case, "rank": 1}
        for arm in ("hcc_133_1", "hcc_26_3"):
            b = float(np.clip(0.6 + rng.normal(0, 0.05), 0.1, 1)); base.append({"arm": arm, **key, "dice_full": b, "dice_within_10mm": b, "coverage": 0.8, "pred_over_lesion_volume": 1.0, "pred_frac_in_lesion": 0.8, "point_x": 1, "point_y": 1, "point_z": 1})
            for c, dd in (("leace", d133 if arm == "hcc_133_1" else -0.3), ("random", 0.0), ("hu", -0.005)):
                les.append({"arm": arm, "condition": c, **key, "dice_full": float(np.clip(b + dd + rng.normal(0, 0.03), 0, 1)), "dice_within_10mm": float(np.clip(b + dd + rng.normal(0, 0.03), 0, 1)), "coverage": 0.8, "pred_over_lesion_volume": 1.0, "pred_frac_in_lesion": 0.8})
        b0 = float(np.clip(0.55 + rng.normal(0, 0.1), 0.1, 1))
        for c, dd in (("unhooked", 0.0), ("leace", dpc), ("random", 0.0), ("hu", -0.01)):
            auto.append({"arm": "hcc_26_auto", "condition": c, **key, "dice_within_10mm": float(np.clip(b0 + dd + rng.normal(0, 0.03), 0, 1)), "coverage": 0.7})
    for i in range(0, n, 3):
        org.append({"case": f"h{i}", "dice_unhooked": 0.9, "dice_leace": 0.9 + organ_drop + float(rng.normal(0, 0.01)), "dice_random": 0.9, "dice_hu": 0.9})
    for name, rows in (("base.csv", base), ("les.csv", les), ("auto.csv", auto), ("org.csv", org)):
        with open(os.path.join(d, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _run(*a, **k):
    d = tempfile.mkdtemp(); _gen(d, *a, **k); A.main(f"{d}/base.csv", f"{d}/les.csv", f"{d}/auto.csv", f"{d}/org.csv", d); return json.load(open(f"{d}/route_results.json"))["route_verdict"]


def test_organ_differs_when_the_m2_used_criteria_are_met_on_133_1():
    assert _run(-0.20, -0.30, -0.01)["verdict"] == "ORGAN DIFFERS"


def test_route_explains_m2_when_133_1_is_nearly_inert_and_the_positive_control_is_large():
    v = _run(-0.01, -0.30, 0.0); assert v["verdict"] == "ROUTE EXPLAINS M2" and v["positive_control_large_effect"] and v["dD_ci_lower_above_minus_0.10"]


def test_inconclusive_when_the_positive_control_fails():
    v = _run(-0.01, -0.02, 0.0); assert v["verdict"] == "INCONCLUSIVE" and not v["positive_control_large_effect"]


def test_inconclusive_when_erasure_is_too_blunt():
    v = _run(-0.20, -0.30, -0.20); assert v["verdict"] == "INCONCLUSIVE" and v["erasure_too_blunt"] and not v["dD_ci_lower_above_minus_0.10"]


def test_dice_within_10mm_ignores_prediction_beyond_10mm_but_counts_a_nearby_shell():
    from glassbox.seg.w2_lib import dice_within
    g = np.indices((60, 60, 60)).astype(float); lesion = ((g[0] - 30) ** 2 + (g[1] - 30) ** 2 + (g[2] - 30) ** 2) <= 8 ** 2; sp = (1.0, 1.0, 1.0)
    far = lesion.copy(); far[50:58, 50:58, 50:58] = True                                   # a blob more than 10 mm from the lesion does not change the metric
    near = ((g[0] - 30) ** 2 + (g[1] - 30) ** 2 + (g[2] - 30) ** 2) <= 14 ** 2           # a shell 6 mm wide around the lesion lowers it
    assert dice_within(lesion, lesion, sp) == 1.0 and dice_within(far, lesion, sp) == 1.0 and 0.2 < dice_within(near, lesion, sp) < 0.8
