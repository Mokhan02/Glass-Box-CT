import importlib.util
import os

import numpy as np

from glassbox.seg import w6_stats as S

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def synth(n_pat=120, seed=0, b_sign=-2.3, b_phase=np.log(0.15), b_prop=np.log(0.7)):
    rng = np.random.default_rng(seed); pat = np.repeat(np.arange(n_pat), 2); n = len(pat); ca = rng.normal(10, 20, n); cp = rng.normal(0, 20, n); drawn = rng.random(n) < 0.65
    ba, bp = ca > 0, cp > 0; u = rng.normal(0, 1.0, n_pat)[pat]; e = rng.normal(0, 1.1, n); al = 1.1 + u + e
    pa = 1 / (1 + np.exp(-(al + b_phase + b_sign * ba + b_prop * ~drawn))); pp = 1 / (1 + np.exp(-(al + b_sign * bp + b_prop * drawn)))
    return pat, ca, cp, drawn, rng.random(n) < pa, rng.random(n) < pp


def test_theta_matches_the_power_script_estimator_and_recovers_a_penalty():
    spec = importlib.util.spec_from_file_location("pw", os.path.join(HERE, "results", "w4b_waw", "w4b_did_power.py")); PW = importlib.util.module_from_spec(spec); spec.loader.exec_module(PW)
    pat, ca, cp, drawn, da, dp = synth(); f1, nf, f2 = S.classes(ca, cp)
    assert S.theta_hat(da, dp, f1, nf, drawn) == PW.theta_hat(da, dp, f1, nf, drawn)      # the same estimator as the one whose power was simulated
    t = S.theta_ci(pat, da, dp, f1, nf, drawn, "t", B=300); assert t["theta"] > 0.15 and t["lo"] < t["theta"] < t["hi"] and S.label_theta({**t, "lo": 0.05, "theta": 0.3}) == "BRIGHT PENALTY"
    pat0, ca0, cp0, dr0, da0, dp0 = synth(b_sign=0.0, seed=1); f10, nf0, _ = S.classes(ca0, cp0); t0 = S.theta_ci(pat0, da0, dp0, f10, nf0, dr0, "t0", B=300); assert abs(t0["theta"]) < 0.2


def test_paired_difference_uses_shared_resamples_and_labels():
    pat, ca, cp, drawn, da, dp = synth(seed=2, b_sign=-2.3); _, _, _, _, da0, dp0 = synth(seed=2, b_sign=0.0); f1, nf, _ = S.classes(ca, cp)
    r = S.theta_diff_ci(pat, (da0, dp0), (da, dp), f1, nf, drawn, "diff", B=300)      # model 1 has no penalty, model 0 has: d < 0
    assert r["d"] < 0 and np.isfinite(r["lo"]) and S.label_reduces({"d": -0.3, "lo": -0.5, "hi": -0.1}) == "REDUCES THE PENALTY" and S.label_reduces({"d": -0.05, "lo": -0.05, "hi": 0.1}) == "DOES NOT REDUCE" and S.label_reduces({"d": -0.2, "lo": -0.4, "hi": 0.05}) == "INCONCLUSIVE"
    assert S.label_equivalence({"theta": 0.0, "lo": -0.2, "hi": 0.2}) == "NO PENALTY" and S.label_equivalence({"theta": 0.0, "lo": -0.3, "hi": 0.2}) == "INCONCLUSIVE"
    assert S.label_theta({"theta": 0.1, "lo": -0.2, "hi": 0.3}) == "INCONCLUSIVE" and S.label_theta({"theta": -0.3, "lo": -0.5, "hi": -0.1}) == "REVERSED"


def test_gap_ratio_labels_and_not_evaluable():
    rng = np.random.default_rng(3); n = 600; pat = np.repeat(np.arange(300), 2); br = rng.random(n) < 0.7
    plain = np.where(br, rng.random(n) < 0.15, rng.random(n) < 0.95).astype(float); bright = np.where(br, rng.random(n) < 0.85, rng.random(n) < 0.95).astype(float); dec = np.where(br, rng.random(n) < 0.80, rng.random(n) < 0.95).astype(float)
    r = S.gap_ratio(pat, br, plain, dec, bright, "r", B=400); assert r["label"] == "DECODER SUFFICES" and r["R"] > 0.8
    none = S.gap_ratio(pat, br, plain, plain, plain, "r2", B=300); assert none["label"] == "NOT EVALUABLE"      # BRIGHT does not reduce the gap: no ratio
    weak = np.where(br, rng.random(n) < 0.20, rng.random(n) < 0.95).astype(float); r3 = S.gap_ratio(pat, br, plain, weak, bright, "r3", B=400); assert r3["label"] in ("ENCODER NEEDED", "INCONCLUSIVE") and r3["R"] < 0.3


def test_penalised_conditional_logit_runs_and_is_finite():
    pat, ca, cp, drawn, da, dp = synth(seed=4); ba, bp = (ca > 0).astype(float), (cp > 0).astype(float); D = np.column_stack([ba - bp, np.abs(ca) / 10 - np.abs(cp) / 10, (~drawn).astype(float) - drawn.astype(float)])
    r = S.cond_logit_penalised(pat, da, dp, D, ["sign", "abs_contrast", "is_propagated"], "cl", b=100); assert r["status"].startswith("OK") and np.isfinite(r["coef"]["sign"]["OR"]) and r["coef"]["sign"]["OR"] < 1


def test_w6_analyze_end_to_end_on_synthetic_scores(tmp_path):
    import csv
    import json
    spec = importlib.util.spec_from_file_location("w6an", os.path.join(HERE, "scripts", "w6_analyze.py")); AN = importlib.util.module_from_spec(spec); spec.loader.exec_module(AN)
    from glassbox.seg import w4_waw as W
    rng = np.random.default_rng(7); paired = tmp_path / "paired"; paired.mkdir(); tr = []; pairs = []
    for p in range(1, 121):
        for rk in (1, 2):
            ca, cp = rng.normal(10, 20), rng.normal(0, 20); drawn = int(rng.integers(1, 3)); tr.append({"patient": p, "mask_phase": drawn, "target_phase": 3 - drawn, "rank": rk, "diam_mm": 20.0, "c_arterial": ca, "c_portal": cp, "c_arterial_eroded": ca, "c_portal_eroded": cp, "c_arterial_median": ca, "c_portal_median": cp, "category": "K"}); pairs.append((p, rk, ca, cp, drawn))
    with open(paired / "tumours_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tr[0])); w.writeheader(); w.writerows(tr)
    pen = {"SHAM": -2.3, "PLAIN": -2.3, "BRIGHT": 0.0, "BRIGHT-DEC": -0.7}; args = []
    for arm, b in pen.items():
        d = tmp_path / arm; d.mkdir(); rows = []
        for p, rk, ca, cp, drawn in pairs:
            for sp, c in ((1, ca), (2, cp)):
                det = int(rng.random() < 1 / (1 + np.exp(-(1.2 + b * (c > 0) + (-1.0 if sp == 1 else 0)))))
                rows.append({"arm": arm, "patient": p, "series_phase": sp, "drawn_phase": drawn, "is_propagated": int(sp != drawn), "rank_drawn_phase": rk, "label_id": rk, "diam_mm_drawn_phase": 20.0, "diam_mm_series": 20.0, "fov_inside_fraction": 1.0, "n_voxels": 100,
                             "coverage": 0.5 * det, "dice_full": 0.6 * det, "dice_within_10mm": 0.6 * det, "coverage_dil2": 0.5 * det, "detected": det, "detected_dil2": det, "pred_diam_mm": 1, "mode": "OK"})
        with open(d / f"scores_{arm}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=W.FIELDS); w.writeheader(); w.writerows(rows)
        json.dump({"series_gate_fail": [], "per_series": []}, open(d / f"alignment_{arm}.json", "w")); args += ["--arm", f"{arm}={d / f'scores_{arm}.csv'}"]
    AN.main([str(paired), str(tmp_path / "out")] + args + ["--b", "60"]); pr = json.load(open(tmp_path / "out" / "primary.json"))
    assert pr["w4b"]["primary"]["d"] < 0 and pr["w4b"]["models"]["SHAM"]["theta"] > 0 and "Q2_decoder_only_ratio" in pr["w6"] and pr["w6"]["BRIGHT_vs_SHAM (PRIMARY, W6)"]["label"].startswith(("CONFIRMED", "INCONCLUSIVE", "NOT CONFIRMED"))
    assert (tmp_path / "out" / "sensitivity_table.json").exists() and (tmp_path / "out" / "ANALYSIS.md").read_text().startswith("# W4b / W6 analysis")
