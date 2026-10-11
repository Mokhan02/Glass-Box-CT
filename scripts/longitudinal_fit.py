"""Fit the two-timepoint bias model from the real per-lesion measurements (results/trust/lesion_measurements.csv, written by scripts/gated_recal_run.py) and write
glassbox/trust/longitudinal_model.json (what the app uses) and results/trust/LONGITUDINAL.md. The pairs it later generates are SIMULATED; the model JSON says so.
Usage: python scripts/longitudinal_fit.py [--csv results/trust/lesion_measurements.csv] [--out glassbox/trust/longitudinal_model.json] [--fixture]"""
import argparse
import csv
import itertools
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.trust.longitudinal import CATEGORIES, SIMULATED, fit_bias_model, simulate  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--csv", default="results/trust/lesion_measurements.csv"); ap.add_argument("--out", default="glassbox/trust/longitudinal_model.json"); ap.add_argument("--md", default="results/trust/LONGITUDINAL.md"); ap.add_argument("--fixture", action="store_true", help="the CSV is made-up test data: stamp the model as a fixture (the app shows a warning and refuses it in a deployment)"); a = ap.parse_args()
rows = [{"gt": float(r["gt_diam_mm"]), "default": float(r["measured_default_mm"]), "recal": float(r["measured_recal_mm"]), "bright": float(r["contrast_hu"]) > 0} for r in csv.DictReader(open(a.csv))]
model = fit_bias_model(rows); model.update({"simulated": True, "notice": SIMULATED, "fixture": bool(a.fixture), "source": "per-lesion measurements of the NV-Segment-CT hepatic-tumour class on the clean 89-case HCC-TACE-Seg cohort (saved logits; default logit > 0 vs recalibrated logit > -3.0)" if not a.fixture else "TEST FIXTURE (made-up data)"})
APPEAR = {"dark to dark": (False, False), "bright to bright": (True, True), "bright to dark (phase change or response)": (True, False), "dark to bright (phase change)": (False, True)}
grid = []
for base, chg, (ap_name, (bb, bf)) in itertools.product([[25.0], [40.0], [60.0, 30.0], [80.0, 40.0]], [round(x, 2) for x in np.arange(-0.5, 0.41, 0.05)], APPEAR.items()):
    r = simulate(model, base, chg, bb, bf, n=800); grid.append({"baseline_mm": base, "change": chg, "appearance": ap_name, "bright_baseline": bb, "bright_followup": bf, "truth": r["truth"]["category"], "p_differ": r["calibration_dependent"]["probability_default_and_recalibrated_differ"],
                                                                 "wrong_default": r["methods"]["default"]["wrong_probability"], "wrong_recal": r["methods"]["recal"]["wrong_probability"], "modal": r["calibration_dependent"]["modal"]})
LEVELS = (0.0, 0.5, 1.0); share = {}; pdiff = {}
for lv in LEVELS:  # the same scenario grid at each persistence assumption: what share is calibration-dependent, and how often is each threshold wrong
    rr = [simulate(model, g["baseline_mm"], g["change"], g["bright_baseline"], g["bright_followup"], n=600, persistence=lv) for g in grid]
    pdiff[lv] = np.array([r["calibration_dependent"]["probability_default_and_recalibrated_differ"] for r in rr]); share[str(lv)] = {"share_calibration_dependent": float(np.mean([r["calibration_dependent"]["flag"] for r in rr])), "default_wrong_more_often_than_recal": float(np.mean([r["methods"]["default"]["wrong_probability"] > r["methods"]["recal"]["wrong_probability"] for r in rr]))}
model["grid_summary"] = {"n_scenarios": len(grid), "by_persistence": share, "mean_abs_change_in_p_differ_0_to_1": float(np.mean(np.abs(pdiff[0.0] - pdiff[1.0]))), "note": "share of simulated scenarios in which the default and recalibrated thresholds give different RECIST categories in at least 25% of draws"}
flip = sorted((g for g in grid if g["p_differ"] >= 0.25 and g["modal"]["default"] != g["modal"]["recal"]), key=lambda g: -g["p_differ"]); presets, seen = [], set()
for g in flip:  # a few distinct illustrative scenarios where the category depends on calibration, then one where it does not
    key = (g["truth"], g["appearance"], len(g["baseline_mm"]))
    if key not in seen and len(presets) < 3:
        seen.add(key); presets.append(g)
calm = [g for g in grid if g["p_differ"] < 0.05 and g["appearance"] == "dark to dark" and g["modal"]["default"] == g["modal"]["recal"] == g["truth"]]
if calm:
    presets.append(calm[len(calm) // 2])
model["presets"] = [{"label": f"Simulated patient {chr(65 + i)}: true {p['truth']}, {p['appearance']}", "baseline_mm": p["baseline_mm"], "change": p["change"], "bright_baseline": p["bright_baseline"], "bright_followup": p["bright_followup"]} for i, p in enumerate(presets)]
json.dump(model, open(a.out, "w"), indent=1)
d, r = model["methods"]["default"], model["methods"]["recal"]
L = ["# Two-timepoint check: bias model and simulated scenarios (SIMULATED)", "", SIMULATED, "", "Modelling assumption: a lesion's detection draw carries over between visits with probability 0.5 (the `persistence` parameter; 0 = independent misses, which exaggerates category flips; 1 = identical). It is a stated assumption, not a fitted quantity, and is adjustable in the app.", "", f"Bias model fitted on {model['n_lesions']} real per-lesion measurements ({model['n_bright']} in bright-tumour cases). Per threshold: detection = logistic(1, log true size, bright); detected diameter = a + b * true + c * bright + noise.", "",
     "| threshold | detected lesions | median error of detected (mm) | detection coefficients | size model (a, b, c) | noise sd (mm) |", "|---|---|---|---|---|---|"]
for k, m in (("default", d), ("recalibrated", r)):
    L.append(f"| {k} | {m['n_detected']} | {m['median_error_mm_detected']:+.1f} | {', '.join(f'{x:+.2f}' for x in m['detect_coef'])} | {', '.join(f'{x:+.2f}' for x in m['size_coef'])} | {m['size_sd']:.1f} |")
tot = len(grid); L += ["", f"Across {tot} simulated scenarios (true change -50% to +40%, four baselines, four appearance patterns), the share in which the default and recalibrated thresholds give different RECIST categories in at least 25% of draws, at each persistence assumption:", "", "| persistence | share of scenarios that are calibration-dependent | default wrong more often than recalibrated |", "|---|---|---|"]
L += [f"| {float(k):.1f} | {v['share_calibration_dependent']:.0%} | {v['default_wrong_more_often_than_recal']:.0%} |" for k, v in share.items()]
hi, lo, mc = share["0.0"]["share_calibration_dependent"], share["1.0"]["share_calibration_dependent"], model["grid_summary"]["mean_abs_change_in_p_differ_0_to_1"]
L += ["", f"Persistence 0 means a lesion's detection is independent between visits (exaggerates flips); 1 means the same lesion is found or missed in the same way both times. Reality is in between. The overall share moves only moderately across the whole range (from {hi:.0%} at 0 to {lo:.0%} at 1), so the qualitative statement is robust to this assumption: in a substantial share of simulated scenarios the response call depends on the calibration. For an individual scenario the probability that the two thresholds disagree changes by {100 * mc:.0f} percentage points on average between persistence 0 and 1.", "", "Illustrative scenarios shown in the app (persistence 0.5):", ""]
L += [f"- {lab['label']}: baseline {p['baseline_mm']} mm, true change {p['change']:+.0%}; default most likely {p['modal']['default']}, recalibrated most likely {p['modal']['recal']} (differ in {p['p_differ']:.0%} of draws)." for lab, p in zip(model['presets'], presets)]
open(a.md, "w").write("\n".join(L) + "\n"); print("\n".join(L))
