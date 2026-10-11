"""Evaluate the phase heuristic on the cohort (descriptive, pre-registered). Reads results/trust/phase_estimates.csv and results/readout_refit/bright_threshold_lesions.csv.
Writes results/trust/PHASE_RESULTS.md and phase_results.json. Usage: python scripts/phase_stats.py"""
import csv
import json
import sys

sys.path.insert(0, ".")
from glassbox.trust.phase_eval import evaluate  # noqa: E402

est = {r["case"]: r for r in csv.DictReader(open("results/trust/phase_estimates.csv"))}; big = {}
for r in csv.DictReader(open("results/readout_refit/bright_threshold_lesions.csv")):
    if r["case"] not in big or float(r["diam_mm"]) > float(big[r["case"]]["diam_mm"]):
        big[r["case"]] = r
rows = [{"estimate": est[c]["estimate"], "pred_contrast": float(est[c]["pred_contrast_hu"]) if est[c]["pred_contrast_hu"] != "" else float("nan"), "ref_contrast": float(big[c]["contrast_hu"]), "missed": float(big[c]["cov_orig"]) < 0.10} for c in est if c in big]
res = evaluate(rows); json.dump(res, open("results/trust/phase_results.json", "w"), indent=1)
f = lambda x: f"{100 * x:.0f}%"  # noqa: E731
m = res["miss_rate"]; a = res["arterial_vs_ref_contrast_positive"]
L = ["# Contrast-phase heuristic on HCC-TACE-Seg (descriptive; rule and thresholds pre-registered in PREREG.md section A)", "", f"{res['n_cases']} clean-cohort cases; phase estimated from the model's own aorta and portal-vein masks. A research heuristic: no threshold was tuned, nothing is gated on this.", "",
     "Phase estimates: " + ", ".join(f"{k} {v}" for k, v in res["estimates"].items()) + ".", ""]
if a:
    L += [f"**Arterial-like vs reference contrast > 0** ({a['cases']} cases with a definite estimate): agreement {f(a['agreement'])} (95% CI {f(a['agreement_ci'][0])} to {f(a['agreement_ci'][1])}), kappa {a['kappa']:.2f} (CI {a['kappa_ci'][0]:.2f} to {a['kappa_ci'][1]:.2f}); both {a['both']}, arterial-like only {a['x_only']}, bright only {a['y_only']}, neither {a['neither']}.", ""]
L += ["**Largest reference lesion missed (covered < 10% by the default prediction):**", "", "| group | cases | miss rate (95% CI) |", "|---|---|---|"]
for k, lab in (("warning_fires", "blind-spot warning fires"), ("warning_quiet", "warning quiet"), ("arterial_like", "phase arterial-like"), ("not_arterial_like", "phase not arterial-like")):
    L.append(f"| {lab} | {m[k]['n']} | {f(m[k]['rate'])} ({f(m[k]['ci'][0])} to {f(m[k]['ci'][1])}) |")
d = m["difference_fires_minus_quiet"]; L += ["", f"Difference in miss rate, warning fires minus quiet: {100 * d['mean']:+.0f} points (95% CI {100 * d['ci'][0]:+.0f} to {100 * d['ci'][1]:+.0f}).", "", "Limits: one disease, one dataset; agreement is with a physical proxy (contrast sign), not true phase labels, which do not exist here."]
open("results/trust/PHASE_RESULTS.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
