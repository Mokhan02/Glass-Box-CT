"""Re-report the size-binned RECIST results three ways (detection, bias among detected, combined), plus where the sum-of-diameters shortfall comes from.
Input: cache/pulled/recist_lesions.csv (one row per RECIST target lesion; written by scripts/recist_shift.py, 'detected' = predicted voxels cover >= 10% of the
lesion, predicted diameter = largest overlapping predicted component, a miss has predicted diameter 0). Usage: python scripts/exp1_recist_split.py"""
import csv
import json
import os

import numpy as np

rows = list(csv.DictReader(open("cache/pulled/recist_lesions.csv")))
for r in rows:
    r["gt"], r["pred"] = float(r["gt"]), float(r["pred"]); r["det"] = r["detected"] == "True"; r["any"] = r["detected_any"] == "True"
    r["err"] = r["pred"] - r["gt"]; r["rel"] = r["err"] / r["gt"]
cases = sorted({r["case"] for r in rows}); by_case = {c: [r for r in rows if r["case"] == c] for c in cases}
rng = np.random.default_rng(0); B = rng.integers(0, len(cases), size=(1000, len(cases)))
BINS = ((10, 30), (30, 60), (60, 100), (100, 1e9))


def boot(stat, sel_rows):
    """patient(case)-bootstrap 95% CI of stat(list of lesion rows)."""
    idx = {c: [r for r in sel_rows if r["case"] == c] for c in cases}; out = []
    for b in B:
        rs = [r for i in b for r in idx[cases[i]]]
        if rs:
            v = stat(rs)
            if np.isfinite(v):
                out.append(v)
    return tuple(np.percentile(out, [2.5, 97.5])) if len(out) > 50 else (np.nan, np.nan)


med = lambda f: (lambda rs: float(np.median([f(r) for r in rs])) if rs else np.nan)  # noqa: E731
out = []
for lo, hi in BINS:
    sel = [r for r in rows if lo <= r["gt"] < hi]; det = [r for r in sel if r["det"]]
    name = f"{lo}-{'inf' if hi > 1e8 else int(hi)}"
    d = {"bin_mm": name, "n_lesions": len(sel), "n_cases": len({r["case"] for r in sel}),
         "detection_rate": len(det) / len(sel), "detection_ci": boot(lambda rs: np.mean([r["det"] for r in rs]), sel), "detected_any_overlap_rate": float(np.mean([r["any"] for r in sel])),
         "n_detected": len(det), "detected_median_error_mm": med(lambda r: r["err"])(det), "detected_error_mm_ci": boot(lambda rs: np.median([r["err"] for r in rs if r["det"]]) if any(r["det"] for r in rs) else np.nan, sel),
         "detected_median_rel_error": med(lambda r: r["rel"])(det), "detected_share_rel_le_minus30": float(np.mean([r["rel"] <= -0.3 for r in det])) if det else np.nan,
         "combined_median_rel_error_misses_as_minus100": med(lambda r: r["rel"])(sel), "combined_share_rel_le_minus30": float(np.mean([r["rel"] <= -0.3 for r in sel])),
         "share_of_mm_shortfall_from_misses": sum(r["gt"] for r in sel if not r["det"]) / max(sum(r["gt"] - r["pred"] for r in sel), 1e-9)}
    out.append(d)
tot = {"bin_mm": "all", "n_lesions": len(rows), "n_cases": len(cases), "detection_rate": float(np.mean([r["det"] for r in rows])), "detection_ci": boot(lambda rs: np.mean([r["det"] for r in rs]), rows),
       "detected_any_overlap_rate": float(np.mean([r["any"] for r in rows])), "n_detected": sum(r["det"] for r in rows), "detected_median_error_mm": med(lambda r: r["err"])([r for r in rows if r["det"]]),
       "detected_error_mm_ci": boot(lambda rs: np.median([r["err"] for r in rs if r["det"]]) if any(r["det"] for r in rs) else np.nan, rows),
       "detected_median_rel_error": med(lambda r: r["rel"])([r for r in rows if r["det"]]), "detected_share_rel_le_minus30": float(np.mean([r["rel"] <= -0.3 for r in rows if r["det"]])),
       "combined_median_rel_error_misses_as_minus100": med(lambda r: r["rel"])(rows), "combined_share_rel_le_minus30": float(np.mean([r["rel"] <= -0.3 for r in rows])),
       "share_of_mm_shortfall_from_misses": sum(r["gt"] for r in rows if not r["det"]) / sum(r["gt"] - r["pred"] for r in rows)}
out.append(tot)
with open("results/exp1_audit/recist_split_lesions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows([{k: (round(v, 3) if isinstance(v, float) else (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v)) for k, v in d.items()} for d in out])

# Case level: sum of target diameters (RECIST's quantity), three versions
cs = []
for c in cases:
    rs = by_case[c]; gs = sum(r["gt"] for r in rs); ps = sum(r["pred"] for r in rs); dr = [r for r in rs if r["det"]]
    cs.append({"case": c, "gt_sum": gs, "rel_A": (ps - gs) / gs, "all_det": len(dr) == len(rs),
               "rel_B": (sum(r["pred"] for r in dr) - sum(r["gt"] for r in dr)) / sum(r["gt"] for r in dr) if dr else np.nan,
               "miss_mm": sum(r["gt"] for r in rs if not r["det"]), "under_mm": sum(r["gt"] - r["pred"] for r in dr)})
A = np.array([c["rel_A"] for c in cs]); Bv = np.array([c["rel_B"] for c in cs]); gsum = np.array([c["gt_sum"] for c in cs])
q = np.quantile(gsum, [0, 1 / 3, 2 / 3, 1]); case_rows = []
for lab, m in (("all cases", np.ones(len(cs), bool)), *[(f"GT sum {q[i]:.0f}-{q[i + 1]:.0f} mm (tercile {i + 1})", (gsum >= q[i]) & (gsum <= q[i + 1])) for i in range(3)]):
    a, b = A[m], Bv[m][np.isfinite(Bv[m])]
    case_rows.append({"cases": lab, "n": int(m.sum()), "A_median_rel_err_misses_as_0mm": float(np.median(a)), "A_share_abs_ge_20": float((np.abs(a) >= .2).mean()), "A_share_abs_ge_30": float((np.abs(a) >= .3).mean()),
                      "B_n_cases_with_a_detected_target": int(len(b)), "B_median_rel_err_detected_targets_only": float(np.median(b)), "B_share_abs_ge_20": float((np.abs(b) >= .2).mean()), "B_share_abs_ge_30": float((np.abs(b) >= .3).mean()),
                      "share_of_total_mm_shortfall_from_misses": float(sum(c["miss_mm"] for c, mm in zip(cs, m) if mm) / max(sum(c["miss_mm"] + c["under_mm"] for c, mm in zip(cs, m) if mm), 1e-9))})
with open("results/exp1_audit/recist_split_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(case_rows[0])); w.writeheader(); w.writerows([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()} for d in case_rows])
fmt = lambda t: f"[{t[0]:.2f},{t[1]:.2f}]" if np.all(np.isfinite(t)) else "[n/a]"  # noqa: E731
print("LESION LEVEL (target lesions >= 10 mm; detected = predicted voxels cover >= 10% of the lesion)")
print(f"{'bin (mm)':>9s} {'n':>4s} | {'detection rate [95% CI]':>26s} {'(any overlap)':>14s} | {'detected only: median err mm [CI]':>36s} {'median rel':>10s} | {'combined median rel (miss=-100%)':>34s}")
for d in out:
    print(f"{d['bin_mm']:>9s} {d['n_lesions']:4d} | {d['detection_rate']:>8.2f} {fmt(d['detection_ci']):>16s} {d['detected_any_overlap_rate']:>14.2f} | {d['detected_median_error_mm']:>+10.1f} {fmt(d['detected_error_mm_ci']):>16s} (n={d['n_detected']:3d}) {d['detected_median_rel_error']:>+8.2f} | {d['combined_median_rel_error_misses_as_minus100']:>+30.2f}   misses = {d['share_of_mm_shortfall_from_misses']:.0%} of mm shortfall")
print("\nCASE LEVEL: sum of target diameters (A = misses counted as 0 mm, as in the earlier reports; B = detected targets only)")
for r in case_rows:
    print(f"  {r['cases']:42s} n={r['n']:3d} | A median {r['A_median_rel_err_misses_as_0mm']:+.2f}, >=20%: {r['A_share_abs_ge_20']:.2f}, >=30%: {r['A_share_abs_ge_30']:.2f} | B (n={r['B_n_cases_with_a_detected_target']}) median {r['B_median_rel_err_detected_targets_only']:+.2f}, >=20%: {r['B_share_abs_ge_20']:.2f}, >=30%: {r['B_share_abs_ge_30']:.2f} | misses = {r['share_of_total_mm_shortfall_from_misses']:.0%} of mm shortfall")
