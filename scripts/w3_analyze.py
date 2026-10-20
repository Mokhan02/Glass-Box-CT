"""W3 Stage 1 pre-registered analysis (results/w3_models/PREREG_stage1.md), applied as written. Statistic helpers come from scripts/w2v3_analyze.py (case-level bootstrap, 1000 resamples, seed 0, percentile CIs).
Input: per-lesion CSVs written by scripts/w3_modelA_run.py and scripts/w3_vista_ref_run.py (one row per lesion and arm; columns arm, organ, case, rank, group, stratum, diam_mm, contrast_hu, coverage, dice_full,
dice_within_10mm, pred_diam_mm, pred_over_lesion_volume, pred_frac_in_lesion).
Per arm: detection gap (detection = coverage >= 0.10) and Dice gap (dark minus bright), each with the margin-0.10 verdict, the model-level label, size-adjusted secondaries (OLS Dice, unpenalised logistic detection),
strata, coverage / Dice-within-10-mm gaps, per-case RECIST sum-of-diameters error (largest overlapping predicted component, a miss counts 0 mm), detected-lesion diameter error.
Model A: if the shipped and raw variants disagree, the cumulative post-processing variants name the step with the largest change in each gap. Reference row: VISTA-3D class 26 reproduced against cache/pulled/recist_lesions.csv.
Usage: python scripts/w3_analyze.py <outdir> <lesions csv> [<lesions csv> ...] [--reference-table cache/pulled/recist_lesions.csv --uid-table results/exp8_lesions/lesion_table.csv] [--exclude case1,case2]"""
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w2v3_analyze as W  # noqa: E402

B, SEED, MARGIN, DET_COV = W.B, W.SEED, W.MARGIN, 0.10
PRIMARY_MODELS = {"A": "A_swin_shipped", "D": "D_nnunet"}
STEP_ORDER = ["A_swin_raw", "A_swin_step_a", "A_swin_step_ab", "A_swin_shipped"]      # raw -> (a) -> (a)+(b) -> (a)+(b)+(c) = shipped
REF_ARM = "vista_26_auto"
# reference-row tolerances (fixed in the PREREG before the run): per-lesion detection flags differ on at most 2 of 86 lesions; the UID-route case-level median sum-of-diameters error within 1.5 points of -19.2%
# (the stored value on the same 75 cases); the detected-lesion median diameter error within 1.0 mm of the stored value on the same lesions
REF_TOL = {"max_flag_disagreements": 2, "sld_median_pts": 0.015, "detected_median_mm": 1.0}


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            rows.append({"arm": r["arm"], "organ": r["organ"], "case": r["case"], "rank": int(r["rank"]), "bright": r["group"] == "bright", "diam": float(r["diam_mm"]), "c": float(r["contrast_hu"]),
                         "cov": float(r["coverage"]), "dice": float(r["dice_full"]), "dice10": float(r["dice_within_10mm"]), "pdiam": float(r["pred_diam_mm"]), "vol": float(r["pred_over_lesion_volume"]),
                         "inside": float(r["pred_frac_in_lesion"])})
    return rows


def det(r):
    return float(r["cov"] >= DET_COV)


def model_label(det_verdict, dice_verdict):
    """The PREREG model-level label from the two per-metric verdicts."""
    g = (det_verdict == "GAP", dice_verdict == "GAP")
    if all(g):
        return "GAP (detection and Dice)"
    if g[1]:
        return "GAP (Dice only)"
    if g[0]:
        return "GAP (detection only)"
    if det_verdict == "NO MEANINGFUL GAP" and dice_verdict == "NO MEANINGFUL GAP":
        return "NO MEANINGFUL GAP (both)"
    return "INCONCLUSIVE"


def logit_coef(rs):
    """Unpenalised logistic regression of detection on [1, bright, log(diameter)] by Newton-Raphson; returns the `bright` coefficient, or nan on separation / non-convergence."""
    X = np.array([[1.0, float(r["bright"]), math.log(r["diam"])] for r in rs]); y = np.array([det(r) for r in rs])
    if X[:, 1].sum() in (0, len(X)) or y.sum() in (0, len(y)):
        return float("nan")
    b = np.zeros(3)
    with np.errstate(all="ignore"):
        return _newton(X, y, b)


def _newton(X, y, b):
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-10 * np.eye(3)
        step = np.linalg.solve(H, X.T @ (y - p)); b = b + step
        if np.abs(step).max() < 1e-8:
            return float(b[1]) if np.abs(b).max() < 15 else float("nan")
    return float("nan")


def logit_adjusted(rs, rng, headline_gap):
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}
    pt = logit_coef(rs); draws = [d for d in (logit_coef([r for i in row for r in by[cases[i]]]) for row in rng.integers(0, len(cases), (B, len(cases)))) if np.isfinite(d)]
    if not np.isfinite(pt) or len(draws) < 100:
        return {"note": "not estimable (separation or too few valid resamples)", "n_valid_resamples": len(draws)}
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)); adj, alo, ahi = -pt, -hi, -lo         # positive adjusted effect = bright detected less often (same sign as the detection gap)
    return {"coef_bright_logit": pt, "adjusted_effect_logit": adj, "adjusted_lo": alo, "adjusted_hi": ahi, "n_valid_resamples": len(draws), "adjusted_effect_gone": bool(alo <= 0 <= ahi or np.sign(adj) != np.sign(headline_gap))}


def recist(rs, rng):
    """Per case: sum of GT diameters vs sum of predicted diameters (largest overlapping predicted component; a miss counts 0 mm). Median relative error over cases with a case bootstrap, plus detected-lesion errors."""
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}
    rel = np.array([(sum(r["pdiam"] for r in by[c]) - sum(r["diam"] for r in by[c])) / sum(r["diam"] for r in by[c]) for c in cases])
    draws = np.array([np.median(rel[row]) for row in rng.integers(0, len(cases), (B, len(cases)))])
    d = [r for r in rs if r["cov"] >= DET_COV]; err = np.array([r["pdiam"] - r["diam"] for r in d]); out = {"n_cases": len(cases), "sld_median_rel_error": float(np.median(rel)), "sld_lo": float(np.percentile(draws, 2.5)), "sld_hi": float(np.percentile(draws, 97.5)),
                                                                                                           "share_abs_ge_20": float(np.mean(np.abs(rel) >= 0.2)), "share_over_ge_20": float(np.mean(rel >= 0.2)), "n_detected": len(d)}
    if len(d):
        out.update({"detected_median_error_mm": float(np.median(err)), "detected_mean_error_mm": float(err.mean()), "detected_median_rel_error": float(np.median([r["pdiam"] / r["diam"] - 1 for r in d])), "detected_n_over_estimates": int((err > 0).sum())})
    out["by_group"] = {}
    for name, b in (("dark", False), ("bright", True)):
        g = [r for r in rs if r["bright"] == b]; gd = [r for r in g if r["cov"] >= DET_COV]
        if g:
            out["by_group"][name] = {"n": len(g), "median_rel_error_lesion_level_misses_as_0mm": float(np.median([r["pdiam"] / r["diam"] - 1 for r in g])),
                                     "detected_median_error_mm": float(np.median([r["pdiam"] - r["diam"] for r in gd])) if gd else None}
    return out


def analyse(rs, rng):
    res = {}
    res["detection_gap"], _ = W.diff_stat(rs, det, rng); res["detection_verdict"] = W.verdict(res["detection_gap"])
    res["dice_gap"], _ = W.diff_stat(rs, lambda r: r["dice"], rng); res["dice_verdict"] = W.verdict(res["dice_gap"])
    res["label"] = model_label(res["detection_verdict"], res["dice_verdict"])
    res["coverage_gap"], _ = W.diff_stat(rs, lambda r: r["cov"], rng); res["dice10_gap"], _ = W.diff_stat(rs, lambda r: r["dice10"], rng)
    res["size_adjusted_dice"] = W.ols_gap(rs, rng, res["dice_gap"]["gap"]); res["size_adjusted_detection"] = logit_adjusted(rs, rng, res["detection_gap"]["gap"])
    res["strata"] = {}
    for n, f in (("<30 mm", lambda d: d < 30), ("30-60 mm", lambda d: 30 <= d <= 60), (">60 mm", lambda d: d > 60)):
        sub = [r for r in rs if f(r["diam"])]
        res["strata"][n] = {"detection_gap": W.diff_stat(sub, det, rng)[0], "dice_gap": W.diff_stat(sub, lambda r: r["dice"], rng)[0]} if sub else {"note": "empty"}
    res["by_group"] = {n: {"n": int(sum(r["bright"] == b for r in rs)), "detection": float(np.mean([det(r) for r in rs if r["bright"] == b])), "dice": float(np.mean([r["dice"] for r in rs if r["bright"] == b])),
                           "coverage": float(np.mean([r["cov"] for r in rs if r["bright"] == b]))} for n, b in (("dark", False), ("bright", True)) if any(r["bright"] == b for r in rs)}
    res["recist"] = recist(rs, rng)
    return res


def step_attribution(R):
    """Cumulative variants raw -> (a) -> (a)+(b) -> (a)+(b)+(c) = shipped: change of each gap between consecutive variants; the step with the largest absolute change is named for each gap."""
    if not all(a in R for a in STEP_ORDER):
        return {"note": "cumulative variants not all present"}
    names = {"A_swin_step_a": "(a) intersect with the largest predicted-liver component", "A_swin_step_ab": "(b) closing", "A_swin_shipped": "(c) dilation"}; out = {"changes": []}
    for prev, cur in zip(STEP_ORDER[:-1], STEP_ORDER[1:]):
        out["changes"].append({"step": names[cur], "delta_detection_gap": R[cur]["detection_gap"]["gap"] - R[prev]["detection_gap"]["gap"], "delta_dice_gap": R[cur]["dice_gap"]["gap"] - R[prev]["dice_gap"]["gap"]})
    for m in ("detection", "dice"):
        out[f"largest_{m}"] = max(out["changes"], key=lambda c: abs(c[f"delta_{m}_gap"]))["step"]
    return out


def disagree(R, a="A_swin_shipped", b="A_swin_raw"):
    """True if the primary and the raw variant differ in the model-level label or in either per-metric verdict category."""
    return a in R and b in R and (R[a]["label"] != R[b]["label"] or R[a]["detection_verdict"] != R[b]["detection_verdict"] or R[a]["dice_verdict"] != R[b]["dice_verdict"])


def reference_check(rows, ref_csv, uid_csv, rng):
    """Reproduce the stored VISTA-3D class-26 numbers on the UID-route lesions: per-lesion detection flags and predicted diameters against cache/pulled/recist_lesions.csv, and the UID-route summary."""
    uid = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open(uid_csv)) if r["route"] == "uid"}
    stored = [r for r in csv.DictReader(open(ref_csv))]; new = {(r["case"], r["rank"]): r for r in rows}; m = []
    for s in stored:
        for (c, rk), u in uid.items():
            if c == s["case"] and abs(float(u["diam_mm"]) - float(s["gt"])) < 0.01 and (c, rk) in new:
                m.append((s, new[(c, rk)]))
    flags = [(s["detected"] == "True") != (n["cov"] >= DET_COV) for s, n in m]; ds = [(s, n) for s, n in m if s["detected"] == "True"]
    st_rows = [{"case": s["case"], "diam": float(s["gt"]), "pdiam": float(s["pred"]), "cov": float(s["cover"])} for s, _ in m]
    rel = lambda rs: float(np.median([(sum(r["pdiam"] for r in rs if r["case"] == c) - sum(r["diam"] for r in rs if r["case"] == c)) / sum(r["diam"] for r in rs if r["case"] == c) for c in sorted({r["case"] for r in rs})]))      # noqa: E731
    new_rows = [n for _, n in m]; stored_sld = rel(st_rows); new_sld = rel(new_rows)
    stored_med = float(np.median([float(s["pred"]) - float(s["gt"]) for s, _ in ds])); new_med = float(np.median([n["pdiam"] - n["diam"] for n in new_rows if n["cov"] >= DET_COV]))
    out = {"n_lesions_matched": len(m), "n_flag_disagreements": int(sum(flags)), "max_abs_coverage_difference": float(max(abs(float(s["cover"]) - n["cov"]) for s, n in m)) if m else None,
           "stored_detected": int(sum(s["detected"] == "True" for s, _ in m)), "new_detected": int(sum(n["cov"] >= DET_COV for n in new_rows)), "stored_sld_median": stored_sld, "new_sld_median": new_sld,
           "stored_detected_median_error_mm": stored_med, "new_detected_median_error_mm": new_med, "tolerances": REF_TOL}
    out["reproduced"] = bool(len(m) == 86 and out["n_flag_disagreements"] <= REF_TOL["max_flag_disagreements"] and abs(new_sld - stored_sld) <= REF_TOL["sld_median_pts"] and abs(new_med - stored_med) <= REF_TOL["detected_median_mm"])
    return out


def main(out, csvs, ref=None, uid=None, exclude=()):
    rows = [r for r in load(csvs) if r["case"] not in exclude]; tag = "_sensitivity_excl_alignment_failures" if exclude else ""; arms = sorted({r["arm"] for r in rows}); rng = np.random.default_rng(SEED); R = {}
    for arm in arms:
        R[arm] = analyse([r for r in rows if r["arm"] == arm], rng)
    R["_primary_models"] = {k: (R[v]["label"] if v in R else "not run") for k, v in PRIMARY_MODELS.items()}
    R["_replication_both_gap_detection_and_dice"] = bool(all(v == "GAP (detection and Dice)" for v in R["_primary_models"].values()))
    R["_A_shipped_vs_raw_disagree"] = disagree(R); R["_A_step_attribution"] = step_attribution(R) if R["_A_shipped_vs_raw_disagree"] or all(a in R for a in STEP_ORDER) else {"note": "not applicable"}
    if ref and uid and REF_ARM in R:
        R["_reference_check"] = reference_check([r for r in rows if r["arm"] == REF_ARM], ref, uid, rng)
    json.dump(R, open(os.path.join(out, f"w3_results{tag}.json"), "w"), indent=1, default=float)
    f = lambda g: f"{100 * g['gap']:+.1f} pts [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}] ({g['n_bright']} bright, {g['n_dark']} dark, {g['n_cases']} cases)"      # noqa: E731
    L = ["# W3 Stage 1 results (generated by scripts/w3_analyze.py, applied as written)", "", "Gaps = dark minus bright (positive = bright segmented worse); detection = coverage >= 0.10; verdict margin 0.10 (10 percentage points for detection).", ""]
    for arm in arms:
        r = R[arm]; L += [f"## {arm}", f"- **Detection gap: {f(r['detection_gap'])} -> {r['detection_verdict']}**", f"- **Dice gap: {f(r['dice_gap'])} -> {r['dice_verdict']}**", f"- **Model-level label: {r['label']}**"]
        for g, v in r["by_group"].items():
            L.append(f"- {g}: n {v['n']}, detection {100 * v['detection']:.1f}%, mean Dice {v['dice']:.3f}, coverage {v['coverage']:.3f}")
        a = r["size_adjusted_dice"]; L.append(f"- size-adjusted Dice gap {100 * a['adjusted_gap']:+.1f} pts [{100 * a['adjusted_lo']:+.1f}, {100 * a['adjusted_hi']:+.1f}]" + ("; **the adjusted effect disappears**" if a["adjusted_effect_gone"] else "; the adjusted effect remains"))
        a = r["size_adjusted_detection"]
        L.append("- size-adjusted detection (logit, positive = bright detected less): " + (f"{a['adjusted_effect_logit']:+.2f} [{a['adjusted_lo']:+.2f}, {a['adjusted_hi']:+.2f}]" + ("; **the adjusted effect disappears**" if a["adjusted_effect_gone"] else "; the adjusted effect remains") if "coef_bright_logit" in a else a["note"]))
        L.append("- strata (detection gap; Dice gap): " + "; ".join(f"{s}: " + (f"{100 * v['detection_gap']['gap']:+.1f}; {100 * v['dice_gap']['gap']:+.1f} pts (n {v['detection_gap']['n_bright']} bright, {v['detection_gap']['n_dark']} dark)" if "dice_gap" in v else v["note"]) for s, v in r["strata"].items()))
        L.append(f"- coverage gap {f(r['coverage_gap'])}; Dice within 10 mm gap {f(r['dice10_gap'])}")
        c = r["recist"]; L.append(f"- RECIST sum of diameters: median case error {100 * c['sld_median_rel_error']:+.1f}% [{100 * c['sld_lo']:+.1f}, {100 * c['sld_hi']:+.1f}] over {c['n_cases']} cases; detected lesions (n {c['n_detected']}): median error {c.get('detected_median_error_mm', float('nan')):+.1f} mm; by group: "
                                  + "; ".join(f"{g} median lesion-level {100 * v['median_rel_error_lesion_level_misses_as_0mm']:+.0f}%" for g, v in c["by_group"].items()) + "")
        L.append("")
    L += ["## Primary models", *[f"- {k}: {v}" for k, v in R["_primary_models"].items()], f"- replication (both A and D 'GAP (detection and Dice)'): {R['_replication_both_gap_detection_and_dice']}", "",
          f"## Model A: shipped vs raw\n- disagree: {R['_A_shipped_vs_raw_disagree']}", f"- step attribution: {json.dumps(R['_A_step_attribution'], default=float)}"]
    if "_reference_check" in R:
        L += ["", "## Reference row (VISTA-3D class 26, automatic)", f"- {json.dumps(R['_reference_check'], default=float)}"]
    open(os.path.join(out, f"w3_results{tag}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    ref = a[a.index("--reference-table") + 1] if "--reference-table" in a else None; uid = a[a.index("--uid-table") + 1] if "--uid-table" in a else None
    paths = [x for i, x in enumerate(a[1:], 1) if x.endswith(".csv") and a[i - 1] not in ("--reference-table", "--uid-table", "--exclude")]
    main(a[0], paths, ref, uid)
    if "--exclude" in a:                       # sensitivity: drop the (at most 2) cases that failed the alignment check; the primary above keeps them
        main(a[0], paths, ref, uid, tuple(a[a.index("--exclude") + 1].split(",")))
