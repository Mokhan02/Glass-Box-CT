"""W2 v3 pre-registered analysis (results/w2_kidney/PREREG_v3.md), applied as written.
Inputs: main-set CSVs w2v2_main_kidney.csv / w2v2_main_hcc.csv (arms kidney_133_1, hcc_26_3, hcc_133_1) and, for the sensitivity, the pilot CSVs w2v2_pilot_*.csv (same code and arms).
Primary: Dice gap = mean Dice(dark) - mean Dice(bright) per primary arm, case-level bootstrap (1000 resamples, seed 0, percentile 95%). Verdict in order: GAP (CI lower > 0 and point >= 0.10),
NO MEANINGFUL GAP (CI upper < 0.10), INCONCLUSIVE. Like-for-like replication = kidney_133_1 and hcc_133_1 both GAP; hcc_26_3 reported separately.
Secondary: (a) LEAK / UNDER / OK rates by brightness, (b) size-adjusted Dice ~ bright + log(diameter) with clustered bootstrap, (c) coverage gap, (d) sensitivity including the pilot lesions.
Usage: python scripts/w2v3_analyze.py <csv dir> <outdir> [--smoke]   (--smoke treats the PILOT CSVs as the main set: a pipeline test only, never a result)"""
import csv
import json
import math
import sys

import numpy as np

B, SEED, MARGIN = 1000, 0, 0.10
PRIMARY = ["kidney_133_1", "hcc_26_3", "hcc_133_1"]


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({"arm": r["arm"], "case": r["case"], "rank": int(r["rank"]), "bright": r["group"] == "bright", "diam": float(r["diam_mm"]), "c": float(r["contrast_hu"]), "cov": float(r["coverage"]),
                    "dice": float(r["dice_full"]), "dice10": float(r["dice_within_10mm"]), "pdiam": float(r["pred_diam_mm"]), "vol": float(r["pred_over_lesion_volume"]), "inside": float(r["pred_frac_in_lesion"])})
    return out


def mode(r):
    """Failure mode, mutually exclusive, in this order: LEAK, UNDER, OK."""
    if r["vol"] >= 3.0 or r["inside"] <= 0.3:
        return "LEAK"
    return "UNDER" if r["cov"] < 0.5 else "OK"


def diff_stat(rs, fn, rng):
    """mean(fn | dark) - mean(fn | bright) with a case-level bootstrap. Returns the stats dict and the bootstrap draws."""
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}
    cnt = np.array([[sum(fn(r) for r in by[c] if not r["bright"]), sum(fn(r) for r in by[c] if r["bright"]), sum(1 for r in by[c] if not r["bright"]), sum(1 for r in by[c] if r["bright"])] for c in cases], float)
    ok = cnt[:, 2].sum() > 0 and cnt[:, 3].sum() > 0
    pt = cnt[:, 0].sum() / cnt[:, 2].sum() - cnt[:, 1].sum() / cnt[:, 3].sum() if ok else float("nan")
    s = cnt[rng.integers(0, len(cases), (B, len(cases)))].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = s[:, 0] / s[:, 2] - s[:, 1] / s[:, 3]
    draws = draws[np.isfinite(draws)]
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if len(draws) else (float("nan"), float("nan"))
    return {"gap": float(pt), "lo": lo, "hi": hi, "n_dark": int(cnt[:, 2].sum()), "n_bright": int(cnt[:, 3].sum()), "n_cases": len(cases)}, draws


def verdict(g):
    if not np.isfinite(g["gap"]):
        return "INCONCLUSIVE (a group is empty)"
    if g["lo"] > 0 and g["gap"] >= MARGIN:
        return "GAP"
    if g["hi"] < MARGIN:
        return "NO MEANINGFUL GAP"
    return "INCONCLUSIVE"


def ols_gap(rs, rng, headline_gap):
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}

    def coef(sample):
        X = np.array([[1.0, float(r["bright"]), math.log(r["diam"])] for r in sample]); y = np.array([r["dice"] for r in sample])
        return np.linalg.lstsq(X, y, rcond=None)[0][1] if 0 < X[:, 1].sum() < len(X) else float("nan")
    pt = coef(rs); draws = [d for d in (coef([r for i in row for r in by[cases[i]]]) for row in rng.integers(0, len(cases), (B, len(cases)))) if np.isfinite(d)]
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)); adj, alo, ahi = -pt, -hi, -lo         # adjusted gap = dark minus bright = minus the bright coefficient
    gone = bool(alo <= 0 <= ahi or np.sign(adj) != np.sign(headline_gap))          # CI includes 0, or opposite sign to the headline gap (PREREG)
    return {"coef_bright": float(pt), "adjusted_gap": float(adj), "adjusted_lo": float(alo), "adjusted_hi": float(ahi), "adjusted_effect_gone": gone}


def strata(rs, rng):
    out = {}
    for n, f in (("<30 mm", lambda d: d < 30), ("30-60 mm", lambda d: 30 <= d <= 60), (">60 mm", lambda d: d > 60)):
        sub = [r for r in rs if f(r["diam"])]; out[n] = diff_stat(sub, lambda r: r["dice"], rng)[0] if sub else {"note": "empty"}
    return out


def analyse(rs, rng):
    res = {}; res["primary_dice_gap"], draws = diff_stat(rs, lambda r: r["dice"], rng); res["verdict"] = verdict(res["primary_dice_gap"])
    res["coverage_gap"] = diff_stat(rs, lambda r: r["cov"], rng)[0]
    res["failure_modes"] = {m: {"dark_rate": float(np.mean([mode(r) == m for r in rs if not r["bright"]])), "bright_rate": float(np.mean([mode(r) == m for r in rs if r["bright"]]))} for m in ("LEAK", "UNDER", "OK")}
    for m in ("LEAK", "UNDER"):
        g = diff_stat(rs, lambda r, m=m: float(mode(r) == m), rng)[0]       # dark minus bright; the reported difference is bright minus dark = -(dark - bright)
        res["failure_modes"][m]["bright_minus_dark"] = {"diff": -g["gap"], "lo": -g["hi"], "hi": -g["lo"]}
    res["size_adjusted"] = ols_gap(rs, rng, res["primary_dice_gap"]["gap"]); res["strata_dice_gap"] = strata(rs, rng)
    res["by_group"] = {n: {"n": int(sum(r["bright"] == b for r in rs)), "coverage": float(np.mean([r["cov"] for r in rs if r["bright"] == b])), "dice": float(np.mean([r["dice"] for r in rs if r["bright"] == b])),
                           "dice_within_10mm": float(np.mean([r["dice10"] for r in rs if r["bright"] == b])), "median_pred_gt_diameter_ratio": float(np.median([r["pdiam"] / r["diam"] for r in rs if r["bright"] == b]))}
                       for n, b in (("dark", False), ("bright", True)) if any(r["bright"] == b for r in rs)}
    return res, draws


def main(d, out, smoke):
    pre = "w2v2_pilot" if smoke else "w2v2_main"
    rows = [r for f in (f"{pre}_kidney.csv", f"{pre}_hcc.csv") for r in load(f"{d}/{f}") if r["arm"] in PRIMARY]
    pilot = [r for f in ("w2v2_pilot_kidney.csv", "w2v2_pilot_hcc.csv") for r in load(f"{d}/{f}") if r["arm"] in PRIMARY]
    rng = np.random.default_rng(SEED); R = {"smoke_test_only": smoke}; draws = {}
    for arm in PRIMARY:
        rs = [r for r in rows if r["arm"] == arm]; R[arm], draws[arm] = analyse(rs, rng)
        R[arm]["sensitivity_including_pilot"] = analyse(rs + ([] if smoke else [r for r in pilot if r["arm"] == arm]), rng)[0]
    k, h = draws["kidney_133_1"], draws["hcc_133_1"]; m = min(len(k), len(h)); diff = k[:m] - h[:m]
    R["replication_like_for_like"] = {"kidney_133_1": R["kidney_133_1"]["verdict"], "hcc_133_1": R["hcc_133_1"]["verdict"], "replicates": bool(R["kidney_133_1"]["verdict"] == "GAP" and R["hcc_133_1"]["verdict"] == "GAP"),
                                      "hcc_26_3_reported_separately": R["hcc_26_3"]["verdict"], "kidney_minus_hcc_matched_gap_difference_descriptive": {"diff": R["kidney_133_1"]["primary_dice_gap"]["gap"] - R["hcc_133_1"]["primary_dice_gap"]["gap"], "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}}
    json.dump(R, open(f"{out}/w2v3_results{'_SMOKE' if smoke else ''}.json", "w"), indent=1, default=float)
    f = lambda g: f"{100 * g['gap']:+.1f} pts [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}] ({g['n_bright']} bright, {g['n_dark']} dark, {g['n_cases']} cases)"
    L = ["# W2 v3 results (generated by scripts/w2v3_analyze.py, applied as written)" if not smoke else "# SMOKE TEST ONLY: pipeline run on the PILOT predictions, NOT a result", "",
         "All of C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K). Point prompts give location. Dice gap = mean Dice dark minus bright (positive = bright segmented worse).", ""]
    for arm in PRIMARY:
        r = R[arm]; L += [f"## {arm}", f"- **Primary Dice gap: {f(r['primary_dice_gap'])} -> {r['verdict']}**"]
        for g, v in r["by_group"].items():
            L.append(f"- {g}: n {v['n']}, mean Dice {v['dice']:.3f}, Dice within 10 mm {v['dice_within_10mm']:.3f}, coverage {v['coverage']:.3f}, median pred/GT diameter ratio {v['median_pred_gt_diameter_ratio']:.2f}")
        L.append("- (a) failure modes (dark -> bright): " + "; ".join(f"{m} {100 * v['dark_rate']:.0f}% -> {100 * v['bright_rate']:.0f}%" for m, v in r["failure_modes"].items())
                 + "; bright-minus-dark LEAK " + (lambda x: f"{100 * x['diff']:+.1f} pts [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]")(r["failure_modes"]["LEAK"]["bright_minus_dark"])
                 + ", UNDER " + (lambda x: f"{100 * x['diff']:+.1f} pts [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]")(r["failure_modes"]["UNDER"]["bright_minus_dark"]))
        a = r["size_adjusted"]; L.append(f"- (b) size-adjusted Dice gap {100 * a['adjusted_gap']:+.1f} pts [{100 * a['adjusted_lo']:+.1f}, {100 * a['adjusted_hi']:+.1f}]" + ("; **the adjusted effect disappears (CI includes 0 or opposite sign to the headline)**" if a["adjusted_effect_gone"] else "; the adjusted effect remains"))
        L.append("- strata Dice gap: " + "; ".join(f"{s}: " + (f(v) if "gap" in v else v["note"]) for s, v in r["strata_dice_gap"].items()))
        L.append(f"- (c) coverage gap: {f(r['coverage_gap'])}")
        s = r["sensitivity_including_pilot"]; L += [f"- (d) including pilot lesions: Dice gap {f(s['primary_dice_gap'])} -> {s['verdict']}; coverage gap {f(s['coverage_gap'])}", ""]
    rp = R["replication_like_for_like"]; d_ = rp["kidney_minus_hcc_matched_gap_difference_descriptive"]
    L += ["## Like-for-like replication (kidney 133/1 and HCC 133/1 both GAP)", f"- kidney 133/1: {rp['kidney_133_1']}; HCC 133/1: {rp['hcc_133_1']}; **replicates: {rp['replicates']}**", f"- HCC 26/3 (reported separately): {rp['hcc_26_3_reported_separately']}",
          f"- kidney minus HCC matched gap difference (descriptive): {100 * d_['diff']:+.1f} pts [{100 * d_['lo']:+.1f}, {100 * d_['hi']:+.1f}]"]
    open(f"{out}/w2v3_results{'_SMOKE' if smoke else ''}.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--smoke" in sys.argv)
