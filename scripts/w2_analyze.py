"""W2 Gate 3 analysis (results/w2_kidney/PREREG.md, applied as written). Inputs: w2_kidney_lesions.csv (class 129), w2_hcc_lesions.csv (class 130, primary HCC arm), w2_hcc26_lesions.csv (class 26 + point, sensitivity).
Headline: gap = mean coverage in dark lesions minus mean coverage in bright lesions, case-level bootstrap (1000 resamples, seed 0, percentile 95%).
Verdict in order: GAP (CI lower > 0 and gap >= 0.10); NO MEANINGFUL GAP (CI upper < 0.10); INCONCLUSIVE. Replication = kidney and HCC (class 130) both GAP. Size-adjusted secondary: strata <30/30-60/>60 mm and OLS coverage ~ bright + log(diameter), clustered bootstrap.
Usage: python scripts/w2_analyze.py <dir with the three CSVs> <outdir>   |   python scripts/w2_analyze.py --selftest"""
import csv
import json
import math
import sys

import numpy as np

B, SEED, MARGIN = 1000, 0, 0.10


def load(path):
    out = []
    for r in csv.DictReader(open(path)):
        out.append({"case": r["case"], "rank": int(r["rank"]), "diam": float(r["diam_mm"]), "c": float(r["contrast_hu"]), "cov": float(r["coverage"]), "dice": float(r["dice_within_10mm"]),
                    "det": int(r["detected_cov_ge_0.10"]), "pdiam": float(r["pred_diam_mm"])})
    return out


def groups(rs):
    cases = sorted({r["case"] for r in rs}); by = {c: [r for r in rs if r["case"] == c] for c in cases}
    return cases, by


def boot_idx(n, rng):
    return rng.integers(0, n, (B, n))


def gap_stat(rs, rng, strat=None):
    """Return gap point, CI, and arrays; gap = mean cov (dark) - mean cov (bright), lesion-level means within each resample of cases."""
    cases, by = groups(rs)
    cnt = np.array([[sum(r["cov"] for r in by[c] if (r["c"] > 0) == b) for b in (False, True)] + [sum(1 for r in by[c] if (r["c"] > 0) == b) for b in (False, True)] for c in cases], float)
    pt = cnt[:, 0].sum() / max(cnt[:, 2].sum(), 1) - cnt[:, 1].sum() / max(cnt[:, 3].sum(), 1) if cnt[:, 2].sum() and cnt[:, 3].sum() else float("nan")
    idx = boot_idx(len(cases), rng); s = cnt[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = s[:, 0] / s[:, 2] - s[:, 1] / s[:, 3]
    draws = draws[np.isfinite(draws)]
    return {"gap": float(pt), "lo": float(np.percentile(draws, 2.5)) if len(draws) else float("nan"), "hi": float(np.percentile(draws, 97.5)) if len(draws) else float("nan"),
            "n_bright": int(cnt[:, 3].sum()), "n_dark": int(cnt[:, 2].sum()), "n_cases": len(cases)}, draws


def verdict(g):
    if not np.isfinite(g["gap"]):
        return "INCONCLUSIVE (a group is empty)"
    if g["lo"] > 0 and g["gap"] >= MARGIN:
        return "GAP"
    if g["hi"] < MARGIN:
        return "NO MEANINGFUL GAP"
    return "INCONCLUSIVE"


def ols_bright(rs, rng):
    cases, by = groups(rs)
    def fit(sample):
        X = np.array([[1.0, float(r["c"] > 0), math.log(r["diam"])] for r in sample]); y = np.array([r["cov"] for r in sample])
        return np.linalg.lstsq(X, y, rcond=None)[0][1] if len({x[1] for x in X}) > 1 else float("nan")
    pt = fit(rs); draws = []
    for row in boot_idx(len(cases), rng):
        d = fit([r for i in row for r in by[cases[i]]])
        if np.isfinite(d):
            draws.append(d)
    return {"coef_bright": float(pt), "lo": float(np.percentile(draws, 2.5)), "hi": float(np.percentile(draws, 97.5))}


def recist(rs, rng):
    """Per-case relative error of the sum of up-to-2 target diameters (predicted = largest overlapping predicted component, a miss = 0), split by the sign of the rank-1 lesion."""
    cases, by = groups(rs); rel, sign = [], []
    for c in cases:
        g = sum(r["diam"] for r in by[c]); p = sum(r["pdiam"] for r in by[c]); rel.append((p - g) / g); sign.append(min(by[c], key=lambda r: r["rank"])["c"] > 0)
    rel, sign = np.array(rel), np.array(sign); out = {}
    for name, m in (("all", np.ones(len(rel), bool)), ("bright (rank-1 lesion)", sign), ("dark (rank-1 lesion)", ~sign)):
        if m.sum():
            idx = rng.integers(0, m.sum(), (B, m.sum())); meds = np.median(rel[m][idx], axis=1)
            out[name] = {"median_signed_error": float(np.median(rel[m])), "lo": float(np.percentile(meds, 2.5)), "hi": float(np.percentile(meds, 97.5)), "n_cases": int(m.sum())}
    return out


def summarize(rs, rng):
    res = {}; res["headline"], draws = gap_stat(rs, rng); res["verdict"] = verdict(res["headline"]); res["_draws"] = draws
    res["by_group"] = {n: {"coverage_mean": float(np.mean([r["cov"] for r in rs if (r["c"] > 0) == b])), "dice_within_10mm_mean": float(np.mean([r["dice"] for r in rs if (r["c"] > 0) == b])),
                           "detected_share": float(np.mean([r["det"] for r in rs if (r["c"] > 0) == b])), "n": int(sum((r["c"] > 0) == b for r in rs)),
                           "lesion_diameter_rel_error_median": float(np.median([(r["pdiam"] - r["diam"]) / r["diam"] for r in rs if (r["c"] > 0) == b]))} for n, b in (("dark", False), ("bright", True)) if any((r["c"] > 0) == b for r in rs)}
    res["recist_per_case"] = recist(rs, rng)
    res["size_strata"] = {n: (lambda sub: gap_stat(sub, rng)[0] if sub else {"note": "empty"})([r for r in rs if f(r["diam"])]) for n, f in (("<30 mm", lambda d: d < 30), ("30-60 mm", lambda d: 30 <= d <= 60), (">60 mm", lambda d: d > 60))}
    adj = ols_bright(rs, rng); res["size_adjusted"] = {**adj, "adjusted_gap_equals_minus_coef": -adj["coef_bright"], "adjusted_gap_lo": -adj["hi"], "adjusted_gap_hi": -adj["lo"],
                                                       "adjusted_effect_gone": bool((-adj["hi"] <= 0 <= -adj["lo"]) or np.sign(-adj["coef_bright"]) != np.sign(res["headline"]["gap"]))}   # CI includes 0, or the sign differs from the headline gap (PREREG)
    return res


def main(d, out):
    rng = np.random.default_rng(SEED); arms = {"kidney (class 129)": load(f"{d}/w2_kidney_lesions.csv"), "HCC (class 130, primary)": load(f"{d}/w2_hcc_lesions.csv"), "HCC (class 26 + point, sensitivity)": load(f"{d}/w2_hcc26_lesions.csv")}
    R = {n: summarize(rs, rng) for n, rs in arms.items()}
    k, h = list(R)[0], list(R)[1]
    dk, dh = R[k]["_draws"], R[h]["_draws"]; m = min(len(dk), len(dh)); diff = dk[:m] - dh[:m]
    R["replication"] = {"kidney_verdict": R[k]["verdict"], "hcc130_verdict": R[h]["verdict"], "replicates": bool(R[k]["verdict"] == "GAP" and R[h]["verdict"] == "GAP"),
                        "kidney_minus_hcc_gap_difference_descriptive": {"diff": R[k]["headline"]["gap"] - R[h]["headline"]["gap"], "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}}
    for n in arms:
        R[n].pop("_draws")
    json.dump(R, open(f"{out}/gate3_results.json", "w"), indent=1, default=float)
    f = lambda g: f"{100 * g['gap']:+.1f} pts [{100 * g['lo']:+.1f}, {100 * g['hi']:+.1f}] ({g['n_bright']} bright, {g['n_dark']} dark lesions, {g['n_cases']} cases)"
    L = ["# W2 Gate 3 results (generated by scripts/w2_analyze.py; verdicts per PREREG.md, applied as written)", "",
         "All of C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K). Point prompts give location: this measures appearance-driven under-segmentation given location, not detection. gap = mean coverage dark minus bright (positive = bright covered less).", ""]
    for n in arms:
        r = R[n]; L += [f"## {n}", f"- **Headline gap: {f(r['headline'])}  -> {r['verdict']}**"]
        for g, v in r["by_group"].items():
            L.append(f"- {g}: n {v['n']}, mean coverage {v['coverage_mean']:.3f}, mean Dice(10 mm) {v['dice_within_10mm_mean']:.3f}, detected (cov >= 0.10) {100 * v['detected_share']:.1f}%, median lesion diameter error {100 * v['lesion_diameter_rel_error_median']:+.1f}%")
        for g, v in r["recist_per_case"].items():
            L.append(f"- RECIST sum-of-diameters median signed error, {g}: {100 * v['median_signed_error']:+.1f}% [{100 * v['lo']:+.1f}, {100 * v['hi']:+.1f}] ({v['n_cases']} cases)")
        L.append("- Size strata (secondary): " + "; ".join(f"{s}: " + (f(v) if "gap" in v else v["note"]) for s, v in r["size_strata"].items()))
        a = r["size_adjusted"]; L += [f"- Size-adjusted (coverage ~ bright + log diameter): coefficient on bright {a['coef_bright']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] -> adjusted gap {100 * a['adjusted_gap_equals_minus_coef']:+.1f} pts [{100 * a['adjusted_gap_lo']:+.1f}, {100 * a['adjusted_gap_hi']:+.1f}]"
                                       + ("; **the adjusted effect disappears (CI includes 0 or has the opposite sign to the headline)**" if a["adjusted_effect_gone"] else "; the adjusted effect remains"), ""]
    rp = R["replication"]; d_ = rp["kidney_minus_hcc_gap_difference_descriptive"]
    L += ["## Replication (kidney and HCC class 130 both GAP)", f"- kidney: {rp['kidney_verdict']}; HCC (130): {rp['hcc130_verdict']}; **replicates: {rp['replicates']}**",
          f"- kidney minus HCC gap difference (descriptive only): {100 * d_['diff']:+.1f} pts [{100 * d_['lo']:+.1f}, {100 * d_['hi']:+.1f}]"]
    open(f"{out}/gate3_results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


def selftest():
    import os, tempfile
    rng = np.random.default_rng(1); d = tempfile.mkdtemp()
    for name, gap in (("kidney", 0.25), ("hcc", 0.0), ("hcc26", 0.15)):
        with open(f"{d}/w2_{name}_lesions.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["case", "rank", "diam_mm", "contrast_hu", "coverage", "dice_within_10mm", "detected_cov_ge_0.10", "pred_diam_mm"])
            for i in range(120):
                for rank in (1, 2):
                    c = float(rng.normal(-10, 30)); diam = float(rng.uniform(12, 90)); cov = float(np.clip(0.8 - (gap if c > 0 else 0) + rng.normal(0, 0.1), 0, 1))
                    w.writerow([f"c{i}", rank, diam, c, cov, cov, int(cov >= 0.1), diam * cov])
    main(d, d)
    R = json.load(open(f"{d}/gate3_results.json"))
    assert R["kidney (class 129)"]["verdict"] == "GAP" and R["HCC (class 130, primary)"]["verdict"] == "NO MEANINGFUL GAP", R["replication"]
    assert R["HCC (class 26 + point, sensitivity)"]["headline"]["gap"] > 0.05
    # a NEGATIVE headline gap (bright covered more) with a same-sign adjusted gap whose CI excludes 0 must NOT be flagged as "adjusted effect gone"
    rs = [{"case": f"n{i}", "rank": 1, "diam": 40.0 + (i % 7), "c": (20.0 if i % 2 else -20.0), "cov": (0.9 if i % 2 else 0.7), "dice": 0.5, "det": 1, "pdiam": 40.0} for i in range(160)]
    r = summarize(rs, np.random.default_rng(0)); assert r["headline"]["gap"] < 0 and not r["size_adjusted"]["adjusted_effect_gone"], r["size_adjusted"]
    print("SELFTEST PASSED")


if __name__ == "__main__":
    selftest() if sys.argv[1] == "--selftest" else main(sys.argv[1], sys.argv[2])
