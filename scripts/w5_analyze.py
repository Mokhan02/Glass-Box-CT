"""W5 pre-registered analysis (results/w5_training/PREREG.md, as written; single-seed arms). Reads the per-lesion CSVs the committed Stage 1 scorer wrote for D0 (model D) and for each W5 arm, restricted to the primary arm label D_nnunet
(the HCC cohort 74 cases / 85 lesions; HCC_019 scored against its corrected label as in W3).
Per arm: detection gap and Dice gap (dark minus bright), mean Dice and median signed case sum-of-diameters error. Contrasts: Delta = gap(BRIGHT-50) - gap(comparator) for comparator D0 and SHAM-50, paired case-level bootstrap (the same resampled
cases for all arms in a replicate; 1000 resamples; seed = first 8 hex digits of SHA-256 of 'W5_<contrast>_<metric>'), 95% percentile CIs. Verdict per metric and comparator (margin 0.10): CLOSES if CI upper < 0 and point <= -0.10; DOES NOT CLOSE if CI lower > -0.10;
INCONCLUSIVE otherwise. Overall label for the prediction (BRIGHT-50 closes the gap vs both D0 and SHAM-50): CONFIRMED (detection and Dice) if all four cells CLOSE; CONFIRMED (detection only) / (Dice only) if only that metric CLOSES against both comparators;
NOT CONFIRMED if, against either comparator, both metrics DO NOT CLOSE; INCONCLUSIVE otherwise. An arm whose sanity gate failed is excluded (label NOT EVALUABLE). D0 versus SHAM-50 is reported descriptively (procedure effect), without a verdict.
Usage: python scripts/w5_analyze.py <outdir> --arm D0=<csv> --arm B50=<csv> --arm S50=<csv> [--gate B50=<gate json> --gate S50=<gate json>]"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

B, MARGIN, DET_COV, ARM_LABEL = 1000, 0.10, 0.10, "D_nnunet"
LABEL = "SINGLE-SEED: one training run per arm; training-run variance is not measured and the patient bootstrap does not cover it"


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        if r["arm"] != ARM_LABEL:
            continue
        rows[(r["case"], int(r["rank"]))] = {"case": r["case"], "bright": r["group"] == "bright", "det": float(float(r["coverage"]) >= DET_COV), "dice": float(r["dice_full"]), "diam": float(r["diam_mm"]), "pdiam": float(r["pred_diam_mm"])}
    return rows


def case_table(rows, cases):
    """Per case: [det dark, det bright, n dark, n bright, dice dark, dice bright]."""
    t = np.zeros((len(cases), 6))
    for i, c in enumerate(cases):
        for r in (r for k, r in rows.items() if k[0] == c):
            j = 1 if r["bright"] else 0
            t[i, j] += r["det"]; t[i, 2 + j] += 1; t[i, 4 + j] += r["dice"]
    return t


def gaps(t):
    s = np.atleast_2d(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s[..., 0] / s[..., 2] - s[..., 1] / s[..., 3], s[..., 4] / s[..., 2] - s[..., 5] / s[..., 3]


def ci(d):
    d = d[np.isfinite(d)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [float("nan")] * 2


def verdict(point, lo, hi):
    if hi < 0 and point <= -MARGIN:
        return "CLOSES"
    if lo > -MARGIN:
        return "DOES NOT CLOSE"
    return "INCONCLUSIVE"


def overall(v):
    """v[(comparator, metric)] -> verdict string, comparators D0 and S50, metrics det and dice."""
    cl = lambda m: all(v[(c, m)] == "CLOSES" for c in ("D0", "S50"))      # noqa: E731
    if cl("det") and cl("dice"):
        return "CONFIRMED (detection and Dice)"
    if cl("det"):
        return "CONFIRMED (detection only)"
    if cl("dice"):
        return "CONFIRMED (Dice only)"
    if any(v[(c, "det")] == "DOES NOT CLOSE" and v[(c, "dice")] == "DOES NOT CLOSE" for c in ("D0", "S50")):
        return "NOT CONFIRMED"
    return "INCONCLUSIVE"


def median_sld_error(rows, cases):
    rel = []
    for c in cases:
        rs = [r for k, r in rows.items() if k[0] == c]
        rel.append((sum(r["pdiam"] for r in rs) - sum(r["diam"] for r in rs)) / sum(r["diam"] for r in rs))
    return float(np.median(rel))


def main(outdir, arms, gates):
    os.makedirs(outdir, exist_ok=True); data = {k: load(p) for k, p in arms.items()}
    keys = {k: set(v) for k, v in data.items()}
    if len({frozenset(s) for s in keys.values()}) != 1:
        raise SystemExit("arms do not cover the same lesions: " + json.dumps({k: len(s) for k, s in keys.items()}))
    cases = sorted({k[0] for k in next(iter(data.values()))}); tabs = {k: case_table(v, cases) for k, v in data.items()}
    res = {"label": LABEL, "n_cases": len(cases), "n_lesions": len(next(iter(data.values()))), "arms": {}, "gates": {}, "contrasts": {}, "verdicts": {}}
    evaluable = {k: True for k in data}
    for k, g in gates.items():
        gj = json.load(open(g)); res["gates"][k] = {"liver_dice": gj["liver_dice"], "tumour_dice": gj["tumour_dice"], "pass": gj["pass"]}; evaluable[k] = bool(gj["pass"])
    for k, t in tabs.items():
        gd, gc = (float(x[0]) for x in gaps(t.sum(0)))
        res["arms"][k] = {"detection_gap": gd, "dice_gap": gc, "n_dark": int(t[:, 2].sum()), "n_bright": int(t[:, 3].sum()), "detection_dark": float(t[:, 0].sum() / t[:, 2].sum()), "detection_bright": float(t[:, 1].sum() / t[:, 3].sum()),
                          "dice_dark": float(t[:, 4].sum() / t[:, 2].sum()), "dice_bright": float(t[:, 5].sum() / t[:, 3].sum()), "mean_dice_all": float(sum(r["dice"] for r in data[k].values()) / len(data[k])), "median_signed_sld_error": median_sld_error(data[k], cases)}

    def contrast(a, b, name):
        out = {}
        for mi, metric in enumerate(("det", "dice")):
            rng = np.random.default_rng(seed(f"W5_{name}_{metric}")); idx = rng.integers(0, len(cases), (B, len(cases)))
            da = np.array([gaps(tabs[a][i].sum(0))[mi][0] for i in idx]); db = np.array([gaps(tabs[b][i].sum(0))[mi][0] for i in idx])
            pt = float(gaps(tabs[a].sum(0))[mi][0] - gaps(tabs[b].sum(0))[mi][0]); lo, hi = ci(da - db); out[metric] = {"delta": pt, "lo": lo, "hi": hi}
        return out
    if all(k in data for k in ("B50", "D0", "S50")):
        v = {}
        for comp in ("D0", "S50"):
            c = contrast("B50", comp, f"B50_vs_{comp}"); res["contrasts"][f"B50_minus_{comp}"] = c
            for m in ("det", "dice"):
                v[(comp, m)] = verdict(c[m]["delta"], c[m]["lo"], c[m]["hi"]) if (evaluable["B50"] and evaluable[comp]) else "NOT EVALUABLE (sanity gate failed)"
        res["verdicts"] = {f"{c}_{m}": x for (c, m), x in v.items()}
        res["prediction"] = overall(v) if all(evaluable[k] for k in ("B50", "D0", "S50")) else "NOT EVALUABLE (an arm failed its sanity gate)"
        res["procedure_effect_S50_minus_D0_EXPLORATORY"] = contrast("S50", "D0", "S50_vs_D0")
    json.dump(res, open(os.path.join(outdir, "w5_results.json"), "w"), indent=1)
    lines = [f"# W5 results ({LABEL})", ""] + [f"- {k}: detection gap {a['detection_gap']:+.3f}, Dice gap {a['dice_gap']:+.3f}, mean Dice {a['mean_dice_all']:.3f}, median signed SLD error {a['median_signed_sld_error']:+.3f}" for k, a in res["arms"].items()]
    for k, c in res["contrasts"].items():
        lines.append(f"- {k}: detection {c['det']['delta']:+.3f} [{c['det']['lo']:+.3f}, {c['det']['hi']:+.3f}], Dice {c['dice']['delta']:+.3f} [{c['dice']['lo']:+.3f}, {c['dice']['hi']:+.3f}]")
    lines.append(f"- verdicts: {res['verdicts']}; prediction: {res.get('prediction')}")
    open(os.path.join(outdir, "w5_results.md"), "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    return res


if __name__ == "__main__":
    a = sys.argv[1:]; arms = {x.split("=", 1)[0]: x.split("=", 1)[1] for i, x in enumerate(a) if i and a[i - 1] == "--arm"}; gates = {x.split("=", 1)[0]: x.split("=", 1)[1] for i, x in enumerate(a) if i and a[i - 1] == "--gate"}
    main(a[0], arms, gates)
