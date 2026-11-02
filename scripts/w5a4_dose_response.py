"""W5 Amendment 4 analysis (CPU; results/w5_training/PREREG.md, Amendment 4): dose-response of the bright-versus-dark gap across p = 0 (D0, D0-seed2), 0.25 (BRIGHT-25) and 0.5 (BRIGHT-50, BRIGHT-50-s2).
Reads the per-lesion CSVs the committed Stage 1 scorer wrote for each run (through the unchanged functions of scripts/w5_analyze.py: same cohort, same gap definition, same case-level bootstrap with the same resampled cases for every arm).
g0 = mean gap of D0 and D0S2, g5 = mean gap of B50 and B50S2, g25 = gap of B25 (gap = dark minus bright; detection rate and mean Dice). C_low = g25 - g0 (predicted < 0), C_high = g25 - g5 (predicted > 0).
Label per metric (in this order): DOSE-RESPONSE if g25 lies strictly between g5 and g0 AND the CI of C_low is entirely below 0; PARTIAL if it lies between but that CI includes 0; NONE otherwise; a metric whose separation g0 - g5 is below 0.20 is capped at PARTIAL
("not resolvable at this separation"). Curve shape (descriptive, not part of the label): "graded" if the CI of C_high is entirely above 0, "saturating" otherwise. Overall: DOSE-RESPONSE (detection and Dice) / (detection only) / (Dice only), else PARTIAL if any metric is PARTIAL, else NONE.
A run whose sanity gate failed makes every comparison that uses it NOT EVALUABLE. Descriptive, no verdict: trend fit of each gap on p over the five runs (slope, intercept, leave-BRIGHT-25-out prediction and residual, patient-bootstrap CIs), the baseline seed variance
|gap(D0) - gap(D0S2)| (and the same for BRIGHT-50), and the W5 contrasts BRIGHT-50 / BRIGHT-50-s2 minus the baseline recomputed against D0, D0S2 and their mean.
Usage: python scripts/w5a4_dose_response.py <outdir> --arm D0=<csv> --arm D0S2=<csv> --arm B25=<csv> --arm B50=<csv> --arm B50S2=<csv> [--gate NAME=<gate json> ...]"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("w5_analyze", os.path.join(HERE, "w5_analyze.py")); W5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(W5)

B, RESOLVABLE = W5.B, 0.20
P = {"D0": 0.0, "D0S2": 0.0, "B25": 0.25, "B50": 0.5, "B50S2": 0.5}
NEEDED = tuple(P)
METRICS = ("det", "dice")


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def label_metric(g0, g25, g5, c_low, c_high):
    """c_low, c_high: (lo, hi) CIs. Returns (label, shape)."""
    between = g5 < g25 < g0
    shape = ("graded" if c_high[0] > 0 else "saturating") if between else None
    if not between:
        return "NONE", shape
    if c_low[1] < 0:
        if g0 - g5 < RESOLVABLE:
            return "PARTIAL (not resolvable at this separation)", shape
        return "DOSE-RESPONSE", shape
    return "PARTIAL", shape


def overall(lab):
    dr = [m for m in METRICS if lab[m].startswith("DOSE-RESPONSE")]
    if len(dr) == 2:
        return "DOSE-RESPONSE (detection and Dice)"
    if dr:
        return f"DOSE-RESPONSE ({'detection' if dr[0] == 'det' else 'Dice'} only)"
    return "PARTIAL" if any(lab[m].startswith("PARTIAL") for m in METRICS) else "NONE"


def trend(gs, ps):
    """OLS of gap on p: (slope, intercept)."""
    slope, icpt = np.polyfit(np.asarray(ps, float), np.asarray(gs, float), 1)
    return float(slope), float(icpt)


def analyse(tabs, gates=None, boot=B):
    """tabs: {arm: per-case table [n_cases x 6]} for the five arms (same cases). Returns the result dict."""
    n = len(next(iter(tabs.values()))); res = {"n_cases": n, "arms": {}, "gates": gates or {}, "labels": {}, "contrasts": {}, "trend": {}, "seed_variance": {}, "w5_contrasts_vs_baselines": {}}
    ok = {k: (gates or {}).get(k, {}).get("pass", True) for k in NEEDED}; bad = [k for k, v in ok.items() if not v]
    G = {k: np.array([float(x[0]) for x in W5.gaps(tabs[k].sum(0))]) for k in NEEDED}      # [det gap, dice gap] per arm (full sample)
    for k in NEEDED:
        res["arms"][k] = {"p": P[k], "detection_gap": G[k][0], "dice_gap": G[k][1]}
    rng = np.random.default_rng(seed("w5a4_dose_response")); idx = rng.integers(0, n, (boot, n))
    Gb = {k: np.array([[float(x[0]) for x in W5.gaps(tabs[k][i].sum(0))] for i in idx]) for k in NEEDED}      # [boot, 2]
    g = lambda D, arms: np.mean([D[a] for a in arms], axis=0)      # noqa: E731
    g0, g5, g25 = g(G, ("D0", "D0S2")), g(G, ("B50", "B50S2")), G["B25"]; g0b, g5b, g25b = g(Gb, ("D0", "D0S2")), g(Gb, ("B50", "B50S2")), Gb["B25"]
    lab = {}
    for mi, m in enumerate(METRICS):
        cl, ch = W5.ci(g25b[:, mi] - g0b[:, mi]), W5.ci(g25b[:, mi] - g5b[:, mi])
        res["contrasts"][m] = {"g0": float(g0[mi]), "g25": float(g25[mi]), "g5": float(g5[mi]), "separation_g0_minus_g5": float(g0[mi] - g5[mi]), "C_low": {"delta": float(g25[mi] - g0[mi]), "lo": cl[0], "hi": cl[1]}, "C_high": {"delta": float(g25[mi] - g5[mi]), "lo": ch[0], "hi": ch[1]}}
        lab[m], shape = label_metric(g0[mi], g25[mi], g5[mi], cl, ch); res["contrasts"][m]["label"] = lab[m]; res["contrasts"][m]["shape"] = shape
    res["labels"] = {"per_metric": lab, "overall": "NOT EVALUABLE (gate failed: " + ", ".join(bad) + ")" if bad else overall(lab)}
    ps = [P[k] for k in NEEDED]
    for mi, m in enumerate(METRICS):
        gs = [G[k][mi] for k in NEEDED]; sl, ic = trend(gs, ps)
        others = [k for k in NEEDED if k != "B25"]; sl2, ic2 = trend([G[k][mi] for k in others], [P[k] for k in others]); pred = ic2 + sl2 * 0.25
        sb = np.array([trend([Gb[k][b, mi] for k in NEEDED], ps)[0] for b in range(boot)]); ib = np.array([trend([Gb[k][b, mi] for k in NEEDED], ps)[1] for b in range(boot)])
        rb = np.array([Gb["B25"][b, mi] - (lambda t: t[1] + t[0] * 0.25)(trend([Gb[k][b, mi] for k in others], [P[k] for k in others])) for b in range(boot)])
        res["trend"][m] = {"slope": sl, "slope_ci": W5.ci(sb), "intercept": ic, "intercept_ci": W5.ci(ib), "leave_B25_out_prediction_at_0.25": float(pred), "B25_residual": float(G["B25"][mi] - pred), "B25_residual_ci": W5.ci(rb), "label": "descriptive, no verdict"}
        d0, d0s = G["D0"][mi], G["D0S2"][mi]; b5, b5s = G["B50"][mi], G["B50S2"][mi]
        res["seed_variance"][m] = {"abs_gap_D0_minus_D0S2": float(abs(d0 - d0s)), "diff_D0_minus_D0S2_ci": W5.ci(Gb["D0"][:, mi] - Gb["D0S2"][:, mi]), "abs_gap_B50_minus_B50S2": float(abs(b5 - b5s)), "diff_B50_minus_B50S2_ci": W5.ci(Gb["B50"][:, mi] - Gb["B50S2"][:, mi]),
                                   "note": "one pair of runs per dose end: a seed-variance estimate, not a bound; SHAM-50 still has one run"}
        out = {}
        for bright in ("B50", "B50S2"):
            for base, bg, bgb in (("D0", G["D0"][mi], Gb["D0"][:, mi]), ("D0S2", G["D0S2"][mi], Gb["D0S2"][:, mi]), ("mean_of_baselines", g0[mi], g0b[:, mi])):
                c = W5.ci(Gb[bright][:, mi] - bgb); out[f"{bright}_minus_{base}"] = {"delta": float(G[bright][mi] - bg), "lo": c[0], "hi": c[1]}
        res["w5_contrasts_vs_baselines"][m] = out
    return res


def main(outdir, arms, gates):
    os.makedirs(outdir, exist_ok=True); data = {k: W5.load(p) for k, p in arms.items()}
    missing = [k for k in NEEDED if k not in data]
    if missing:
        raise SystemExit("missing arms: " + ", ".join(missing))
    if len({frozenset(v) for v in data.values()}) != 1:
        raise SystemExit("arms do not cover the same lesions: " + json.dumps({k: len(v) for k, v in data.items()}))
    cases = sorted({k[0] for k in next(iter(data.values()))}); tabs = {k: W5.case_table(v, cases) for k, v in data.items()}; gj = {}
    for k, path in gates.items():
        j = json.load(open(path)); gj[k] = {"liver_dice": j["liver_dice"], "tumour_dice": j["tumour_dice"], "pass": bool(j["pass"])}
    res = analyse(tabs, gj); res["inputs"] = arms; json.dump(res, open(os.path.join(outdir, "w5a4_results.json"), "w"), indent=1, default=float)
    f = lambda d: f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"      # noqa: E731
    L = ["# W5 Amendment 4: dose-response of the bright-versus-dark gap", "", f"**Overall label: {res['labels']['overall']}** ({res['n_cases']} cases; gaps = dark minus bright)", "",
         "| arm | p | detection gap | Dice gap |", "|---|---|---|---|"] + [f"| {k} | {v['p']} | {v['detection_gap']:+.3f} | {v['dice_gap']:+.3f} |" for k, v in res["arms"].items()] + [""]
    for m, name in (("det", "detection"), ("dice", "Dice")):
        c = res["contrasts"][m]
        L.append(f"- **{name}**: g0 {c['g0']:+.3f}, g25 {c['g25']:+.3f}, g5 {c['g5']:+.3f} (separation {c['separation_g0_minus_g5']:.3f}); C_low = g25 - g0 {f(c['C_low'])}; C_high = g25 - g5 {f(c['C_high'])} -> **{c['label']}**" + (f", curve {c['shape']}" if c["shape"] else ""))
    L += ["", "Descriptive (no verdict):"]
    for m, name in (("det", "detection"), ("dice", "Dice")):
        t, s = res["trend"][m], res["seed_variance"][m]
        L.append(f"- {name}: slope per unit p {t['slope']:+.3f} [{t['slope_ci'][0]:+.3f}, {t['slope_ci'][1]:+.3f}]; B25 residual from the leave-out line {t['B25_residual']:+.3f} [{t['B25_residual_ci'][0]:+.3f}, {t['B25_residual_ci'][1]:+.3f}]; |gap(D0) - gap(D0S2)| = {s['abs_gap_D0_minus_D0S2']:.3f}; |gap(B50) - gap(B50S2)| = {s['abs_gap_B50_minus_B50S2']:.3f}")
    open(os.path.join(outdir, "w5a4_results.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    a, g, out = {}, {}, sys.argv[1]
    for i, x in enumerate(sys.argv):
        if x in ("--arm", "--gate"):
            k, v = sys.argv[i + 1].split("=", 1); (a if x == "--arm" else g)[k] = v
    main(out, a, g)
