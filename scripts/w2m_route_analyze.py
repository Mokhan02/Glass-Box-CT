"""W2-M route check analysis (results/w2_mech/PREREG_route.md, applied as written). Question: is M2's null due to the zero-shot route rather than the organ?
Primary arm HCC 133/1 (W2 v3 arm hcc_133_1): dD = mean change in dice_full (hooked minus the W2 v3 unhooked run) under the cross-fitted HCC LEACE, case-level bootstrap (1000 resamples, seed 0, 95% percentile CI).
Positive control: class 26 AUTOMATIC route, change in lesion Dice within 10 mm under LEACE on the lesions its unhooked run detects (coverage >= 10%), same bootstrap.
ORGAN DIFFERS: the M2 USED criteria on 133/1 ((1) dD <= -0.05 with CI upper < 0; (2) LEACE minus random and LEACE minus HU CI uppers < -0.02; (3) liver organ Dice change CI lower >= -0.05).
ROUTE EXPLAINS M2: dD CI lower > -0.10 AND the positive control CI upper < -0.10. INCONCLUSIVE otherwise (including a failed positive control or an erasure too blunt). Order: ORGAN DIFFERS, then ROUTE EXPLAINS, then INCONCLUSIVE.
Secondary: the M2-style verdict on 133/1, the same statistics for the 26/3 arm, descriptive comparison with M2 and mech B.
Usage: python scripts/w2m_route_analyze.py <w2v3 hcc csv> <route lesions csv> <route auto csv> <route organ csv> <outdir>"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w2m_m2_analyze as M2  # noqa: E402  (boot, decide and the failure-mode rule are shared with M2)

COND = ["leace", "random", "hu"]


def arm_stats(base, rows, organ, arm, rng):
    by = {c: {(r["case"], int(r["rank"])): r for r in rows if r["arm"] == arm and r["condition"] == c} for c in COND}; keys = sorted(by["leace"]); assert all(sorted(by[c]) == keys for c in COND)
    b = {k: base[(arm, k[0], k[1])] for k in keys}; cases = [k[0] for k in keys]
    dd = {c: np.array([float(by[c][k]["dice_full"]) - float(b[k]["dice_full"]) for k in keys]) for c in COND}
    R = {"n_lesions": len(keys), "n_cases": len(set(cases)), "delta_dice": {c: M2.boot(dd[c], cases, rng) for c in COND}, "delta_dice10": {c: M2.boot([float(by[c][k]["dice_within_10mm"]) - float(b[k]["dice_within_10mm"]) for k in keys], cases, rng) for c in COND},
         "delta_coverage": {c: M2.boot([float(by[c][k]["coverage"]) - float(b[k]["coverage"]) for k in keys], cases, rng) for c in COND},
         "paired": {"leace_minus_random": M2.boot(dd["leace"] - dd["random"], cases, rng), "leace_minus_hu": M2.boot(dd["leace"] - dd["hu"], cases, rng)}}
    ocases = [r["case"] for r in organ]; R["organ_leace"] = M2.boot([float(r["dice_leace"]) - float(r["dice_unhooked"]) for r in organ], ocases, rng); R["organ_n_cases"] = len(organ)
    R["m2_style"] = M2.decide(R["delta_dice"]["leace"], R["paired"]["leace_minus_random"], R["paired"]["leace_minus_hu"], R["organ_leace"])
    bm = {k: M2.mode(b[k]) for k in keys}; R["failure_modes"] = {c: {m: {"baseline_rate": float(np.mean([bm[k] == m for k in keys])), "hooked_rate": float(np.mean([M2.mode(by[c][k]) == m for k in keys]))} for m in ("LEAK", "UNDER", "OK")} for c in COND}
    return R


def positive_control(auto, rng):
    g = {(r["condition"], r["case"], int(r["rank"])): r for r in auto}; keys = sorted({k[1:] for k in g}); det = [k for k in keys if float(g[("unhooked", k[0], k[1])]["coverage"]) >= 0.10]
    out = {"n_lesions": len(keys), "n_detected_unhooked": len(det)}
    for name, ks in (("detected", det), ("all", keys)):
        out[name] = {c: M2.boot([float(g[(c, k[0], k[1])]["dice_within_10mm"]) - float(g[("unhooked", k[0], k[1])]["dice_within_10mm"]) for k in ks], [k[0] for k in ks], rng) for c in COND}
    return out


def route_verdict(primary, pc):
    dD = primary["delta_dice"]["leace"]; organ_differs = primary["m2_style"]["verdict"] == "USED"; large = pc["detected"]["leace"]["hi"] < -0.10; no_large_drop = dD["lo"] > -0.10
    if organ_differs:
        v = "ORGAN DIFFERS"
    elif no_large_drop and large:
        v = "ROUTE EXPLAINS M2"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "organ_differs_criteria_met": bool(organ_differs), "dD_ci_lower_above_minus_0.10": bool(no_large_drop), "positive_control_large_effect": bool(large), "erasure_too_blunt": bool(primary["m2_style"]["erasure_too_blunt"])}


def main(base_csv, les_csv, auto_csv, org_csv, out):
    rng = np.random.default_rng(0)
    base = {(r["arm"], r["case"], int(r["rank"])): r for r in csv.DictReader(open(base_csv))}
    # the W2 v3 CSV has no dice_within_10mm for lesions in the new column set: v3 stored it as dice_within_10mm; coverage and dice_full are present
    rows = list(csv.DictReader(open(les_csv))); organ = list(csv.DictReader(open(org_csv))); auto = list(csv.DictReader(open(auto_csv)))
    R = {"hcc_133_1": arm_stats(base, rows, organ, "hcc_133_1", rng), "hcc_26_3": arm_stats(base, rows, organ, "hcc_26_3", rng)}; R["positive_control_auto_26"] = positive_control(auto, rng)
    R["route_verdict"] = route_verdict(R["hcc_133_1"], R["positive_control_auto_26"]); json.dump(R, open(f"{out}/route_results.json", "w"), indent=1)
    f = lambda m, s=1.0: f"{m['mean'] * s:+.3f} [{m['lo'] * s:+.3f}, {m['hi'] * s:+.3f}]"; V = R["route_verdict"]; P = R["hcc_133_1"]; S = R["hcc_26_3"]; PC = R["positive_control_auto_26"]
    L = ["# W2-M route check results (generated by scripts/w2m_route_analyze.py, applied as written)", "", "HCC main-set lesions (76 lesions, 68 cases), cross-fitted HCC mech-B erasers at the mid-encoder. M2 reference (kidney, 133/1): lesion Dice -0.015 [-0.028, -0.002]; mech B reference (HCC, class-26 automatic route, 10 mm Dice): -0.33 [-0.38, -0.29].", "",
         f"## Verdict: **{V['verdict']}**", f"- primary dD (HCC 133/1, LEACE, dice_full): {f(P['delta_dice']['leace'])}; CI lower > -0.10: {V['dD_ci_lower_above_minus_0.10']}",
         f"- positive control (class 26 automatic route, LEACE, 10 mm Dice, {PC['n_detected_unhooked']} of {PC['n_lesions']} lesions detected unhooked): {f(PC['detected']['leace'])}; CI upper < -0.10 (large effect): {V['positive_control_large_effect']} (all {PC['n_lesions']} lesions: {f(PC['all']['leace'])})",
         f"- ORGAN DIFFERS criteria (the M2 USED criteria on 133/1) met: {V['organ_differs_criteria_met']}; erasure too blunt: {V['erasure_too_blunt']}", "",
         "## Secondary: M2-style verdict on each point arm", "| arm | verdict | (1) lesion Dice change | (2) LEACE minus random / minus HU | (3) liver organ Dice change (30-case subset) |", "|---|---|---|---|---|"]
    for name, A in (("HCC 133/1 (primary arm)", P), ("HCC 26/3 (secondary arm)", S)):
        d = A["m2_style"]; L.append(f"| {name} | {d['verdict']}{' (too blunt)' if d['erasure_too_blunt'] else ''}{' (improves)' if d['erasure_improves'] else ''} | {f(A['delta_dice']['leace'])} | {f(A['paired']['leace_minus_random'])} / {f(A['paired']['leace_minus_hu'])} | {f(A['organ_leace'])} |")
    L += ["", "## Lesion Dice change by eraser (dice_full; 10 mm Dice in the second table)", "| arm | LEACE | random | HU |", "|---|---|---|---|"] + [f"| {n} | " + " | ".join(f(A["delta_dice"][c]) for c in COND) + " |" for n, A in (("133/1", P), ("26/3", S))]
    L += ["", "| arm (10 mm Dice) | LEACE | random | HU |", "|---|---|---|---|"] + [f"| {n} | " + " | ".join(f(A["delta_dice10"][c]) for c in COND) + " |" for n, A in (("133/1", P), ("26/3", S))] + [f"| class 26 automatic (positive control, detected lesions) | " + " | ".join(f(PC["detected"][c]) for c in COND) + " |"]
    L += ["", "## Failure-mode rates under LEACE (baseline -> hooked)", "| arm | LEAK | UNDER | OK |", "|---|---|---|---|"] + [f"| {n} | " + " | ".join(f"{100 * A['failure_modes']['leace'][m]['baseline_rate']:.0f}% -> {100 * A['failure_modes']['leace'][m]['hooked_rate']:.0f}%" for m in ("LEAK", "UNDER", "OK")) + " |" for n, A in (("133/1", P), ("26/3", S))]
    open(f"{out}/route_results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:6])
