"""W2-M M2 analysis (results/w2_mech/PREREG.md, applied as written; M1 branch = ORGAN-SPECIFIC, primary eraser = kidney-native LEACE fitted on training folds, applied to held-out patients).
Baseline = the W2 v3 unhooked kidney arm (kidney_133_1) for the same 209 main-set lesions. Per lesion: change in dice_full (hooked minus unhooked) under leace_native, random_native, hu_native (and the secondary leace_hcc).
Case-level bootstrap, 1000 resamples, seed 0, percentile 95% CIs. USED iff (1) mean lesion Dice change under leace_native <= -0.05 with a CI excluding zero; AND (2) the paired differences (leace - random) and (leace - hu) each have a CI upper bound < -0.02;
AND (3) the kidney organ Dice change under leace_native (seeded 60-case subset, per case) has a CI lower bound >= -0.05. If (3) fails: "erasure too blunt" in addition. NOT NECESSARY otherwise. A positive mean change with CI lower > 0: "erasure improves".
Secondary: change in LEAK / UNDER / OK rates under each hook. Usage: python scripts/w2m_m2_analyze.py <w2v3 baseline csv> <m2 lesions csv> <m2 organ csv> <outdir>"""
import csv
import json
import sys

import numpy as np

B, SEED = 1000, 0
COND = ["leace_native", "random_native", "hu_native", "leace_hcc"]


def mode(r):
    v, i, c = float(r["pred_over_lesion_volume"]), float(r["pred_frac_in_lesion"]), float(r["coverage"])
    return "LEAK" if (v >= 3.0 or i <= 0.3) else ("UNDER" if c < 0.5 else "OK")


def boot(values, cases, rng):
    """Mean of per-lesion `values` with a case-level bootstrap: returns (mean, lo, hi)."""
    uc = sorted(set(cases)); idx = {c: np.flatnonzero(np.array(cases) == c) for c in uc}; v = np.asarray(values, float)
    s = np.array([[v[idx[c]].sum(), len(idx[c])] for c in uc]); pt = s[:, 0].sum() / s[:, 1].sum(); r = rng.integers(0, len(uc), (B, len(uc))); d = s[r, 0].sum(1) / s[r, 1].sum(1)
    return {"mean": float(pt), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "n": int(len(v)), "n_cases": len(uc)}


def decide(res_leace, d_rand, d_hu, organ):
    c1 = res_leace["mean"] <= -0.05 and res_leace["hi"] < 0
    c2 = d_rand["hi"] < -0.02 and d_hu["hi"] < -0.02
    c3 = organ["lo"] >= -0.05
    v = "USED" if (c1 and c2 and c3) else "NOT NECESSARY"
    return {"verdict": v, "cond1_lesion_dice_drop": bool(c1), "cond2_beyond_both_controls": bool(c2), "cond3_organ_dice_holds": bool(c3), "erasure_too_blunt": bool(not c3), "erasure_improves": bool(res_leace["lo"] > 0)}


def main(base_csv, les_csv, org_csv, out):
    rng = np.random.default_rng(SEED)
    base = {(r["case"], int(r["rank"])): r for r in csv.DictReader(open(base_csv)) if r["arm"] == "kidney_133_1"}
    rows = list(csv.DictReader(open(les_csv))); by = {c: {(r["case"], int(r["rank"])): r for r in rows if r["condition"] == c} for c in COND}
    keys = sorted(by["leace_native"]); assert all(sorted(by[c]) == keys for c in COND), "conditions cover different lesions"; assert all(k in base for k in keys), "lesion missing from the v3 baseline"
    cases = [k[0] for k in keys]; bright = np.array([float(by["leace_native"][k]["contrast_hu"]) > 0 for k in keys])
    dd = {c: np.array([float(by[c][k]["dice_full"]) - float(base[k]["dice_full"]) for k in keys]) for c in COND}
    R = {"n_lesions": len(keys), "n_cases": len(set(cases)), "delta_dice": {c: boot(dd[c], cases, rng) for c in COND}}
    R["delta_dice_by_brightness"] = {c: {g: boot(dd[c][m], [x for x, mm in zip(cases, m) if mm], rng) for g, m in (("bright", bright), ("dark", ~bright))} for c in COND}
    R["delta_coverage"] = {c: boot([float(by[c][k]["coverage"]) - float(base[k]["coverage"]) for k in keys], cases, rng) for c in COND}
    R["paired"] = {"leace_native_minus_random_native": boot(dd["leace_native"] - dd["random_native"], cases, rng), "leace_native_minus_hu_native": boot(dd["leace_native"] - dd["hu_native"], cases, rng)}
    org = list(csv.DictReader(open(org_csv))); ocases = [r["case"] for r in org]
    R["organ"] = {"n_cases": len(org), "unhooked_mean_dice": float(np.mean([float(r["dice_unhooked"]) for r in org])), **{f"delta_{c}": boot([float(r[f"dice_{c}"]) - float(r["dice_unhooked"]) for r in org], ocases, rng) for c in COND}}
    R["decision_primary"] = decide(R["delta_dice"]["leace_native"], R["paired"]["leace_native_minus_random_native"], R["paired"]["leace_native_minus_hu_native"], R["organ"]["delta_leace_native"])
    bm = {k: mode(base[k]) for k in keys}; R["failure_modes"] = {}
    for c in COND:
        em = {k: mode(by[c][k]) for k in keys}
        R["failure_modes"][c] = {m: {"baseline_rate": float(np.mean([bm[k] == m for k in keys])), "hooked_rate": float(np.mean([em[k] == m for k in keys])),
                                     "change": boot([float(em[k] == m) - float(bm[k] == m) for k in keys], cases, rng)} for m in ("LEAK", "UNDER", "OK")}
    json.dump(R, open(f"{out}/m2_results.json", "w"), indent=1)
    f = lambda m, s=1.0: f"{m['mean'] * s:+.3f} [{m['lo'] * s:+.3f}, {m['hi'] * s:+.3f}]"
    D = R["decision_primary"]
    L = ["# W2-M M2 results (generated by scripts/w2m_m2_analyze.py, applied as written)", "", "C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K). Kidney arm = label 133 + point label 1, main-set lesions; baseline = W2 v3 unhooked run. Primary eraser = kidney-native LEACE (training folds -> held-out patients); the HCC-direction arm is secondary.", "",
         f"## Verdict (primary, kidney-native): **{D['verdict']}**" + (" (erasure too blunt)" if D["erasure_too_blunt"] else "") + (" (erasure improves)" if D["erasure_improves"] else ""),
         f"- (1) mean lesion Dice change under kidney-native LEACE {f(R['delta_dice']['leace_native'])}: needs <= -0.05 with CI upper < 0 -> {D['cond1_lesion_dice_drop']}",
         f"- (2) paired differences: LEACE minus random {f(R['paired']['leace_native_minus_random_native'])}, LEACE minus HU {f(R['paired']['leace_native_minus_hu_native'])}: both CI uppers < -0.02 -> {D['cond2_beyond_both_controls']}",
         f"- (3) kidney organ Dice change under LEACE (60-case subset, {R['organ']['n_cases']} cases, unhooked mean Dice {R['organ']['unhooked_mean_dice']:.3f}) {f(R['organ']['delta_leace_native'])}: CI lower >= -0.05 -> {D['cond3_organ_dice_holds']}", "",
         "## Lesion Dice change (hooked minus unhooked), 209 lesions", "| eraser | all | bright | dark | coverage change |", "|---|---|---|---|---|"]
    for c in COND:
        L.append(f"| {c}{' (secondary)' if c == 'leace_hcc' else ''} | {f(R['delta_dice'][c])} | {f(R['delta_dice_by_brightness'][c]['bright'])} | {f(R['delta_dice_by_brightness'][c]['dark'])} | {f(R['delta_coverage'][c])} |")
    L += ["", "## Kidney organ Dice change (60-case subset)", "| eraser | change |", "|---|---|"] + [f"| {c} | {f(R['organ']['delta_' + c])} |" for c in COND]
    L += ["", "## Failure-mode rates (secondary): baseline -> under erasure, change with case-level CI", "| eraser | LEAK | UNDER | OK |", "|---|---|---|---|"]
    for c in COND:
        L.append(f"| {c} | " + " | ".join(f"{100 * R['failure_modes'][c][m]['baseline_rate']:.0f}% -> {100 * R['failure_modes'][c][m]['hooked_rate']:.0f}% ({f(R['failure_modes'][c][m]['change'], 100)} pts)" for m in ("LEAK", "UNDER", "OK")) + " |")
    open(f"{out}/m2_results.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(*sys.argv[1:5])
