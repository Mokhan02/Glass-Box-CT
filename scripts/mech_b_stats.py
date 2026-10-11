"""Mechanistic experiment B statistics (results/mech/PREREG.md section B). Reads cache/mech_b/run_shard*_{lesions,cases}.csv and results/mech/B_probe_check.json.
Usage: python scripts/mech_b_stats.py [--src cache/mech_b] [--out results/mech]"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="cache/mech_b"); ap.add_argument("--out", default="results/mech"); a = ap.parse_args()
rd = lambda pat: [r for f in sorted(glob.glob(os.path.join(a.src, pat))) for r in csv.DictReader(open(f))]  # noqa: E731
les, cas = rd("run_shard*_lesions.csv"), rd("run_shard*_cases.csv")
CONDS = ("leace", "random", "hu")
L = {}
for r in les:
    L.setdefault((r["case"], r["rank"]), {})[r["cond"]] = (float(r["coverage"]), float(r["lesion_dice"]))
C = {}
for r in cas:
    C.setdefault(r["case"], {})[r["cond"]] = (float(r["tumour_dice"]), float(r["liver_dice"]), float(r["max_abs_diff_vs_stored"]))
det = [k for k, d in L.items() if len(d) == 4 and d["none"][0] >= 0.10]
cases = sorted({k[0] for k in det}); rng = np.random.default_rng(1); BI = rng.integers(0, len(cases), size=(2000, len(cases))); by = {c: [k for k in det if k[0] == c] for c in cases}
print(f"{len(L)} lesions, {len(det)} detected by the unhooked model in {len(cases)} cases; max none-vs-stored diff {max(v['none'][2] for v in C.values() if 'none' in v):.3f}")


def stat(fn, boots=True):
    pt = fn(det); bs = [fn([k for c in b for k in by[cases[c]]]) for b in BI] if boots else []
    return pt, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


dD = lambda cd: (lambda ks: float(np.mean([L[k][cd][1] - L[k]["none"][1] for k in ks])))  # noqa: E731
dC = lambda cd: (lambda ks: float(np.mean([L[k][cd][0] - L[k]["none"][0] for k in ks])))  # noqa: E731
still = lambda cd: (lambda ks: float(np.mean([L[k][cd][0] >= 0.10 for k in ks])))  # noqa: E731
res = {"n_lesions_detected_by_original": len(det), "n_cases": len(cases)}
for cd in CONDS:
    res[cd] = {"delta_lesion_dice": stat(dD(cd)), "delta_coverage": stat(dC(cd)), "still_detected": stat(still(cd))}
res["leace_minus_random"] = stat(lambda ks: dD("leace")(ks) - dD("random")(ks)); res["leace_minus_hu"] = stat(lambda ks: dD("leace")(ks) - dD("hu")(ks))
ccs = sorted(c for c in C if len(C[c]) == 4); cb = np.random.default_rng(1).integers(0, len(ccs), size=(2000, len(ccs)))
for cd in ("leace", "random", "hu"):
    dl = np.array([C[c][cd][1] - C[c]["none"][1] for c in ccs]); bs = [dl[b].mean() for b in cb]
    res[cd]["delta_liver_dice"] = [float(dl.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
    dt = np.array([C[c][cd][0] - C[c]["none"][0] for c in ccs]); bs = [dt[b].mean() for b in cb]; res[cd]["delta_case_tumour_dice"] = [float(dt.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]]
probe = json.load(open(os.path.join(a.out, "B_probe_check.json"))); res["probe_near_chance_after_leace"] = probe["probe_near_chance_after_leace"]
neg = lambda t: t[1][1] < 0  # CI entirely below zero  # noqa: E731
used = neg(res["leace"]["delta_lesion_dice"]) and neg(res["leace_minus_random"]) and neg(res["leace_minus_hu"])
blunt = res["leace"]["delta_liver_dice"][0] < -0.05
res["verdict"] = ("USED" if used else "NOT NECESSARY") + (" (but erasure too blunt: liver Dice collapses too)" if blunt else "") + ("" if probe["probe_near_chance_after_leace"] else " [probe not near chance: necessity not tested]")
json.dump(res, open(os.path.join(a.out, "B_results.json"), "w"), indent=1); print(json.dumps(res, indent=1))
rows = []
for cd in CONDS:
    r = res[cd]; rows.append({"condition": cd, "delta_lesion_dice": round(r["delta_lesion_dice"][0], 4), "ci_low": round(r["delta_lesion_dice"][1][0], 4), "ci_high": round(r["delta_lesion_dice"][1][1], 4),
                              "delta_coverage": round(r["delta_coverage"][0], 4), "still_detected": round(r["still_detected"][0], 3), "delta_liver_dice": round(r["delta_liver_dice"][0], 4), "liver_ci_low": round(r["delta_liver_dice"][1][0], 4), "liver_ci_high": round(r["delta_liver_dice"][1][1], 4)})
with open(os.path.join(a.out, "B_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for i, cd in enumerate(CONDS):
    r = res[cd]["delta_lesion_dice"]; ax[0].errorbar(i, r[0], yerr=[[r[0] - r[1][0]], [r[1][1] - r[0]]], fmt="o", capsize=4)
    r = res[cd]["delta_liver_dice"]; ax[1].errorbar(i, r[0], yerr=[[r[0] - r[1][0]], [r[1][1] - r[0]]], fmt="o", capsize=4)
for axx, t in zip(ax, (f"Change in lesion Dice (lesions the model detects, n={len(det)})", "Change in liver Dice (check)")):
    axx.axhline(0, color="grey", lw=0.8); axx.set_xticks(range(3)); axx.set_xticklabels(["LEACE\n(tumour concept)", "random\ndirection", "local-HU\nconcept"]); axx.set_title(t, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "B_figure.png"), dpi=150)
