"""Mechanistic experiment D statistics (results/mech/PREREG.md section D). Reads cache/mech_d/patch_shard*.csv.
Usage: python scripts/mech_d_stats.py [--src cache/mech_d] [--out results/mech]"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="cache/mech_d"); ap.add_argument("--out", default="results/mech"); ap.add_argument("--n-boot", type=int, default=2000); a = ap.parse_args()
rows = [r for f in sorted(glob.glob(os.path.join(a.src, "patch_shard*.csv"))) for r in csv.DictReader(open(f))]
les = {}
for r in rows:
    k = (r["case"], r["rank"]); d = les.setdefault(k, {"patch": {}, "control": {}})
    v = {"cov": float(r["cov"]), "logit": float(r["logit_core"]), "dice": float(r["dice"])}
    if r["cond"] in ("dark", "bright"):
        d[r["cond"]] = v
    elif r["cond"] == "patch":
        d["patch"][(r["layer"], r["region"])] = v
    elif r["cond"] == "control_unrelated":
        d["control"][(r["layer"], r["region"])] = v
keys = [k for k, d in les.items() if "dark" in d and "bright" in d and len(d["patch"]) == 9 and len(d["control"]) == 3]
elig = [k for k in keys if les[k]["dark"]["cov"] - les[k]["bright"]["cov"] >= 0.20]
print(f"{len(les)} lesions run, {len(keys)} complete, {len(elig)} eligible (coverage drop >= 0.20)")
cases = sorted({k[0] for k in elig}); rng = np.random.default_rng(1); BI = rng.integers(0, len(cases), size=(a.n_boot, len(cases))); by_case = {c: [k for k in elig if k[0] == c] for c in cases}


def frac(sel, layer, region, ctl, q):
    num = den = 0.0
    for k in sel:
        d = les[k]; p = (d["control"] if ctl else d["patch"]).get((layer, region)); num += p[q] - d["bright"][q]; den += d["dark"][q] - d["bright"][q]
    return num / den if den else float("nan")


res = []
for q, qn in (("cov", "coverage"), ("logit", "core logit")):
    for layer in ("mid", "bottleneck", "decoder_F"):
        for region, ctl in (("R1", False), ("R2", False), ("R3", False), ("R1", True)):
            pt = frac(elig, layer, region, ctl, q)
            bs = [frac([k for c in b for k in by_case[cases[c]]], layer, region, ctl, q) for b in BI]
            res.append({"outcome": qn, "layer": layer, "region": "R1 unrelated-case control" if ctl else {"R1": "R1 dilated tumour", "R2": "R2 shell only", "R3": "R3 far context"}[region], "restored_fraction": round(pt, 3),
                        "ci_low": round(float(np.nanpercentile(bs, 2.5)), 3), "ci_high": round(float(np.nanpercentile(bs, 97.5)), 3), "n_lesions": len(elig)})
with open(os.path.join(a.out, "D_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
read = {}
for region in ("R1 dilated tumour", "R2 shell only", "R3 far context"):
    hit = [r for r in res if r["outcome"] == "coverage" and r["region"] == region and r["restored_fraction"] >= 0.5]
    read[region] = hit[0]["layer"] if hit else "none reaches 0.50"
ctrl_ok = {l: next(r for r in res if r["outcome"] == "coverage" and r["layer"] == l and r["region"] == "R1 unrelated-case control")["restored_fraction"] < next(r for r in res if r["outcome"] == "coverage" and r["layer"] == l and r["region"] == "R1 dilated tumour")["restored_fraction"] for l in ("mid", "bottleneck", "decoder_F")}
json.dump({"n_lesions_run": len(les), "n_complete": len(keys), "n_eligible": len(elig), "earliest_layer_restoring_ge_50pct_coverage": read, "unrelated_control_restores_less_than_R1": ctrl_ok}, open(os.path.join(a.out, "D_results.json"), "w"), indent=1)
print(open(os.path.join(a.out, "D_results.json")).read())
for r in res:
    if r["outcome"] == "coverage":
        print(r["layer"].ljust(11), r["region"].ljust(28), r["restored_fraction"], [r["ci_low"], r["ci_high"]])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for axx, qn in zip(ax, ("coverage", "core logit")):
    regs = ["R1 dilated tumour", "R2 shell only", "R3 far context", "R1 unrelated-case control"]
    for i, reg in enumerate(regs):
        for j, layer in enumerate(("mid", "bottleneck", "decoder_F")):
            r = next(x for x in res if x["outcome"] == qn and x["layer"] == layer and x["region"] == reg); xpos = j + (i - 1.5) * 0.18
            axx.errorbar(xpos, r["restored_fraction"], yerr=[[r["restored_fraction"] - r["ci_low"]], [r["ci_high"] - r["restored_fraction"]]], fmt="o", color=f"C{i}", capsize=3, label=reg if j == 0 else None)
    axx.axhline(0.5, color="grey", lw=0.8, ls="--"); axx.set_xticks(range(3)); axx.set_xticklabels(["mid", "bottleneck", "decoder_F"]); axx.set_title(f"Fraction of the dark-to-bright {qn} drop restored (n={len(elig)})", fontsize=9)
ax[0].set_ylabel("restored fraction"); ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(a.out, "D_figure.png"), dpi=150)
