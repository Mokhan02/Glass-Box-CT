"""Mechanistic experiment C statistics (results/mech/PREREG.md section C). Reads cache/mech_c/heldout_shard*.csv and chosen.json.
Usage: python scripts/mech_c_stats.py [--src cache/mech_c] [--out results/mech]"""
import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="cache/mech_c"); ap.add_argument("--out", default="results/mech"); a = ap.parse_args()
rows = [r for f in sorted(glob.glob(os.path.join(a.src, "heldout_shard*.csv"))) for r in csv.DictReader(open(f))]
cases = sorted({r["case"] for r in rows}); P = len(cases); ix = {c: i for i, c in enumerate(cases)}
BI = np.random.default_rng(1).integers(0, P, size=(2000, P))
bright = np.zeros(P, bool)
for r in rows:
    bright[ix[r["case"]]] = float(r["contrast_hu"]) > 0
T = {}  # (layer, arm) -> arrays over cases
for r in rows:
    key = (r["layer"], r["arm"]); d = T.setdefault(key, {k: np.full(P, np.nan) for k in ("dice", "fp", "nt", "nd", "rel")}); i = ix[r["case"]]
    d["dice"][i] = float(r["dice"]); d["fp"][i] = float(r["fp_cm3"]); d["nt"][i] = float(r["n_targets"]); d["nd"][i] = float(r["n_detected"]); d["rel"][i] = float(r["relerr"])
orig = T[("-", "orig")]
res = {"n_cases": P, "chosen": json.load(open(os.path.join(a.src, "chosen.json")))}


def summ(d, sel):
    ok = sel & ~np.isnan(d["rel"]); rel = d["rel"][ok]
    return {"dice": float(np.mean(d["dice"][sel])), "detection": float(d["nd"][sel].sum() / max(d["nt"][sel].sum(), 1)), "sld_median_rel_error": float(np.median(rel)) if len(rel) else float("nan"),
            "over20": float(np.mean(rel >= 0.2)) if len(rel) else float("nan"), "fp_cm3": float(np.mean(d["fp"][sel]))}


def ci(fn):
    bs = [fn(b) for b in BI]; return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def arm_table(d):
    out = {}
    for nm, sel in (("all", np.ones(P, bool)), ("bright", bright), ("dark", ~bright)):
        out[nm] = summ(d, sel)
    out["all_ci"] = {"dice": ci(lambda b: np.mean(d["dice"][b])), "detection": ci(lambda b: d["nd"][b].sum() / max(d["nt"][b].sum(), 1)),
                     "over20": ci(lambda b: np.nanmean(d["rel"][b][~np.isnan(d["rel"][b])] >= 0.2))}
    return out


res["orig"] = arm_table(orig); table = []
for layer in ("mid", "bottleneck"):
    L = {arm: T[(layer, arm)] for arm in ("thr", "probe", "random", "meandiff")}; R = {"layer": layer, "arms": {k: arm_table(v) for k, v in L.items()}}
    for other in ("thr", "random", "meandiff"):
        dd = L["probe"]["dice"] - L[other]["dice"]; R[f"probe_minus_{other}"] = {"dice_gain": float(dd.mean()), "ci": ci(lambda b, dd=dd: dd[b].mean())}
    over_p, over_t = R["arms"]["probe"]["all"]["over20"], R["arms"]["thr"]["all"]["over20"]
    gt, gr = R["probe_minus_thr"]["ci"], R["probe_minus_random"]["ci"]
    R["verdict"] = "BEATS THRESHOLD" if (gt[0] > 0 and gr[0] > 0 and (over_p - over_t) <= 0.05) else ("WORSE THAN THRESHOLD" if gt[1] < 0 else "EQUIVALENT TO THRESHOLD")
    res[layer] = R
    for arm in ("thr", "probe", "random", "meandiff"):
        for grp in ("all", "bright", "dark"):
            s = R["arms"][arm][grp]; table.append({"layer": layer, "arm": arm, "group": grp, **{k: round(v, 4) for k, v in s.items()}})
for grp in ("all", "bright", "dark"):
    table.append({"layer": "-", "arm": "orig", "group": grp, **{k: round(v, 4) for k, v in res["orig"][grp].items()}})
json.dump(res, open(os.path.join(a.out, "C_results.json"), "w"), indent=1)
with open(os.path.join(a.out, "C_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
print(json.dumps({k: res[k] for k in ("mid", "bottleneck")}, indent=1)[:6000]); print("orig", res["orig"]["all"])
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for axx, layer in zip(ax, ("mid", "bottleneck")):
    for i, arm in enumerate(("thr", "probe", "random", "meandiff")):
        d = T[(layer, arm)]["dice"]; m = d.mean(); lo, hi = ci(lambda b, d=d: d[b].mean()); axx.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", capsize=4)
    axx.axhline(orig["dice"].mean(), color="grey", ls="--", lw=0.8); axx.set_xticks(range(4)); axx.set_xticklabels(["threshold\n(matched FP)", "probe\ndirection", "random\ndirection", "mean-diff\ndirection"], fontsize=8)
    axx.set_title(f"{layer}: held-out tumour Dice (dashed = original)", fontsize=9)
ax[0].set_ylabel("Dice"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "C_figure.png"), dpi=150)
