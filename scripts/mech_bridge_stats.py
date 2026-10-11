"""EXPLORATORY, post hoc (results/mech/DEVIATIONS.md #5). Usage: python scripts/mech_bridge_stats.py [--bridge cache/mech_bridge/bridge.csv] [--d cache/mech_d] [--out results/mech]"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("--bridge", default="cache/mech_bridge/bridge.csv"); ap.add_argument("--d", default="cache/mech_d"); ap.add_argument("--out", default="results/mech"); a = ap.parse_args()
cov = {}
for f in glob.glob(os.path.join(a.d, "patch_shard*.csv")):
    for r in csv.DictReader(open(f)):
        if r["cond"] in ("dark", "bright"):
            cov.setdefault((r["case"], r["rank"]), {})[r["cond"]] = float(r["cov"])
rows = []
for r in csv.DictReader(open(a.bridge)):
    k = (r["case"], r["rank"])
    if int(r["n_cells"]) > 0 and np.isfinite(float(r["t_dark"])) and k in cov and len(cov[k]) == 2:
        rows.append({"case": r["case"], "td": float(r["t_dark"]), "tb": float(r["t_bright"]), "cd": cov[k]["dark"], "cb": cov[k]["bright"]})
n_all = len(list(csv.DictReader(open(a.bridge)))); cases = sorted({r["case"] for r in rows}); BI = np.random.default_rng(1).integers(0, len(cases), size=(2000, len(cases))); by = {c: [r for r in rows if r["case"] == c] for c in cases}
bs_rows = lambda b: [r for c in b for r in by[cases[c]]]  # noqa: E731


def analyse(sel):
    cs = sorted({r["case"] for r in sel}); idx = np.random.default_rng(1).integers(0, len(cs), size=(2000, len(cs))); bc = {c: [r for r in sel if r["case"] == c] for c in cs}
    pick = lambda b: [r for c in b for r in bc[cs[c]]]  # noqa: E731
    d = np.array([r["tb"] - r["td"] for r in sel]); out = {"n_tumours": len(sel), "n_cases": len(cs), "mean_bright_minus_dark_projection": float(d.mean()), "mean_dark_projection": float(np.mean([r["td"] for r in sel]))}
    bm = [np.mean([r["tb"] - r["td"] for r in pick(b)]) for b in idx]; out["ci"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
    nz = d[d != 0]; neg = int((nz < 0).sum()); out["n_negative"] = neg; out["n_nonzero"] = int(len(nz)); out["sign_test_p"] = float(binomtest(neg, len(nz), 0.5).pvalue) if len(nz) else float("nan")
    x = np.array([r["td"] - r["tb"] for r in sel]); y = np.array([r["cd"] - r["cb"] for r in sel])
    out["pearson"], out["spearman"] = float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0])
    ps, ss = [], []
    for b in idx:
        s2 = pick(b); xb = np.array([r["td"] - r["tb"] for r in s2]); yb = np.array([r["cd"] - r["cb"] for r in s2])
        if len(xb) > 3 and xb.std() > 0 and yb.std() > 0:
            ps.append(pearsonr(xb, yb)[0]); ss.append(spearmanr(xb, yb)[0])
    out["pearson_ci"], out["spearman_ci"] = [float(np.percentile(ps, 2.5)), float(np.percentile(ps, 97.5))], [float(np.percentile(ss, 2.5)), float(np.percentile(ss, 97.5))]
    return out


res = {"n_pairs_run": n_all, "n_complete_with_cells": len(rows), "all_complete": analyse(rows), "eligible_drop_ge_0.20": analyse([r for r in rows if r["cd"] - r["cb"] >= 0.20])}
e = res["eligible_drop_ge_0.20"]; res["verdict"] = "projection tracks the brightness effect" if (e["ci"][1] < 0 and e["spearman_ci"][0] > 0) else "does not meet the fixed rule"
json.dump(res, open(os.path.join(a.out, "bridge_exploratory.json"), "w"), indent=1); print(json.dumps(res, indent=1))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8)); x = np.array([r["td"] - r["tb"] for r in rows]); y = np.array([r["cd"] - r["cb"] for r in rows])
ax[0].scatter([r["td"] for r in rows], [r["tb"] for r in rows], s=14); lim = [min(min(x * 0 + [r["td"] for r in rows]), min(r["tb"] for r in rows)), max(max(r["td"] for r in rows), max(r["tb"] for r in rows))]; ax[0].plot(lim, lim, "k--", lw=0.8)
ax[0].set_xlabel("tumour-concept projection, dark"); ax[0].set_ylabel("bright version"); ax[0].set_title(f"Mid-encoder projection per tumour (n={len(rows)}), exploratory", fontsize=9)
ax[1].scatter(x, y, s=14); ax[1].set_xlabel("projection drop (dark - bright)"); ax[1].set_ylabel("coverage drop (dark - bright)"); ax[1].set_title(f"Spearman {res['all_complete']['spearman']:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "bridge_figure.png"), dpi=150)
