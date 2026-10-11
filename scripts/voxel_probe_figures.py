"""Voxel-probe experiment, step 3: the detection-across-layers figure and four case overlays.
Overlay case choice (fixed rule): among cases with >= 50 held-out interior-FN mid-layer tokens, the 2 with the HIGHEST and the 2 with the LOWEST
per-case interior-FN detection rate of the trained mid-layer Probe A. Usage: python scripts/voxel_probe_figures.py --vp cache/vp --res results/voxel_probe"""
import argparse
import csv
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vp", default="cache/vp"); ap.add_argument("--res", default="results/voxel_probe")
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.res, "probe_table.csv"))))
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan  # noqa: E731
LAY = ["mid", "bottleneck", "decoder_F"]


def pick(model, feats, layer, subset="pooled"):
    return next(r for r in rows if r["model"] == model and r["features"] == feats and r["layer"] == layer and r["probe"] == "A" and r["subset"] == subset)


fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for a_, (title, rk, lo, hi) in zip(ax, (("Interior FN (missed tumour tissue)", "interior_FN_rate", "ci_low", "ci_high"), ("TP (found tumour tissue)", "TP_rate", "TP_ci_low", "TP_ci_high"),
                                         ("LIV (liver parenchyma: false-positive rate)", "LIV_false_positive_rate", "LIV_fpr_ci_low", "LIV_fpr_ci_high"))):
    x = np.arange(len(LAY))
    t = [pick("trained", "activations", l) for l in LAY]; h = [pick("trained", "rawHU", l) for l in LAY]; r = [pick("random", "activations", l) for l in LAY]
    a_.bar(x - 0.2, [f(q, rk) for q in t], 0.4, yerr=[[f(q, rk) - f(q, lo) for q in t], [f(q, hi) - f(q, rk) for q in t]], color="#2a6f97", capsize=3, label="trained activations")
    a_.bar(x + 0.2, [f(q, rk) for q in r], 0.4, color="#bbbbbb", label="random-init activations")
    a_.plot(x, [f(q, rk) for q in h], "o--", color="#d1495b", lw=2, label="raw-HU control")
    a_.axhline(0.05, color="k", ls=":", lw=1); a_.set_xticks(x); a_.set_xticklabels(LAY); a_.set_title(title, fontsize=11); a_.set_ylim(0, 1.05)
ax[0].set_ylabel("detection rate at 95% specificity (patient-bootstrap 95% CI)"); ax[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Probe A (tumour vs liver tokens), held-out HCC-TACE-Seg, in_distribution=False; dotted line = 5% chance level", fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(a.res, "figure_detection.png"), dpi=110); plt.close()

S = np.load(os.path.join(a.res, "_scores_mid_trained.npz")); cases = list(S["cases"]); pid = S["pid"]; sc, thr = S["score"], S["thr"]
tum = S["is_tum"].astype(bool); fn_int = tum & (S["pred_cov"].astype(float) < 0.5) & (S["edt"].astype(float) > 2.0) & ~np.isnan(sc)
stats = [(p, int((fn_int & (pid == p)).sum()), float((sc[fn_int & (pid == p)] > thr[fn_int & (pid == p)]).mean())) for p in range(len(cases)) if (fn_int & (pid == p)).sum() >= 50]
stats.sort(key=lambda s: -s[2]); chosen = [(s, "probe CATCHES missed tissue") for s in stats[:2]] + [(s, "probe does NOT catch it") for s in stats[-2:]]
fig, axes = plt.subplots(2, 2, figsize=(13, 12)); axes = axes.ravel()
for ax_, ((p, n, rate), why) in zip(axes, chosen):
    z = np.load(os.path.join(a.vp, "trained", cases[p] + ".npz")); hu, lab, pred, zz = z["prev_hu"].astype(float), z["prev_lab"], z["prev_pred"], int(z["prev_z"])
    xyz = S["xyz"][pid == p].astype(int); s_ = sc[pid == p]; t_ = thr[pid == p]; tu = tum[pid == p]
    ax_.imshow(np.clip(hu, -150, 250).T, cmap="gray", origin="lower")
    near = (np.abs(xyz[:, 2] - zz) <= 4) & ~np.isnan(s_)
    for (x0, y0, _), sv, th, is_t in zip(xyz[near], s_[near], t_[near], tu[near]):
        if sv > th:
            ax_.add_patch(plt.Rectangle((x0 - 4, y0 - 4), 8, 8, fc="cyan", ec="none", alpha=0.35))
    ax_.contour((lab == 2).T.astype(float), levels=[0.5], colors="lime", linewidths=1.6); ax_.contour(pred.T.astype(float), levels=[0.5], colors="red", linewidths=1.4)
    ax_.set_title(f"{cases[p][:12]} - {why}\nheld-out interior-FN detection {rate:.2f} ({n} tokens)  | green GT tumour, red model, cyan = probe says tumour", fontsize=10); ax_.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(a.res, "figure_overlays.png"), dpi=80)
print("wrote figure_detection.png and figure_overlays.png; overlay cases:", [(cases[p][:12], n, round(r, 2), w) for (p, n, r), w in chosen])
