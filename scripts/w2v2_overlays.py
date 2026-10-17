"""W2 v2 pilot overlays: one PNG per pilot lesion, one panel per arm (axial slice through the prompt point; green = GT lesion, red = prediction, cyan = GT organ label, yellow + = point).
Usage: python scripts/w2v2_overlays.py <dir with pilot npz masks> results/w2_kidney/v2"""
import csv
import glob
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src, out = sys.argv[1:3]; os.makedirs(f"{out}/overlays", exist_ok=True)
rows = {}
for f in ("w2v2_pilot_kidney.csv", "w2v2_pilot_hcc.csv"):
    for r in csv.DictReader(open(f"{out}/{f}")):
        rows[(r["arm"], r["case"], int(r["rank"]))] = r
lesions = sorted({(r["organ"], r["case"], int(r["rank"])) for r in rows.values()})
ORDER = ["kidney_133_1", "hcc_26_3", "hcc_133_1", "hcc_26_1"]
for organ, case, rank in lesions:
    arms = [a for a in ORDER if (a, case, rank) in rows]; fig, ax = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 5.2), squeeze=False)
    for a_, arm in zip(ax[0], arms):
        z = np.load(f"{src}/pilot__{arm}__{case}__r{rank}.npz"); r = rows[(arm, case, rank)]; pt = tuple(int(v) for v in z["point"]); sx, sy = float(z["ct_zooms"][0]), float(z["ct_zooms"][1])
        sl = z["ct_slice"]; X, Y = sl.shape; half = int(round(max(60.0, 0.9 * float(r["diam_mm"])) / sx)); xs = slice(max(pt[0] - half, 0), min(pt[0] + half, X)); ys = slice(max(pt[1] - half, 0), min(pt[1] + half, Y))
        a_.imshow(np.clip(sl[xs, ys], -160, 240).T, cmap="gray", origin="lower", extent=(xs.start, xs.stop, ys.start, ys.stop), aspect=sy / sx)
        for m, col, lw in (((z["seg"][:, :, pt[2]] == 1), "cyan", 0.7), (z["lesion"][:, :, pt[2]], "lime", 1.6), (z["pred"][:, :, pt[2]], "red", 1.3)):
            if m[xs, ys].any():
                a_.contour(np.arange(xs.start, xs.stop) + 0.5, np.arange(ys.start, ys.stop) + 0.5, m[xs, ys].T.astype(float), levels=[0.5], colors=col, linewidths=lw)
        a_.plot([pt[0] + 0.5], [pt[1] + 0.5], marker="+", color="yellow", markersize=12, mew=2); a_.set_xticks([]); a_.set_yticks([])
        ok = float(r["pred_over_lesion_volume"]) < 3 and float(r["pred_frac_in_lesion"]) > 0.3
        a_.set_title(f"{arm} (label {r['label_prompt']}, point label {r['point_label']})\nvol {float(r['pred_over_lesion_volume']):.2f}x, inside {float(r['pred_frac_in_lesion']):.2f}, cov {float(r['coverage']):.2f} -> {'PASS' if ok else 'FAIL'}", fontsize=8.5, color=("green" if ok else "red"))
    r0 = rows[(arms[0], case, rank)]; fig.suptitle(f"{organ} {case} r{rank}  [{r0['group']}, {r0['stratum']}, lesion {float(r0['diam_mm']):.0f} mm, contrast {float(r0['contrast_hu']):+.0f} HU]", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/overlays/pilot__{organ}__{case}__r{rank}.png", dpi=85); plt.close(fig)
print(len(lesions), "overlays written")
