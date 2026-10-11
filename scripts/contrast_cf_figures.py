"""Four overlay figures for the contrast counterfactual (2 per arm), chosen by the fixed rule in PREREG.md: in each arm, the lesion with the largest change in coverage
in the hypothesised direction (edit minus control), and the lesion with the median such change. Each figure: CT slice before and after the edit, GT lesion (green), prediction (red).
Usage: python scripts/contrast_cf_figures.py cache/contrast_cf results/contrast_cf"""
import csv
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cf, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
R = list(csv.DictReader(open(os.path.join(cf, "edits.csv"))))
by = {}
for r in R:
    by.setdefault((r["case"], r["rank"], r["arm"]), {})[r["cond"]] = r
D = []
for (case, rank, arm), v in by.items():
    if "edit" in v and "control" in v:
        e, c = v["edit"], v["control"]; D.append({"case": case, "rank": rank, "arm": arm, "D": (float(e["cov"]) - float(e["cov0"])) - (float(c["cov"]) - float(c["cov0"])), "edit": e})
for arm, sign in (("A", -1), ("B", +1)):
    ls = sorted([d for d in D if d["arm"] == arm], key=lambda d: -sign * d["D"])  # most change in the hypothesised direction first
    if not ls:
        continue
    for tag, d in (("largest effect", ls[0]), ("median effect", ls[len(ls) // 2])):
        b = np.load(os.path.join(cf, "slices", f"{d['case']}_L{d['rank']}_baseline.npz")); e = np.load(os.path.join(cf, "slices", f"{d['case']}_L{d['rank']}_edit.npz"))
        ys, xs = np.nonzero(b["lesion_mask"].T); cx, cy = (int(xs.mean()), int(ys.mean())) if len(xs) else (b["ct"].shape[0] // 2, b["ct"].shape[1] // 2)
        half = max(int(max(np.ptp(xs), np.ptp(ys)) * 1.2) if len(xs) else 60, 50); sl = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
        for a_, ct, pred, ttl, cov in ((ax[0], b["ct"], b["pred"], "before edit", d["edit"]["cov0"]), (ax[1], e["ct"], e["pred"], "after edit", d["edit"]["cov"])):
            a_.imshow(np.clip(np.asarray(ct, float), -150, 250).T[sl], cmap="gray", origin="lower")
            a_.contour(np.asarray(b["lesion_mask"], float).T[sl], levels=[0.5], colors="lime", linewidths=1.8); a_.contour(np.asarray(pred, float).T[sl], levels=[0.5], colors="red", linewidths=1.5)
            a_.set_title(f"{ttl}: coverage {float(cov):.2f}", fontsize=11); a_.axis("off")
        arm_txt = "dark tumour made bright" if arm == "A" else "bright, missed tumour made dark"
        fig.suptitle(f"Arm {arm} ({arm_txt}) | {d['case'][:11]} lesion {d['rank']} | {tag} | edit minus control change in coverage {d['D']:+.2f}\nHU: tumour {float(d['edit']['T']):.0f}, local liver {float(d['edit']['L']):.0f}, edit target {float(d['edit']['target_mean']):.0f} | green GT lesion, red prediction (class 26)", fontsize=10)
        plt.tight_layout(); plt.savefig(os.path.join(out, f"overlay_arm{arm}_{tag.split()[0]}.png"), dpi=90); plt.close()
        print("saved", f"overlay_arm{arm}_{tag.split()[0]}.png", d["case"][:11], f"D={d['D']:+.2f}")
