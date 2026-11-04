"""W4 alignment diagnosis: overlays of the ground-truth organ (green contour) and the predicted liver (red contour) on the CT for series saved by scripts/w4_waw_align_diag.py --overlay (axial and coronal slice, one row per model).
Usage: python scripts/w4_waw_align_overlay.py <diag dir> <out dir> <case> [<case> ...]"""
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main(diag, out, cases):
    os.makedirs(out, exist_ok=True)
    for c in cases:
        arms = [a for a in ("vista", "A", "D0", "B50") if os.path.exists(os.path.join(diag, f"{a}__{c}.npz"))]
        fig, ax = plt.subplots(len(arms), 2, figsize=(8, 4 * len(arms)))
        ax = np.atleast_2d(ax)
        for i, a in enumerate(arms):
            z = np.load(os.path.join(diag, f"{a}__{c}.npz")); sp = z["spacing"]
            for j, (k, asp) in enumerate((("ax", sp[1] / sp[0]), ("cor", sp[2] / sp[0]))):
                ct = np.clip(z["ct_" + k].T.astype(float), -150, 250); g = z["gt_" + k].T; p = z["pr_" + k].T
                ax[i, j].imshow(ct, cmap="gray", origin="lower", aspect=asp if k == "cor" else 1)
                for m, col in ((g, "lime"), (p, "red")):
                    if m.any():
                        ax[i, j].contour(m.astype(float), levels=[0.5], colors=col, linewidths=0.9)
                ax[i, j].set_title(f"{c} {a} {'axial z=' + str(int(z['z'])) if k == 'ax' else 'coronal y=' + str(int(z['y']))}  (green GT organ, red predicted liver)", fontsize=7); ax[i, j].axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"overlay_{c}.png"), dpi=70); plt.close(fig); print("wrote", c, arms)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
