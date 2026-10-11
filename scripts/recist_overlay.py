"""Visual check of the RECIST-shift measurements. Picks 5 informative target lesions from cache/heldout_pred/recist_lesions.csv
(a large under-measured lesion, an accurate one, a missed one, a mid-size under-measured one, one near -30%), and for each draws the
slice where the ground-truth diameter is longest: CT (soft-tissue window), GT contour (green) with its longest-diameter line, the
prediction contour (red) with the longest diameter of the overlapping predicted component on that slice, and a 50 mm scale bar built
from the CT's in-plane voxel size so spacing errors would be visible. Usage: python scripts/recist_overlay.py cache/recist_check.png"""
import csv
import os
import sys

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

root, predd = "cache/heldout_nii/hcc_tace_seg", "cache/heldout_pred/hcc_tace_seg"
rows = [r for r in csv.DictReader(open("cache/heldout_pred/recist_lesions.csv"))]
for r in rows:
    r["gt"], r["pred"] = float(r["gt"]), float(r["pred"]); r["rel"] = (r["pred"] - r["gt"]) / r["gt"]


def feret(mask2d, sx, sy):
    pts = np.argwhere(mask2d).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0, None
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    d = np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)); i, j = np.unravel_index(d.argmax(), d.shape)
    return float(d[i, j]), (h[i], h[j])


def pick():
    out = []
    def add(c, why):
        c = [r for r in c if r["case"] not in {o[0]["case"] for o in out}]
        if c:
            out.append((c[0], why))
    add(sorted([r for r in rows if r["gt"] >= 80 and r["detected"] == "True" and 0.4 <= r["pred"] / r["gt"] <= 0.7], key=lambda r: -r["gt"]), "large, under-measured")
    add(sorted([r for r in rows if r["detected"] == "True" and abs(r["pred"] - r["gt"]) < 4 and r["gt"] >= 30], key=lambda r: abs(r["pred"] - r["gt"])), "accurate")
    add(sorted([r for r in rows if r["detected"] == "False" and 20 <= r["gt"] < 60], key=lambda r: -r["gt"]), "missed")
    add(sorted([r for r in rows if r["detected"] == "True" and 20 <= r["gt"] < 50 and r["rel"] < -0.4], key=lambda r: r["rel"]), "mid-size, under-measured")
    add(sorted([r for r in rows if r["detected"] == "True" and -0.35 <= r["rel"] <= -0.25 and r["gt"] >= 40], key=lambda r: abs(r["rel"] + 0.3)), "about -30%")
    return out


sel = pick()
fig, axes = plt.subplots(2, 3, figsize=(18, 12)); axes = axes.ravel()
for ax, (r, why) in zip(axes, sel):
    case = r["case"]; img = nib.load(os.path.join(root, case, "ct.nii.gz")); sx, sy = (float(z) for z in img.header.get_zooms()[:2])
    ct = np.asarray(img.dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    pred = np.load(os.path.join(predd, case + ".npz"))["pred"]
    ann = (seg > 0).any(axis=(0, 1)); zi = np.flatnonzero(ann)
    gt3 = np.zeros_like(seg, bool); gt3[:, :, ann] = (seg == 2)[:, :, ann]
    lab, n = ndimage.label(gt3, structure=np.ones((3, 3, 3)))
    best = (0, None, None)
    for i in range(1, n + 1):
        comp = lab == i
        for z in np.flatnonzero(comp.any(axis=(0, 1))):
            d, ep = feret(comp[:, :, z], sx, sy)
            if abs(d - r["gt"]) < 0.5 and d > best[0]:
                best = (d, z, ep, comp[:, :, z])
    d, z, ep, gm = best
    if z is None:
        ax.set_title(f"{case}: could not re-find lesion"); continue
    pm = pred[:, :, z] & ndimage.binary_dilation(gm, iterations=15)  # prediction near the lesion on this slice
    pd_, pep = feret(pm, sx, sy) if pm.any() else (0.0, None)
    H, W = ct.shape[:2]; ext = [0, H * sx, 0, W * sy]  # axis 0 -> x (mm) drawn horizontally after transpose
    ax.imshow(np.clip(ct[:, :, z], -150, 250).T, cmap="gray", origin="lower", extent=ext, aspect="equal")
    for m, c in ((gm, "lime"), (pm, "red")):
        if m.any():
            ax.contour(np.arange(H) * sx + sx / 2, np.arange(W) * sy + sy / 2, m.T.astype(float), levels=[0.5], colors=c, linewidths=1.6)
    for e, c in ((ep, "lime"), (pep, "red")):
        if e is not None:
            ax.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], "-", color=c, lw=2.2)
    cx, cy = np.argwhere(gm).mean(0) * np.array([sx, sy])
    ax.set_xlim(cx - 90, cx + 90); ax.set_ylim(cy - 90, cy + 90)
    x0, y0 = cx - 80, cy - 82; ax.plot([x0, x0 + 50], [y0, y0], "y-", lw=4); ax.text(x0, y0 + 3, "50 mm", color="yellow", fontsize=11)
    ax.set_title(f"{case[:12]} [{why}]\nGT {d:.0f} mm (green) | model {pd_:.0f} mm (red) | in-plane {sx:.2f} mm/px", fontsize=11); ax.axis("off")
for ax in axes[len(sel):]:
    ax.axis("off")
plt.tight_layout(); plt.savefig(sys.argv[1], dpi=80)
print("saved", sys.argv[1], "; lesions:", [(r["case"][:12], why, round(r["gt"], 1), round(r["pred"], 1)) for r, why in sel])
