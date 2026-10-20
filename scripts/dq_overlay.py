"""GT-liver overlay on the CT (CPU only) for visual alignment review: green = GT liver (labels 1 and 2) as converted; red dashed = the same contour shifted by the edge-screen estimate (scripts/dq_label_offset.py).
Usage (cwd = cache root): python scripts/dq_overlay.py <out dir> <label_offset csv> <case> [<case> ...]"""
import csv
import os
import sys

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out, offcsv, cases = sys.argv[1], sys.argv[2], sys.argv[3:]; os.makedirs(out, exist_ok=True); off = {r["case"]: r for r in csv.DictReader(open(offcsv))}
for case in cases:
    root = "heldout_nii_clean" if os.path.isdir(f"heldout_nii_clean/hcc_tace_seg/{case}") else "heldout_nii"; d = f"{root}/hcc_tace_seg/{case}"; nii = nib.load(f"{d}/ct.nii.gz"); ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(f"{d}/seg.nii.gz").dataobj)
    gt = (seg == 1) | (seg == 2); zs = np.flatnonzero(gt.any(axis=(0, 1))); sp = nii.header.get_zooms()[:3]; o = off[case]; sx, sy = float(o["shift_x_mm"]) / sp[0], float(o["shift_y_mm"]) / sp[1]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    for a, z in zip(ax, [zs[int(len(zs) * f)] for f in (0.25, 0.5, 0.75)]):
        a.imshow(np.clip(ct[:, :, z].T, -150, 250), cmap="gray", origin="lower", aspect=sp[1] / sp[0]); a.contour(gt[:, :, z].T, levels=[0.5], colors="lime", linewidths=1.2)
        a.contour(np.roll(gt[:, :, z], (int(round(sx)), int(round(sy))), axis=(0, 1)).T, levels=[0.5], colors="red", linewidths=1.0, linestyles="dashed"); a.set_title(f"z={z}", fontsize=9); a.axis("off")
    fig.suptitle(f"{case}: GT liver (green); shifted by edge-screen estimate x {o['shift_x_mm']} mm, y {o['shift_y_mm']} mm (red dashed); gain {o['gain_over_zero']}", fontsize=10); fig.tight_layout(); fig.savefig(f"{out}/gt_overlay_{case}.png", dpi=80); plt.close(fig)
