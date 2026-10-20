"""Data-quality screen (CPU only, model-free): is the ground-truth liver label displaced relative to its CT? For each case the GT liver (labels 1 and 2) boundary voxels on the annotated slices are shifted over a grid of in-plane shifts (+-40 mm) and
slice shifts (+-3) and the mean in-plane CT gradient magnitude (Gaussian sigma 1 voxel, HU clipped to [-150, 250]) at the shifted boundary is computed; a label that is aligned with the anatomy has its boundary on strong edges, so the shift that maximises this mean
is the estimated displacement of the label (mm, array axes x/y, and slices). Reported: estimated shift per axis, the relative gain of the best shift over zero shift. Flag: |shift| > 10 mm on x or y AND gain > 25% (threshold set after a 3-case test on HCC_019, HCC_004 and HCC_009 and before the cohort run: a 10% threshold falsely flagged HCC_004, where a far-away strong edge won the search).
Stored VISTA-3D liver Dice (hackathon evaluation tables, annotated slices) is joined in by scripts/dq_screen_report.py. Usage: python scripts/dq_label_offset.py <out csv> <case id> [...] (cwd = cache root with heldout_nii/ heldout_nii_clean/)"""
import csv
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

out, cases = sys.argv[1], sys.argv[2:]
rows = []
for case in cases:
    root = os.environ.get("DQ_ROOT") or ("heldout_nii_clean" if os.path.isdir(f"heldout_nii_clean/hcc_tace_seg/{case}") else "heldout_nii"); d = f"{root}/hcc_tace_seg/{case}"
    nii = nib.load(f"{d}/ct.nii.gz"); sp = np.array(nii.header.get_zooms()[:3], float); ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(f"{d}/seg.nii.gz").dataobj)
    liver = (seg == 1) | (seg == 2); ann = (seg > 0).any(axis=(0, 1)); liver &= ann[None, None, :]
    g = ndimage.gaussian_filter(np.clip(ct, -150, 250), sigma=(1, 1, 0)); gm = np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1))
    bnd = np.zeros_like(liver)
    for z in np.flatnonzero(liver.any(axis=(0, 1))):
        m = liver[:, :, z]; bnd[:, :, z] = m & ~ndimage.binary_erosion(m, iterations=2)
    pts = np.argwhere(bnd); pts = pts[np.random.default_rng(0).permutation(len(pts))[:8000]]; X, Y, Z = gm.shape
    steps_x = np.arange(-int(40 / sp[0]), int(40 / sp[0]) + 1, 2); steps_y = np.arange(-int(40 / sp[1]), int(40 / sp[1]) + 1, 2); best = (-1.0, 0, 0, 0); score0 = None
    def score(dx, dy, dz):
        x, y, z = pts[:, 0] + dx, pts[:, 1] + dy, pts[:, 2] + dz; ok = (x >= 0) & (x < X) & (y >= 0) & (y < Y) & (z >= 0) & (z < Z)
        return float(gm[x[ok], y[ok], z[ok]].mean()) if ok.mean() > 0.9 else -1.0
    for dz in range(-3, 4):
        for dx in steps_x:
            for dy in steps_y:
                s = score(dx, dy, dz)
                if s > best[0]:
                    best = (s, dx, dy, dz)
    s, bx, by, bz = best
    for dx in range(bx - 1, bx + 2):                                                    # refine to 1 voxel
        for dy in range(by - 1, by + 2):
            s2 = score(dx, dy, bz)
            if s2 > s:
                s, bx, by = s2, dx, dy
    score0 = score(0, 0, 0); r = {"case": case, "nii_root": root, "shift_x_mm": round(bx * sp[0], 1), "shift_y_mm": round(by * sp[1], 1), "shift_z_slices": bz, "score_zero_shift": round(score0, 3), "score_best": round(s, 3), "gain_over_zero": round(s / score0 - 1, 3),
                                  "n_boundary_points": int(len(pts))}
    r["flag"] = bool((abs(r["shift_x_mm"]) > 10 or abs(r["shift_y_mm"]) > 10) and r["gain_over_zero"] > 0.25); rows.append(r); print(r, flush=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
