"""W2 validity audit (CPU): overlays and a connected-component audit of the 16 lesions rerun with masks saved (results/w2_kidney/validity_selection.csv).
For each lesion: components of the prediction (26-connectivity, annotated slices, as in the Gate 3 measurement), which components overlap the lesion, which one the original measurement used (largest axial Feret
among overlapping components), the component containing the prompt point, the distance of non-overlapping components from the lesion, the composition of the predicted voxels, and the CT / GT / prediction geometry.
Usage: python scripts/w2_validity_audit.py <dir with the npz files> results/w2_kidney"""
import csv
import glob
import os
import sys

import matplotlib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

src, out = sys.argv[1:3]; os.makedirs(f"{out}/overlays", exist_ok=True)
sel = {(r["organ"], r["case"], int(r["rank"])): r for r in csv.DictReader(open(f"{out}/validity_selection.csv"))}


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def axial_diam(mask, sx, sy):
    return max((feret(mask[:, :, z], sx, sy) for z in np.flatnonzero(mask.any(axis=(0, 1)))), default=0.0)


rows = []
for f in sorted(glob.glob(f"{src}/*.npz")):
    org, case, rk = os.path.basename(f)[:-4].split("__"); rank = int(rk[1:]); z = np.load(f); s = sel[(org, case, rank)]
    pred, seg, lesion, pt, ann = z["pred"], z["seg"], z["lesion"], tuple(int(v) for v in z["point"]), z["ann"]; zoom = z["ct_zooms"]; sx, sy, sz = (float(v) for v in zoom)
    organ_lab = 1; pa, la, sa = pred[:, :, ann], lesion[:, :, ann], seg[:, :, ann]; zidx = np.flatnonzero(ann); pz = int(np.searchsorted(zidx, pt[2]))
    plab, n = ndimage.label(pa, structure=np.ones((3, 3, 3))); sizes = np.bincount(plab.ravel())[1:] if n else np.array([])
    ov = [i for i in np.unique(plab[la]) if i > 0]; dia = {i: axial_diam(plab == i, sx, sy) for i in ov}
    meas = max(dia, key=dia.get) if dia else None; pcomp = int(plab[pt[0], pt[1], pz]); pdia = axial_diam(plab == pcomp, sx, sy) if pcomp > 0 else float("nan")
    # distance of non-overlapping components to the lesion (mm), by nearest voxel pair on subsampled voxels
    lt = cKDTree(np.argwhere(la & ~ndimage.binary_erosion(la)) * zoom); extra = []
    for i in range(1, n + 1):
        if i in ov:
            continue
        v = np.argwhere(plab == i); v = v[:: max(1, len(v) // 20000)] * zoom; extra.append((int(sizes[i - 1]), float(lt.query(v)[0].min())))
    vox_ml = sx * sy * sz / 1000.0; tot = int(pa.sum()); in_les = int((pa & la).sum()); in_org = int((pa & ~la & (sa == organ_lab)).sum()); out_org = tot - in_les - in_org
    rows.append({"organ": org, "case": case, "rank": rank, "group": s["group"], "stratum": s["stratum"], "gt_diam_mm": float(s["diam_mm"]), "csv_pred_diam_mm": float(s["pred_diam_mm"]),
                 "n_pred_components": n, "n_overlapping": len(ov), "measured_component_voxels": int(sizes[meas - 1]) if meas else 0, "measured_diam_mm": round(dia[meas], 3) if meas else 0.0,
                 "point_in_prediction": bool(pcomp > 0), "point_component_voxels": int(sizes[pcomp - 1]) if pcomp > 0 else 0, "point_component_diam_mm": round(pdia, 3),
                 "point_component_is_measured": bool(pcomp == meas), "n_extra_components": len(extra), "extra_components_min_dist_mm": round(min((d for _, d in extra), default=float("nan")), 1),
                 "extra_components_total_voxels": int(sum(v for v, _ in extra)), "pred_ml": round(tot * vox_ml, 1), "lesion_ml": round(int(la.sum()) * vox_ml, 1), "pred_over_lesion_volume": round(tot / max(int(la.sum()), 1), 2),
                 "pred_frac_in_lesion": round(in_les / max(tot, 1), 3), "pred_frac_in_organ_label_excl_lesion": round(in_org / max(tot, 1), 3), "pred_frac_outside_organ_label": round(out_org / max(tot, 1), 3),
                 "ct_seg_affine_equal": bool(np.allclose(z["ct_affine"], z["seg_affine"], atol=1e-4)), "ct_seg_spacing_equal": bool(np.allclose(z["ct_zooms"], z["seg_zooms"], atol=1e-4)),
                 "pred_ct_seg_shape_equal": bool(tuple(pred.shape) == tuple(z["ct_shape"]) == tuple(z["seg_shape"])), "spacing_mm": f"{sx:.3f}x{sy:.3f}x{sz:.3f}", "recomputed_matches_csv": bool(abs(round(dia[meas], 3) - float(s["pred_diam_mm"])) < 2e-3) if meas else bool(float(s["pred_diam_mm"]) == 0)})
    # overlay: axial slice through the prompt point
    sl = z["ct_slice"]; X, Y = sl.shape; half = int(round(70.0 / sx)); x0, x1 = max(pt[0] - half, 0), min(pt[0] + half, X); y0, y1 = max(pt[1] - half, 0), min(pt[1] + half, Y)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for k, (a_, crop) in enumerate(zip(ax, (True, False))):
        xs, ys = (slice(x0, x1), slice(y0, y1)) if crop else (slice(0, X), slice(0, Y))
        a_.imshow(np.clip(sl[xs, ys], -160, 240).T, cmap="gray", origin="lower", extent=(xs.start, xs.stop, ys.start, ys.stop), aspect=sy / sx)
        for m, col, lw in ((seg[:, :, pt[2]] == organ_lab, "cyan", 0.7), (lesion[:, :, pt[2]], "lime", 1.6), (pred[:, :, pt[2]], "red", 1.3)):
            if m[xs, ys].any():
                a_.contour(np.arange(xs.start, xs.stop) + 0.5, np.arange(ys.start, ys.stop) + 0.5, m[xs, ys].T.astype(float), levels=[0.5], colors=col, linewidths=lw)
        a_.plot([pt[0] + 0.5], [pt[1] + 0.5], marker="+", color="yellow", markersize=12, mew=2); a_.set_xticks([]); a_.set_yticks([]); a_.set_title("zoomed (140 mm)" if crop else "full axial slice", fontsize=9)
    r = rows[-1]; fig.suptitle(f"{org} {case} r{rank} [{s['group']}, {s['stratum'].split('(')[0]}]  GT lesion {r['gt_diam_mm']:.0f} mm | measured {r['measured_diam_mm']:.0f} mm | point component {r['point_component_diam_mm']:.0f} mm | "
                               f"{n} pred components, pred {r['pred_over_lesion_volume']:.0f}x lesion volume\ngreen = GT lesion, red = prediction, cyan = GT organ label, yellow + = prompt point", fontsize=8.5)
    fig.tight_layout(); fig.savefig(f"{out}/overlays/{org}__{case}__r{rank}.png", dpi=95); plt.close(fig)
with open(f"{out}/validity_audit.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print({k: r[k] for k in ("organ", "case", "rank", "group", "stratum", "gt_diam_mm", "measured_diam_mm", "point_component_diam_mm", "n_pred_components", "n_overlapping", "n_extra_components", "pred_over_lesion_volume", "pred_frac_in_lesion", "pred_frac_in_organ_label_excl_lesion", "pred_frac_outside_organ_label", "recomputed_matches_csv")})
print("geometry: affine equal", all(r["ct_seg_affine_equal"] for r in rows), "| spacing equal", all(r["ct_seg_spacing_equal"] for r in rows), "| shapes equal", all(r["pred_ct_seg_shape_equal"] for r in rows), "| measured==csv", all(r["recomputed_matches_csv"] for r in rows))
