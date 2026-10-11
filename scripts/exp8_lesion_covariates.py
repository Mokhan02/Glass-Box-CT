"""Experiment 8 (lesion covariates). For each RECIST target lesion (same definition as scripts/recist_shift.py: GT hepatic-tumour components on annotated
slices, longest axial diameter >= 10 mm, the 2 largest per case; detected = predicted voxels cover >= 10% of the lesion) compute:
  size            longest axial diameter (mm), volume (mm^3)
  contrast        mean HU in the lesion minus mean HU of non-tumour liver in a 5-20 mm shell around it (fallback: whole liver)
  phase proxies   HU of the labelled aorta fragment and portal-vein segment, whole-liver HU, series-description phase count, number of acquisitions
  window position distance (mm) from the lesion centroid to the nearest edge of the best covering 128^3 sliding window, number of covering windows,
                  whether any single window contains the whole lesion box (windows reconstructed with MONAI's scan interval, overlap 0.3)
  window content  GT liver and tumour volume inside the best window (scaled by the annotated fraction of the window)
Usage (instance, low priority): nice python scripts/exp8_lesion_covariates.py cache/exp8_lesions.csv"""
import csv
import glob
import json
import math
import os
import re
import sys

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.data.preprocess import build_pair_transforms  # noqa: E402

root, predd, held = "cache/heldout_nii/hcc_tace_seg", "cache/heldout_pred/hcc_tace_seg", "cache/heldout/hcc_tace_seg"
acq = {r["case"]: r for r in csv.DictReader(open("cache/hcc_acq_check.csv"))}
ROI, OVERLAP, VOX = 128, 0.3, 1.5


def feret(m2, sx, sy):
    pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def starts_1d(size):
    if size <= ROI:
        return [-((ROI - size) // 2)]  # single window, symmetric padding
    interval = max(int(ROI * (1 - OVERLAP)), 1); num = int(math.ceil((size - ROI) / interval)) + 1
    return sorted({min(i * interval, size - ROI) for i in range(num)})


rows = []
for case in sorted(os.listdir(predd)):
    case = case[:-4]
    nii = nib.load(os.path.join(root, case, "ct.nii.gz")); aff0 = nii.affine; sx, sy, sz = (float(z) for z in nii.header.get_zooms()[:3])
    ct = np.asarray(nii.dataobj).astype(np.float32); seg = np.asarray(nib.load(os.path.join(root, case, "seg.nii.gz")).dataobj)
    ann = (seg > 0).any(axis=(0, 1)); zidx = np.flatnonzero(ann)
    pred = np.load(os.path.join(predd, case + ".npz"))["pred"]
    gt_c, pr_c = (seg == 2)[:, :, ann], pred[:, :, ann]
    lab, n = ndimage.label(gt_c, structure=np.ones((3, 3, 3))); les = []
    for i in range(1, n + 1):
        comp = lab == i; zs = np.flatnonzero(comp.any(axis=(0, 1)))
        les.append((max(feret(comp[:, :, z], sx, sy) for z in zs), i))
    les = sorted([l for l in les if l[0] >= 10.0], key=lambda l: -l[0])[:2]
    if not les:
        continue
    # native (1.5 mm, RAS, cropped) frame: affine and GT label map, for window geometry
    lab0 = np.zeros(seg.shape, np.int16); lab0[seg == 1] = 1; lab0[seg == 2] = 2; lab0[:, :, ~ann] = 3
    img = torch.from_numpy(ct)[None]; im = MetaTensor(img, meta={"affine": torch.tensor(aff0, dtype=torch.float64), "spatial_shape": ct.shape}, affine=torch.tensor(aff0, dtype=torch.float64))
    from monai.transforms import LoadImage
    ld = LoadImage(image_only=True); I0, S0 = ld(os.path.join(root, case, "ct.nii.gz")), ld(os.path.join(root, case, "seg.nii.gz"))
    pair = build_pair_transforms()({"image": I0, "label": MetaTensor(torch.from_numpy(lab0), meta=I0.meta.copy(), affine=I0.affine)})
    L = pair["label"][0].as_tensor().numpy().astype(np.int16); aff_n = np.array(pair["image"].affine, float); inv_n = np.linalg.inv(aff_n); shp = L.shape
    wstarts = [starts_1d(s) for s in shp]
    desc = ""; pid = re.match(r"^(HCC_\d+)", case).group(1)
    m = json.load(open(os.path.join(root, case, "segments.json")))
    for f in glob.glob(f"{held}/{pid}/*/{m['ct_series']}/*.dcm")[:1]:
        import pydicom
        desc = str(pydicom.dcmread(f, stop_before_pixels=True).get("SeriesDescription", ""))
    ph = re.search(r"(\d)\s*PHASE", desc.upper()); nph = int(ph.group(1)) if ph else np.nan
    liver_mask = (seg == 1); aorta, portal = ct[(seg == 4) & ann[None, None, :]], ct[(seg == 3) & ann[None, None, :]]
    for rank, (diam, cid) in enumerate(les, 1):
        comp_c = lab == cid; xs, ys, ks = np.nonzero(comp_c); zs = zidx[ks]
        det_cover = float((pr_c & comp_c).sum()) / max(int(comp_c.sum()), 1)
        thu = float(ct[xs, ys, zs].mean())
        # shell of non-tumour liver, 5-20 mm from the lesion (3D, true spacing), inside a padded bounding box
        pad = int(25 / min(sx, sy, sz)); X, Y, Z = ct.shape  # clip the padded box to the volume on both sides
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X); y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y); z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
        les3 = np.zeros((x1 - x0, y1 - y0, z1 - z0), bool); les3[xs - x0, ys - y0, zs - z0] = True
        dist = ndimage.distance_transform_edt(~les3, sampling=(sx, sy, sz)); shell = liver_mask[x0:x1, y0:y1, z0:z1] & ann[None, None, z0:z1] & (dist >= 5) & (dist <= 20)
        hu_all = ct[liver_mask & ann[None, None, :]]
        shell_hu = float(ct[x0:x1, y0:y1, z0:z1][shell].mean()) if shell.sum() >= 200 else float(hu_all.mean())
        # window geometry: centroid and box of the lesion in the native frame
        cen = aff0 @ np.array([xs.mean(), ys.mean(), zs.mean(), 1.0]); cn = (inv_n @ cen)[:3]
        corners = np.array([[a, b, c_, 1.0] for a in (xs.min(), xs.max()) for b in (ys.min(), ys.max()) for c_ in (zs.min(), zs.max())]); cb = (inv_n @ (aff0 @ corners.T)).T[:, :3]
        lo, hi = cb.min(0), cb.max(0)
        best, ncov, contained = -1.0, 0, False; bw = None
        for a in wstarts[0]:
            for b in wstarts[1]:
                for c_ in wstarts[2]:
                    s0 = np.array([a, b, c_], float)
                    if np.all(cn >= s0) and np.all(cn < s0 + ROI):
                        ncov += 1; e = float(min((cn - s0).min(), (s0 + ROI - cn).min()))
                        if e > best:
                            best, bw = e, (a, b, c_)
                    if np.all(lo >= s0) and np.all(hi < s0 + ROI):
                        contained = True
        w = L[max(bw[0], 0):bw[0] + ROI, max(bw[1], 0):bw[1] + ROI, max(bw[2], 0):bw[2] + ROI] if bw is not None else L[:0, :0, :0]
        known = float((w != 3).mean()) if w.size else np.nan
        rows.append({"case": case, "route": acq[case]["route"], "n_acq": int(acq[case]["n_acq_numbers"]), "rank": rank, "diam_mm": diam, "volume_mm3": float(comp_c.sum() * sx * sy * sz),
                     "detected": det_cover >= 0.10, "cover": det_cover, "tumour_hu": thu, "liver_hu_shell": shell_hu, "liver_hu_all": float(hu_all.mean()), "contrast_hu": thu - shell_hu,
                     "aorta_hu": float(aorta.mean()) if len(aorta) > 100 else np.nan, "portal_hu": float(portal.mean()) if len(portal) > 100 else np.nan, "n_phase_in_desc": nph, "series_desc": desc[:40],
                     "best_edge_mm": best * VOX if best >= 0 else np.nan, "n_windows": ncov, "contained_in_a_window": contained,
                     "win_liver_mm3": float(((w == 1) | (w == 2)).sum() * VOX ** 3 / max(known, 1e-6)) if w.size else np.nan, "win_tumour_mm3": float((w == 2).sum() * VOX ** 3 / max(known, 1e-6)) if w.size else np.nan,
                     "win_known_frac": known, "native_shape": "x".join(map(str, shp))})
    print(f"{case} lesions {len(les)} " + " ".join(f"d={r['diam_mm']:.0f}mm det={int(r['detected'])} contrast={r['contrast_hu']:.0f}HU edge={r['best_edge_mm']:.0f}mm" for r in rows[-len(les):]), flush=True)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} lesions; detected {sum(r['detected'] for r in rows)}")
