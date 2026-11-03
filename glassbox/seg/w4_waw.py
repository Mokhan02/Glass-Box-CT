"""W4 (WAW-TACE) detection scoring (results/w4_waw/PREREG.md). Detection is EXACTLY the W3 Stage 1 definition: the lesion's coverage (share of the lesion's voxels predicted positive, on the annotated slices) from
`glassbox.seg.w2_lib.lesion_metrics`, called as `glassbox.seg.w3_lib.lesion_rows` calls it, with detection = coverage >= 0.10. The only extension is the lesion set: every kept tumour of at least 10 mm is scored, not only the two largest.
Ground truth per series (written by `scripts/w4_waw_paired_contrast.py --save-dir`): `seg.nii.gz` (1 reference parenchyma, 2 tumour, 3 portal/splenic vein; the slices holding any label are the annotated slices, as in W3), `tumour_labels.nii.gz`
(uint16 component ids, identical on both phases), optional `tumour_labels_dil2.nii.gz` (the propagated labels grown by 2 mm: the dilation sensitivity), `tumours.csv`, `ct.nii.gz`. Pure numpy / scipy apart from `read_series` (nibabel)."""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy import ndimage

from glassbox.audit.contrast import feret
from glassbox.seg.w2_lib import dice_within, lesion_metrics

DET_COV = 0.10
FIELDS = ["arm", "patient", "series_phase", "drawn_phase", "is_propagated", "rank_drawn_phase", "label_id", "diam_mm_drawn_phase", "diam_mm_series", "fov_inside_fraction", "n_voxels", "coverage", "dice_full", "dice_within_10mm", "coverage_dil2", "detected", "detected_dil2", "pred_diam_mm", "mode"]


def component_diameter(comp, spacing):
    """Longest axial (Feret) diameter of a boolean component, the maximum over its axial slices (the definition inside `select_lesions`)."""
    zs = np.flatnonzero(comp.any(axis=(0, 1)))
    return max((feret(comp[:, :, z], spacing[0], spacing[1]) for z in zs), default=0.0)


def read_series(gt_root, case):
    """The ground truth of one series: dict(spacing, seg, labels, labels_dil2 or None, tumours (list of dicts), ct_shape)."""
    import nibabel as nib
    d = os.path.join(gt_root, case); ct = nib.load(os.path.join(d, "ct.nii.gz")); sp = tuple(float(z) for z in ct.header.get_zooms()[:3])
    seg = np.asarray(nib.load(os.path.join(d, "seg.nii.gz")).dataobj); lab = np.asarray(nib.load(os.path.join(d, "tumour_labels.nii.gz")).dataobj)
    dp = os.path.join(d, "tumour_labels_dil2.nii.gz"); dil = np.asarray(nib.load(dp).dataobj) if os.path.exists(dp) else None
    return {"spacing": sp, "seg": seg, "labels": lab, "labels_dil2": dil, "tumours": list(csv.DictReader(open(os.path.join(d, "tumours.csv")))), "ct_shape": tuple(ct.shape)}


def score_arrays(arm, seg, labels, labels_dil2, tumours, pred_tumour, spacing):
    """Rows (one per tumour of this series) for one prediction. pred_tumour: bool array on the input grid (may be larger than the CT by padding; cropped as in `lesion_rows`)."""
    ann = (seg > 0).any(axis=(0, 1)); pa = np.asarray(pred_tumour)[:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, ann].astype(bool); lab_a = labels[:, :, ann]; dil_a = labels_dil2[:, :, ann] if labels_dil2 is not None else None; rows = []
    for t in tumours:
        i = int(t["label_id"]); comp = lab_a == i; n = int(comp.sum()); r = {"arm": arm, "patient": t["patient"], "series_phase": t["series_phase"], "drawn_phase": t["drawn_phase"], "is_propagated": int(t["is_propagated"]), "rank_drawn_phase": t["rank_drawn_phase"],
                                                                         "label_id": i, "diam_mm_drawn_phase": t["diam_mm_drawn_phase"], "fov_inside_fraction": t["fov_inside_fraction"], "n_voxels": n}
        if n == 0:      # the propagated label left the scanned volume
            rows.append({**r, "diam_mm_series": 0.0, "coverage": "", "dice_full": "", "dice_within_10mm": "", "coverage_dil2": "", "detected": "", "detected_dil2": "", "pred_diam_mm": "", "mode": "NO_VOXELS"}); continue
        m = lesion_metrics(pa, comp, spacing); cov = m["coverage"]
        if dil_a is not None:
            c2 = dil_a == i; cov2 = float((pa & c2).sum() / max(int(c2.sum()), 1))
        else:
            cov2 = cov      # a drawn mask is not dilated: the sensitivity changes the propagated series only
        rows.append({**r, "diam_mm_series": round(component_diameter(comp, spacing), 3), "coverage": round(cov, 5), "dice_full": round(m["dice_full"], 5), "dice_within_10mm": round(dice_within(pa, comp, spacing), 5), "coverage_dil2": round(cov2, 5),
                     "detected": int(cov >= DET_COV), "detected_dil2": int(cov2 >= DET_COV), "pred_diam_mm": round(m["pred_diam_mm"], 3), "mode": m["mode"]})
    return rows
