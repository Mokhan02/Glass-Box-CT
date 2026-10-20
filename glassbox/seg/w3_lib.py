"""Shared helpers for the W3 Stage 1 runners (model A, the VISTA-3D reference row, later model D): the cohort lesion tables, the per-lesion rows (same metrics as W2 v3 via glassbox.seg.w2_lib),
and the alignment check. No torch / monai imports, so it can be tested anywhere."""
from __future__ import annotations

import csv
import os

import numpy as np

from glassbox.seg.w2_lib import dice_within, lesion_metrics

ALIGN_MIN_DICE = 0.85          # PREREG_stage1 edit 1: organ Dice vs the GT organ on every case, array indices; stop if more than 2 cases fail
ALIGN_MAX_FAILURES = 2
HCC_ROOT = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/hcc_tace_seg"
KITS_ROOT = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"
FIELDS = ["arm", "organ", "case", "rank", "group", "stratum", "diam_mm", "contrast_hu", "coverage", "dice_full", "dice_within_10mm", "pred_diam_mm", "pred_over_lesion_volume", "pred_frac_in_lesion", "mode"]


def stratum(d):
    return "<30 mm" if d < 30 else ("30-60 mm" if d <= 60 else ">60 mm")


def lesion_table(organ, repo="."):
    """{(case, rank): row} of the cohort: HCC UID route (exp8 lesion table, 86 lesions) or C4KC-KiTS (gt_lesions.csv, two largest per case, 219 lesions)."""
    if organ == "hcc":
        return {(r["case"], int(r["rank"])): r for r in csv.DictReader(open(os.path.join(repo, "results/exp8_lesions/lesion_table.csv"))) if r["route"] == "uid"}
    return {(r["case"], int(r["rank"])): r for r in csv.DictReader(open(os.path.join(repo, "results/w2_kidney/gt_lesions.csv"))) if int(r["rank"]) <= 2}


def gt_organ(seg, organ):
    """Ground-truth organ used for the alignment check: HCC liver = labels 1 (liver) and 2 (tumour); KiTS kidney = labels 1 and 2 (kidney and mass)."""
    return (seg == 1) | (seg == 2)


def dice(a, b):
    return float(2 * (a & b).sum() / max(int(a.sum()) + int(b.sum()), 1))


def alignment(pred_organ_ann, seg_ann, organ):
    """Organ Dice on the annotated slices, array indices, no header used."""
    return dice(pred_organ_ann.astype(bool), gt_organ(seg_ann, organ))


def lesion_rows(organ, case, root, table, preds):
    """preds: {arm: bool prediction on the ORIGINAL array grid (may be larger than the CT by padding; cropped)}. Returns (rows, geometry dict with seg, ann, spacing)."""
    from glassbox.seg.w2_lib import case_geometry
    nii, sp, seg, ann, zidx, lab, sel = case_geometry(root, case); rows = []
    for rank, (diam, cid) in enumerate(sel[:2], 1):
        if (case, rank) not in table:
            continue
        ref = table[(case, rank)]; assert abs(float(ref["diam_mm"]) - diam) < 0.01, f"lesion mismatch {case} {rank}"
        comp_c = lab == cid
        for arm, pred in preds.items():
            pa = pred[:seg.shape[0], :seg.shape[1], :seg.shape[2]][:, :, ann]; m = lesion_metrics(pa, comp_c, sp)
            rows.append({"arm": arm, "organ": organ, "case": case, "rank": rank, "group": "bright" if float(ref["contrast_hu"]) > 0 else "dark", "stratum": stratum(diam), "diam_mm": round(diam, 3),
                         "contrast_hu": float(ref["contrast_hu"]), "coverage": round(m["coverage"], 5), "dice_full": round(m["dice_full"], 5), "dice_within_10mm": round(dice_within(pa, comp_c, sp), 5),
                         "pred_diam_mm": round(m["pred_diam_mm"], 3), "pred_over_lesion_volume": round(m["pred_over_lesion_volume"], 4), "pred_frac_in_lesion": round(m["pred_frac_in_lesion"], 4), "mode": m["mode"]})
    return rows, {"seg": seg, "ann": ann, "spacing": sp}
