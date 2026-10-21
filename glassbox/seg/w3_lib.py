"""Shared helpers for the W3 Stage 1 runners (model A, the VISTA-3D reference row, later model D): the cohort lesion tables, the per-lesion rows (same metrics as W2 v3 via glassbox.seg.w2_lib),
and the alignment check. No torch / monai imports, so it can be tested anywhere."""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from glassbox.seg.w2_lib import dice_within, lesion_metrics

ALIGN_MIN_DICE = 0.85          # PREREG_stage1 edit 1 (superseded as the gate by Amendment 2; kept as a descriptive flag: cases below it are listed and form the sensitivity exclusion)
ALIGN_MAX_FAILURES = 2         # stop if more than 2 cases fail the alignment check
ALIGN_MAX_OFFSET_MM = 15.0     # Amendment 2: predicted-vs-GT organ centroid offset limit (the limit of the 2026-10-20 diagnostic rule, fixed before that diagnostic ran)
HCC_ROOT = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/hcc_tace_seg"
KITS_ROOT = "/lambda/nfs/CTGate/glassbox/cache/heldout_nii/c4kc_kits"
FIELDS = ["arm", "organ", "case", "rank", "group", "stratum", "diam_mm", "contrast_hu", "coverage", "dice_full", "dice_within_10mm", "pred_diam_mm", "pred_over_lesion_volume", "pred_frac_in_lesion", "mode"]


def stratum(d):
    return "<30 mm" if d < 30 else ("30-60 mm" if d <= 60 else ">60 mm")


def corrected_cases(repo="."):
    """Cases whose label the original converter displaced (results/dq_hcc_labels/corrected_cases.json; HCC_019_773068)."""
    d = json.load(open(os.path.join(repo, "results/dq_hcc_labels/corrected_cases.json"))); return {k: v for k, v in d.items() if not k.startswith("_")}


def excluded_cases(repo="."):
    """Data-quality exclusions (results/dq_hcc_labels/excluded_cases.json; HCC_027_967743): excluded from the primary analysis, included in the sensitivity."""
    d = json.load(open(os.path.join(repo, "results/dq_hcc_labels/excluded_cases.json"))); return {k: v for k, v in d.items() if not k.startswith("_")}


def case_root(organ, case, base, as_converted=False, repo="."):
    """Directory of the converted case: the corrected conversion for the corrected cases (HCC only), otherwise `base`."""
    return corrected_cases(repo)[case]["corrected_root"] if (organ == "hcc" and not as_converted and case in corrected_cases(repo)) else base


def lesion_table(organ, repo=".", as_converted=False, include_excluded=False):
    """{(case, rank): row} of the cohort: HCC UID route (exp8 lesion table, 86 lesions, minus the data-quality exclusions unless include_excluded; the contrast of corrected cases recomputed on the corrected label unless as_converted) or C4KC-KiTS (gt_lesions.csv, two largest per case, 219 lesions)."""
    if organ == "hcc":
        t = {(r["case"], int(r["rank"])): dict(r) for r in csv.DictReader(open(os.path.join(repo, "results/exp8_lesions/lesion_table.csv"))) if r["route"] == "uid"}
        if not include_excluded:
            t = {k: v for k, v in t.items() if k[0] not in excluded_cases(repo)}
        if not as_converted:
            for c, v in corrected_cases(repo).items():
                for rank, contrast in v["contrast_hu"].items():
                    t[(c, int(rank))]["contrast_hu"] = str(contrast)
        return t
    return {(r["case"], int(r["rank"])): r for r in csv.DictReader(open(os.path.join(repo, "results/w2_kidney/gt_lesions.csv"))) if int(r["rank"]) <= 2}


def gt_organ(seg, organ):
    """Ground-truth organ used for the alignment check: HCC liver = labels 1 (liver) and 2 (tumour); KiTS kidney = labels 1 and 2 (kidney and mass)."""
    return (seg == 1) | (seg == 2)


def dice(a, b):
    return float(2 * (a & b).sum() / max(int(a.sum()) + int(b.sum()), 1))


def alignment(pred_organ_ann, seg_ann, organ):
    """Organ Dice on the annotated slices, array indices, no header used."""
    return dice(pred_organ_ann.astype(bool), gt_organ(seg_ann, organ))


def alignment_report(pred_organ_full, seg, ann, spacing, input_shape):
    """Amendment 2 alignment check, array indices, GT spacing for mm: (1) the output array has the input array's shape; (2) no axis flip of the predicted organ raises the organ Dice (orientation); (3) predicted-vs-GT organ centroid offset <= 15 mm on the annotated slices.
    The organ Dice is reported (and flagged below 0.85) but is no longer the gate. pred_organ_full: bool array on the input grid (before any cropping)."""
    shape_ok = tuple(pred_organ_full.shape) == tuple(input_shape); sh = seg.shape
    pa = pred_organ_full[:sh[0], :sh[1], :sh[2]][:, :, ann].astype(bool); gt = gt_organ(seg[:, :, ann], "hcc"); d0 = dice(pa, gt)
    flips = {f"flip_{a}": dice(np.flip(pa, a), gt) for a in (0, 1, 2)}; flip_better = any(v > d0 for v in flips.values())
    if pa.any() and gt.any():
        sp = np.array(spacing, float); off = np.argwhere(pa).mean(0) - np.argwhere(gt).mean(0); off_mm = off * sp; offset = float(np.linalg.norm(off_mm))
    else:
        off_mm, offset = np.full(3, np.nan), float("inf")
    return {"organ_dice": round(d0, 4), "low_dice_lt_0.85": bool(d0 < ALIGN_MIN_DICE), "centroid_offset_mm": round(offset, 1), "centroid_offset_xyz_mm": [round(float(v), 1) for v in off_mm], "shape_ok": shape_ok, "flip_dice": {k: round(v, 4) for k, v in flips.items()},
            "flip_improves_dice": bool(flip_better), "pass": bool(shape_ok and not flip_better and offset <= ALIGN_MAX_OFFSET_MM)}


SHIFT_MAX_MM = 5.0             # Amendment 2 as stated by the researcher: best Dice-maximising translation <= 5 mm per axis
SHIFT_MIN_GAIN = 0.05          # "substantially raises liver Dice": best-shift Dice minus unshifted Dice >= 0.05 (the margin of the 2026-10-20 diagnostic rule)


def shift_test(pred_full, seg, ann, spacing, search_mm=(40.0, 40.0, 7.5), search_vox=None):
    """Dice-maximising translation test of the predicted organ against the GT organ (labels 1 and 2, annotated slices), array indices, GT spacing for mm. Dice(s) = 2 sum_x G(x) P(x - s) / (|P_ann| + |G|) for every integer shift s inside the search window,
    by FFT cross-correlation (P = predicted organ on the whole grid so z shifts can use unannotated slices, G = GT organ on the annotated slices). Returns the unshifted Dice, the best shift (voxels and mm per axis, smallest |s| among ties), the best Dice and the verdict:
    ALIGNED if every axis shift <= 5 mm; MISALIGNED if some axis shift > 5 mm AND the best Dice exceeds the unshifted Dice by >= 0.05; otherwise MODEL-SIDE (no translation substantially fixes it)."""
    from scipy.signal import fftconvolve
    sh = seg.shape; P = np.asarray(pred_full)[:sh[0], :sh[1], :sh[2]].astype(np.float32); G = np.zeros(sh, np.float32); G[:, :, ann] = gt_organ(seg[:, :, ann], "hcc")
    Pa = float(P[:, :, ann].sum()); Gs = float(G.sum()); den = max(Pa + Gs, 1.0)
    if Pa == 0 or Gs == 0:
        return {"dice_unshifted": 0.0, "best_dice": 0.0, "gain": 0.0, "best_shift_voxels": [0, 0, 0], "best_shift_mm": [0.0, 0.0, 0.0], "max_axis_shift_mm": 0.0, "verdict": "MODEL-SIDE"}
    C = fftconvolve(G, P[::-1, ::-1, ::-1], mode="full"); ctr = np.array(sh) - 1; sp = np.array(spacing, float); lim = np.array(search_vox, int) if search_vox is not None else np.floor(np.array(search_mm) / sp).astype(int)
    lim = np.minimum(lim, np.array(sh) - 1)
    sl = tuple(slice(ctr[i] - lim[i], ctr[i] + lim[i] + 1) for i in range(3)); W = C[sl] * 2 / den; d0 = float(C[tuple(ctr)] * 2 / den); best = W.max()
    cand = np.argwhere(W >= best - 1e-6); idx = cand[np.argmin(np.linalg.norm((cand - lim) * sp, axis=1))]; s = idx - lim; mm = s * sp; gain = float(best - d0); mx = float(np.abs(mm).max())
    verdict = "ALIGNED" if mx <= SHIFT_MAX_MM else ("MISALIGNED" if gain >= SHIFT_MIN_GAIN else "MODEL-SIDE")
    return {"dice_unshifted": round(d0, 4), "best_dice": round(float(best), 4), "gain": round(gain, 4), "best_shift_voxels": [int(v) for v in s], "best_shift_mm": [round(float(v), 1) for v in mm], "max_axis_shift_mm": round(mx, 1), "verdict": verdict}


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
