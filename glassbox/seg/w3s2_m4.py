"""W3 Stage 2, M4 (readout-fix generality): threshold shift on a model's tumour decision-margin maps, chosen by nested patient-level cross-validation (the VISTA-3D recalibration analysis, glassbox/trust/gating.py, applied to the policy "always use the shifted threshold"). Pure numpy / scipy.
Maps: margin (float16, annotated slab in [x, y, z] order, model grid), label map (0 bg, 1 liver, 2 tumour, 3 unknown slice), spacing [x, y, z] mm, as written by scripts/w3s2_m2_run.py. Everything is measured on the model's own preprocessed grid on annotated slices (as the VISTA `rr` cache), lesions as in the hackathon RECIST analysis:
26-connected tumour components, longest axial (x-y) Feret diameter >= 10 mm, the two largest per case; detected = predicted voxels cover >= 10% of the lesion; predicted diameter = largest overlapping predicted component (0 for a miss); case error = (sum predicted - sum GT) / sum GT."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from glassbox.audit.contrast import select_lesions
from glassbox.seg.w3s2_stats import arm_seed, fold_of_cases

T_GRID = (-0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0)          # threshold on the margin (0 = the model's default decision)
OVER = 0.20
STRUCT = np.ones((3, 3, 3))


def feret(m2, sx, sy):
    from scipy.spatial import ConvexHull
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


class CaseMap:
    """One case: annotated slices only, GT lesions (two largest >= 10 mm) and a function giving per-lesion coverage / predicted diameter and the tumour Dice at any threshold."""

    def __init__(self, margin, label, spacing, contrasts=None):
        keep = (label != 3).any(axis=(0, 1)); self.m = margin[:, :, keep].astype(np.float32); lab = label[:, :, keep]; self.sp = tuple(float(s) for s in spacing); self.tum = lab == 2
        self.glab, sel = select_lesions(self.tum, self.sp, 2, 10.0); self.lesions = [{"id": i, "diam": d, "mask": self.glab == i} for d, i in sel]; self.contrast = contrasts

    def at(self, t):
        pred = self.m > t; plab, _ = ndimage.label(pred, structure=STRUCT); out = []
        for les in self.lesions:
            cov = float((pred & les["mask"]).sum()) / max(int(les["mask"].sum()), 1); det = cov >= 0.10; pd_ = 0.0
            if det:
                ov = [i for i in np.unique(plab[les["mask"]]) if i > 0]; pd_ = max((axial_diam(plab == i, self.sp[0], self.sp[1]) for i in ov), default=0.0)
            out.append({"cov": cov, "det": det, "pdiam": pd_, "diam": les["diam"]})
        dice = 2 * float((pred & self.tum).sum()) / max(int(pred.sum()) + int(self.tum.sum()), 1)
        gs = sum(o["diam"] for o in out); rel = (sum(o["pdiam"] for o in out) - gs) / gs if gs > 0 else float("nan")
        return {"lesions": out, "rel": rel, "dice": dice}


def policy_table(cases, per_case, group):
    """per_case[c] = result of CaseMap.at for the policy; group[c] = list of 'bright' / 'dark' per lesion rank. Returns pooled summary numbers."""
    rel = np.array([per_case[c]["rel"] for c in cases]); ok = np.isfinite(rel)
    det_b = [l["det"] for c in cases for l, g in zip(per_case[c]["lesions"], group[c]) if g == "bright"]; det_d = [l["det"] for c in cases for l, g in zip(per_case[c]["lesions"], group[c]) if g == "dark"]
    gap = (np.mean(det_d) - np.mean(det_b)) if det_b and det_d else float("nan")
    return {"n_cases_with_target": int(ok.sum()), "median_signed_error": float(np.median(rel[ok])), "mean_abs_error": float(np.mean(np.abs(rel[ok]))), "over_20pct": float(np.mean(rel[ok] >= OVER)), "abs_ge_20pct": float(np.mean(np.abs(rel[ok]) >= OVER)),
            "mean_dice": float(np.mean([per_case[c]["dice"] for c in cases])), "detection_dark": float(np.mean(det_d)) if det_d else float("nan"), "detection_bright": float(np.mean(det_b)) if det_b else float("nan"), "detection_gap": float(gap)}


def nested_threshold(cases, maps, fold, t_grid=T_GRID, cache=None):
    """Outer 5-fold (fold = index in the sorted primary list mod 5). Inside each outer training set the threshold minimising the mean absolute sum-of-diameters error over the training cases is chosen (inner 4-fold CV of a one-parameter policy with nothing to fit
    equals the training-set mean; ties go to the less negative threshold). Returns {case: threshold} and the chosen threshold per outer fold."""
    cache = {} if cache is None else cache

    def rel(c, t):
        if (c, t) not in cache:
            cache[(c, t)] = maps[c].at(t)["rel"]
        return cache[(c, t)]
    chosen, per_fold = {}, {}
    for f in range(5):
        tr = [c for c in cases if fold[c] != f]; best, bt = None, None
        for t in sorted(t_grid, reverse=True):
            e = np.array([abs(rel(c, t)) for c in tr]); e = e[np.isfinite(e)]; m = float(e.mean()) if len(e) else float("inf")
            if best is None or m < best - 1e-12:
                best, bt = m, t
        per_fold[f] = bt
        for c in cases:
            if fold[c] == f:
                chosen[c] = bt
    return chosen, per_fold


def m4_run(cases, maps, group, B=1000, arm="M4"):
    """Policies default (t = 0) and shifted (nested threshold); pooled tables; paired case-level bootstrap of the primary quantities (detection-gap change, change of |median signed error|, over-estimation change)."""
    fold = fold_of_cases(cases); chosen, per_fold = nested_threshold(cases, maps, fold); d0 = {c: maps[c].at(0.0) for c in cases}; ds = {c: maps[c].at(chosen[c]) for c in cases}
    tab = {"default": policy_table(cases, d0, group), "shifted": policy_table(cases, ds, group)}; rng = np.random.default_rng(arm_seed(arm)); n = len(cases); draws = {"gap": [], "absbias": [], "over": [], "bias_d": [], "bias_s": []}
    for row in rng.integers(0, n, (B, n)):
        cs = [cases[i] for i in row]; a = policy_table(cs, {c: d0[c] for c in set(cs)}, group); b = policy_table(cs, {c: ds[c] for c in set(cs)}, group)
        draws["gap"].append(b["detection_gap"] - a["detection_gap"]); draws["absbias"].append(abs(b["median_signed_error"]) - abs(a["median_signed_error"])); draws["over"].append(b["over_20pct"] - a["over_20pct"])
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))      # noqa: E731
    d = {"gap_change": (tab["shifted"]["detection_gap"] - tab["default"]["detection_gap"], *ci(draws["gap"])), "abs_bias_change": (abs(tab["shifted"]["median_signed_error"]) - abs(tab["default"]["median_signed_error"]), *ci(draws["absbias"])),
         "over20_change": (tab["shifted"]["over_20pct"] - tab["default"]["over_20pct"], *ci(draws["over"]))}
    gap_ok = d["gap_change"][2] < 0 and d["gap_change"][0] <= -0.05; bias_ok = d["abs_bias_change"][2] < 0 and d["abs_bias_change"][0] <= -0.05
    verdict = "FIX GENERALISES" if gap_ok and bias_ok else ("PARTIAL" if gap_ok or bias_ok else "DOES NOT GENERALISE")
    return {"tables": tab, "paired": d, "chosen_thresholds_per_fold": per_fold, "gap_criterion_met": bool(gap_ok), "bias_criterion_met": bool(bias_ok), "verdict": verdict, "over_estimation_cost_flag": bool(d["over20_change"][0] >= 0.10 or tab["shifted"]["over_20pct"] >= 0.15)}
