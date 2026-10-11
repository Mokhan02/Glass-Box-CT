"""Flag-gated recalibration on the saved logits of the clean 89-case cohort (results/trust/PREREG.md section B). CPU, instance-side (needs cache/rr).
Writes results/trust/gated_recal.json and GATED_RECAL.md. Usage: python scripts/gated_recal_run.py [--rr cache/rr] [--out results/trust]"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.trust.gating import run_gating  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--rr", default="cache/rr"); ap.add_argument("--out", default="results/trust"); ap.add_argument("--recal", type=float, default=-3.0); a = ap.parse_args()
VOX, K = 1.5, 5
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz")); fold_of = {c: i % K for i, c in enumerate(cases)}


def feret(m2):
    pts = np.argwhere(m2).astype(float) * VOX
    if len(pts) < 2:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:
        h = pts
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def lesions(m3):
    lab, n = ndimage.label(m3, structure=np.ones((3, 3, 3))); out = []
    for i in range(1, n + 1):
        out.append({"id": i, "diam": max(feret(lab[:, :, z] == i) for z in np.flatnonzero((lab == i).any(axis=(0, 1))))})
    return lab, out


def lesion_rows(pred, glab, gl):
    """Per target lesion: the reference diameter and what this prediction measured for it (0 if it covers < 10% of the lesion): the data the two-timepoint bias model is fitted on."""
    plab, pl = lesions(pred); pd_ = {l["id"]: l["diam"] for l in pl}; out = []
    for l in gl:
        gm = glab == l["id"]; ov = np.unique(plab[gm & (plab > 0)]); det = (pred & gm).sum() / max(gm.sum(), 1) >= 0.10
        out.append({"gt_diam": l["diam"], "measured": max([pd_[o] for o in ov], default=0.0) if det else 0.0})
    return out


def rel_error(pred, glab, gl):
    """Relative error of the sum of target diameters (as results/readout_refit): a target counts if >= 10% covered; nan if there is no measurable target."""
    if not gl:
        return float("nan")
    plab, pl = lesions(pred); pd_ = {l["id"]: l["diam"] for l in pl}; sg = sp = 0.0
    for l in gl:
        gm = glab == l["id"]; ov = np.unique(plab[gm & (plab > 0)]); det = (pred & gm).sum() / max(gm.sum(), 1) >= 0.10
        sg += l["diam"]; sp += max([pd_[o] for o in ov], default=0.0) if det else 0.0
    return (sp - sg) / sg


def job(c):
    z = np.load(os.path.join(a.rr, c + ".npz")); L = z["L"]; shape = tuple(z["shape"]); base = z["logit_full"].astype(np.float32).ravel(); base[z["roi_idx"].astype(np.int64)] = z["logit_roi"]; base = base.reshape(shape)
    m = base > 0; feats = None
    if m.any():  # the shipped flag's features, on the whole default prediction
        q = np.clip(1.0 / (1.0 + np.exp(-base[m].astype(np.float64))), 1e-6, 1 - 1e-6); feats = [float(np.log1p(m.sum() * VOX ** 3 / 1000.0)), float(q.mean()), float((-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean())]
    ann = (L != 3).any(axis=(0, 1)); gt = (L == 2)[:, :, ann]; glab, gl = lesions(gt); gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]
    out = {"case": c, "fold": fold_of[c], "features": feats, "contrast_hu": float(z["contrast_hu"]), "lesion_rows": []}; per = {}
    for name, t in (("default", 0.0), ("recal", a.recal)):
        pred = (base > t)[:, :, ann]; out[f"dice_{name}"] = float(2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)); out[f"rel_{name}"] = rel_error(pred, glab, gl); per[name] = lesion_rows(pred, glab, gl)
    out["lesion_rows"] = [{"gt_diam": d["gt_diam"], "measured_default": d["measured"], "measured_recal": r["measured"]} for d, r in zip(per["default"], per["recal"])]
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(job, cases, chunksize=2))
    import csv  # per-lesion measurements for the two-timepoint bias model (scripts/longitudinal_fit.py)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "lesion_measurements.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "contrast_hu", "gt_diam_mm", "measured_default_mm", "measured_recal_mm"])
        for r in rows:
            for l in r["lesion_rows"]:
                w.writerow([r["case"], r["contrast_hu"], round(l["gt_diam"], 2), round(l["measured_default"], 2), round(l["measured_recal"], 2)])
    res = run_gating([{k: v for k, v in r.items() if k != "lesion_rows"} for r in rows]); res["n_cases"] = len(rows); res["recal_logit"] = a.recal; os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "gated_recal.json"), "w"), indent=1)
    t = res["policies"]; L = ["# Flag-gated recalibration (pre-registered, results/trust/PREREG.md section B)", "", f"{res['n_cases']} clean-cohort cases ({res['n_cases_with_target']} with a measurable target); 5-fold patient-level CV; the flag is refitted on training folds; {res['n_gated_to_recalibrated']} cases were gated to the recalibrated threshold.", "",
         "| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |", "|---|---|---|---|---|---|"]
    for k in ("default", "recalibrated", "gated"):
        r = t[k]; L.append(f"| {k} | {100 * r['median_signed_error']:+.1f}% | {100 * r['mean_abs_error']:.1f}% | {100 * r['over_20pct']:.1f}% | {100 * r['abs_error_ge_20pct']:.1f}% | {r['mean_dice']:.3f} |")
    g1, g2 = res["gated_minus_recal_over20"], res["gated_minus_default_abs_error"]
    L += ["", f"Gated minus always-recalibrated, over-estimation indicator: {g1['mean']:+.3f} (95% CI {g1['ci'][0]:+.3f} to {g1['ci'][1]:+.3f}).", f"Gated minus always-default, absolute sum-of-diameters error: {g2['mean']:+.3f} (95% CI {g2['ci'][0]:+.3f} to {g2['ci'][1]:+.3f}).", "", f"**Decision (pre-registered rule): {res['decision']}.**"]
    open(os.path.join(a.out, "GATED_RECAL.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
