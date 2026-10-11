"""SENSITIVITY ANALYSIS (not pre-registered): flag-gated recalibration with nested cross-validation. The recalibrated threshold AND the flag's operating point are chosen inside each
training fold only. CPU, instance-side (saved logits in cache/rr). Writes results/trust/{gated_nested.json,GATED_NESTED.md,gated_quote.json}.
gated_quote.json holds the numbers the app and README quote: the nested analysis (see results/trust/DEVIATIONS.md #1).
Usage: python scripts/gated_recal_nested.py [--rr cache/rr] [--out results/trust]"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

sys.path.insert(0, ".")
from glassbox.trust.gating import T_GRID, run_nested  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--rr", default="cache/rr"); ap.add_argument("--out", default="results/trust"); a = ap.parse_args()
VOX, K = 1.5, 5; TS = (0.0,) + tuple(T_GRID)
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


def rel_error(pred, glab, gl):
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
    if m.any():
        q = np.clip(1.0 / (1.0 + np.exp(-base[m].astype(np.float64))), 1e-6, 1 - 1e-6); feats = [float(np.log1p(m.sum() * VOX ** 3 / 1000.0)), float(q.mean()), float((-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean())]
    ann = (L != 3).any(axis=(0, 1)); gt = (L == 2)[:, :, ann]; glab, gl = lesions(gt); gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]; rel, dice = {}, {}
    for t in TS:
        pred = (base > t)[:, :, ann]; dice[t] = float(2 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)); rel[t] = rel_error(pred, glab, gl)
    return {"fold": fold_of[c], "features": feats, "dice": dice, "rel": rel}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=24) as ex:
        rows = list(ex.map(job, cases, chunksize=2))
    res = run_nested(rows); os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, "gated_nested.json"), "w"), indent=1)
    pre = json.load(open(os.path.join(a.out, "gated_recal.json")))
    ci = lambda x, pc=True: f"{100 * x['value']:+.1f}% [{100 * x['ci'][0]:+.1f}, {100 * x['ci'][1]:+.1f}]" if pc else f"{x['value']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"  # noqa: E731
    L = ["# Flag-gated recalibration: nested-CV SENSITIVITY analysis (not pre-registered)", "", "The recalibrated threshold (grid -1.0 to -4.0 logit) and the flag's sensitivity target (0.80 to 0.95) are chosen inside each outer training fold only, by an inner 4-fold cross-validation of the gated policy's mean absolute error; always-recalibrated gets its own threshold chosen inside the training fold. 95% patient-bootstrap CIs in brackets.", "",
         f"{res['n_cases_with_target']} cases with a measurable target.", "", "| policy | median signed error | mean abs error | over-estimated >= 20% | abs error >= 20% | mean Dice |", "|---|---|---|---|---|---|"]
    for k in ("default", "recalibrated", "gated"):
        p = res["policies"][k]; L.append(f"| {k} | {ci(p['median_signed_error'])} | {100 * p['mean_abs_error']['value']:.1f}% [{100 * p['mean_abs_error']['ci'][0]:.1f}, {100 * p['mean_abs_error']['ci'][1]:.1f}] | {100 * p['over_20pct']['value']:.1f}% [{100 * p['over_20pct']['ci'][0]:.1f}, {100 * p['over_20pct']['ci'][1]:.1f}] | {100 * p['abs_error_ge_20pct']['value']:.1f}% | {p['mean_dice']['value']:.3f} [{p['mean_dice']['ci'][0]:.3f}, {p['mean_dice']['ci'][1]:.3f}] |")
    g1, g2 = res["gated_minus_recal_over20"], res["gated_minus_default_abs_error"]
    L += ["", f"Gated minus always-recalibrated, over-estimation indicator: {g1['mean']:+.3f} (95% CI {g1['ci'][0]:+.3f} to {g1['ci'][1]:+.3f}). Gated minus always-default, absolute error: {g2['mean']:+.3f} (95% CI {g2['ci'][0]:+.3f} to {g2['ci'][1]:+.3f}). The pre-registered rule applied to this analysis would {'ship' if res['pre_registered_rule_would_ship'] else 'NOT ship'} the gated policy.", "", "Parameters chosen per outer fold:", ""]
    L += [f"- fold {f}: gated threshold {v['gated_recalibrated_logit']}, flag sensitivity {v['gated_flag_sensitivity']} ({v['n_gated']} held-out cases gated); always-recalibrated threshold {v['recalibrated_logit_for_always_recal']}" for f, v in res["chosen_per_fold"].items()]
    pg, ng = pre["policies"]["gated"]["mean_abs_error"], res["policies"]["gated"]["mean_abs_error"]["value"]; po, no = pre["policies"]["gated"]["over_20pct"], res["policies"]["gated"]["over_20pct"]["value"]
    q = {"quoted": "nested sensitivity analysis", "rule": "the nested analysis is quoted: it is the stricter test (no threshold or flag operating point is chosen with any information from the held-out fold) and it gives the less favourable over-estimation rate and fails the pre-registered shipping rule",
         "gated_mean_abs_error": {"pre_registered": pg, "nested": ng}, "gated_over_20pct": {"pre_registered": po, "nested": no}, "pre_registered_rule_would_ship": {"pre_registered": pre["ship_gated_policy"], "nested": res["pre_registered_rule_would_ship"]}, "n": res["n_cases_with_target"]}
    q.update({k: {m: res["policies"][k][m]["value"] for m in ("median_signed_error", "mean_abs_error", "over_20pct", "mean_dice")} for k in ("default", "recalibrated", "gated")}); q["gated_over_20pct_ci"] = res["policies"]["gated"]["over_20pct"]["ci"]
    json.dump(q, open(os.path.join(a.out, "gated_quote.json"), "w"), indent=1)
    L += ["", f"**Quoted in the app and README: the nested sensitivity analysis.** Gated over-estimation {100 * no:.1f}% here vs {100 * po:.1f}% pre-registered; gated mean absolute error {100 * ng:.1f}% vs {100 * pg:.1f}%; this analysis does {'' if res['pre_registered_rule_would_ship'] else 'NOT '}meet the pre-registered shipping rule (see results/trust/DEVIATIONS.md #1 for why the quoting rule was changed after both results were seen)."]
    open(os.path.join(a.out, "GATED_NESTED.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
