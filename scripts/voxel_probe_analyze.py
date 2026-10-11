"""Voxel-probe experiment, step 2. Implements results/voxel_probe/PREREG.md exactly; the decision rule is computed here from the pre-registered numbers.
Usage: python scripts/voxel_probe_analyze.py --vp cache/vp --out results/voxel_probe [--n-boot 1000]
Inputs: per-case token files from voxel_probe_extract.py (trained and random), probability maps in cache/vp_prob, the converted HCC cases."""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--vp", default="cache/vp")
ap.add_argument("--prob", default="cache/vp_prob")
ap.add_argument("--root", default="cache/heldout_nii/hcc_tace_seg")
ap.add_argument("--out", default="results/voxel_probe")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--skip-recovery", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
# ---- pre-registered constants (PREREG.md) ----
X_RATE, LOST_LATE_DROP, LOGIT_FAR, SPEC, CAP, K, C_REG, SEED = 0.50, 0.15, -2.0, 0.95, 200, 5, 0.1, 0
LAYERS = ["mid", "bottleneck", "decoder_F"]; INTERIOR_MM = 2.0
rng = np.random.default_rng(SEED)

cases = sorted(f[:-4] for f in os.listdir(os.path.join(a.vp, "trained")) if f.endswith(".npz"))
P = len(cases); fold_of = {c: i % K for i, c in enumerate(cases)}  # fixed grouped folds: sorted case id, round-robin
route = {c: json.load(open(os.path.join(a.root, c, "segments.json"))).get("label_method", "uid") for c in cases}
B = rng.integers(0, P, size=(a.n_boot, P))  # one set of patient resamples, reused by every row (paired comparisons)


def load(model):
    D = {h: {k: [] for k in ("x", "xyz", "is_tum", "pred_cov", "edt", "hu14", "pid")} for h in LAYERS}; per_case = {}
    for pi, c in enumerate(cases):
        z = np.load(os.path.join(a.vp, model, c + ".npz"))
        for h in LAYERS:
            n = len(z[h + "_is_tum"])
            for k, src in (("x", "x"), ("xyz", "xyz"), ("is_tum", "is_tum"), ("pred_cov", "pred_cov"), ("edt", "edt"), ("hu14", "hu14")):
                D[h][k].append(z[f"{h}_{src}"])
            D[h]["pid"].append(np.full(n, pi))
        per_case[c] = {k: z[k] for k in ("n_comp", "comp_mm3", "gt_tumour_mm3", "logit_fn", "logit_tp", "logit_fn_n", "logit_tp_n", "prev_hu", "prev_lab", "prev_pred", "prev_z")}
        per_case[c].update({f"{h}_comp_retained": z[f"{h}_comp_retained"] for h in LAYERS})
    return {h: {k: np.concatenate(v) for k, v in d.items()} for h, d in D.items()}, per_case


def groups(d):
    tum = d["is_tum"].astype(bool); pc = d["pred_cov"].astype(float); inter = d["edt"].astype(float) > INTERIOR_MM
    return {"TUM": tum, "LIV": ~tum, "TP": tum & (pc >= 0.5), "FN": tum & (pc < 0.5), "FN_int": tum & (pc < 0.5) & inter, "TP_int": tum & (pc >= 0.5) & inter}


def cap_idx(pid, mask):
    out = []
    for p in np.unique(pid[mask]):
        i = np.flatnonzero(mask & (pid == p)); out.append(rng.choice(i, CAP, replace=False) if len(i) > CAP else i)
    return np.concatenate(out) if out else np.array([], int)


def run_probe(Xf, d, g, kind):
    """kind 'A' (TUM vs LIV) or 'B' (TP vs LIV). Returns held-out score and threshold for every token (cross-fitted)."""
    pid = d["pid"]; fold = np.array([fold_of[cases[p]] for p in pid]); score = np.full(len(pid), np.nan); thr_tok = np.full(len(pid), np.nan)
    pos = g["TUM"] if kind == "A" else g["TP"]
    for f in range(K):
        tr, te = fold != f, fold == f
        i1, i0 = cap_idx(pid, pos & tr), cap_idx(pid, g["LIV"] & tr)
        if len(i1) < 20 or len(i0) < 20:
            continue
        sc = StandardScaler().fit(Xf[np.r_[i1, i0]]); y = np.r_[np.ones(len(i1)), np.zeros(len(i0))]
        clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(sc.transform(Xf[np.r_[i1, i0]]), y)
        liv_tr = np.flatnonzero(g["LIV"] & tr); thr = np.percentile(clf.decision_function(sc.transform(Xf[liv_tr])), 100 * SPEC)
        te_i = np.flatnonzero(te); score[te_i] = clf.decision_function(sc.transform(Xf[te_i])); thr_tok[te_i] = thr
    return score, thr_tok


def counts(mask, det, pid):
    n = np.bincount(pid[mask], minlength=P).astype(float); k = np.bincount(pid[mask & det], minlength=P).astype(float); return n, k


def rate_ci(n, k, sel=None):
    sel = np.ones(P, bool) if sel is None else sel; n, k = n * sel, k * sel
    if n.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    boots = (k[B].sum(1) / np.maximum(n[B].sum(1), 1)); ok = n[B].sum(1) > 0
    return k.sum() / n.sum(), *np.percentile(boots[ok], [2.5, 97.5]), int((n > 0).sum())


def auroc(s1, s0):
    r = rankdata(np.r_[s1, s0]); return (r[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))


def auroc_ci(score, mask1, mask0, pid):
    ok = ~np.isnan(score); m1, m0 = mask1 & ok, mask0 & ok
    if m1.sum() < 5 or m0.sum() < 5:
        return np.nan, np.nan, np.nan
    by1 = [score[m1 & (pid == p)] for p in range(P)]; by0 = [score[m0 & (pid == p)] for p in range(P)]; vals = []
    for b in B[:min(len(B), 300)]:  # 300 resamples for AUROC (cost); rates use the full set
        s1, s0 = np.concatenate([by1[i] for i in b]), np.concatenate([by0[i] for i in b])
        if len(s1) and len(s0):
            vals.append(auroc(s1, s0))
    return auroc(score[m1], score[m0]), *np.percentile(vals, [2.5, 97.5])


print(f"{P} patients (uid {sum(r == 'uid' for r in route.values())}, geometry {sum(r == 'geometry' for r in route.values())}); loading tokens ...", flush=True)
data = {m: load(m) for m in ("trained", "random")}
vol = np.array([data["trained"][1][c]["gt_tumour_mm3"] for c in cases]); small = vol < np.median(vol)
is_geo = np.array([route[c] == "geometry" for c in cases]); is_uid = ~is_geo
rows, store = [], {}
for model in ("trained", "random"):
    for h in LAYERS:
        d = data[model][0][h]; g = groups(d)
        for fs_name, Xf in (("activations", d["x"].astype(np.float32)), ("rawHU", d["hu14"].astype(np.float32))):
            if model == "random" and fs_name == "rawHU":
                continue  # the raw-HU control does not depend on the model
            for kind in ("A", "B"):
                score, thr = run_probe(Xf, d, g, kind); det = score > thr; valid = ~np.isnan(score)
                store[(model, h, fs_name, kind)] = (score, thr)
                nI, kI = counts(g["FN_int"] & valid, det, d["pid"]); nT, kT = counts(g["TP"] & valid, det, d["pid"]); nL, kL = counts(g["LIV"] & valid, det, d["pid"])
                variants = (("pooled", None), ("uid", is_uid), ("geometry", is_geo), ("small tumours", small), ("large tumours", ~small))
                for vname, sel in variants:
                    r, lo, hi, npat = rate_ci(nI, kI, sel); rT = rate_ci(nT, kT, sel); rL = rate_ci(nL, kL, sel)
                    pr = np.where(nI > 0, kI / np.maximum(nI, 1), np.nan); selp = np.ones(P, bool) if sel is None else sel
                    row = {"in_distribution": False, "model": model, "layer": h, "features": fs_name, "probe": kind, "subset": vname, "interior_FN_rate": r, "ci_low": lo, "ci_high": hi,
                           "n_patients_with_interior_FN": npat, "interior_FN_tokens": int((nI * selp).sum()), "patient_avg_rate": float(np.nanmean(pr[selp])) if np.isfinite(pr[selp]).any() else np.nan,
                           "TP_rate": rT[0], "TP_ci_low": rT[1], "TP_ci_high": rT[2], "LIV_false_positive_rate": rL[0], "LIV_fpr_ci_low": rL[1], "LIV_fpr_ci_high": rL[2]}
                    if vname == "pooled":
                        row["auroc_FN_vs_LIV"], row["auroc_ci_low"], row["auroc_ci_high"] = auroc_ci(score, g["FN"], g["LIV"], d["pid"])
                    rows.append(row)
                print(f"  {model:7s} {h:10s} {fs_name:11s} probe {kind}: interior-FN detection {kI.sum() / max(nI.sum(), 1):.3f} (n={int(nI.sum())} tokens, {int((nI > 0).sum())} patients)", flush=True)
cols = ["in_distribution", "model", "layer", "features", "probe", "subset", "interior_FN_rate", "ci_low", "ci_high", "n_patients_with_interior_FN", "interior_FN_tokens", "patient_avg_rate",
        "TP_rate", "TP_ci_low", "TP_ci_high", "LIV_false_positive_rate", "LIV_fpr_ci_low", "LIV_fpr_ci_high", "auroc_FN_vs_LIV", "auroc_ci_low", "auroc_ci_high"]
with open(os.path.join(a.out, "probe_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

# ---- resolution loss ----
res = []
for h in LAYERS:
    tot = kept = 0; vanish_cases = 0
    for c in cases:
        rt = data["trained"][1][c][f"{h}_comp_retained"]; tot += len(rt); kept += int(rt.sum()); vanish_cases += int(len(rt) > 0 and not rt.any())
    res.append({"layer": h, "tumour_components": tot, "retained_with_a_tumour_token": kept, "vanished": tot - kept, "cases_with_no_tumour_token": vanish_cases})
    for rn in ("uid", "geometry"):
        t2 = k2 = 0
        for c in cases:
            if route[c] == rn:
                rt = data["trained"][1][c][f"{h}_comp_retained"]; t2 += len(rt); k2 += int(rt.sum())
        res[-1][f"vanished_{rn}"] = f"{t2 - k2}/{t2}"
    mm3 = np.concatenate([data["trained"][1][c]["comp_mm3"] for c in cases]); ret = np.concatenate([data["trained"][1][c][f"{h}_comp_retained"] for c in cases])
    res[-1]["median_mm3_vanished"] = float(np.median(mm3[~ret])) if (~ret).any() else np.nan; res[-1]["median_mm3_retained"] = float(np.median(mm3[ret])) if ret.any() else np.nan
with open(os.path.join(a.out, "resolution_loss.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)

# ---- head check: class-26 logits of interior GT-tumour voxels ----
fn_all = np.concatenate([data["trained"][1][c]["logit_fn"].astype(np.float32) for c in cases]); tp_all = np.concatenate([data["trained"][1][c]["logit_tp"].astype(np.float32) for c in cases])
case_med = np.array([np.median(data["trained"][1][c]["logit_fn"].astype(np.float32)) if data["trained"][1][c]["logit_fn_n"] > 0 else np.nan for c in cases])
head = {"interior_FN_voxels_sampled": int(len(fn_all)), "interior_FN_logit_median": float(np.median(fn_all)), "interior_FN_logit_IQR": [float(x) for x in np.percentile(fn_all, [25, 75])],
        "share_FN_within_1_logit_below_0": float(((fn_all > -1.0) & (fn_all <= 0)).mean()), "share_FN_below_minus_2": float((fn_all < -2.0).mean()), "interior_TP_logit_median": float(np.median(tp_all)) if len(tp_all) else None,
        "per_case_median_FN_logit_quartiles": [float(x) for x in np.nanpercentile(case_med, [25, 50, 75])], "cases_with_interior_FN": int(np.isfinite(case_med).sum()),
        "median_by_route": {rn: float(np.nanmedian(case_med[[route[c] == rn for c in cases]])) for rn in ("uid", "geometry")}}
json.dump(head, open(os.path.join(a.out, "head_check.json"), "w"), indent=1)

# ---- threshold recovery (same folds; threshold chosen on training folds only) ----
if not a.skip_recovery:
    import nibabel as nib
    GRID = [0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0]
    def feret(m2, sx, sy):
        pts = np.argwhere(m2).astype(float) * np.array([sx, sy])
        if len(pts) < 2: return 0.0
        try: hh = pts[ConvexHull(pts).vertices]
        except Exception: hh = pts
        return float(np.sqrt(((hh[:, None] - hh[None]) ** 2).sum(-1)).max())
    def lesions(m3, sx, sy):
        lab, n = ndimage.label(m3, structure=np.ones((3, 3, 3))); out = []
        for i in range(1, n + 1):
            comp = lab == i; out.append({"id": i, "diam": max(feret(comp[:, :, z], sx, sy) for z in np.flatnonzero(comp.any(axis=(0, 1))))})
        return lab, out
    def dice(p, g):
        return 2 * (p & g).sum() / max(p.sum() + g.sum(), 1)
    cache = {}
    for c in cases:
        seg = np.asarray(nib.load(os.path.join(a.root, c, "seg.nii.gz")).dataobj); ann = (seg > 0).any(axis=(0, 1))
        p = np.load(os.path.join(a.prob, c + ".npz"))["p"].astype(np.float32)[:, :, ann]; gt = (seg == 2)[:, :, ann]
        sx, sy = (float(v) for v in nib.load(os.path.join(a.root, c, "ct.nii.gz")).header.get_zooms()[:2]); cache[c] = (p, gt, sx, sy)
    sig = lambda t: 1 / (1 + np.exp(-t))  # noqa: E731
    DM = np.array([[dice(cache[c][0] > sig(t), cache[c][1]) for t in GRID] for c in cases])
    chosen = np.zeros(P, int)
    for f in range(K):
        tr = np.array([fold_of[c] != f for c in cases]); best = int(DM[tr].mean(0).argmax()); chosen[~tr] = best
    d0, d1 = DM[:, 0], DM[np.arange(P), chosen]
    def recist(c, t):
        p, gt, sx, sy = cache[c]; pred = p > sig(t); glab, gl = lesions(gt, sx, sy); gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]
        if not gl: return None
        plab, pl = lesions(pred, sx, sy); pd_ = {l["id"]: l["diam"] for l in pl}; sg = sp = 0.0
        for l in gl:
            gm = glab == l["id"]; ov = np.unique(plab[gm & (plab > 0)]); det = (pred & gm).sum() / max(gm.sum(), 1) >= 0.10
            sg += l["diam"]; sp += max([pd_[o] for o in ov], default=0.0) if det else 0.0
        return (sp - sg) / sg
    r0 = [recist(c, 0.0) for c in cases]; r1 = [recist(c, GRID[chosen[i]]) for i, c in enumerate(cases)]
    def summ(r, sel):
        v = np.array([x for x, s in zip(r, sel) if x is not None and s], float)
        return {"n": int(len(v)), "median_rel_error": float(np.median(v)), "frac_abs_ge_20pct": float((np.abs(v) >= 0.2).mean()), "frac_abs_ge_30pct": float((np.abs(v) >= 0.3).mean()), "frac_overestimate_ge_20pct": float((v >= 0.2).mean())}
    rec_rows = []
    for vname, sel in (("pooled", np.ones(P, bool)), ("uid", is_uid), ("geometry", is_geo)):
        diff = d1[sel] - d0[sel]; bd = [(d1[sel][b] - d0[sel][b]).mean() for b in rng.integers(0, sel.sum(), size=(a.n_boot, sel.sum()))]
        rec_rows.append({"subset": vname, "n_cases": int(sel.sum()), "mean_dice_threshold_0": float(d0[sel].mean()), "mean_dice_recovered": float(d1[sel].mean()), "dice_gain": float(diff.mean()),
                         "gain_ci_low": float(np.percentile(bd, 2.5)), "gain_ci_high": float(np.percentile(bd, 97.5)),
                         **{f"recist_{k}_at_0": v for k, v in summ(r0, sel).items()}, **{f"recist_{k}_recovered": v for k, v in summ(r1, sel).items()}})
    with open(os.path.join(a.out, "threshold_recovery.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec_rows[0])); w.writeheader(); w.writerows(rec_rows)
    head["thresholds_chosen_per_fold"] = {int(f): GRID[int(chosen[[fold_of[c] == f for c in cases]][0])] for f in range(K)}
    json.dump(head, open(os.path.join(a.out, "head_check.json"), "w"), indent=1)

# ---- decision rule (mechanical, per PREREG) ----
def get(model, layer, fs, kind="A", subset="pooled"):
    return next(r for r in rows if r["model"] == model and r["layer"] == layer and r["features"] == fs and r["probe"] == kind and r["subset"] == subset)
enc, hu = get("trained", "mid", "activations"), get("trained", "mid", "rawHU"); dec = get("trained", "decoder_F", "activations"); rnd = get("random", "mid", "activations")
represents = enc["interior_FN_rate"] >= X_RATE and enc["ci_low"] > hu["ci_high"]
never = enc["ci_low"] <= hu["ci_high"]
# paired bootstrap of (decoder_F - mid) interior-FN rate
sE, tE = store[("trained", "mid", "activations", "A")]; sD, tD = store[("trained", "decoder_F", "activations", "A")]
dE, dD = data["trained"][0]["mid"], data["trained"][0]["decoder_F"]
nE, kE = counts(groups(dE)["FN_int"] & ~np.isnan(sE), sE > tE, dE["pid"]); nD, kD = counts(groups(dD)["FN_int"] & ~np.isnan(sD), sD > tD, dD["pid"])
diffs = (kD[B].sum(1) / np.maximum(nD[B].sum(1), 1)) - (kE[B].sum(1) / np.maximum(nE[B].sum(1), 1)); dlo, dhi = np.percentile(diffs, [2.5, 97.5]); dpoint = kD.sum() / nD.sum() - kE.sum() / nE.sum()
late_a = (dpoint <= -LOST_LATE_DROP) and dhi < 0; late_b = head["interior_FN_logit_median"] <= LOGIT_FAR
if represents and (late_a or late_b): verdict = "LOST LATE: the encoder represents the missed tissue and the loss happens later (decoder_F detection or head logits clearly lower)"
elif represents: verdict = "ENCODER REPRESENTS the missed tissue (and the pre-registered 'lost late' criteria are not met)"
elif never: verdict = "NEVER REPRESENTED: the mid-encoder probe does not beat the raw-HU control"
else: verdict = "INCONCLUSIVE"
rand_note = "random-init CI overlaps trained CI" if rnd["ci_high"] >= enc["ci_low"] and rnd["ci_low"] <= enc["ci_high"] else "random-init CI separate from trained"
dec_out = {"rule": "PREREG.md", "R_enc": enc["interior_FN_rate"], "CI_enc": [enc["ci_low"], enc["ci_high"]], "R_hu": hu["interior_FN_rate"], "CI_hu": [hu["ci_low"], hu["ci_high"]], "X": X_RATE,
           "represents": bool(represents), "never": bool(never), "decoder_F_rate": dec["interior_FN_rate"], "decoder_minus_mid": [float(dpoint), float(dlo), float(dhi)], "late_criterion_a": bool(late_a),
           "median_FN_logit": head["interior_FN_logit_median"], "late_criterion_b": bool(late_b), "random_init_rate": rnd["interior_FN_rate"], "random_init_CI": [rnd["ci_low"], rnd["ci_high"]], "random_note": rand_note, "verdict": verdict}
json.dump(dec_out, open(os.path.join(a.out, "decision.json"), "w"), indent=1, default=float)
print("\nDECISION:", json.dumps(dec_out, indent=1, default=float))
np.savez(os.path.join(a.out, "_scores_mid_trained.npz"), score=sE, thr=tE, pid=dE["pid"], xyz=dE["xyz"], is_tum=dE["is_tum"], pred_cov=dE["pred_cov"], edt=dE["edt"], cases=np.array(cases))
