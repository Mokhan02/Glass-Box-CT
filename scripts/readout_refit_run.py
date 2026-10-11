"""Readout re-fit, step 2 (results/readout_refit/PREREG.md + DEVIATIONS.md): fit, 5-fold patient-level CV, evaluate, decide.
Reads cache/rr/*.npz written by readout_refit_extract.py. Writes results/readout_refit/{per_case.csv,table.csv,TABLE.md,DECISION.json,
descriptive.json,figure_refit.png} and cache/rr_models.npz (per-fold readouts, for the overlays).
Usage: python scripts/readout_refit_run.py [--rr cache/rr] [--out results/readout_refit]"""
import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage
from scipy.spatial import ConvexHull

ap = argparse.ArgumentParser()
ap.add_argument("--rr", default="cache/rr")
ap.add_argument("--out", default="results/readout_refit")
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
VOX = 1.5
K = 5
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
TGRID = [-0.25 * i for i in range(0, 21)]  # 0 ... -5.0
STEPS, LR = 300, 1e-2
SEED = 0
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w_orig = np.load(os.path.join(a.rr, "w_orig.npy")).astype(np.float32)
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz") and f != "w_orig.npz")
cases = cases[: a.limit or None]
fold_of = {c: i % K for i, c in enumerate(cases)}
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- sampling
def sample_case(c):
    z = np.load(os.path.join(a.rr, c + ".npz"))
    L = z["L"]; idx = z["roi_idx"].astype(np.int64); F = z["F"]
    Lr = L.ravel()[idx]; known = Lr != 3; tum = Lr == 2
    dist = ndimage.distance_transform_edt(L != 2) * VOX  # distance to the nearest GT tumour voxel (annotated or not: unknown slices hold no label 2)
    dr = dist.ravel()[idx]
    ti = np.flatnonzero(tum)
    shell = np.flatnonzero(known & ~tum & (dr > 5) & (dr <= 15))
    liv = np.flatnonzero(known & (Lr == 1))
    r = np.random.default_rng([SEED, int(c.split("_")[1])])
    ns = min(len(shell), max(3000, len(ti))); nl = min(len(liv), max(3000, 3 * len(ti)))
    sel = np.concatenate([ti, r.choice(shell, ns, replace=False) if ns else shell[:0], r.choice(liv, nl, replace=False) if nl else liv[:0]])
    y = np.concatenate([np.ones(len(ti), np.float32), np.zeros(ns + nl, np.float32)])
    kind = np.concatenate([np.zeros(len(ti), np.int8), np.ones(ns, np.int8), np.full(nl, 2, np.int8)])
    return F[sel], y, kind, float(z["contrast_hu"]), str(z["route"])


print(f"{len(cases)} cases; sampling", flush=True)
S = {}
for c in cases:
    S[c] = sample_case(c)
contrast = {c: S[c][3] for c in cases}
print("sampled voxels:", sum(len(S[c][1]) for c in cases), "of which tumour", int(sum(S[c][1].sum() for c in cases)), flush=True)


# ----------------------------------------------------------------- fitting
def gather(cs):
    X = torch.from_numpy(np.concatenate([S[c][0] for c in cs])).to(dev).float()
    y = torch.from_numpy(np.concatenate([S[c][1] for c in cs])).to(dev)
    return X, y


def soft_dice(p, y):
    return 2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6)


def fit(X, y, lam):
    w0 = torch.from_numpy(w_orig).to(dev)
    w = w0.clone().requires_grad_(True); b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        z = X @ w + b
        loss = Fn.binary_cross_entropy_with_logits(z, y) + (1 - soft_dice(torch.sigmoid(z), y)) + lam * ((w - w0) ** 2).sum()
        loss.backward(); opt.step()
    return w.detach(), b.detach()


models, lam_chosen, inner_scores = {}, {}, {}
for f in range(K):
    tr = [c for c in cases if fold_of[c] != f]
    order = rng.permutation(len(tr)); inner = {c: i % 4 for i, c in enumerate(np.array(tr)[order])}
    sc = np.zeros(len(LAMS))
    for g in range(4):
        itr, iva = [c for c in tr if inner[c] != g], [c for c in tr if inner[c] == g]
        Xt, yt = gather(itr); Xv, yv = gather(iva)
        for j, lam in enumerate(LAMS):
            w, b = fit(Xt, yt, lam); sc[j] += float(soft_dice(torch.sigmoid(Xv @ w + b), yv)) / 4
        del Xt, Xv
    lam = LAMS[int(sc.argmax())]; lam_chosen[f] = lam; inner_scores[f] = sc.tolist()
    X, y = gather(tr); w, b = fit(X, y, lam); del X
    models[f] = (w.cpu().numpy(), float(b.cpu()))
    print(f"fold {f}: lambda {lam} (inner soft-Dice {np.round(sc, 3).tolist()}); |w-w0|={np.linalg.norm(models[f][0] - w_orig):.3f} b={models[f][1]:.3f}", flush=True)
np.savez(os.path.join("cache", "rr_models.npz") if os.path.isdir("cache") else os.path.join(a.out, "rr_models.npz"),
         w=np.stack([models[f][0] for f in range(K)]), b=np.array([models[f][1] for f in range(K)]), w_orig=w_orig)


# ----------------------------------------------------------------- measurement (workers)
def feret(m2):
    pts = np.argwhere(m2).astype(float) * VOX
    if len(pts) < 2:
        return 0.0
    try:
        hh = pts[ConvexHull(pts).vertices]
    except Exception:
        hh = pts
    return float(np.sqrt(((hh[:, None] - hh[None]) ** 2).sum(-1)).max())


def lesions(m3):
    lab, n = ndimage.label(m3, structure=np.ones((3, 3, 3))); out = []
    for i in range(1, n + 1):
        comp = lab == i
        out.append({"id": i, "diam": max(feret(comp[:, :, z]) for z in np.flatnonzero(comp.any(axis=(0, 1))))})
    return lab, out


def measure(pred, gt, glab, gl):
    """Dice (annotated slices, cropped box), sum-of-diameters relative error (None if no >=10 mm target), n targets, n detected."""
    dice = 2.0 * (pred & gt).sum() / max(pred.sum() + gt.sum(), 1)
    if not gl:
        return dice, None, 0, 0
    plab, pl = lesions(pred); pd_ = {l["id"]: l["diam"] for l in pl}; sg = sp = 0.0; nd = 0
    for l in gl:
        gm = glab == l["id"]; ov = np.unique(plab[gm & (plab > 0)]); det = (pred & gm).sum() / max(gm.sum(), 1) >= 0.10
        sg += l["diam"]; nd += int(det); sp += max([pd_[o] for o in ov], default=0.0) if det else 0.0
    return dice, (sp - sg) / sg, len(gl), nd


def eval_job(args):
    c, rr, W, B, tgrid = args
    z = np.load(os.path.join(rr, c + ".npz"))
    L = z["L"]; shape = tuple(z["shape"]); idx = z["roi_idx"].astype(np.int64)
    base = z["logit_full"].astype(np.float32).ravel(); base[idx] = z["logit_roi"]
    roi = np.zeros(base.shape, bool); roi[idx] = True
    ann = (L != 3).any(axis=(0, 1))
    box = np.argwhere((roi | (L == 2).ravel() | (base > 0)).reshape(shape))
    lo, hi = box.min(0), box.max(0) + 1
    sl = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(None))
    zsel = np.flatnonzero(ann)
    crop = lambda v: v.reshape(shape)[sl][:, :, zsel]  # noqa: E731
    gt = crop((L == 2).ravel()); roi_c = crop(roi); base_c = crop(base)
    glab, gl = lesions(gt); gl = sorted([l for l in gl if l["diam"] >= 10.0], key=lambda l: -l["diam"])[:2]
    out = {"case": c, "orig": measure(base_c > 0, gt, glab, gl), "thr": {}, "refit": {}}
    for t in tgrid:
        out["thr"][t] = measure(np.where(roi_c, base_c > t, base_c > 0), gt, glab, gl)
    F = z["F"].astype(np.float32)
    for f, (w, b) in enumerate(zip(W, B)):
        v = base.copy(); v[idx] = F @ w + b
        out["refit"][f] = measure(crop(v) > 0, gt, glab, gl)
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=20) as ex:
        res = {r["case"]: r for r in ex.map(eval_job, [(c, a.rr, [models[f][0] for f in range(K)], [models[f][1] for f in range(K)], TGRID) for c in cases], chunksize=1)}
    print("evaluated", len(res), "cases", flush=True)

    def over(vals):  # fraction with relative error >= +20% among cases with a measurable target
        v = [m[1] for m in vals if m[1] is not None]
        return float(np.mean([x >= 0.2 for x in v])) if v else float("nan")

    # ---- per-fold matched threshold (training folds only)
    t_match, t_dice, train_rate = {}, {}, {}
    for f in range(K):
        tr = [c for c in cases if fold_of[c] != f]
        rate_refit = over([res[c]["refit"][f] for c in tr]); train_rate[f] = rate_refit
        rates = np.array([over([res[c]["thr"][t] for c in tr]) for t in TGRID])
        t_match[f] = TGRID[int(np.argmin(np.abs(rates - rate_refit)))]  # argmin returns the first = least negative on ties
        dm = np.array([np.mean([res[c]["thr"][t][0] for c in tr]) for t in TGRID]); t_dice[f] = TGRID[int(dm.argmax())]
    held = {c: {"orig": res[c]["orig"], "refit": res[c]["refit"][fold_of[c]], "thr": res[c]["thr"][t_match[fold_of[c]]],
                "thrdice": res[c]["thr"][t_dice[fold_of[c]]]} for c in cases}
    arms = ["orig", "thr", "thrdice", "refit"]
    names = {"orig": "original readout", "thr": "threshold shift (matched over-estimation)", "thrdice": "threshold shift (Dice-optimal, extra)", "refit": "re-fit readout"}
    with open(os.path.join(a.out, "per_case.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh); w_.writerow(["case", "fold", "contrast_hu"] + [f"{m}_{k}" for m in arms for k in ("dice", "relerr", "n_targets", "n_detected")])
        for c in cases:
            w_.writerow([c, fold_of[c], contrast[c]] + [x for m in arms for x in held[c][m]])
    P = len(cases); brng = np.random.default_rng(1); BI = brng.integers(0, P, size=(a.n_boot, P))

    def metrics(arm, ix):
        cs = [cases[i] for i in ix]; v = [held[c][arm] for c in cs]
        rel = np.array([m[1] for m in v if m[1] is not None]); nt = sum(m[2] for m in v); nd = sum(m[3] for m in v)
        return {"dice": float(np.mean([m[0] for m in v])), "detection": nd / max(nt, 1), "sld_median_rel_error": float(np.median(rel)),
                "frac_abs_ge_20": float((np.abs(rel) >= 0.2).mean()), "frac_over_ge_20": float((rel >= 0.2).mean())}

    rows = []
    allix = np.arange(P)
    for arm in arms:
        pt = metrics(arm, allix); boots = [metrics(arm, b) for b in BI[:1000]]
        row = {"arm": names[arm], "n_cases": P}
        for k, v in pt.items():
            lo_, hi_ = np.percentile([b[k] for b in boots], [2.5, 97.5]); row[k] = v; row[k + "_ci_low"] = float(lo_); row[k + "_ci_high"] = float(hi_)
        rows.append(row)
    with open(os.path.join(a.out, "table.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    with open(os.path.join(a.out, "TABLE.md"), "w") as fh:
        fh.write("| arm | tumour Dice | lesion detection | SLD median error | % abs err >= 20% | % OVER-estimated >= 20% |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f3 = lambda k, p=False: (f"{100 * r[k]:.1f}%" if p else f"{r[k]:.3f}") + f" [{(100 if p else 1) * r[k + '_ci_low']:.{1 if p else 3}f}, {(100 if p else 1) * r[k + '_ci_high']:.{1 if p else 3}f}]"  # noqa: E731
            fh.write(f"| {r['arm']} | {f3('dice')} | {f3('detection', True)} | {f3('sld_median_rel_error', True)} | {f3('frac_abs_ge_20', True)} | {f3('frac_over_ge_20', True)} |\n")

    # ---- decision (rule fixed in PREREG): per-case Dice, re-fit minus matched threshold, patient bootstrap
    dd = np.array([held[c]["refit"][0] - held[c]["thr"][0] for c in cases]); gain = float(dd.mean())
    lo_, hi_ = np.percentile([dd[b].mean() for b in BI], [2.5, 97.5])
    over_refit, over_thr = metrics("refit", allix)["frac_over_ge_20"], metrics("thr", allix)["frac_over_ge_20"]
    if lo_ > 0 and (over_refit - over_thr) <= 0.05:
        verdict = "BEATS THRESHOLD"
    elif hi_ < 0:
        verdict = "WORSE THAN THRESHOLD"
    else:
        verdict = "NO BETTER THAN THRESHOLD"
    dec = {"rule": "results/readout_refit/PREREG.md", "dice_gain_refit_minus_matched_threshold": gain, "ci": [float(lo_), float(hi_)],
           "heldout_over20_refit": over_refit, "heldout_over20_threshold": over_thr, "over_diff_pp": 100 * (over_refit - over_thr),
           "verdict": verdict, "lambda_per_fold": lam_chosen, "inner_scores": inner_scores, "matched_t_per_fold": t_match, "dice_opt_t_per_fold": t_dice,
           "train_fold_refit_over20": train_rate, "n_cases": P}
    json.dump(dec, open(os.path.join(a.out, "DECISION.json"), "w"), indent=1)
    print(json.dumps({k: dec[k] for k in ("dice_gain_refit_minus_matched_threshold", "ci", "heldout_over20_refit", "heldout_over20_threshold", "verdict")}, indent=1))
    print(open(os.path.join(a.out, "TABLE.md")).read())

    # ---- descriptive extras
    Xp = np.concatenate([S[c][0] for c in cases]).astype(np.float32); yp = np.concatenate([S[c][1] for c in cases])
    Xg, yg = torch.from_numpy(Xp).to(dev), torch.from_numpy(yp).to(dev)
    wp = torch.zeros(48, device=dev, requires_grad=True); bp = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([wp, bp], lr=LR)
    for _ in range(300):
        opt.zero_grad(); Fn.binary_cross_entropy_with_logits(Xg @ wp + bp, yg).backward(); opt.step()
    wprobe = wp.detach().cpu().numpy()  # logistic probe on decoder_F, tumour vs (shell + liver); no pre-trained init
    cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))  # noqa: E731
    desc = {"cos_probe_vs_original": cos(wprobe, w_orig), "cos_refit_vs_original_per_fold": [cos(models[f][0], w_orig) for f in range(K)]}
    uo, up = w_orig / np.linalg.norm(w_orig), wprobe / np.linalg.norm(wprobe)
    grp = {"bright (contrast>0)": [c for c in cases if contrast[c] > 0], "dark (contrast<=0)": [c for c in cases if contrast[c] <= 0]}
    proj = {}
    for gname, cs in grp.items():
        acc = {"missed": [[], []], "detected": [[], []]}
        for c in cs:
            Fc, yc = S[c][0].astype(np.float32), S[c][1]; tf = Fc[yc == 1]
            lg = tf @ w_orig; miss = lg <= 0
            for nm, m in (("missed", miss), ("detected", ~miss)):
                if m.sum():
                    acc[nm][0].append(float((tf[m] @ uo).mean())); acc[nm][1].append(float((tf[m] @ up).mean()))
        proj[gname] = {nm: {"n_cases": len(v[0]), "mean_projection_onto_original_readout": float(np.mean(v[0])) if v[0] else None,
                            "mean_projection_onto_probe_direction": float(np.mean(v[1])) if v[1] else None} for nm, v in acc.items()}
    desc["projection_by_group"] = proj
    sub = {}
    for gname, cs in grp.items():
        ii = [cases.index(c) for c in cs]
        sub[gname] = {"n": len(ii), "dice_orig": float(np.mean([held[c]["orig"][0] for c in cs])), "dice_thr": float(np.mean([held[c]["thr"][0] for c in cs])), "dice_refit": float(np.mean([held[c]["refit"][0] for c in cs]))}
    desc["exploratory_subgroups_dice"] = sub
    json.dump(desc, open(os.path.join(a.out, "descriptive.json"), "w"), indent=1)
    print(json.dumps(desc, indent=1))

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for axx, k, ttl, pc in ((ax[0], "dice", "Tumour Dice (held-out folds)", False), (ax[1], "sld_median_rel_error", "Sum-of-diameters median error", True)):
        for i, r in enumerate(rows):
            m = r[k] * (100 if pc else 1); axx.errorbar(i, m, yerr=[[m - r[k + "_ci_low"] * (100 if pc else 1)], [r[k + "_ci_high"] * (100 if pc else 1) - m]], fmt="o", capsize=4)
        axx.set_xticks(range(len(rows))); axx.set_xticklabels(["original", "threshold\n(matched)", "threshold\n(Dice-opt.)", "re-fit"], fontsize=8)
        axx.set_title(ttl, fontsize=10); axx.set_ylabel("%" if pc else "Dice")
        if pc: axx.axhline(0, color="grey", lw=0.8)
    fig.suptitle(f"Class-26 readout re-fit, 5-fold patient-level CV, n={P} (verdict: {verdict})", fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "figure_refit.png"), dpi=150)
