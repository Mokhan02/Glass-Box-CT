"""Mechanistic experiment A: direct logit attribution (results/mech/PREREG.md, section A). CPU only.
Reads cache/rr/*.npz (window-averaged decoder_F at ROI voxels, labels, original class-26 logits) and w_orig.
Writes results/mech/A_*.csv|json|png and A_SUMMARY.md.
Usage: python scripts/mech_a_dla.py [--rr cache/rr] [--out results/mech]"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--rr", default="cache/rr")
ap.add_argument("--out", default="results/mech")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
VOX, K, TOPK, MINV = 1.5, 5, 8, 20
w = np.load(os.path.join(a.rr, "w_orig.npy")).astype(np.float32)
cases = sorted(f[:-4] for f in os.listdir(a.rr) if f.endswith(".npz") and not f.endswith(".part.npz") and f != "w_orig.npz")
fold_of = {c: i % K for i, c in enumerate(cases)}
P = len(cases)


def job(c):
    z = np.load(os.path.join(a.rr, c + ".npz")); L = z["L"]; idx = z["roi_idx"].astype(np.int64)
    F = z["F"].astype(np.float32); lg = z["logit_roi"]; Lr = L.ravel()[idx]; known = Lr != 3; tum = (Lr == 2); liv = (Lr == 1)
    C = F * w  # per-channel contributions
    recon = float(np.abs(C.sum(1) - lg).max())
    groups = {"det": tum & (lg > 0), "miss": tum & (lg <= 0), "tum": tum, "liver": liv & known}
    out = {"case": c, "contrast": float(z["contrast_hu"]), "recon_err": recon, "n": {}, "mean": {}}
    for g, m in groups.items():
        out["n"][g] = int(m.sum()); out["mean"][g] = C[m].mean(0) if m.sum() >= MINV else None
    out["logit_mean"] = {g: float(lg[m].mean()) if m.sum() >= MINV else None for g, m in groups.items()}
    # probe sample (tumour vs 5-15 mm shell + liver), capped per case
    dist = ndimage.distance_transform_edt(L != 2) * VOX; dr = dist.ravel()[idx]; r = np.random.default_rng(int(c.split("_")[1]))
    shell = np.flatnonzero(known & ~tum & (dr > 5) & (dr <= 15)); lv = np.flatnonzero(known & liv & ~tum); ti = np.flatnonzero(tum)
    pick = lambda ix, n: r.choice(ix, min(n, len(ix)), replace=False) if len(ix) else ix  # noqa: E731
    s = np.concatenate([pick(ti, 4000), pick(shell, 4000), pick(lv, 4000)]); y = np.concatenate([np.ones(min(4000, len(ti))), np.zeros(min(4000, len(shell)) + min(4000, len(lv)))])
    out["probeX"] = F[s]; out["probey"] = y
    return out


def probe_dir(X, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1.0, max_iter=300).fit(X, y); return m.coef_[0]


cos = lambda u, v: float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))  # noqa: E731

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=16) as ex:
        R = list(ex.map(job, cases, chunksize=2))
    print("max |sum c - logit| over all cases:", max(r["recon_err"] for r in R))
    bright = np.array([r["contrast"] > 0 for r in R])
    # ---- per-case missed-vs-detected gap
    has_md = np.array([r["mean"]["det"] is not None and r["mean"]["miss"] is not None for r in R])
    Dc = np.array([r["mean"]["det"] - r["mean"]["miss"] if h else np.zeros(48) for r, h in zip(R, has_md)])
    fold = np.array([fold_of[r["case"]] for r in R])
    top_by_fold = {}
    for f in range(K):
        tr = has_md & (fold != f); top_by_fold[f] = np.argsort(-Dc[tr].mean(0))[:TOPK]
    num = lambda ix: sum(Dc[i][top_by_fold[fold[i]]].sum() for i in ix if has_md[i])  # noqa: E731
    den = lambda ix: sum(Dc[i].sum() for i in ix if has_md[i])  # noqa: E731
    ids = np.arange(P); share = num(ids) / den(ids)
    rng = np.random.default_rng(1); BI = rng.integers(0, P, size=(a.n_boot, P))
    bs = [num(b) / den(b) for b in BI if den(b) != 0]; share_ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    D_all = Dc[has_md].mean(0); order = np.argsort(-D_all); cum = np.cumsum(D_all[order]) / D_all.sum(); kmin = int(np.argmax(cum >= 0.5)) + 1
    # ---- bright vs dark gap (all tumour voxels)
    has_t = np.array([r["mean"]["tum"] is not None for r in R]); Tm = np.array([r["mean"]["tum"] if h else np.zeros(48) for r, h in zip(R, has_t)])
    def bgap(ix):
        b = [i for i in ix if has_t[i] and bright[i]]; d = [i for i in ix if has_t[i] and not bright[i]]
        return Tm[b].mean(0) - Tm[d].mean(0)
    B_all = bgap(ids); top_all = np.argsort(-D_all)[:TOPK]
    shareB = B_all[top_all].sum() / B_all.sum()
    def boot_stats(b):
        Db = Dc[[i for i in b if has_md[i]]].mean(0); Bb = bgap(b); t = np.argsort(-Db)[:TOPK]
        return Bb[t].sum() / Bb.sum(), spearmanr(Db, Bb)[0]
    bb = np.array([boot_stats(b) for b in BI]); rho = float(spearmanr(D_all, B_all)[0])
    shareB_ci = [float(np.nanpercentile(bb[:, 0], 2.5)), float(np.nanpercentile(bb[:, 0], 97.5))]; rho_ci = [float(np.nanpercentile(bb[:, 1], 2.5)), float(np.nanpercentile(bb[:, 1], 97.5))]
    # ---- group means per channel
    def gmean(g, sel=None):
        ix = [i for i, r in enumerate(R) if r["mean"][g] is not None and (sel is None or sel[i])]
        return np.mean([R[i]["mean"][g] for i in ix], 0), len(ix)
    rows = []
    for name, g, sel in (("liver (reference)", "liver", None), ("tumour detected, bright", "det", bright), ("tumour detected, dark", "det", ~bright), ("tumour missed, bright", "miss", bright), ("tumour missed, dark", "miss", ~bright)):
        m, n = gmean(g, sel); rows.append((name, n, m))
    with open(os.path.join(a.out, "A_channel_means.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["channel", "w_j", "D_missed_vs_detected", "B_bright_minus_dark", "selected_top8_all_data"] + [f"{nm} (n={n})" for nm, n, _ in rows])
        for j in range(48):
            wr.writerow([j, f"{w[j]:.4f}", f"{D_all[j]:.4f}", f"{B_all[j]:.4f}", int(j in top_all)] + [f"{m[j]:.4f}" for _, _, m in rows])
    # ---- cosines
    X = np.concatenate([r["probeX"] for r in R]); y = np.concatenate([r["probey"] for r in R]); fo = np.concatenate([[fold[i]] * len(R[i]["probey"]) for i in range(P)])
    wp_all = probe_dir(X, y); cos_all = cos(wp_all, w); cos_f = [cos(probe_dir(X[fo != f], y[fo != f]), w) for f in range(K)]
    # ---- summary
    verdict = ("CLEARLY CONCENTRATED" if share_ci[0] >= 0.5 else "CONCENTRATED") if share >= 0.5 else "NOT CONCENTRATED"
    same = bool(rho_ci[0] > 0 and shareB >= 0.5)
    res = {"n_cases": P, "n_cases_with_both_det_and_miss_voxels": int(has_md.sum()), "max_abs_recon_error": max(r["recon_err"] for r in R),
           "mean_logit_gap_detected_minus_missed": float(D_all.sum()), "heldout_top8_share": float(share), "heldout_top8_share_ci": share_ci, "verdict_concentration": verdict,
           "min_k_for_50pct_all_data_optimistic": kmin, "top8_channels_all_data": [int(j) for j in top_all],
           "bright_minus_dark_gap_total": float(B_all.sum()), "bright_dark_share_same_top8": float(shareB), "bright_dark_share_ci_reselected_each_bootstrap": shareB_ci,
           "spearman_D_vs_B": rho, "spearman_ci": rho_ci, "same_channels_for_bright_dark": same,
           "cos_w_vs_probe_direction_all": cos_all, "cos_w_vs_probe_direction_per_fold": cos_f,
           "counts": {"bright_cases": int(bright.sum()), "dark_cases": int((~bright).sum())}}
    json.dump(res, open(os.path.join(a.out, "A_results.json"), "w"), indent=1); print(json.dumps(res, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); o = np.argsort(-D_all)
    ax[0].bar(range(48), D_all[o], color=["tab:red" if j in top_all else "tab:gray" for j in o]); ax[0].set_title("Missed-vs-detected logit gap by channel (sorted)\nred = top 8", fontsize=9); ax[0].set_xlabel("channel (sorted)"); ax[0].set_ylabel("detected - missed contribution (logit)")
    ax[1].scatter(D_all, B_all, s=14); ax[1].set_xlabel("D_j missed-vs-detected"); ax[1].set_ylabel("B_j bright-minus-dark"); ax[1].set_title(f"Same channels? Spearman {rho:.2f} [{rho_ci[0]:.2f}, {rho_ci[1]:.2f}]", fontsize=9)
    for j in top_all: ax[1].annotate(str(j), (D_all[j], B_all[j]), fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "A_figure.png"), dpi=150)
