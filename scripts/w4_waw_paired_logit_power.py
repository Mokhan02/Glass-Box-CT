"""W4 scouting (CPU): power of the redesigned paired test (PREREG draft, analysis (b)) by simulation, using the real covariates of the 297 kept paired tumours (`paired_host1/tumours_paired.csv`).
Design: conditional logistic regression with the tumour as stratum, detected ~ phase + bright + |contrast| + is_propagated (log diameter treated as constant within the tumour here; is_propagated = 1 on the series that carries the registered rather than the drawn mask, identified by arterial-drawn against portal-venous-drawn tumours). With two observations per tumour only the DISCORDANT tumours (detected on exactly one phase)
contribute, and the model reduces to an ordinary logistic regression of z = [arterial detected, portal missed] on the differences (bright_A - bright_P, |c_A| - |c_P|, prop_A - prop_P) with an intercept = the phase effect; patient-clustered sandwich CI for the sign coefficient.
Generating model: logit P(detect, tumour i, phase) = alpha_i + b_phase [arterial] + b_sign [bright on that phase] + b_mag |c| / 10 HU + b_prop [propagated mask], alpha_i ~ N(mu, 1.5^2) (mu set for about 70% overall detection). Power = share of simulations with the CI of the sign odds ratio entirely above 1.
Usage: python scripts/w4_waw_paired_logit_power.py <tumours_paired.csv> <out json> [sims]"""
import csv
import json
import sys

import numpy as np


def fit_logit(X, z, ridge=1e-6, iters=50):
    """IRLS logistic regression with a tiny ridge for separation; X includes the intercept column. Returns beta, cov (patient-clustered sandwich is built by the caller)."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9; H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (z - p) - ridge * b
        step = np.linalg.solve(H, g); b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def sandwich_ci(X, z, b, clusters):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9; A = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])); S = np.zeros((X.shape[1],) * 2)
    for c in np.unique(clusters):
        m = clusters == c; s = X[m].T @ (z[m] - p[m]); S += np.outer(s, s)
    n_c = len(np.unique(clusters)); V = A @ S @ A * (n_c / max(n_c - 1, 1)); return np.sqrt(np.diag(V))


def main(path, out, sims=1000):
    rows = [r for r in csv.DictReader(open(path)) if r["category"] == "K"]
    ca = np.array([float(r["c_arterial"]) for r in rows]); cp = np.array([float(r["c_portal"]) for r in rows]); pat = np.array([int(r["patient"]) for r in rows]); n = len(rows)
    prop_a = np.array([r["mask_phase"] == "2" for r in rows], float); prop_p = 1 - prop_a      # the arterial series carries the propagated mask when the mask was drawn on the portal-venous phase, and the reverse
    ba, bp = (ca > 0).astype(float), (cp > 0).astype(float); ma, mp = np.abs(ca) / 10, np.abs(cp) / 10; flip_ap = (ba == 1) & (bp == 0); flip_pa = (ba == 0) & (bp == 1); rng = np.random.default_rng(20261103)
    res = {"kept_tumours": n, "patients": int(len(set(pat))), "tumours_non_flip": int((~(flip_ap | flip_pa)).sum()), "flip_A_bright_P_dark": int(flip_ap.sum()), "flip_A_dark_P_bright": int(flip_pa.sum()), "scenarios": []}
    for b_mag in (0.3,):
        for or_phase, or_prop in ((1.0, 1.0), (1.0, 0.7), (1.5, 0.7), (1.5, 1.0)):
            for or_sign in (1.0, 3.0, 5.0, 10.0, 25.0):
                bs, bph, bpr = np.log(or_sign), np.log(or_phase), np.log(or_prop); hit = 0; disc_tot = []; disc_flip = []; ok = 0
                # mu so that the mean detection is about 0.70 under this scenario (found once by bisection on a draw)
                lo, hi = -6.0, 6.0
                for _ in range(30):
                    mu = (lo + hi) / 2; la = mu + bph + bs * ba + b_mag * ma + bpr * prop_a; lp = mu + bs * bp + b_mag * mp + bpr * prop_p; pm = (1 / (1 + np.exp(-la)) + 1 / (1 + np.exp(-lp))).mean() / 2
                    lo, hi = (mu, hi) if pm < 0.70 else (lo, mu)
                for _ in range(sims):
                    al = rng.normal(mu, 1.5, n); ya = rng.random(n) < 1 / (1 + np.exp(-(al + bph + bs * ba + b_mag * ma + bpr * prop_a))); yp = rng.random(n) < 1 / (1 + np.exp(-(al + bs * bp + b_mag * mp + bpr * prop_p)))
                    d = ya != yp; k = int(d.sum()); disc_tot.append(k); disc_flip.append(int((d & (flip_ap | flip_pa)).sum()))
                    if k < 8:
                        continue
                    X = np.column_stack([np.ones(k), (ba - bp)[d], (ma - mp)[d], (prop_a - prop_p)[d]]); z = ya[d].astype(float)
                    if np.linalg.matrix_rank(X) < 4 or (ba - bp)[d].std() == 0:
                        continue
                    b = fit_logit(X, z); se = sandwich_ci(X, z, b, pat[d]); ok += 1; hit += (b[1] - 1.96 * se[1] > 0)
                res["scenarios"].append({"b_mag_per_10HU": b_mag, "OR_phase": or_phase, "OR_propagated": or_prop, "OR_sign": or_sign, "power_CI_of_sign_OR_above_1": round(hit / sims, 3), "fits_ok": ok, "mean_discordant_tumours": round(float(np.mean(disc_tot)), 1), "mean_discordant_among_sign_flippers": round(float(np.mean(disc_flip)), 1)})
                print(res["scenarios"][-1], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
