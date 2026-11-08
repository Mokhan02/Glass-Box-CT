"""W4b power simulation (CPU, synthetic outcomes; only the covariates of the 297 kept W4 tumours are used: patient, drawn phase, sign of the contrast on each series).
Estimator (PREREG W4b draft): bright penalty theta = standardised difference-in-differences of (detected on the portal-venous series - detected on the arterial series) between the A-bright/P-dark flippers (F1) and the non-flipping tumours (NF),
standardised over the drawn-phase strata with the F1 stratum shares (this holds the propagated-mask asymmetry fixed). Positive theta = bright detected less. Patient-clustered percentile bootstrap CI (B bootstrap draws).
Generating model: logit P(detect) = u_patient + e_tumour (total SD 1.5) + b_phase [arterial] + b_sign [bright on that series] + b_prop [propagated mask]; the mean detection of the dark, non-propagated, portal series is set to about 0.75.
Usage: python results/w4b_waw/w4b_did_power.py <tumours_paired.csv> <out.json> [sims] [B]"""
import csv
import json
import sys

import numpy as np


def theta_hat(det_a, det_p, f1, nf, drawn_a):
    """Standardised DiD of d = det_p - det_a; strata = drawn phase (True = arterial-drawn); weights = F1 stratum shares."""
    d = det_p.astype(float) - det_a.astype(float); tot = f1.sum(); th = 0.0
    for s in (True, False):
        a = f1 & (drawn_a == s); b = nf & (drawn_a == s)
        if a.sum() == 0:
            continue
        if b.sum() == 0:
            return np.nan
        th += a.sum() / tot * (d[a].mean() - d[b].mean())
    return th


def main(path, out, sims=300, B=300):
    rows = [r for r in csv.DictReader(open(path)) if r["category"] == "K"]; ca = np.array([float(r["c_arterial"]) for r in rows]); cp = np.array([float(r["c_portal"]) for r in rows]); pat = np.array([int(r["patient"]) for r in rows]); drawn_a = np.array([r["mask_phase"] == "1" for r in rows])
    ba, bp = ca > 0, cp > 0; f1 = ba & ~bp; nf = (ba == bp); n = len(rows); up = np.unique(pat); idx = [np.flatnonzero(pat == p) for p in up]
    prop_a = ~drawn_a; prop_p = drawn_a      # the arterial series carries the propagated mask when the mask was drawn on the portal-venous phase
    rng = np.random.default_rng(20261108); res = {"tumours": int(n), "patients": int(len(up)), "F1_A_bright_P_dark": int(f1.sum()), "NF": int(nf.sum()), "A_dark_P_bright_descriptive_only": int((~ba & bp).sum()), "scenarios": []}
    for b_sign in (0.0, -0.5, -1.0, -2.3):
        b_phase, b_prop = np.log(0.15), np.log(0.7); mu = 1.5
        lo, hi = -2.0, 6.0
        for _ in range(25):      # mu so that the mean detection of the dark portal-venous non-propagated series is about 0.75
            mu = (lo + hi) / 2; p = 1 / (1 + np.exp(-mu)); lo, hi = (mu, hi) if p < 0.75 else (lo, mu)
        pen_hit = null_hit = null25 = rev_hit = 0; ths = []; ok = 0
        for _ in range(sims):
            u = rng.normal(0, 1.0, len(up))[np.searchsorted(up, pat)]; e = rng.normal(0, 1.1, n); al = mu + u + e
            pa = 1 / (1 + np.exp(-(al + b_phase + b_sign * ba + b_prop * prop_a))); pp = 1 / (1 + np.exp(-(al + b_sign * bp + b_prop * prop_p)))
            da = rng.random(n) < pa; dp = rng.random(n) < pp; th = theta_hat(da, dp, f1, nf, drawn_a)
            if not np.isfinite(th):
                continue
            bs = []
            for _b in range(B):
                pick = rng.integers(0, len(up), len(up)); r = np.concatenate([idx[i] for i in pick]); t = theta_hat(da[r], dp[r], f1[r], nf[r], drawn_a[r])
                if np.isfinite(t):
                    bs.append(t)
            lo_, hi_ = np.percentile(bs, [2.5, 97.5]); ok += 1; ths.append(th)
            pen_hit += (lo_ > 0 and th >= 0.20); null_hit += (lo_ > -0.20 and hi_ < 0.20); null25 += (lo_ > -0.25 and hi_ < 0.25); rev_hit += (hi_ < 0 and th <= -0.20)
        res["scenarios"].append({"b_sign_logit": b_sign, "OR_sign_bright_vs_dark": round(float(np.exp(b_sign)), 3), "mean_theta": round(float(np.mean(ths)), 3), "P_BRIGHT_PENALTY_label": round(pen_hit / ok, 3), "P_NO_PENALTY_label_(CI within +-0.20)": round(null_hit / ok, 3), "P_NO_PENALTY_label_(CI within +-0.25)": round(null25 / ok, 3), "P_REVERSED_label": round(rev_hit / ok, 3), "sims": ok})
        print(res["scenarios"][-1], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300, int(sys.argv[4]) if len(sys.argv) > 4 else 300)
