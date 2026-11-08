"""W4b primary-test power (CPU, synthetic outcomes on the real covariates of the 297 kept W4 tumours). Primary test: paired difference of the bright penalty, d_theta = theta(BRIGHT-FT) - theta(SHAM-FT), the SAME tumours and the SAME resampled patients;
label BRIGHT-FT REDUCES THE PENALTY if the 95% patient-clustered bootstrap CI upper bound is < 0 (one-sided alpha 0.025) and d_theta <= -0.10. Generating model as in w4b_did_power.py; the two models share the tumour and patient random effects (what makes a tumour hard is shared) and add an independent model-specific tumour term (SD 0.7).
SHAM-FT true sign OR 0.10 (the W4 non-augmented models); BRIGHT-FT true sign OR varied. Usage: python results/w4b_waw/w4b_primary_power.py <tumours_paired.csv> <out.json> [sims] [B]"""
import csv
import json
import sys

import numpy as np

sys.path.insert(0, "results/w4b_waw")
from w4b_did_power import theta_hat  # noqa: E402


def main(path, out, sims=200, B=200):
    rows = [r for r in csv.DictReader(open(path)) if r["category"] == "K"]; ca = np.array([float(r["c_arterial"]) for r in rows]); cp = np.array([float(r["c_portal"]) for r in rows]); pat = np.array([int(r["patient"]) for r in rows]); drawn_a = np.array([r["mask_phase"] == "1" for r in rows])
    ba, bp = ca > 0, cp > 0; f1 = ba & ~bp; nf = (ba == bp); n = len(rows); up = np.unique(pat); idx = [np.flatnonzero(pat == p) for p in up]; prop_a, prop_p = ~drawn_a, drawn_a
    rng = np.random.default_rng(20261109); res = {"tumours": int(n), "scenarios": []}; mu = 1.1
    for or_b in (1.0, 0.61, 0.37, 0.10):
        b_s, b_b = np.log(0.10), np.log(or_b); b_phase, b_prop = np.log(0.15), np.log(0.7); hit = ok = 0; dts = []
        for _ in range(sims):
            u = rng.normal(0, 1.0, len(up))[np.searchsorted(up, pat)]; e = rng.normal(0, 1.1, n); out_ = {}
            for name, bs in (("sham", b_s), ("bright", b_b)):
                al = mu + u + e + rng.normal(0, 0.7, n)
                pa = 1 / (1 + np.exp(-(al + b_phase + bs * ba + b_prop * prop_a))); pp = 1 / (1 + np.exp(-(al + bs * bp + b_prop * prop_p))); out_[name] = (rng.random(n) < pa, rng.random(n) < pp)
            ts, tb = theta_hat(*out_["sham"], f1, nf, drawn_a), theta_hat(*out_["bright"], f1, nf, drawn_a)
            if not (np.isfinite(ts) and np.isfinite(tb)):
                continue
            dt = tb - ts; bs_ = []
            for _b in range(B):
                pick = rng.integers(0, len(up), len(up)); r = np.concatenate([idx[i] for i in pick]); a1 = theta_hat(out_["sham"][0][r], out_["sham"][1][r], f1[r], nf[r], drawn_a[r]); a2 = theta_hat(out_["bright"][0][r], out_["bright"][1][r], f1[r], nf[r], drawn_a[r])
                if np.isfinite(a1) and np.isfinite(a2):
                    bs_.append(a2 - a1)
            hi = np.percentile(bs_, 97.5); ok += 1; dts.append(dt); hit += (hi < 0 and dt <= -0.10)
        res["scenarios"].append({"BRIGHT_FT_true_sign_OR": or_b, "SHAM_FT_true_sign_OR": 0.10, "mean_d_theta": round(float(np.mean(dts)), 3), "P_REDUCES_label": round(hit / ok, 3), "sims": ok}); print(res["scenarios"][-1], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200, int(sys.argv[4]) if len(sys.argv) > 4 else 200)
