"""EXPLORATORY, post hoc (results/mech/DEVIATIONS.md #2): bright-minus-dark per-channel contribution within detected-only and missed-only tumour voxels. CPU.
Usage: python scripts/mech_a_strat.py"""
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RR, OUT, K, MINV = "cache/rr", "results/mech", 5, 20
TOP = [36, 20, 10, 22, 18, 30, 17, 7]
w = np.load(os.path.join(RR, "w_orig.npy")).astype(np.float32)
cases = sorted(f[:-4] for f in os.listdir(RR) if f.endswith(".npz") and not f.endswith(".part.npz"))


def job(c):
    z = np.load(os.path.join(RR, c + ".npz")); Lr = z["L"].ravel()[z["roi_idx"].astype(np.int64)]; F = z["F"].astype(np.float32); lg = z["logit_roi"]; tum = Lr == 2
    out = {"contrast": float(z["contrast_hu"])}
    for nm, m in (("det", tum & (lg > 0)), ("miss", tum & (lg <= 0))):
        out[nm] = (F[m] * w).mean(0) if m.sum() >= MINV else None; out[nm + "_n"] = int(m.sum())
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=12) as ex:
        R = list(ex.map(job, cases, chunksize=2))
    bright = np.array([r["contrast"] > 0 for r in R]); rng = np.random.default_rng(1); BI = rng.integers(0, len(R), size=(2000, len(R))); res = {}; rows = []
    for st in ("det", "miss"):
        ok = np.array([r[st] is not None for r in R]); M = np.array([r[st] if r[st] is not None else np.zeros(48) for r in R])

        def gap(ix):
            b = [i for i in ix if ok[i] and bright[i]]; d = [i for i in ix if ok[i] and not bright[i]]
            return M[b].mean(0) - M[d].mean(0)
        S = gap(np.arange(len(R))); bs = np.array([gap(b) for b in BI]); tot = bs.sum(1); sh = bs[:, TOP].sum(1) / tot
        res[st] = {"n_bright": int((ok & bright).sum()), "n_dark": int((ok & ~bright).sum()), "total_gap_logit": float(S.sum()), "total_gap_ci": [float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))],
                   "fixed8_gap": float(S[TOP].sum()), "fixed8_gap_ci": [float(np.percentile(bs[:, TOP].sum(1), 2.5)), float(np.percentile(bs[:, TOP].sum(1), 97.5))],
                   "fixed8_share": float(S[TOP].sum() / S.sum()), "fixed8_share_ci": [float(np.nanpercentile(sh, 2.5)), float(np.nanpercentile(sh, 97.5))]}
        for j in range(48):
            rows.append({"stratum": st, "channel": j, "top8": int(j in TOP), "bright_minus_dark": round(float(S[j]), 4), "ci_low": round(float(np.percentile(bs[:, j], 2.5)), 4), "ci_high": round(float(np.percentile(bs[:, j], 97.5)), 4)})
    neg = all(res[s]["total_gap_ci"][1] < 0 for s in res); inc0 = all(res[s]["total_gap_ci"][0] <= 0 <= res[s]["total_gap_ci"][1] for s in res)
    own = neg and all(res[s]["fixed8_share"] >= 0.5 for s in res)
    res["interpretation_by_prefixed_rule"] = "brightness has its OWN effect on those channels" if own else ("overlap mostly reflects miss rate" if inc0 else "mixed")
    json.dump(res, open(os.path.join(OUT, "A_strat_exploratory.json"), "w"), indent=1)
    with open(os.path.join(OUT, "A_strat_exploratory.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(json.dumps(res, indent=1))
