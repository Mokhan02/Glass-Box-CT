"""W3 Stage 2, M2 step 0 (CPU; PREREG_stage2.md): fit the cross-fitted erasers from the M1 token tables of one model. Per fold f (fold = index in the sorted primary case list mod 5) on the TRAINING folds' trained `mid` tokens (tumour and liver tokens of the primary cohort only):
LEACE for the concept 'token is tumour vs liver' (`leace`), LEACE for the local-HU concept = cell mean HU, `hu14[:, 0]`, continuous (`hu`), and a random unit direction erased by orthogonal projection through the training-token mean (`random`, seed 100 + fold). A sixth set `all` (fitted on all primary cases, random seed 105) is used only for
cases outside the primary list (HCC_027, sensitivity), which never contributed to any fit. Eraser x' = x - d (a.(x - mu)) / c. Usage: python scripts/w3s2_m2_fit.py <tokens dir with tokens/*.npz> <primary cases file> <out npz>"""
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from glassbox.seg import w3s2_stats as S  # noqa: E402

tdir, cases_file, out = sys.argv[1:4]; cases = sorted(l.strip() for l in open(cases_file) if l.strip()); fold = S.fold_of_cases(cases); X, y, hu, pid = [], [], [], []
for i, c in enumerate(cases):
    z = np.load(os.path.join(tdir, "tokens", c + ".npz")); X.append(z["mid_x"].astype(np.float32)); y.append(z["mid_is_tum"].astype(np.float32)); hu.append(z["mid_hu14"][:, 0].astype(np.float32)); pid.append(np.full(len(y[-1]), i))
X, y, hu, pid = np.concatenate(X), np.concatenate(y), np.concatenate(hu), np.concatenate(pid); f_of = np.array([fold[cases[p]] for p in pid]); save = {"dim": np.array(X.shape[1])}
for f in list(range(5)) + ["all"]:
    tr = np.ones(len(X), bool) if f == "all" else f_of != f; seed = 105 if f == "all" else 100 + f; mu = X[tr].astype(np.float64).mean(0)
    for name, e in (("leace", S.leace(X[tr], y[tr])), ("hu", S.leace(X[tr], hu[tr])), ("random", S.random_eraser(mu, X.shape[1], seed))):
        for k, v in e.items():
            save[f"{f}_{name}_{k}"] = np.asarray(v)
    print(f"fold {f}: {int(tr.sum())} training tokens, tumour fraction {y[tr].mean():.3f}", flush=True)
np.savez(out, **save); print("saved", out)
