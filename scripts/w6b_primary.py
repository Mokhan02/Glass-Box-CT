"""[W6b PRIMARY criterion 1, results/w6b_distill/PREREG.md section 6] Non-inferiority of bright arterial detection against the reference arm (BRIGHT seed 1): on the arterial-bright tumours (arterial contrast > 0) common to the arms (W6: 242 of 297), the paired difference (arm minus reference) of the W4 `detected` flag, with a patient-clustered bootstrap 95% CI (B = 2,000, percentile). NON-INFERIOR if the CI lower bound is above -0.10. Same tumour build and options as scripts/w6_analyze.py. The guard (criterion 2) comes from the forgetting analysis, not from this script.
Usage: python scripts/w6b_primary.py <paired dir> <out json> --ref NAME=scores_a.csv[,scores_b.csv] --arm NAME=scores_a.csv[,scores_b.csv] [...] [--b 2000] [--margin 0.10]"""
import csv
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import w6_analyze as W  # noqa: E402


def main(argv):
    paired, out = argv[0], argv[1]; opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    B = int(opt("--b", 2000)); margin = float(opt("--margin", 0.10)); arms, ref = {}, None
    for i, a in enumerate(argv):
        if a in ("--arm", "--ref"):
            n, p = argv[i + 1].split("=", 1); arms[n] = W.load_arm(p.split(","))
            if a == "--ref":
                ref = n
    tum = {(r["patient"], r["mask_phase"], r["rank"]): r for r in csv.DictReader(open(os.path.join(paired, "tumours_paired.csv")))}
    base = {"cats": {"K"}, "det": "detected", "excl_pat": set(), "top2": False, "min_diam": 0.0, "gate": False, "contrast": ""}
    Ts = {n: W.AN.build(tum, r, g, base) for n, (r, g) in arms.items()}; res = {"reference": ref, "margin": margin, "arms": {}}
    for n in Ts:
        if n == ref:
            continue
        ks = sorted(W.common(Ts, [n, ref])); br = [k for k in ks if Ts[ref][k]["c"][1] > 0]; pat = np.array([Ts[ref][k]["pat"] for k in br])
        d = np.array([Ts[n][k]["obs"][1]["det"] - Ts[ref][k]["obs"][1]["det"] for k in br], float); up = np.unique(pat); idx = {p: np.where(pat == p)[0] for p in up}
        rng = np.random.default_rng(int(hashlib.sha256(f"w6b_primary_{n}".encode()).hexdigest()[:8], 16)); bs = np.array([np.concatenate([d[idx[p]] for p in rng.choice(up, len(up))]).mean() for _ in range(B)])
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res["arms"][n] = {"n_bright_arterial": len(br), "n_patients": int(len(up)), "arm_detection": float(np.mean([Ts[n][k]["obs"][1]["det"] for k in br])), "reference_detection": float(np.mean([Ts[ref][k]["obs"][1]["det"] for k in br])),
                          "diff": float(d.mean()), "lo": lo, "hi": hi, "criterion_1": "NON-INFERIOR" if lo > -margin else "NOT SHOWN NON-INFERIOR"}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
