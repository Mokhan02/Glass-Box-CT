"""[W6b graded secondaries, results/w6b_distill/PREREG.md section 7; DESCRIPTIVE, no pass/fail beyond the primary] For each arm against the baseline AND against the reference arm (BRIGHT): (1) the number of drift classes above the 10 mL floor whose mean Dice to the baseline is below 0.95 (the guard's failing-class count; BRIGHT: 32 of 48), with a 95% series-level bootstrap CI, and its paired difference to the reference with CI (same resampled series); (2) the kidney Dice change (arm minus baseline, and arm minus reference) with a 95% case-level bootstrap CI over the 60 fixed cases.
Same definitions as the frozen guard analysis: per class and series Dice = 1 if both masks empty else 2|a&b|/(|a|+|b|); floor = median baseline volume over the series >= 10 mL; a class fails if its mean Dice < 0.95. The drift masks are read one series at a time (memory only).
Usage: python scripts/w6b_secondary.py <out json> --base <forget dir> --ref NAME=<forget dir> --arm NAME=<forget dir> [...] [--b 2000]"""
import csv
import json
import os
import sys

import numpy as np

FLOOR_ML, DRIFT_MIN = 10.0, 0.95


def dice_matrix(base, arm, meta):
    classes = [str(c) for c in meta["classes"]]; D = np.zeros((len(meta["series"]), len(classes))); V = np.zeros_like(D)
    for i, s in enumerate(meta["series"]):
        fn = s.replace("/", "_") + ".npz"; zb, za = np.load(os.path.join(base, "drift", fn)), np.load(os.path.join(arm, "drift", fn)); v = float(np.prod(zb["spacing"])) / 1000.0
        for j, c in enumerate(classes):
            a, b = zb[c], za[c]; t = a.sum() + b.sum(); D[i, j] = 1.0 if t == 0 else 2.0 * (a & b).sum() / t; V[i, j] = float(a.sum()) * v
    return classes, D, V


def kidney(path):
    return {r["case"]: float(r["dice"]) for r in csv.DictReader(open(path))}


def main(argv):
    out = argv[0]; opt = lambda n, d=None: argv[argv.index(n) + 1] if n in argv else d      # noqa: E731
    base = opt("--base"); B = int(opt("--b", 2000)); arms = {}; ref = None
    for i, a in enumerate(argv):
        if a in ("--arm", "--ref"):
            n, p = argv[i + 1].split("=", 1); arms[n] = p
            if a == "--ref":
                ref = n
    meta = json.load(open(os.path.join(base, "drift_meta.json"))); rng = np.random.default_rng(20261109); ns = len(meta["series"]); idx = rng.integers(0, ns, (B, ns))
    Dm, floor = {}, None
    for n, d in arms.items():
        classes, D, V = dice_matrix(base, d, meta); Dm[n] = D
        if floor is None:
            floor = np.median(V, axis=0) >= FLOOR_ML; fc = [c for c, f in zip(classes, floor) if f]
    def fails(D, ix=None):      # number of above-floor classes with mean Dice < 0.95 (ix = resampled series)
        m = (D if ix is None else D[ix]).mean(0); return int(((m < DRIFT_MIN) & floor).sum())
    res = {"n_series": ns, "n_above_floor": int(floor.sum()), "reference": ref, "arms": {}}
    kb = kidney(os.path.join(base, "kidney.csv")); kr = kidney(os.path.join(arms[ref], "kidney.csv")) if ref else None
    cases = sorted(kb); kidx = rng.integers(0, len(cases), (B, len(cases)))
    for n, d in arms.items():
        D = Dm[n]; bs = np.array([fails(D, ix) for ix in idx]); r = {"failing_classes": fails(D), "failing_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "lowest_class_mean_dice_above_floor": float(D.mean(0)[floor].min())}
        if ref and n != ref:
            bd = bs - np.array([fails(Dm[ref], ix) for ix in idx]); r["failing_minus_reference"] = fails(D) - fails(Dm[ref]); r["failing_minus_reference_ci"] = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
        ka = kidney(os.path.join(d, "kidney.csv")); a = np.array([ka[c] for c in cases]); b = np.array([kb[c] for c in cases]); ch = a - b
        r["kidney_change_vs_baseline"] = {"mean": float(ch.mean()), "ci": [float(np.percentile(ch[kidx].mean(1), 2.5)), float(np.percentile(ch[kidx].mean(1), 97.5))], "arm_mean": float(a.mean()), "baseline_mean": float(b.mean())}
        if kr is not None and n != ref:
            rr = np.array([kr[c] for c in cases]); df = a - rr; r["kidney_change_vs_reference"] = {"mean": float(df.mean()), "ci": [float(np.percentile(df[kidx].mean(1), 2.5)), float(np.percentile(df[kidx].mean(1), 97.5))]}
        res["arms"][n] = r
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
