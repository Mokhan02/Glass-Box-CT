"""[DESCRIPTIVE, requested by the researcher 2026-11-09] Detection counts per arm on the W4b tumour set (297 tumours, same build and options as scripts/w6_analyze.py): arterial series bright/dark and portal-venous series bright/dark, bright = contrast on that series > 0. Detection = the W4 `detected` flag.
Usage: python scripts/w6_detection_table.py <paired dir> <out json> --arm NAME=scores_a.csv[,scores_b.csv] [...]"""
import csv, importlib.util, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import w6_analyze as W  # noqa: E402

def main(argv):
    paired, out = argv[0], argv[1]; arms = {}
    for i, a in enumerate(argv):
        if a == "--arm":
            n, p = argv[i + 1].split("=", 1); arms[n] = W.load_arm(p.split(","))
    tum = {(r["patient"], r["mask_phase"], r["rank"]): r for r in csv.DictReader(open(os.path.join(paired, "tumours_paired.csv")))}
    base = {"cats": {"K"}, "det": "detected", "excl_pat": set(), "top2": False, "min_diam": 0.0, "gate": False, "contrast": ""}
    Ts = {n: W.AN.build(tum, r, g, base) for n, (r, g) in arms.items()}; ks = sorted(W.common(Ts, list(Ts))); res = {"n_tumours": len(ks)}
    for n, T in Ts.items():
        e = {}
        for ph, name in ((1, "arterial"), (2, "portal")):
            for grp, sel in (("bright", lambda c: c > 0), ("dark", lambda c: c <= 0)):
                idx = [k for k in ks if sel(T[k]["c"][ph])]; d = [T[k]["obs"][ph]["det"] for k in idx]
                e[f"{name}_{grp}"] = {"n": len(idx), "detected": int(sum(d)), "pct": round(100 * float(np.mean(d)), 1)}
        res[n] = e
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res))

if __name__ == "__main__":
    main(sys.argv[1:])
