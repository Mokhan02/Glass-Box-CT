"""W3 Stage 2, M1 analysis (results/w3_models/PREREG_stage2.md): the voxel probe of results/voxel_probe/PREREG.md, procedure and decision rule unchanged (copied from scripts/voxel_probe_analyze.py with the same random-number call order), on per-case token files from
scripts/w3s2_m1_extract.py. Per layer: trained network activations vs raw-HU control vs randomly initialised network (token groups of the random-init control are those of the trained model on the identical tokens), probes A (TUMOUR vs LIV) and B (TP vs LIV),
detection of interior-FN tokens at 95% specificity of the training folds, patient-bootstrap CIs, strata small / large, head check, decision rule with the four verdict categories LOST LATE / REPRESENTED / NEVER REPRESENTED / INCONCLUSIVE.
`--vista-compat` reads the stored VISTA-3D token layout (cache/vp/{trained,random}/<case>.npz, random-init groups from the random model's own predictions as in the original run) and, with `--seed 0 --check results/voxel_probe/probe_table.csv`, reproduces the original probe table (sanity check 1 of the PREREG).
Usage: python scripts/w3s2_m1_analyze.py --tokens <dir with tokens/*.npz> --cases <cases.txt> --out <outdir> --arm <name> [--layers mid,bottleneck,decoder_F] [--sens-layer mid_s4] [--n-boot 1000]
       python scripts/w3s2_m1_analyze.py --vista-compat --tokens cache/vp --out <outdir> --arm vista --seed 0 --check results/voxel_probe/probe_table.csv"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--tokens", required=True); ap.add_argument("--cases"); ap.add_argument("--out", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--layers", default="mid,bottleneck,decoder_F")
ap.add_argument("--sens-layer", default=None, help="extra probe-only layer (A: mid_hs2, D: mid_s4), same rule"); ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=None, help="default: first 8 hex digits of SHA-256 of --arm")
ap.add_argument("--vista-compat", action="store_true"); ap.add_argument("--check", default=None)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
X_RATE, LOST_LATE_DROP, LOGIT_FAR, SPEC, CAP, K, C_REG = 0.50, 0.15, -2.0, 0.95, 200, 5, 0.1; INTERIOR_MM = 2.0
SEED = a.seed if a.seed is not None else int(hashlib.sha256(a.arm.encode()).hexdigest()[:8], 16); rng = np.random.default_rng(SEED)
LAYERS = a.layers.split(","); EXTRA = [a.sens_layer] if a.sens_layer else []
tdir = os.path.join(a.tokens, "tokens") if not a.vista_compat else a.tokens
cases = [l.strip() for l in open(a.cases)] if a.cases else sorted(f[:-4] for f in os.listdir(os.path.join(a.tokens, "trained")) if f.endswith(".npz"))
cases = sorted(cases); P = len(cases); fold_of = {c: i % K for i, c in enumerate(cases)}
B = rng.integers(0, P, size=(a.n_boot, P))      # one set of patient resamples reused by every row (paired comparisons), as the original


def load(model):
    """{layer: arrays} and per-case head arrays. W3 layout: one npz per case (`<layer>_x` trained, `<layer>_xr` random-init, shared groups). VISTA layout: one npz per case and model."""
    D = {h: {k: [] for k in ("x", "xyz", "is_tum", "pred_cov", "edt", "hu14", "pid")} for h in LAYERS + EXTRA}; per_case = {}
    for pi, c in enumerate(cases):
        z = np.load(os.path.join(tdir, model, c + ".npz")) if a.vista_compat else np.load(os.path.join(tdir, c + ".npz"))
        for h in LAYERS + EXTRA:
            n = len(z[h + "_is_tum"])
            D[h]["x"].append(z[f"{h}_x"] if (a.vista_compat or model == "trained") else z[f"{h}_xr"])
            for k in ("xyz", "is_tum", "pred_cov", "edt", "hu14"):
                D[h][k].append(z[f"{h}_{k}"])
            D[h]["pid"].append(np.full(n, pi))
        per_case[c] = {k: z[k] for k in ("gt_tumour_mm3", "logit_fn", "logit_tp", "logit_fn_n", "logit_tp_n")}
    return {h: {k: np.concatenate(v) for k, v in d.items()} for h, d in D.items()}, per_case


def groups(d):
    tum = d["is_tum"].astype(bool); pc = d["pred_cov"].astype(float); inter = d["edt"].astype(float) > INTERIOR_MM
    return {"TUM": tum, "LIV": ~tum, "TP": tum & (pc >= 0.5), "FN": tum & (pc < 0.5), "FN_int": tum & (pc < 0.5) & inter, "TP_int": tum & (pc >= 0.5) & inter}


def cap_idx(pid, mask):
    out = []
    for p in np.unique(pid[mask]):
        i = np.flatnonzero(mask & (pid == p)); out.append(rng.choice(i, CAP, replace=False) if len(i) > CAP else i)
    return np.concatenate(out) if out else np.array([], int)


def run_probe(Xf, d, g, kind):
    pid = d["pid"]; fold = np.array([fold_of[cases[p]] for p in pid]); score = np.full(len(pid), np.nan); thr_tok = np.full(len(pid), np.nan); pos = g["TUM"] if kind == "A" else g["TP"]
    for f in range(K):
        tr, te = fold != f, fold == f; i1, i0 = cap_idx(pid, pos & tr), cap_idx(pid, g["LIV"] & tr)
        if len(i1) < 20 or len(i0) < 20:
            continue
        sc = StandardScaler().fit(Xf[np.r_[i1, i0]]); y = np.r_[np.ones(len(i1)), np.zeros(len(i0))]
        clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(sc.transform(Xf[np.r_[i1, i0]]), y)
        liv_tr = np.flatnonzero(g["LIV"] & tr); thr = np.percentile(clf.decision_function(sc.transform(Xf[liv_tr])), 100 * SPEC); te_i = np.flatnonzero(te)
        score[te_i] = clf.decision_function(sc.transform(Xf[te_i])); thr_tok[te_i] = thr
    return score, thr_tok


def counts(mask, det, pid):
    return np.bincount(pid[mask], minlength=P).astype(float), np.bincount(pid[mask & det], minlength=P).astype(float)


def rate_ci(n, k, sel=None):
    sel = np.ones(P, bool) if sel is None else sel; n, k = n * sel, k * sel
    if n.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    boots = (k[B].sum(1) / np.maximum(n[B].sum(1), 1)); ok = n[B].sum(1) > 0
    return k.sum() / n.sum(), *np.percentile(boots[ok], [2.5, 97.5]), int((n > 0).sum())


def auroc(s1, s0):
    r = rankdata(np.r_[s1, s0]); return (r[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0))


def auroc_ci(score, mask1, mask0, pid):
    ok = ~np.isnan(score); m1, m0 = mask1 & ok, mask0 & ok
    if m1.sum() < 5 or m0.sum() < 5:
        return np.nan, np.nan, np.nan
    by1 = [score[m1 & (pid == p)] for p in range(P)]; by0 = [score[m0 & (pid == p)] for p in range(P)]; vals = []
    for b in B[:min(len(B), 300)]:
        s1, s0 = np.concatenate([by1[i] for i in b]), np.concatenate([by0[i] for i in b])
        if len(s1) and len(s0):
            vals.append(auroc(s1, s0))
    return auroc(score[m1], score[m0]), *np.percentile(vals, [2.5, 97.5])


print(f"{P} patients; loading tokens ...", flush=True)
data = {m: load(m) for m in ("trained", "random")}
if not a.vista_compat:      # DEVIATION 3 of the VISTA work made the default here: the random-init groups are those of the trained model on identical tokens
    for h in LAYERS + EXTRA:
        t, r = data["trained"][0][h], data["random"][0][h]
        assert np.array_equal(t["xyz"], r["xyz"]) and np.array_equal(t["pid"], r["pid"]) and np.array_equal(t["is_tum"], r["is_tum"]), f"trained and random tokens differ at {h}"
vol = np.array([data["trained"][1][c]["gt_tumour_mm3"] for c in cases]); small = vol < np.median(vol)
rows, store = [], {}
for layer_set in (LAYERS, EXTRA):      # base layers first (the original call order), then the probe-only sensitivity layer
    for model in ("trained", "random"):
        for h in layer_set:
            d = data[model][0][h]; g = groups(d)
            for fs_name, Xf in (("activations", d["x"].astype(np.float32)), ("rawHU", d["hu14"].astype(np.float32))):
                if model == "random" and fs_name == "rawHU":
                    continue
                for kind in ("A", "B"):
                    score, thr = run_probe(Xf, d, g, kind); det = score > thr; valid = ~np.isnan(score); store[(model, h, fs_name, kind)] = (score, thr)
                    nI, kI = counts(g["FN_int"] & valid, det, d["pid"]); nT, kT = counts(g["TP"] & valid, det, d["pid"]); nL, kL = counts(g["LIV"] & valid, det, d["pid"])
                    for vname, sel in (("pooled", None), ("small tumours", small), ("large tumours", ~small)):
                        r, lo, hi, npat = rate_ci(nI, kI, sel); rT = rate_ci(nT, kT, sel); rL = rate_ci(nL, kL, sel); pr = np.where(nI > 0, kI / np.maximum(nI, 1), np.nan); selp = np.ones(P, bool) if sel is None else sel
                        row = {"in_distribution": False, "model": model, "layer": h, "features": fs_name, "probe": kind, "subset": vname, "interior_FN_rate": r, "ci_low": lo, "ci_high": hi, "n_patients_with_interior_FN": npat,
                               "interior_FN_tokens": int((nI * selp).sum()), "patient_avg_rate": float(np.nanmean(pr[selp])) if np.isfinite(pr[selp]).any() else np.nan, "TP_rate": rT[0], "TP_ci_low": rT[1], "TP_ci_high": rT[2],
                               "LIV_false_positive_rate": rL[0], "LIV_fpr_ci_low": rL[1], "LIV_fpr_ci_high": rL[2]}
                        if vname == "pooled":
                            row["auroc_FN_vs_LIV"], row["auroc_ci_low"], row["auroc_ci_high"] = auroc_ci(score, g["FN"], g["LIV"], d["pid"])
                        rows.append(row)
                    print(f"  {model:7s} {h:10s} {fs_name:11s} probe {kind}: interior-FN detection {kI.sum() / max(nI.sum(), 1):.3f} (n={int(nI.sum())} tokens, {int((nI > 0).sum())} patients)", flush=True)
cols = ["in_distribution", "model", "layer", "features", "probe", "subset", "interior_FN_rate", "ci_low", "ci_high", "n_patients_with_interior_FN", "interior_FN_tokens", "patient_avg_rate", "TP_rate", "TP_ci_low", "TP_ci_high",
        "LIV_false_positive_rate", "LIV_fpr_ci_low", "LIV_fpr_ci_high", "auroc_FN_vs_LIV", "auroc_ci_low", "auroc_ci_high"]
with open(os.path.join(a.out, "probe_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

# ---- head check ----
fn_all = np.concatenate([data["trained"][1][c]["logit_fn"].astype(np.float32) for c in cases]); tp_all = np.concatenate([data["trained"][1][c]["logit_tp"].astype(np.float32) for c in cases])
head = {"interior_FN_voxels_sampled": int(len(fn_all)), "interior_FN_margin_median": float(np.median(fn_all)), "interior_FN_margin_IQR": [float(x) for x in np.percentile(fn_all, [25, 75])], "share_FN_within_1_unit_below_0": float(((fn_all > -1.0) & (fn_all <= 0)).mean()),
        "share_FN_below_minus_2": float((fn_all < -2.0).mean()), "interior_TP_margin_median": float(np.median(tp_all)) if len(tp_all) else None}
json.dump(head, open(os.path.join(a.out, "head_check.json"), "w"), indent=1)


def get(model, layer, fs, kind="A", subset="pooled"):
    return next(r for r in rows if r["model"] == model and r["layer"] == layer and r["features"] == fs and r["probe"] == kind and r["subset"] == subset)


def decide(layer):
    """The pre-registered rule at `layer` (decoder_F is always the late layer)."""
    enc, hu = get("trained", layer, "activations"), get("trained", layer, "rawHU"); dec = get("trained", "decoder_F", "activations"); rnd = get("random", layer, "activations")
    represents = bool(enc["interior_FN_rate"] >= X_RATE and enc["ci_low"] > hu["ci_high"]); never = bool(enc["ci_low"] <= hu["ci_high"])
    sE, tE = store[("trained", layer, "activations", "A")]; sD, tD = store[("trained", "decoder_F", "activations", "A")]; dE, dD = data["trained"][0][layer], data["trained"][0]["decoder_F"]
    nE, kE = counts(groups(dE)["FN_int"] & ~np.isnan(sE), sE > tE, dE["pid"]); nD, kD = counts(groups(dD)["FN_int"] & ~np.isnan(sD), sD > tD, dD["pid"])
    diffs = (kD[B].sum(1) / np.maximum(nD[B].sum(1), 1)) - (kE[B].sum(1) / np.maximum(nE[B].sum(1), 1)); dlo, dhi = np.percentile(diffs, [2.5, 97.5]); dpoint = kD.sum() / nD.sum() - kE.sum() / nE.sum()
    late_a = bool(dpoint <= -LOST_LATE_DROP and dhi < 0); late_b = bool(head["interior_FN_margin_median"] <= LOGIT_FAR)
    verdict = ("LOST LATE" if (represents and (late_a or late_b)) else "REPRESENTED" if represents else "NEVER REPRESENTED" if never else "INCONCLUSIVE")
    note = "random-init control not available (empty FN group)" if not np.isfinite(rnd["interior_FN_rate"]) else ("random-init CI overlaps the trained CI: 'represents' qualified as separable in features a random network also provides" if (rnd["ci_high"] >= enc["ci_low"] and rnd["ci_low"] <= enc["ci_high"]) else "random-init CI separate from the trained CI")
    return {"layer": layer, "verdict": verdict, "R_enc": enc["interior_FN_rate"], "CI_enc": [enc["ci_low"], enc["ci_high"]], "R_hu": hu["interior_FN_rate"], "CI_hu": [hu["ci_low"], hu["ci_high"]], "represents": represents, "never": never,
            "decoder_F_rate": dec["interior_FN_rate"], "decoder_minus_layer": [float(dpoint), float(dlo), float(dhi)], "late_criterion_a": late_a, "median_FN_margin": head["interior_FN_margin_median"], "late_criterion_b": late_b,
            "random_init_rate": rnd["interior_FN_rate"], "random_init_CI": [rnd["ci_low"], rnd["ci_high"]], "random_note": note}


out = {"arm": a.arm, "seed": SEED, "n_cases": P, "primary_layer": "mid", "verdict_categories": ["LOST LATE", "REPRESENTED", "NEVER REPRESENTED", "INCONCLUSIVE"], "primary": decide("mid"), "sensitivity_layers": {h: decide(h) for h in EXTRA}}
json.dump(out, open(os.path.join(a.out, "decision.json"), "w"), indent=1, default=float); print("\nDECISION:", json.dumps(out["primary"], indent=1, default=float))
if a.check:      # sanity check 1: reproduce the stored probe table
    ref = list(csv.DictReader(open(a.check))); bad = []; n = 0
    for r in rows:
        m = [x for x in ref if x["model"] == r["model"] and x["layer"] == r["layer"] and x["features"] == r["features"] and x["probe"] == r["probe"] and x["subset"] == r["subset"]]
        if not m:
            continue
        for k in ("interior_FN_rate", "ci_low", "ci_high", "TP_rate", "LIV_false_positive_rate", "auroc_FN_vs_LIV"):
            if m[0].get(k, "") not in ("", None) and np.isfinite(r.get(k, np.nan)):
                n += 1; bad += [(r["model"], r["layer"], r["features"], r["probe"], r["subset"], k, m[0][k], round(r[k], 4))] if abs(float(m[0][k]) - round(r[k], 4)) > 1e-4 else []
    json.dump({"compared_values": n, "mismatches": bad[:50], "n_mismatches": len(bad), "reproduced": len(bad) == 0 and n > 0}, open(os.path.join(a.out, "reproduction_check.json"), "w"), indent=1); print("REPRODUCTION:", n, "values compared,", len(bad), "mismatches")
