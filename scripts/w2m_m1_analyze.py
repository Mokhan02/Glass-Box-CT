"""W2-M M1 analysis (results/w2_mech/PREREG.md, applied as written): is the HCC tumour concept shared with kidney tumour tissue at the mid-encoder?
Transfer probe = the HCC probe A (tumour vs liver tokens; StandardScaler + L2 LogisticRegression(C=0.1, class_weight=balanced, max_iter=2000), <= 200 tokens per (patient, group), seed 0) fitted ONCE on all HCC cases
and applied UNCHANGED (weights and HCC scaler) to kidney tokens (tumour vs kidney parenchyma). Transfer controls: the same pipeline with HCC random-init tokens -> kidney random-init tokens and HCC raw-HU (14 features) -> kidney raw-HU.
Kidney-native ceiling: the same probe spec with 5-fold GroupKFold by case (sorted case id, fold = index mod 5), with its own random-init and raw-HU controls.
Metrics on interior tumour tokens vs kidney-parenchyma tokens: AUROC (primary) and detection at 95% specificity (threshold per fold from the KID tokens of the other folds; detection at the HCC-derived threshold also reported);
bright vs dark by the lesion's sign. CIs: case-level bootstrap, 1000 resamples, seed 0, percentile.
Verdict: SHARED (transfer CI-lower beats BOTH control CI-uppers, > 0.5, and (AUC_T - 0.5)/(AUC_N - 0.5) >= 0.70); PARTIAL (first two hold, fraction < 0.70); ORGAN-SPECIFIC otherwise.
Usage: python scripts/w2m_m1_analyze.py --hcc-vp <cache/vp> --rr <cache/rr> --kid <dir with trained/ and random/> --out <outdir>"""
import argparse
import json
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

B, SEED, K, CAP, C_REG, SPEC, FRAC = 1000, 0, 5, 200, 0.1, 0.95, 0.70
KINDS = ("trained", "random", "hu")


def auroc(s_pos, s_neg):
    if len(s_pos) == 0 or len(s_neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([s_pos, s_neg])); return float((r[: len(s_pos)].sum() - len(s_pos) * (len(s_pos) + 1) / 2) / (len(s_pos) * len(s_neg)))


def cap_idx(pid, mask, rng):
    out = []
    for p in np.unique(pid[mask]):
        idx = np.flatnonzero(mask & (pid == p)); out.append(idx if len(idx) <= CAP else rng.choice(idx, CAP, replace=False))
    return np.concatenate(out) if out else np.array([], int)


def fit_probe(X, y, pid, seed=SEED):
    """HCC / kidney-native probe A: tumour (y = 1) vs the other tokens, capped per (patient, group)."""
    rng = np.random.default_rng(seed); i1, i0 = cap_idx(pid, y == 1, rng), cap_idx(pid, y == 0, rng); idx = np.r_[i1, i0]
    sc = StandardScaler().fit(X[idx]); clf = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(sc.transform(X[idx]), y[idx]); return sc, clf


def score(model, X):
    sc, clf = model; return clf.decision_function(sc.transform(X))


def load_hcc(vp, rr):
    cases = sorted(f[:-4] for f in os.listdir(rr) if f.endswith(".npz") and not f.endswith(".part.npz")); D = {k: [] for k in KINDS}; y, pid = [], []
    for i, c in enumerate(cases):
        t = np.load(os.path.join(vp, "trained", c + ".npz")); r = np.load(os.path.join(vp, "random", c + ".npz"))
        yy = t["mid_is_tum"].astype(int); D["trained"].append(t["mid_x"].astype(np.float32)); D["hu"].append(t["mid_hu14"].astype(np.float32))
        assert len(r["mid_x"]) == len(yy), f"random-init and trained token tables differ for {c}"            # token groups come from the trained run (HCC DEVIATIONS #3)
        D["random"].append(r["mid_x"].astype(np.float32)); y.append(yy); pid.append(np.full(len(yy), i))
    return {k: np.concatenate(v) for k, v in D.items()}, np.concatenate(y), np.concatenate(pid), cases


def load_kidney(kid):
    cases = sorted(f[:-4] for f in os.listdir(os.path.join(kid, "trained")) if f.endswith(".npz") and not f.endswith(".part.npz"))
    D = {k: [] for k in KINDS}; M = {k: [] for k in ("is_t", "is_k", "interior", "bright", "pid")}; lesions = []
    for i, c in enumerate(cases):
        t = np.load(os.path.join(kid, "trained", c + ".npz")); r = np.load(os.path.join(kid, "random", c + ".npz"))
        assert np.array_equal(t["mid_xyz"], r["mid_xyz"]), f"random-init and trained token positions differ for {c}"
        D["trained"].append(t["mid_x"].astype(np.float32)); D["random"].append(r["mid_x"].astype(np.float32)); D["hu"].append(t["mid_hu14"].astype(np.float32))
        n = len(t["mid_xyz"]); M["is_t"].append(t["mid_is_tum"]); M["is_k"].append(t["mid_is_kid"]); M["interior"].append(t["mid_interior"]); M["bright"].append(t["mid_bright"]); M["pid"].append(np.full(n, i))
        lesions += [(c, int(b), bool(h)) for b, h in zip(t["lesion_bright"], t["lesion_has_token"])]
    return {k: np.concatenate(v) for k, v in D.items()}, {k: np.concatenate(v) for k, v in M.items()}, cases, lesions


def metric_ci(draws_fn, point, resamples):
    d = np.array([x for x in (draws_fn(ix) for ix in resamples) if np.isfinite(x)]); return {"value": float(point), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}, d


def analyse(HX, hy, hpid, KX, M, n_kid_cases, rng):
    pid, fold = M["pid"], M["pid"] % K; pos = M["is_t"] & M["interior"]; neg = M["is_k"]; keep = pos | neg
    res, scores = {}, {}
    hcc_models = {k: fit_probe(HX[k], hy, hpid) for k in KINDS}
    hcc_thr = {k: float(np.percentile(score(hcc_models[k], HX[k][hy == 0]), 100 * SPEC)) for k in KINDS}             # HCC-derived threshold: 95th percentile of HCC liver-token scores
    for k in KINDS:
        scores[("transfer", k)] = score(hcc_models[k], KX[k])
        nat = np.full(len(pid), np.nan)
        for f in range(K):
            tr, te = (fold != f), (fold == f); y_tr = M["is_t"].astype(int)
            m_ = fit_probe(KX[k][tr & (M["is_t"] | M["is_k"])], y_tr[tr & (M["is_t"] | M["is_k"])], pid[tr & (M["is_t"] | M["is_k"])])
            nat[te] = score(m_, KX[k][te])
        scores[("native", k)] = nat
    idx_keep = np.flatnonzero(keep); pid_keep = pid[idx_keep]
    by_case = [idx_keep[pid_keep == p] for p in range(n_kid_cases)]
    resamples = [np.concatenate([by_case[p] for p in rng.integers(0, n_kid_cases, n_kid_cases)]) for _ in range(B)]
    draws = {}
    for key, s in scores.items():
        def au(ix, s=s, sub=None):
            ii = ix[(pos[ix] if sub is None else (pos[ix] & (M["bright"][ix] == sub)))]; jj = ix[neg[ix]]; return auroc(s[ii], s[jj])
        point = auroc(s[pos], s[neg]); res[key], draws[key] = metric_ci(au, point, resamples)
        bpos, dpos = pos & M["bright"], pos & ~M["bright"]
        res[key + ("bright",)], db = metric_ci(lambda ix, s=s: au(ix, s, True), auroc(s[bpos], s[neg]), resamples)
        res[key + ("dark",)], dd = metric_ci(lambda ix, s=s: au(ix, s, False), auroc(s[dpos], s[neg]), resamples)
        dff = np.array([au(ix, s, True) - au(ix, s, False) for ix in resamples]); dff = dff[np.isfinite(dff)]
        res[key + ("bright_minus_dark",)] = {"value": float(res[key + ("bright",)]["value"] - res[key + ("dark",)]["value"]), "lo": float(np.percentile(dff, 2.5)), "hi": float(np.percentile(dff, 97.5))}
        # detection at 95% specificity: threshold per fold from the KID tokens of the other folds (the recalibrated one-scalar threshold), and at the HCC-derived threshold (transfer only)
        thr = np.full(len(pid), np.nan)
        for f in range(K):
            tr = (fold != f) & neg; thr[fold == f] = np.percentile(s[tr], 100 * SPEC)
        det = s > thr
        res[key + ("detection",)], _ = metric_ci(lambda ix, det=det: float(det[ix[pos[ix]]].mean()) if pos[ix].any() else float("nan"), float(det[pos].mean()), resamples)
        if key[0] == "transfer":
            res[key + ("detection_hcc_threshold",)], _ = metric_ci(lambda ix, s=s, k=key[1]: float((s[ix[pos[ix]]] > hcc_thr[k]).mean()) if pos[ix].any() else float("nan"), float((s[pos] > hcc_thr[key[1]]).mean()), resamples)
    return res, scores, int(pos.sum()), int(neg.sum())


def verdict(res):
    T, N = res[("transfer", "trained")], res[("native", "trained")]; ctr = (res[("transfer", "random")], res[("transfer", "hu")])
    beats = all(T["lo"] > c["hi"] for c in ctr); above = T["lo"] > 0.5; frac = (T["value"] - 0.5) / (N["value"] - 0.5) if N["value"] > 0.5 else float("nan")
    v = "SHARED" if beats and above and frac >= FRAC else "PARTIAL" if beats and above else "ORGAN-SPECIFIC"
    return {"verdict": v, "transfer_beats_both_controls": bool(beats), "transfer_ci_lower_above_0.5": bool(above), "fraction_of_native_above_chance": float(frac), "transfer_significantly_below_0.5": bool(T["hi"] < 0.5)}


def main(a):
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(SEED)
    HX, hy, hpid, hcases = load_hcc(a.hcc_vp, a.rr); KX, M, kcases, lesions = load_kidney(a.kid)
    res, scores, n_pos, n_neg = analyse(HX, hy, hpid, KX, M, len(kcases), rng); V = verdict(res)
    les = {"lesions": len(lesions), "bright": sum(b for _, b, _ in lesions), "dark": sum(1 - b for _, b, _ in lesions), "lesions_with_no_mid_tumour_token": sum(1 for _, _, h in lesions if not h),
           "bright_with_no_token": sum(1 for _, b, h in lesions if b and not h), "dark_with_no_token": sum(1 for _, b, h in lesions if (not b) and not h)}
    out = {"n_hcc_cases": len(hcases), "n_kidney_cases": len(kcases), "interior_tumour_tokens": n_pos, "kidney_parenchyma_tokens": n_neg, "lesion_resolution": les, "verdict": V,
           "metrics": {"/".join(k): v for k, v in res.items()}}
    json.dump(out, open(os.path.join(a.out, "m1_results.json"), "w"), indent=1)
    f = lambda m: f"{m['value']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"
    L = ["# W2-M M1 results (generated by scripts/w2m_m1_analyze.py, applied as written)", "",
         "C4KC-KiTS: possibly seen in training (images, via AbdomenCT-1K). Kidney main-set lesions only. Tokens are 12 mm mid-encoder cells.", "",
         f"HCC cases {len(hcases)}; kidney cases {len(kcases)}; interior tumour tokens {n_pos}; kidney-parenchyma tokens {n_neg}; lesions {les['lesions']} ({les['bright']} bright, {les['dark']} dark), of which {les['lesions_with_no_mid_tumour_token']} have no tumour token ({les['bright_with_no_token']} bright, {les['dark_with_no_token']} dark)", "",
         f"## Verdict: **{V['verdict']}**", f"- transfer AUROC {f(res[('transfer', 'trained')])}; kidney-native ceiling {f(res[('native', 'trained')])}; fraction of the native above-chance signal {V['fraction_of_native_above_chance']:.2f} (needs >= {FRAC} for SHARED)",
         f"- transfer controls: random-init {f(res[('transfer', 'random')])}; raw-HU {f(res[('transfer', 'hu')])}; transfer CI-lower beats both control CI-uppers: {V['transfer_beats_both_controls']}; CI-lower above 0.5: {V['transfer_ci_lower_above_0.5']}"
         + ("; transfer is significantly BELOW 0.5 (reversed)" if V["transfer_significantly_below_0.5"] else ""), "", "## AUROC (interior tumour tokens vs kidney parenchyma), patient-level bootstrap 95% CI",
         "| probe | trained | random-init | raw-HU |", "|---|---|---|---|"]
    for m in ("transfer", "native"):
        L.append(f"| {m} | " + " | ".join(f(res[(m, k)]) for k in KINDS) + " |")
    L += ["", "## Detection at 95% specificity (threshold per fold from the other folds' kidney-parenchyma tokens)", "| probe | trained | random-init | raw-HU |", "|---|---|---|---|"]
    for m in ("transfer", "native"):
        L.append(f"| {m} | " + " | ".join(f(res[(m, k, 'detection')]) for k in KINDS) + " |")
    L += ["", "Transfer detection at the unchanged HCC-derived threshold: " + "; ".join(f"{k} {f(res[('transfer', k, 'detection_hcc_threshold')])}" for k in KINDS), "", "## Bright vs dark lesions (descriptive)", "| probe | AUROC bright | AUROC dark | bright minus dark |", "|---|---|---|---|"]
    for m in ("transfer", "native"):
        L.append(f"| {m} trained | {f(res[(m, 'trained', 'bright')])} | {f(res[(m, 'trained', 'dark')])} | {f(res[(m, 'trained', 'bright_minus_dark')])} |")
    open(os.path.join(a.out, "m1_results.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hcc-vp", required=True); ap.add_argument("--rr", required=True); ap.add_argument("--kid", required=True); ap.add_argument("--out", required=True)
    main(ap.parse_args())
