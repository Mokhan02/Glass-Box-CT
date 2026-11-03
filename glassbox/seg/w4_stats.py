"""W4 (WAW-TACE) statistics (results/w4_waw/PREREG.md). Pure numpy.
(c) `paired_delta`: gap(arm1) - gap(arm0) with ONE patient resample shared by the arms, gap = mean(dark) - mean(bright) of a per-observation metric; `closes_verdict` is the W5 rule (CLOSES if CI upper < 0 and point <= -0.10; DOES NOT CLOSE if CI lower > -0.10).
(b) `cond_logit`: the conditional logistic regression with the tumour as stratum, reduced to the discordant tumours: z = [detected on the arterial series and missed on the portal-venous series] regressed on the within-tumour differences,
intercept = phase effect; coefficients per column of D (sign, |contrast|/10 HU, is_propagated, optional log diameter). Patient-clustered percentile bootstrap (resampling patients), sandwich Wald, Jeffreys (Firth) fit when a fit is separated.
`effective_n_design`: the v1 design (flip-and-comparable tumours, within-tumour detection difference bright minus dark phase, patient-clustered bootstrap, margin 0.20)."""
from __future__ import annotations

import hashlib

import numpy as np

B_C, B_B, MARGIN_C, MARGIN_EFF = 1000, 2000, 0.10, 0.20


def name_seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def closes_verdict(point, lo, hi, margin=MARGIN_C):
    if not np.isfinite(point):
        return "INCONCLUSIVE (empty group)"
    if hi < 0 and point <= -margin:
        return "CLOSES"
    return "DOES NOT CLOSE" if lo > -margin else "INCONCLUSIVE"


def gap_label(point, lo, hi, margin=MARGIN_C):
    """W3 rule, applied to a gap = dark minus bright."""
    if not np.isfinite(point):
        return "INCONCLUSIVE (empty group)"
    if lo > 0 and point >= margin:
        return "GAP"
    return "NO GAP" if hi < margin else "INCONCLUSIVE"


def model_gap_label(det, dice):
    g = (det == "GAP", dice == "GAP")
    if all(g):
        return "GAP (detection and Dice)"
    if g[1]:
        return "GAP (Dice only)"
    if g[0]:
        return "GAP (detection only)"
    return "NO GAP" if det == "NO GAP" and dice == "NO GAP" else "INCONCLUSIVE"


def overall_closes(v):
    """v[(comparator, metric)] with comparators 'D0' (and optionally others) and metrics 'det', 'dice'; the W5 overall label for one comparator set."""
    comps = sorted({c for c, _ in v}); cl = lambda m: all(v[(c, m)] == "CLOSES" for c in comps)      # noqa: E731
    if cl("det") and cl("dice"):
        return "CONFIRMED (detection and Dice)"
    if cl("det"):
        return "CONFIRMED (detection only)"
    if cl("dice"):
        return "CONFIRMED (Dice only)"
    return "NOT CONFIRMED" if any(v[(c, "det")] == "DOES NOT CLOSE" and v[(c, "dice")] == "DOES NOT CLOSE" for c in comps) else "INCONCLUSIVE"


def _counts(patients, bright, val):
    """Per patient: [sum val dark, sum val bright, n dark, n bright]; returns (patient ids, array)."""
    ids = sorted(set(patients)); ix = {p: i for i, p in enumerate(ids)}; a = np.zeros((len(ids), 4))
    for p, b, v in zip(patients, bright, val):
        i = ix[p]; a[i, 1 if b else 0] += v; a[i, 3 if b else 2] += 1
    return ids, a


def _gap(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 2] - c[..., 1] / c[..., 3]


def gap_ci(patients, bright, val, seed_name, b=B_C):
    ids, c = _counts(patients, bright, val); rng = np.random.default_rng(name_seed(seed_name)); s = c[rng.integers(0, len(ids), (b, len(ids)))].sum(1); d = _gap(s); d = d[np.isfinite(d)]
    pt = float(_gap(c.sum(0)))
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (float("nan"),) * 2
    return {"gap": pt, "lo": lo, "hi": hi, "n_dark": int(c[:, 2].sum()), "n_bright": int(c[:, 3].sum()), "n_patients": len(ids)}


def paired_delta(patients, bright, vals, arm1, arm0, seed_name, b=B_C):
    """vals: {arm: list aligned with patients/bright}. Delta = gap(arm1) - gap(arm0) on the same observations and the same resampled patients."""
    ids, c1 = _counts(patients, bright, vals[arm1]); _, c0 = _counts(patients, bright, vals[arm0]); rng = np.random.default_rng(name_seed(seed_name)); ix = rng.integers(0, len(ids), (b, len(ids)))
    d = _gap(c1[ix].sum(1)) - _gap(c0[ix].sum(1)); d = d[np.isfinite(d)]; pt = float(_gap(c1.sum(0)) - _gap(c0.sum(0)))
    lo, hi = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (float("nan"),) * 2
    return {"delta": pt, "lo": lo, "hi": hi, "gap_arm1": float(_gap(c1.sum(0))), "gap_arm0": float(_gap(c0.sum(0))), "n_dark": int(c1[:, 2].sum()), "n_bright": int(c1[:, 3].sum()), "n_patients": len(ids)}


def fit_logit(X, z, ridge=1e-6, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9; H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); st = np.linalg.solve(H, X.T @ (z - p) - ridge * b); b = b + st
        if np.max(np.abs(st)) < 1e-8:
            break
    return b


def fit_firth(X, z, iters=100):
    """Jeffreys-penalised logistic regression (Firth), the fallback for a separated fit."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = np.maximum(p * (1 - p), 1e-9); A = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1]); Ai = np.linalg.inv(A)
        h = np.einsum("ij,jk,ik->i", X, Ai, X) * W; st = Ai @ (X.T @ (z - p + h * (0.5 - p)))
        st = np.clip(st, -5, 5); b = b + st
        if np.max(np.abs(st)) < 1e-8:
            break
    return b


def separated(X, z, b):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); return bool(np.max(np.abs(b)) > 12 or np.any((p > 1 - 1e-6) & (z == 0)) or np.any((p < 1e-6) & (z == 1)) or np.linalg.matrix_rank(X) < X.shape[1])


def fit_robust(X, z):
    """(beta, used_firth). Ordinary fit; Jeffreys-penalised if the data are separated or the design is rank-deficient (rank-deficient designs use pinv inside Firth)."""
    if np.linalg.matrix_rank(X) < X.shape[1] or len(z) < X.shape[1] + 1:
        return np.full(X.shape[1], np.nan), False
    b = fit_logit(X, z)
    if separated(X, z, b):
        return fit_firth(X, z), True
    return b, False


def sandwich_se(X, z, b, clusters):
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = p * (1 - p) + 1e-9; A = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])); S = np.zeros((X.shape[1],) * 2)
    for c in np.unique(clusters):
        m = clusters == c; s = X[m].T @ (z[m] - p[m]); S += np.outer(s, s)
    n = len(np.unique(clusters)); V = A @ S @ A * (n / max(n - 1, 1)); return np.sqrt(np.diag(V))


def cond_logit(pat, det_a, det_p, D, names, seed_name, b=B_B):
    """pat: patient per tumour; det_a / det_p: 0/1 detection on the arterial / portal-venous series; D: (n, k) within-tumour differences (arterial minus portal-venous) of the covariates, names: their k names.
    Returns per coefficient (intercept = phase) OR and CIs, discordant counts and the number of Firth replicates."""
    pat = np.asarray(pat); det_a = np.asarray(det_a, int); det_p = np.asarray(det_p, int); D = np.asarray(D, float); d = det_a != det_p; z = (det_a[d] == 1).astype(float); Xd = np.column_stack([np.ones(d.sum()), D[d]]); cl = pat[d]
    cols = ["phase"] + list(names); out = {"discordant": int(d.sum()), "tumours": int(len(pat)), "columns": cols}
    beta, firth = fit_robust(Xd, z)
    if not np.all(np.isfinite(beta)):
        out["status"] = "NOT ESTIMABLE"; return out
    se = sandwich_se(Xd, z, beta, cl); rng = np.random.default_rng(name_seed(seed_name)); up = np.unique(pat); idx = {p: np.flatnonzero(pat == p) for p in up}; draws = []; n_firth = 0; n_fail = 0
    for _ in range(b):
        pick = rng.integers(0, len(up), len(up)); rows = np.concatenate([idx[up[i]] for i in pick]); da, dp = det_a[rows], det_p[rows]; dd = da != dp
        if dd.sum() < len(cols) + 1:
            n_fail += 1; continue
        Xb = np.column_stack([np.ones(dd.sum()), D[rows][dd]]); bb, f = fit_robust(Xb, (da[dd] == 1).astype(float))
        if np.all(np.isfinite(bb)):
            draws.append(bb); n_firth += f
        else:
            n_fail += 1
    dr = np.array(draws); out.update({"status": "OK" + (" (point estimate Jeffreys-penalised)" if firth else ""), "bootstrap_replicates_ok": len(draws), "bootstrap_failed": n_fail, "bootstrap_firth_replicates": int(n_firth), "coef": {}})
    for j, c in enumerate(cols):
        lo, hi = (float(np.percentile(dr[:, j], 2.5)), float(np.percentile(dr[:, j], 97.5))) if len(dr) else (float("nan"),) * 2
        out["coef"][c] = {"beta": float(beta[j]), "OR": float(np.exp(beta[j])), "boot_OR_lo": float(np.exp(lo)), "boot_OR_hi": float(np.exp(hi)), "wald_OR_lo": float(np.exp(beta[j] - 1.96 * se[j])), "wald_OR_hi": float(np.exp(beta[j] + 1.96 * se[j]))}
    return out


def sign_verdict(res, col="sign"):
    if res.get("status", "").startswith("NOT"):
        return "INCONCLUSIVE (not estimable)"
    c = res["coef"][col]
    if c["boot_OR_lo"] > 1:
        return "REAL-PHASE BRIGHTNESS EFFECT"
    return "OPPOSITE EFFECT" if c["boot_OR_hi"] < 1 else "INCONCLUSIVE"


def effective_n_design(pat, diff, seed_name, b=B_B):
    """diff: detection(bright series) - detection(dark series) per flip-and-comparable tumour (values -1, 0, 1). Patient-clustered bootstrap; margin 0.20."""
    pat = np.asarray(pat); diff = np.asarray(diff, float)
    if len(diff) == 0:
        return {"n": 0, "label": "INCONCLUSIVE (no tumours)"}
    up = np.unique(pat); idx = {p: np.flatnonzero(pat == p) for p in up}; rng = np.random.default_rng(name_seed(seed_name)); dr = []
    for _ in range(b):
        pick = rng.integers(0, len(up), len(up)); dr.append(diff[np.concatenate([idx[up[i]] for i in pick])].mean())
    pt = float(diff.mean()); lo, hi = float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5)); lab = "SIGN EFFECT" if lo > 0 and pt >= MARGIN_EFF else ("NO SIGN EFFECT" if hi < MARGIN_EFF else "INCONCLUSIVE")
    return {"n": int(len(diff)), "n_patients": int(len(up)), "mean_diff": pt, "lo": lo, "hi": hi, "label": lab}
