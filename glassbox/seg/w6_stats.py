"""W6 / W4b statistics (results/w4b_waw/PREREG.md section 5 and 6; results/w6_vista_finetune/PREREG.md section 8). Pure numpy.
`theta_hat`: the bright penalty, a standardised difference in differences of risk differences: for each tumour d = detected(portal-venous series) - detected(arterial series); theta = mean(d | F1) - mean(d | NF), standardised over the drawn-phase strata with the F1 stratum shares
(F1 = bright on the arterial and not bright on the portal-venous series; NF = same sign on both; F2, the reverse, is excluded). Positive theta = the bright series is detected less. Patient-clustered percentile bootstrap; paired tests use the SAME resampled patients for both models.
`gap_ratio`: the decoder-only ratio R = (gap_plain - gap_dec) / (gap_plain - gap_bright) with shared patient resamples (W6 section 8)."""
from __future__ import annotations

import numpy as np

from glassbox.seg.w4_stats import fit_firth, name_seed, sandwich_se

MARGIN_REDUCE, MARGIN_EFFECT, MARGIN_EQUIV = 0.10, 0.20, 0.25
B_TEST = 2000


def theta_hat(det_a, det_p, f1, nf, drawn_a):
    d = np.asarray(det_p, float) - np.asarray(det_a, float); tot = f1.sum(); th = 0.0
    if tot == 0:
        return float("nan")
    for s in (True, False):
        a = f1 & (drawn_a == s); b = nf & (drawn_a == s)
        if a.sum() == 0:
            continue
        if b.sum() == 0:
            return float("nan")
        th += a.sum() / tot * (d[a].mean() - d[b].mean())
    return float(th)


def classes(ca, cp):
    """F1 / NF masks from the arterial and portal-venous contrasts (bright = contrast > 0)."""
    ba, bp = np.asarray(ca) > 0, np.asarray(cp) > 0; return ba & ~bp, ba == bp, ~ba & bp


def _resamples(patients, B, seed_name):
    patients = np.asarray(patients); up = np.unique(patients); idx = [np.flatnonzero(patients == p) for p in up]; rng = np.random.default_rng(name_seed(seed_name))
    for _ in range(B):
        yield np.concatenate([idx[i] for i in rng.integers(0, len(up), len(up))])


def theta_ci(patients, det_a, det_p, f1, nf, drawn_a, seed_name, B=B_TEST):
    det_a, det_p, drawn_a = np.asarray(det_a), np.asarray(det_p), np.asarray(drawn_a); pt = theta_hat(det_a, det_p, f1, nf, drawn_a); dr = []
    for r in _resamples(patients, B, seed_name):
        t = theta_hat(det_a[r], det_p[r], f1[r], nf[r], drawn_a[r]); dr.append(t) if np.isfinite(t) else None
    lo, hi = (float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))) if dr else (float("nan"),) * 2
    return {"theta": pt, "lo": lo, "hi": hi, "n_F1": int(f1.sum()), "n_NF": int(nf.sum()), "boot_ok": len(dr)}


def theta_diff_ci(patients, m1, m0, f1, nf, drawn_a, seed_name, B=B_TEST):
    """d = theta(m1) - theta(m0) on the same tumours and the same resampled patients; m = (det_a, det_p)."""
    a1, p1, a0, p0 = (np.asarray(x) for x in (*m1, *m0)); drawn_a = np.asarray(drawn_a); pt = theta_hat(a1, p1, f1, nf, drawn_a) - theta_hat(a0, p0, f1, nf, drawn_a); dr = []
    for r in _resamples(patients, B, seed_name):
        t1 = theta_hat(a1[r], p1[r], f1[r], nf[r], drawn_a[r]); t0 = theta_hat(a0[r], p0[r], f1[r], nf[r], drawn_a[r])
        if np.isfinite(t1) and np.isfinite(t0):
            dr.append(t1 - t0)
    lo, hi = (float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))) if dr else (float("nan"),) * 2
    return {"d": float(pt), "lo": lo, "hi": hi, "boot_ok": len(dr)}


def label_theta(t):
    """Per-model label: BRIGHT PENALTY / REVERSED / INCONCLUSIVE (public models only ever get these)."""
    if not np.isfinite(t["theta"]):
        return "INCONCLUSIVE (not estimable)"
    if t["lo"] > 0 and t["theta"] >= MARGIN_EFFECT:
        return "BRIGHT PENALTY"
    return "REVERSED" if t["hi"] < 0 and t["theta"] <= -MARGIN_EFFECT else "INCONCLUSIVE"


def label_equivalence(t, m=MARGIN_EQUIV):
    """S3: NO PENALTY if the whole CI lies within (-m, +m); otherwise INCONCLUSIVE."""
    return "NO PENALTY" if np.isfinite(t["theta"]) and t["lo"] > -m and t["hi"] < m else "INCONCLUSIVE"


def label_reduces(r):
    """Paired difference d = theta(m1) - theta(m0): REDUCES THE PENALTY if the CI upper bound < 0 and d <= -0.10; DOES NOT REDUCE if the CI lower bound > -0.10; else INCONCLUSIVE."""
    if not np.isfinite(r["d"]):
        return "INCONCLUSIVE (not estimable)"
    if r["hi"] < 0 and r["d"] <= -MARGIN_REDUCE:
        return "REDUCES THE PENALTY"
    return "DOES NOT REDUCE" if r["lo"] > -MARGIN_REDUCE else "INCONCLUSIVE"


def raw_components(det_a, det_p, f1, nf):
    """Plain probabilities behind theta (for the report): P(detected) on each series for F1 and NF tumours."""
    out = {}
    for nm, m in (("F1", f1), ("NF", nf)):
        out[nm] = {"n": int(m.sum()), "P_det_arterial": float(np.mean(np.asarray(det_a)[m])) if m.any() else None, "P_det_portal": float(np.mean(np.asarray(det_p)[m])) if m.any() else None}
    return out


def _gap_counts(patients, bright, vals):
    ids = sorted(set(patients)); ix = {p: i for i, p in enumerate(ids)}; c = np.zeros((len(ids), 4))
    for p, b, v in zip(patients, bright, vals):
        i = ix[p]; c[i, 1 if b else 0] += v; c[i, 3 if b else 2] += 1
    return ids, c


def _gap(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 2] - c[..., 1] / c[..., 3]


def gap_ratio(patients, bright, v_plain, v_dec, v_bright, seed_name, B=B_TEST):
    """R = (gap_plain - gap_dec) / (gap_plain - gap_bright) on one metric (gap = dark minus bright). Evaluable only if the CI of the denominator excludes 0. DECODER SUFFICES if R >= 0.5 and its CI lower bound >= 0.25; ENCODER NEEDED if R < 0.5 and the CI upper bound < 0.75; else INCONCLUSIVE."""
    ids, cp = _gap_counts(patients, bright, v_plain); _, cd = _gap_counts(patients, bright, v_dec); _, cb = _gap_counts(patients, bright, v_bright)
    ix = np.random.default_rng(name_seed(seed_name)).integers(0, len(ids), (B, len(ids))); gp, gd, gb = _gap(cp[ix].sum(1)), _gap(cd[ix].sum(1)), _gap(cb[ix].sum(1)); den = gp - gb; num = gp - gd
    den0 = float(_gap(cp.sum(0)) - _gap(cb.sum(0))); num0 = float(_gap(cp.sum(0)) - _gap(cd.sum(0))); dlo, dhi = (float(np.nanpercentile(den, 2.5)), float(np.nanpercentile(den, 97.5)))
    res = {"denominator": den0, "den_lo": dlo, "den_hi": dhi, "numerator": num0}
    if not (dlo > 0 or dhi < 0):
        return {**res, "R": float("nan"), "label": "NOT EVALUABLE"}
    ok = np.isfinite(den) & (np.abs(den) > 1e-9); r = num[ok] / den[ok]; R = num0 / den0; lo, hi = float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
    lab = "DECODER SUFFICES" if (R >= 0.5 and lo >= 0.25) else ("ENCODER NEEDED" if (R < 0.5 and hi < 0.75) else "INCONCLUSIVE")
    return {**res, "R": float(R), "R_lo": lo, "R_hi": hi, "label": lab}


def cond_logit_penalised(pat, det_a, det_p, D, names, seed_name, b=1000):
    """W4b secondary (ii): the conditional logistic regression reduced to the discordant tumours (z = detected arterial and missed portal-venous on the within-tumour differences D), Jeffreys-penalised in EVERY fit (point and bootstrap), so one estimator throughout.
    Returns OR and patient-bootstrap CI per coefficient (first = phase) and the discordant count."""
    pat = np.asarray(pat); det_a = np.asarray(det_a, int); det_p = np.asarray(det_p, int); D = np.asarray(D, float); d = det_a != det_p; z = (det_a[d] == 1).astype(float); X = np.column_stack([np.ones(d.sum()), D[d]]); cols = ["phase"] + list(names)
    if d.sum() < X.shape[1] + 2:
        return {"status": "NOT ESTIMABLE", "discordant": int(d.sum())}
    beta = fit_firth(X, z); se = sandwich_se(X, z, beta, pat[d]); draws = []
    for r in _resamples(pat, b, seed_name):
        da, dp = det_a[r], det_p[r]; dd = da != dp
        if dd.sum() >= X.shape[1] + 2:
            draws.append(fit_firth(np.column_stack([np.ones(dd.sum()), D[r][dd]]), (da[dd] == 1).astype(float)))
    dr = np.array(draws); out = {"status": "OK (penalised)", "discordant": int(d.sum()), "bootstrap_ok": len(draws), "coef": {}}
    for j, c in enumerate(cols):
        lo, hi = (float(np.percentile(dr[:, j], 2.5)), float(np.percentile(dr[:, j], 97.5))) if len(dr) else (float("nan"),) * 2
        out["coef"][c] = {"OR": float(np.exp(beta[j])), "boot_OR_lo": float(np.exp(lo)), "boot_OR_hi": float(np.exp(hi)), "wald_OR_lo": float(np.exp(beta[j] - 1.96 * se[j])), "wald_OR_hi": float(np.exp(beta[j] + 1.96 * se[j]))}
    return out
