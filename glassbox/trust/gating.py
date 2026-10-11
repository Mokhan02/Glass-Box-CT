"""Flag-gated recalibration: the cross-validated comparison and the shipping decision pre-registered in results/trust/PREREG.md (section B). Pure functions on per-case summaries.
Each case: {"fold": int, "features": [log1p vol, mean prob, mean entropy] or None if the default prediction is empty, "dice_default": float,
            "rel_default": float|nan, "rel_recal": float|nan, "dice_default"/"dice_recal"}. rel_* = relative error of the sum of target diameters (nan when no measurable target)."""
from __future__ import annotations

import numpy as np

SENSITIVITY = 0.90
DICE_FAIL = 0.7
OVER = 0.20


def fit_flag(X: np.ndarray, y: np.ndarray) -> dict | None:
    """Logistic regression on standardised features (C = 1) and the flag threshold: the largest training-fold probability whose sensitivity for y is >= 90%."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(y) < 6 or y.min() == y.max():
        return None
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0).fit(sc.transform(X), y); s = clf.predict_proba(sc.transform(X))[:, 1]
    thr = next((float(t) for t in np.unique(s)[::-1] if (s[y == 1] >= t).mean() >= SENSITIVITY), float(s.min()))
    return {"sc": sc, "clf": clf, "thr": thr}


def flagged(model: dict | None, feats) -> bool:
    if feats is None:
        return True  # an empty default prediction predicts under-segmentation
    if model is None:
        return False
    return bool(model["clf"].predict_proba(model["sc"].transform(np.asarray([feats], float)))[0, 1] >= model["thr"])


def cv_flags(cases: list[dict]) -> list[bool]:
    """Out-of-fold flag decisions: the flag is refitted on training folds only (cases with an empty default prediction are not used to fit)."""
    out = [False] * len(cases)
    for f in sorted({c["fold"] for c in cases}):
        tr = [c for c in cases if c["fold"] != f and c["features"] is not None]
        model = fit_flag(np.array([c["features"] for c in tr], float), np.array([int(c["dice_default"] < DICE_FAIL) for c in tr])) if tr else None
        for i, c in enumerate(cases):
            if c["fold"] == f:
                out[i] = flagged(model, c["features"])
    return out


def _ci(vals: np.ndarray, idx: np.ndarray) -> list[float]:
    b = np.array([np.nanmean(vals[i]) for i in idx]); return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run_gating(cases: list[dict], n_boot: int = 2000, seed: int = 1) -> dict:
    fl = np.array(cv_flags(cases)); rd = np.array([c["rel_default"] for c in cases], float); rr = np.array([c["rel_recal"] for c in cases], float)
    rg = np.where(fl, rr, rd); ok = ~np.isnan(rd) & ~np.isnan(rr); P = int(ok.sum()); idx = np.random.default_rng(seed).integers(0, P, size=(n_boot, P))
    rd, rr, rg, fl_ok = rd[ok], rr[ok], rg[ok], fl[ok]
    dd, dr = np.array([c["dice_default"] for c in cases])[ok], np.array([c["dice_recal"] for c in cases])[ok]; dg = np.where(fl_ok, dr, dd)
    pol = {"default": rd, "recalibrated": rr, "gated": rg}; dice = {"default": dd, "recalibrated": dr, "gated": dg}
    table = {k: {"median_signed_error": float(np.median(v)), "mean_abs_error": float(np.mean(np.abs(v))), "over_20pct": float(np.mean(v >= OVER)), "abs_error_ge_20pct": float(np.mean(np.abs(v) >= OVER)), "mean_dice": float(np.mean(dice[k]))} for k, v in pol.items()}
    over_diff = (rg >= OVER).astype(float) - (rr >= OVER).astype(float); acc_diff = np.abs(rg) - np.abs(rd)
    ci_over, ci_acc = _ci(over_diff, idx), _ci(acc_diff, idx)
    ship = bool(ci_over[1] < 0 and ci_acc[1] < 0)
    return {"n_cases_with_target": P, "n_gated_to_recalibrated": int(fl_ok.sum()), "policies": table,
            "gated_minus_recal_over20": {"mean": float(over_diff.mean()), "ci": ci_over}, "gated_minus_default_abs_error": {"mean": float(acc_diff.mean()), "ci": ci_acc},
            "ship_gated_policy": ship, "decision": "SHIPPED: gated policy beats both" if ship else "NOT SHIPPED: keep the side-by-side default / recalibrated view (the gated policy does not beat both with a CI excluding zero)"}
