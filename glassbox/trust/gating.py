"""Flag-gated recalibration: the cross-validated comparison and the shipping decision pre-registered in results/trust/PREREG.md (section B). Pure functions on per-case summaries.
Each case: {"fold": int, "features": [log1p vol, mean prob, mean entropy] or None if the default prediction is empty, "dice_default": float,
            "rel_default": float|nan, "rel_recal": float|nan, "dice_default"/"dice_recal"}. rel_* = relative error of the sum of target diameters (nan when no measurable target)."""
from __future__ import annotations

import numpy as np

SENSITIVITY = 0.90
DICE_FAIL = 0.7
OVER = 0.20


def fit_flag(X: np.ndarray, y: np.ndarray, sens: float = SENSITIVITY) -> dict | None:
    """Logistic regression on standardised features (C = 1) and the flag threshold: the largest training-fold probability whose sensitivity for y is >= `sens` (90% in the pre-registered analysis)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(y) < 6 or y.min() == y.max():
        return None
    sc = StandardScaler().fit(X); clf = LogisticRegression(C=1.0).fit(sc.transform(X), y); s = clf.predict_proba(sc.transform(X))[:, 1]
    thr = next((float(t) for t in np.unique(s)[::-1] if (s[y == 1] >= t).mean() >= sens), float(s.min()))
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


# ------------------------------------------------------------------------------------------------ sensitivity analysis (NOT pre-registered)
T_GRID = (-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0)
S_GRID = (0.80, 0.85, 0.90, 0.95)


def _fit_and_flag(train: list[dict], test: list[dict], sens: float) -> list[bool]:
    tr = [c for c in train if c["features"] is not None]
    model = fit_flag(np.array([c["features"] for c in tr], float), np.array([int(c["dice"][0.0] < DICE_FAIL) for c in tr]), sens) if tr else None
    return [flagged(model, c["features"]) for c in test]


def run_nested(cases: list[dict], t_grid=T_GRID, s_grid=S_GRID, inner_k: int = 4, n_boot: int = 2000, seed: int = 1) -> dict:
    """Nested cross-validation: inside each outer training fold, choose the recalibrated threshold (for always-recalibrated: smallest mean absolute error) and, for the gated
    policy, the recalibrated threshold AND the flag's sensitivity target (smallest mean absolute error of the gated policy under an inner cross-validation); then apply to the held-out
    fold. Each case: {"fold", "features", "dice": {t: float}, "rel": {t: float}} with t = 0.0 the default threshold and every t of the grid present."""
    P = len(cases); rel = lambda c, t: c["rel"][t]  # noqa: E731
    pol = {"default": np.full(P, np.nan), "recalibrated": np.full(P, np.nan), "gated": np.full(P, np.nan)}; dic = {k: np.full(P, np.nan) for k in pol}; chosen = {}
    for f in sorted({c["fold"] for c in cases}):
        train = [c for c in cases if c["fold"] != f]; test_ix = [i for i, c in enumerate(cases) if c["fold"] == f]; test = [cases[i] for i in test_ix]
        mae = lambda vals: float(np.nanmean(np.abs(vals))) if np.isfinite(vals).any() else float("inf")  # noqa: E731
        t_r = min(t_grid, key=lambda t: mae(np.array([rel(c, t) for c in train], float)))
        best = (float("inf"), None, None)
        for t in t_grid:
            for sens in s_grid:
                errs = []
                for g in range(inner_k):
                    itr, iva = [c for j, c in enumerate(train) if j % inner_k != g], [c for j, c in enumerate(train) if j % inner_k == g]
                    errs += [rel(c, t if fl else 0.0) for c, fl in zip(iva, _fit_and_flag(itr, iva, sens))]
                m = mae(np.array(errs, float))
                if m < best[0] - 1e-12:
                    best = (m, t, sens)
        _, t_g, s_g = best; fl = _fit_and_flag(train, test, s_g); chosen[int(f)] = {"recalibrated_logit_for_always_recal": t_r, "gated_recalibrated_logit": t_g, "gated_flag_sensitivity": s_g, "n_gated": int(sum(fl))}
        for i, c, flg in zip(test_ix, test, fl):
            pol["default"][i], dic["default"][i] = rel(c, 0.0), c["dice"][0.0]; pol["recalibrated"][i], dic["recalibrated"][i] = rel(c, t_r), c["dice"][t_r]
            tt = t_g if flg else 0.0; pol["gated"][i], dic["gated"][i] = rel(c, tt), c["dice"][tt]
    ok = ~np.isnan(pol["default"]); n = int(ok.sum()); idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n)); pol = {k: v[ok] for k, v in pol.items()}; dic = {k: v[ok] for k, v in dic.items()}

    def stat(fn, v):
        pt = float(fn(v)); b = np.array([fn(v[i]) for i in idx]); return {"value": pt, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    table = {k: {"median_signed_error": stat(np.median, pol[k]), "mean_abs_error": stat(lambda x: np.mean(np.abs(x)), pol[k]), "over_20pct": stat(lambda x: np.mean(x >= OVER), pol[k]),
                 "abs_error_ge_20pct": stat(lambda x: np.mean(np.abs(x) >= OVER), pol[k]), "mean_dice": stat(np.mean, dic[k])} for k in pol}
    over_diff = (pol["gated"] >= OVER).astype(float) - (pol["recalibrated"] >= OVER).astype(float); acc_diff = np.abs(pol["gated"]) - np.abs(pol["default"])
    ci = lambda v: [float(np.percentile([v[i].mean() for i in idx], 2.5)), float(np.percentile([v[i].mean() for i in idx], 97.5))]  # noqa: E731
    return {"n_cases_with_target": n, "policies": table, "chosen_per_fold": chosen, "gated_minus_recal_over20": {"mean": float(over_diff.mean()), "ci": ci(over_diff)}, "gated_minus_default_abs_error": {"mean": float(acc_diff.mean()), "ci": ci(acc_diff)},
            "pre_registered_rule_would_ship": bool(ci(over_diff)[1] < 0 and ci(acc_diff)[1] < 0), "label": "SENSITIVITY ANALYSIS, not pre-registered"}
