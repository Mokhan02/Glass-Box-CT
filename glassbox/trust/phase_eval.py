"""Descriptive evaluation of the phase heuristic, as pre-registered (results/trust/PREREG.md section A). Nothing here is tuned or gated.
Row: {"estimate": str, "pred_contrast": float|nan, "ref_contrast": float, "missed": bool}  (missed = largest reference lesion covered < 10% by the default prediction)."""
from __future__ import annotations

import numpy as np


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool); po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def _two_by_two(x: np.ndarray, y: np.ndarray) -> dict:
    return {"both": int((x & y).sum()), "x_only": int((x & ~y).sum()), "y_only": int((~x & y).sum()), "neither": int((~x & ~y).sum()), "agreement": float((x == y).mean()), "kappa": kappa(x, y)}


def evaluate(rows: list[dict], n_boot: int = 2000, seed: int = 1) -> dict:
    est = np.array([r["estimate"] for r in rows]); pc = np.array([r["pred_contrast"] for r in rows], float); rc = np.array([r["ref_contrast"] for r in rows], float); miss = np.array([r["missed"] for r in rows], bool)
    art = est == "arterial-like"; warn = art | (np.nan_to_num(pc, nan=-1.0) > 0); refpos = rc > 0; definite = est != "indeterminate"; av = np.isin(est, ["arterial-like", "venous-like"])
    out = {"n_cases": len(rows), "estimates": {k: int((est == k).sum()) for k in ("arterial-like", "venous-like", "unenhanced", "indeterminate")}}
    out["arterial_vs_ref_contrast_positive"] = {"cases": int(definite.sum()), **_two_by_two(art[definite], refpos[definite])} if definite.any() else None
    out["venous_vs_ref_contrast_nonpositive"] = {"cases": int(av.sum()), **_two_by_two((est == "venous-like")[av], (~refpos)[av])} if av.any() else None
    rate = lambda m, s: float(m[s].mean()) if s.any() else float("nan")  # noqa: E731
    out["miss_rate"] = {"warning_fires": {"n": int(warn.sum()), "rate": rate(miss, warn)}, "warning_quiet": {"n": int((~warn).sum()), "rate": rate(miss, ~warn)},
                        "arterial_like": {"n": int(art.sum()), "rate": rate(miss, art)}, "not_arterial_like": {"n": int((~art).sum()), "rate": rate(miss, ~art)}}
    idx = np.random.default_rng(seed).integers(0, len(rows), size=(n_boot, len(rows))); ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]  # noqa: E731
    with np.errstate(all="ignore"):
        b = [{"w": rate(miss[i], warn[i]), "q": rate(miss[i], ~warn[i]), "a": rate(miss[i], art[i]), "na": rate(miss[i], ~art[i])} for i in idx]
        out["miss_rate"]["warning_fires"]["ci"] = ci([x["w"] for x in b]); out["miss_rate"]["warning_quiet"]["ci"] = ci([x["q"] for x in b]); out["miss_rate"]["arterial_like"]["ci"] = ci([x["a"] for x in b]); out["miss_rate"]["not_arterial_like"]["ci"] = ci([x["na"] for x in b])
        out["miss_rate"]["difference_fires_minus_quiet"] = {"mean": out["miss_rate"]["warning_fires"]["rate"] - out["miss_rate"]["warning_quiet"]["rate"], "ci": ci([x["w"] - x["q"] for x in b])}
        if out["arterial_vs_ref_contrast_positive"]:
            ag = [(art[i][definite[i]] == refpos[i][definite[i]]).mean() for i in idx if definite[i].any()]; kp = [kappa(art[i][definite[i]], refpos[i][definite[i]]) for i in idx if definite[i].any()]
            out["arterial_vs_ref_contrast_positive"]["agreement_ci"] = ci(ag); out["arterial_vs_ref_contrast_positive"]["kappa_ci"] = ci(kp)
    return out
