"""Two-timepoint consistency check. ALL OUTPUT OF THIS MODULE IS SIMULATED. HCC-TACE-Seg has no usable follow-up pairs, so the pairs are generated from a bias model fitted to the
real per-lesion measurements (detection probability and size-dependent under-measurement, default vs recalibrated threshold; scripts/longitudinal_fit.py). They are never real patients
and every caller must label them as simulated. Pure numpy; research use only; not a diagnosis.
RECIST 1.1 target-lesion response, from the sum of target diameters: CR all targets gone; PR >= 30% decrease from baseline; PD >= 20% increase from the nadir AND >= 5 mm absolute
(or a new lesion); otherwise SD. With two timepoints the nadir is the baseline."""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

SIMULATED = "SIMULATED: generated from a fitted bias model, not real patients."
CATEGORIES = ("CR", "PR", "SD", "PD")
MEASURABLE_MM = 10.0
DEFAULT_PERSISTENCE = 0.5
DISAGREE_FLAG = 0.25   # "response depends on calibration" when default and recalibrated categories differ in at least this share of simulated draws
MODEL_PATH = os.environ.get("GLASSBOX_LONGITUDINAL_MODEL") or os.path.join(os.path.dirname(__file__), "longitudinal_model.json")


def recist_category(baseline_sum: float, followup_sum: float, nadir: float | None = None, new_lesion: bool = False) -> str:
    n = baseline_sum if nadir is None else min(nadir, baseline_sum)
    if followup_sum <= 0 and not new_lesion:
        return "CR"
    if new_lesion or (followup_sum >= 1.2 * n and followup_sum - n >= 5.0):
        return "PD"
    if followup_sum <= 0.7 * baseline_sum:
        return "PR"
    return "SD"


def match_lesions(a: list[dict], b: list[dict], max_mm: float = 25.0, ratio: tuple[float, float] = (0.3, 3.0)) -> dict:
    """Match lesions between two scans already in a common frame. Each lesion: {"centroid_mm": [x, y, z], "diameter_mm": d}. Optimal assignment by distance plus a size-change penalty
    (exhaustive for up to 7 lesions, greedy beyond); pairs further apart than `max_mm` or with an implausible size ratio stay unmatched. Returns indices into a and b."""
    def cost(i, j):
        d = math.dist(a[i]["centroid_mm"], b[j]["centroid_mm"]); r = (b[j]["diameter_mm"] + 1e-6) / (a[i]["diameter_mm"] + 1e-6)
        return d + 5.0 * abs(math.log(r)) if d <= max_mm and ratio[0] <= r <= ratio[1] else None
    n, m = len(a), len(b); pairs: list[tuple[int, int]] = []
    if n and m and max(n, m) <= 7:
        best = (math.inf, [])
        for cols in itertools.permutations(range(m), min(n, m)) if n <= m else itertools.permutations(range(n), m):
            ps = list(zip(range(n), cols)) if n <= m else [(r, c) for c, r in zip(range(m), cols)]
            cs = [cost(i, j) for i, j in ps]; ok = [(p, c) for p, c in zip(ps, cs) if c is not None]; tot = sum(c for _, c in ok) + 1000.0 * (len(ps) - len(ok))
            if tot < best[0]:
                best = (tot, [p for p, _ in ok])
        pairs = best[1]
    elif n and m:
        cand = sorted((cost(i, j), i, j) for i in range(n) for j in range(m) if cost(i, j) is not None); ui, uj = set(), set()
        for c, i, j in cand:
            if i not in ui and j not in uj:
                pairs.append((i, j)); ui.add(i); uj.add(j)
    mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
    return {"matched": sorted(pairs), "disappeared": [i for i in range(n) if i not in mi], "new": [j for j in range(m) if j not in mj]}


# ---------------------------------------------------------------- bias model
def _logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2, it: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); W = p * (1 - p) + 1e-9; g = X.T @ (y - p) - ridge * w; H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit_bias_model(rows: list[dict]) -> dict:
    """rows: {"gt": reference diameter (mm), "default": measured (0 = missed), "recal": measured, "bright": bool}. For each threshold: P(detected) = logistic(1, log gt, bright);
    among detected lesions, measured = a + b * gt + c * bright + noise(sd). Fitted by logistic / least squares on the cohort's per-lesion measurements."""
    gt = np.array([r["gt"] for r in rows], float); br = np.array([r["bright"] for r in rows], float); out = {"n_lesions": len(rows), "n_bright": int(br.sum()), "methods": {}}
    for k in ("default", "recal"):
        m = np.array([r[k] for r in rows], float); det = m > 0; Xd = np.column_stack([np.ones_like(gt), np.log(gt), br]); wd = _logistic(Xd, det.astype(float))
        Xs = np.column_stack([np.ones(det.sum()), gt[det], br[det]]); beta, *_ = np.linalg.lstsq(Xs, m[det], rcond=None); res = m[det] - Xs @ beta
        out["methods"][k] = {"detect_coef": wd.tolist(), "size_coef": beta.tolist(), "size_sd": float(res.std(ddof=3)) if det.sum() > 3 else float(res.std()), "n_detected": int(det.sum()),
                              "median_error_mm_detected": float(np.median(m[det] - gt[det])) if det.any() else None}
    return out


def _measure(method: dict, true_mm: np.ndarray, bright: bool, u: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Measured diameters for true diameters, using common random numbers u (detection) and z (noise)."""
    wd, beta, sd = np.array(method["detect_coef"]), np.array(method["size_coef"]), method["size_sd"]
    p = 1 / (1 + np.exp(-(wd[0] + wd[1] * np.log(np.maximum(true_mm, 1e-3)) + wd[2] * float(bright)))); mu = beta[0] + beta[1] * true_mm + beta[2] * float(bright)
    return np.where(u < p, np.maximum(mu + sd * z, 0.0), 0.0)


def simulate(model: dict, baseline_mm: list[float], change: float, bright_baseline: bool, bright_followup: bool, n: int = 2000, seed: int = 1, persistence: float = DEFAULT_PERSISTENCE) -> dict:
    """SIMULATED. True target lesions of `baseline_mm` all change by `change` (e.g. -0.25). Measure both timepoints under each threshold with common random numbers and report the
    RECIST category distribution, the mean measured sums, how often each is wrong, and how often the two thresholds disagree.
    `persistence` (0 to 1) is how much a lesion's detection draw carries over between the two visits: 0 = independent misses (exaggerates flips), 1 = the same lesion is missed or found in
    the same way both times unless its size change moves it across the threshold. The truth is in between; the default is a stated modelling assumption, not a fitted quantity."""
    t0 = np.array([d for d in baseline_mm if d >= MEASURABLE_MM][:2], float)
    if t0.size == 0:
        raise ValueError("needs at least one baseline lesion of 10 mm or more")
    t1 = t0 * (1 + change); truth = recist_category(float(t0.sum()), float(t1.sum())); rng = np.random.default_rng(seed)
    u0, u1f, z0, z1 = rng.random((n, t0.size)), rng.random((n, t0.size)), rng.standard_normal((n, t0.size)), rng.standard_normal((n, t0.size)); keep = rng.random((n, t0.size)) < persistence; u1 = np.where(keep, u0, u1f); res: dict = {"simulated": True, "notice": SIMULATED, "truth": {"category": truth, "baseline_sum_mm": float(t0.sum()), "followup_sum_mm": float(t1.sum()), "change": change}, "methods": {}}; cats = {}
    for k in ("default", "recal"):
        m0 = np.array([_measure(model["methods"][k], t0, bright_baseline, u0[i], z0[i]) for i in range(n)]); m1 = np.array([_measure(model["methods"][k], t1, bright_followup, u1[i], z1[i]) for i in range(n)])
        s0, s1 = m0.sum(1), m1.sum(1); c = np.array([recist_category(a, b) if a > 0 else "n/a" for a, b in zip(s0, s1)]); cats[k] = c
        res["methods"][k] = {"category_probability": {x: float((c == x).mean()) for x in CATEGORIES}, "baseline_not_measurable": float((c == "n/a").mean()), "mean_baseline_sum_mm": float(s0.mean()), "mean_followup_sum_mm": float(s1.mean()), "wrong_probability": float((c != truth).mean())}
    res["persistence"] = persistence; res["pair"] = _illustrative_pair(t0, t1, seed)
    both = (cats["default"] != "n/a") & (cats["recal"] != "n/a"); pd = float((cats["default"][both] != cats["recal"][both]).mean()) if both.any() else 0.0
    res["calibration_dependent"] = {"probability_default_and_recalibrated_differ": pd, "flag": pd >= DISAGREE_FLAG, "rule": f"flag when the two thresholds give different categories in at least {DISAGREE_FLAG:.0%} of simulated draws",
                                    "modal": {k: max(CATEGORIES, key=lambda x, k=k: res["methods"][k]["category_probability"][x]) for k in cats}}
    return res


def _illustrative_pair(t0: np.ndarray, t1: np.ndarray, seed: int) -> dict:
    """SIMULATED lesion positions for the display: baseline lesions 60 mm apart, follow-up positions jittered by a few mm, then matched exactly as real scans would be."""
    rng = np.random.default_rng(seed + 7); a = [{"centroid_mm": [60.0 * i, 0.0, 0.0], "diameter_mm": float(d)} for i, d in enumerate(t0)]
    b = [{"centroid_mm": [60.0 * i + float(rng.normal(0, 3)), float(rng.normal(0, 3)), float(rng.normal(0, 3))], "diameter_mm": float(d)} for i, d in enumerate(t1)]
    m = match_lesions(a, b); return {"baseline": a, "followup": b, "matching": {"matched": [[i, j] for i, j in m["matched"]], "disappeared": m["disappeared"], "new": m["new"]}}


def load_model(path: str | None = None) -> dict | None:
    p = path or MODEL_PATH
    return json.load(open(p)) if os.path.exists(p) else None
