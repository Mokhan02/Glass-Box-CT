"""W3 Stage 2 statistics (pure numpy / scipy): case-level bootstrap with a per-arm seed, the LEACE closed form and eraser fitting, and the pre-registered verdict rules of M2 (mech B margins) and M3 (contrast_cf rule). Tested in tests/test_w3s2_stats.py."""
from __future__ import annotations

import hashlib

import numpy as np
from scipy.stats import binomtest

B_DEFAULT = 1000


def arm_seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def leace(X, z):
    """Rank-1 LEACE closed form (covariance ridge 1e-4 x mean eigenvalue): x' = x - d (a.(x - mu)) / c with d = Sigma_xz, a = Sigma^-1 Sigma_xz, c = a.d (scripts/mech_b_fit.py, scripts/w2m_m2_fit.py)."""
    X = np.asarray(X, np.float64); z = np.asarray(z, np.float64); mu = X.mean(0); Xc = X - mu; S = Xc.T @ Xc / len(X); S += np.eye(S.shape[0]) * 1e-4 * np.trace(S) / S.shape[0]
    d = Xc.T @ (z - z.mean()) / len(X); a = np.linalg.solve(S, d); return {"mu": mu, "d": d, "a": a, "c": float(a @ d)}


def random_eraser(mu, dim, seed):
    """Orthogonal projection removing a random unit direction u through the training-token mean: x' = x - u (u.(x - mu)), i.e. d = a = u, c = 1."""
    u = np.random.default_rng(seed).normal(size=dim); u /= np.linalg.norm(u); return {"mu": np.asarray(mu, np.float64), "d": u, "a": u, "c": 1.0}


def apply_eraser(x, e):
    x = np.asarray(x, np.float64); return x - np.outer((x - e["mu"]) @ e["a"], e["d"]) / e["c"]


def fold_of_cases(primary_cases, K=5):
    """fold = index in the sorted primary case list mod K (the PREREG definition)."""
    return {c: i % K for i, c in enumerate(sorted(primary_cases))}


def _groups(cases, values):
    cs = sorted(set(cases)); idx = {c: np.flatnonzero(np.asarray(cases) == c) for c in cs}; v = np.asarray(values, float); return cs, [v[idx[c]] for c in cs]


def boot_stat(cases, values, stat, seed_name, B=B_DEFAULT):
    """Case-level bootstrap of stat(concatenated values of the resampled cases): point, 2.5 and 97.5 percentiles (seeded by the arm name)."""
    cs, per = _groups(cases, values); rng = np.random.default_rng(arm_seed(seed_name)); n = len(cs); draws = []
    for row in rng.integers(0, n, (B, n)):
        v = np.concatenate([per[i] for i in row]); draws.append(stat(v))
    return float(stat(np.concatenate(per))), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_test(d, direction):
    """Exact two-sided binomial test of the number of non-zero d with the hypothesised sign (direction -1: d < 0, +1: d > 0) against 50%."""
    d = np.asarray(d, float); d = d[d != 0]
    if len(d) == 0:
        return {"n": 0, "k": 0, "p": 1.0}
    k = int((d < 0).sum() if direction < 0 else (d > 0).sum()); return {"n": int(len(d)), "k": k, "p": float(binomtest(k, len(d), 0.5).pvalue)}


# ------------------------------------------------------------------ M2
def m2_verdict(lesion_d, liver_d, probe_upper):
    """lesion_d: dict with 'leace', 'random', 'hu' -> (cases, per-lesion change in lesion Dice), liver_d: (cases, per-case change in liver Dice under LEACE), probe_upper: post-erasure held-out probe AUROC CI upper bound.
    USED iff the mean lesion Dice change under LEACE is < 0 with a CI excluding 0 AND the paired differences (LEACE - random) and (LEACE - HU) each have CIs entirely below 0 (mech B). Flags: erasure too blunt (mean liver Dice change < -0.05),
    erasure improves (positive mean with CI excluding 0), not testing necessity (post-erasure probe not near chance: upper CI > 0.60)."""
    mean = lambda v: float(np.mean(v))      # noqa: E731
    cs, dl = lesion_d["leace"]; out = {}
    out["leace"] = boot_stat(cs, dl, mean, "M2_leace"); out["random"] = boot_stat(lesion_d["random"][0], lesion_d["random"][1], mean, "M2_random"); out["hu"] = boot_stat(lesion_d["hu"][0], lesion_d["hu"][1], mean, "M2_hu")
    dr = np.asarray(lesion_d["leace"][1]) - np.asarray(lesion_d["random"][1]); dh = np.asarray(lesion_d["leace"][1]) - np.asarray(lesion_d["hu"][1])
    out["leace_minus_random"] = boot_stat(cs, dr, mean, "M2_leace_minus_random"); out["leace_minus_hu"] = boot_stat(cs, dh, mean, "M2_leace_minus_hu"); out["liver"] = boot_stat(liver_d[0], liver_d[1], mean, "M2_liver")
    used = out["leace"][2] < 0 and out["leace_minus_random"][2] < 0 and out["leace_minus_hu"][2] < 0
    blunt = out["liver"][0] < -0.05; improves = out["leace"][1] > 0; not_testing = probe_upper is not None and probe_upper > 0.60
    verdict = "USED" if used else "NOT NECESSARY"; flags = []
    if used and blunt:
        flags.append("erasure too blunt")
    if improves:
        flags.append("erasure improves")
    if not_testing:
        flags.append("not testing necessity (post-erasure probe not near chance)")
    out.update({"verdict": verdict, "flags": flags, "meets_M2": bool(used and not blunt and not not_testing), "probe_upper_CI": probe_upper}); return out


# ------------------------------------------------------------------ M3
def m3_arm_stats(cases, d_edit, d_diff, name, direction):
    """Median of delta_cov(edit) and of D = delta_cov(edit) - delta_cov(control) with case-bootstrap CIs and the paired sign test in the hypothesised direction (direction -1 for Arm A, +1 for Arm B)."""
    med = lambda v: float(np.median(v))      # noqa: E731
    return {"n": int(len(d_edit)), "n_cases": len(set(cases)), "median_edit": boot_stat(cases, d_edit, med, f"M3_{name}_edit"), "median_D": boot_stat(cases, d_diff, med, f"M3_{name}_D"), "sign_test": sign_test(d_diff, direction)}


def m3_verdict(A, Bm):
    """contrast_cf rule. SUPPORTED only if (1) Arm A: median delta_cov(edit) < 0 with CI entirely below 0 and median D_A < 0 with CI entirely below 0; (2) Arm B: both medians > 0 with CIs entirely above 0; (3) Arm A sign test p < 0.05 (D < 0 direction); (4) Arm B likewise.
    NOT SUPPORTED: neither arm meets (1)/(2) and in both arms the point estimate of median D is zero or in the wrong direction, with at least 10 lesions per arm. INCONCLUSIVE otherwise (including an arm with fewer than 10 lesions)."""
    a_ok = A["median_edit"][2] < 0 and A["median_D"][2] < 0; b_ok = Bm["median_edit"][1] > 0 and Bm["median_D"][1] > 0
    if a_ok and b_ok and A["sign_test"]["p"] < 0.05 and Bm["sign_test"]["p"] < 0.05 and A["n"] >= 10 and Bm["n"] >= 10:      # an arm with fewer than 10 lesions can never support the claim
        return "SUPPORTED"
    if A["n"] >= 10 and Bm["n"] >= 10 and not a_ok and not b_ok and A["median_D"][0] >= 0 and Bm["median_D"][0] <= 0:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"
