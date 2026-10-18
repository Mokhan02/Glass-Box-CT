import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, "scripts")
import w2m_m1_analyze as A  # noqa: E402


def _tokens(direction, rng, n_cases, sep, seed_off=0):
    """Per case: tumour tokens shifted along `direction` by sep, kidney/liver tokens at 0; 192-d; HU feature carries no signal."""
    X, y, pid, bright = [], [], [], []
    for i in range(n_cases):
        nt, nk = 20, 60; t = rng.normal(0, 1, (nt, 192)) + sep * direction; k = rng.normal(0, 1, (nk, 192))
        X += [t, k]; y += [np.ones(nt, int), np.zeros(nk, int)]; pid += [np.full(nt + nk, i)]; bright += [np.arange(nt) % 2 == 0, np.zeros(nk, bool)]
    return np.concatenate(X), np.concatenate(y), np.concatenate(pid), np.concatenate(bright)


def _run(sep_h, sep_k, same_direction, seed=0):
    rng = np.random.default_rng(seed); d1 = rng.normal(0, 1, 192); d1 /= np.linalg.norm(d1); d2 = d1 if same_direction else np.linalg.qr(np.stack([d1, rng.normal(0, 1, 192)]).T)[0][:, 1]
    hx, hy, hp, _ = _tokens(d1 * 6, rng, 30, sep_h); kx, ky, kp, kb = _tokens(d2 * 6, rng, 25, sep_k)
    HX = {"trained": hx, "random": rng.normal(0, 1, hx.shape), "hu": rng.normal(0, 1, (len(hy), 14))}
    KX = {"trained": kx, "random": rng.normal(0, 1, kx.shape), "hu": rng.normal(0, 1, (len(ky), 14))}
    M = {"is_t": ky == 1, "is_k": ky == 0, "interior": np.ones(len(ky), bool), "bright": kb, "pid": kp}
    res, _, npos, nneg = A.analyse(HX, hy, hp, KX, M, 25, np.random.default_rng(0)); return res, A.verdict(res)


def test_shared_when_the_direction_is_the_same():
    res, v = _run(1.0, 1.0, True); assert v["verdict"] == "SHARED" and res[("transfer", "trained")]["value"] > 0.9 and res[("transfer", "random")]["hi"] < 0.6


def test_organ_specific_when_the_direction_differs():
    res, v = _run(1.0, 1.0, False); assert v["verdict"] == "ORGAN-SPECIFIC" and res[("native", "trained")]["value"] > 0.9 and res[("transfer", "trained")]["lo"] < 0.7


def test_partial_when_transfer_is_above_chance_but_below_70_percent_of_native():
    res, v = _run(1.0, 1.0, True); T = dict(res[("transfer", "trained")]); N = res[("native", "trained")]
    # force the three-tier logic with a weaker transfer than native but still above both controls
    res[("transfer", "trained")] = {"value": 0.70, "lo": 0.66, "hi": 0.74}; res[("native", "trained")] = {"value": 0.95, "lo": 0.93, "hi": 0.97}
    res[("transfer", "random")] = {"value": 0.5, "lo": 0.47, "hi": 0.53}; res[("transfer", "hu")] = {"value": 0.52, "lo": 0.49, "hi": 0.55}
    assert A.verdict(res)["verdict"] == "PARTIAL"                                                    # (0.70-0.5)/(0.95-0.5) = 0.44 < 0.70
    res[("transfer", "trained")] = {"value": 0.88, "lo": 0.84, "hi": 0.92}; assert A.verdict(res)["verdict"] == "SHARED"            # 0.38/0.45 = 0.84
    res[("transfer", "hu")] = {"value": 0.86, "lo": 0.83, "hi": 0.89}; assert A.verdict(res)["verdict"] == "ORGAN-SPECIFIC"          # does not beat the HU control
    res[("transfer", "trained")] = {"value": 0.30, "lo": 0.26, "hi": 0.34}; v = A.verdict(res); assert v["verdict"] == "ORGAN-SPECIFIC" and v["transfer_significantly_below_0.5"]


def test_auroc_basic():
    assert A.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0 and A.auroc(np.array([0.0]), np.array([1.0])) == 0.0
