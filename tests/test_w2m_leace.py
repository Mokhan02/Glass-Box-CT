import sys

import numpy as np

sys.path.insert(0, "scripts")
import w2m_m2_fit as F  # noqa: E402
from glassbox.seg.w2_lib import eraser_apply_np, failure_mode  # noqa: E402


def test_leace_removes_the_linear_covariance_with_the_concept_and_keeps_the_mean():
    rng = np.random.default_rng(0); n, p = 4000, 24; z = (rng.random(n) < 0.3).astype(np.float32)
    X = rng.normal(0, 1, (n, p)).astype(np.float32); X[:, :3] += 2.0 * z[:, None]; X[:, 3:6] += 0.5 * rng.normal(0, 1, (n, 1))
    e = F.leace(X, z); Xe = eraser_apply_np(X, e["mu"], e["d"], e["a"], float(e["c"]))
    cov = ((Xe - Xe.mean(0)).T @ (z - z.mean())) / n; cov0 = ((X - X.mean(0)).T @ (z - z.mean())) / n
    assert np.abs(cov).max() < 1e-4 * np.abs(cov0).max() + 1e-5 and np.allclose(Xe.mean(0), X.mean(0), atol=0.05)         # cross-covariance with the concept is gone; the mean is (nearly) preserved


def test_erasure_makes_a_linear_probe_fall_to_chance_and_zero_d_is_the_identity():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1); n, p = 3000, 16; z = (rng.random(n) < 0.4).astype(int); X = rng.normal(0, 1, (n, p)); X[:, :2] += 1.5 * z[:, None]
    e = F.leace(X, z.astype(np.float32)); Xe = eraser_apply_np(X, e["mu"], e["d"], e["a"], float(e["c"]))
    auc = lambda M: roc_auc_score(z[1500:], LogisticRegression(max_iter=500).fit(M[:1500], z[:1500]).decision_function(M[1500:]))
    assert auc(X) > 0.85 and auc(Xe) < 0.58
    assert np.array_equal(eraser_apply_np(X, e["mu"], np.zeros_like(e["d"]), e["a"], float(e["c"])), X)           # d = 0 is the identity (the identity-hook check)


def test_failure_mode_matches_the_w2_v3_definition():
    assert failure_mode(3.0, 0.9, 0.9) == "LEAK" and failure_mode(1.0, 0.3, 0.9) == "LEAK" and failure_mode(1.0, 0.9, 0.49) == "UNDER" and failure_mode(5.0, 0.1, 0.1) == "LEAK" and failure_mode(1.0, 0.9, 0.5) == "OK"
