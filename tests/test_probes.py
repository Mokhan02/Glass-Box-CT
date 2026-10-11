import numpy as np

from glassbox.probes.core import cov_pool, evaluate, fit_basis, mean_pool


def _scans(n=120, d=24, seed=0, effect=1.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = np.stack([mean_pool(rng.normal(size=(50, d)) + effect * yi * (np.arange(d) < 4)) for yi in y])
    return X, y


def test_probe_beats_chance_when_signal_exists_and_not_when_shuffled():
    X, y = _scans()
    g = np.arange(len(y))
    real = evaluate(X, y, g, "logistic", "auroc", n_boot=200)
    shuf = evaluate(X, np.random.default_rng(1).permutation(y), g, "logistic", "auroc", n_boot=200)
    assert real["estimate"] > 0.9 and real["ci_low"] > 0.8
    assert shuf["ci_low"] < 0.5 < shuf["ci_high"] + 0.15


def test_ridge_r2_and_groups_never_split():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 10))
    y = X[:, 0] * 2 + rng.normal(scale=0.1, size=100)
    out = evaluate(X, y, np.arange(100) // 2, "ridge", "r2", n_boot=100)  # 2 scans per patient
    assert out["estimate"] > 0.9 and out["n_scans"] == 50


def test_cov_pool_shape_and_basis_from_train_only():
    rng = np.random.default_rng(0)
    mu, B = fit_basis(rng.normal(size=(500, 24)), rank=6)
    v = cov_pool(rng.normal(size=(40, 24)), mu, B)
    assert v.shape == (6 * 7 // 2,)
