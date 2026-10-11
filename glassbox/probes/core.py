"""Probe machinery shared by RQ1 and RQ2.

Pooling turns a scan's token table [T, C] into one vector:
  mean_pool  - mean token
  cov_pool   - low-rank second moment (covariance pooling): project tokens onto an r-dim basis fit on TRAIN scans only,
               then take the upper triangle of the r x r covariance (r(r+1)/2 features).
Evaluation: grouped K-fold (a group = one patient, never split across folds), out-of-fold predictions, and a
patient-level bootstrap for 95% CIs. Every number reported should come from `evaluate`.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mean_pool(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).mean(0)


def fit_basis(token_sample: np.ndarray, rank: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """PCA basis from a token sample drawn from TRAIN scans only. Returns (mean [C], basis [C, rank])."""
    x = token_sample.astype(np.float32)
    mu = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:rank].T


def cov_pool(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    z = (x.astype(np.float32) - mu) @ basis
    c = (z.T @ z) / max(len(z) - 1, 1)
    return c[np.triu_indices(c.shape[0])]


def _model(kind: str, alpha: float = 10.0):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))


def oof_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions with GroupKFold. kind: 'ridge' (continuous) or 'logistic' (binary or multiclass)."""
    pred = None
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = _model(kind).fit(X[tr], y[tr])
        if kind == "ridge":
            p = m.predict(X[te])
        else:
            p = m.predict_proba(X[te])
            if pred is None:
                pred = np.zeros((len(y), p.shape[1]))
        if pred is None:
            pred = np.zeros(len(y))
        pred[te] = p
    return pred


def score(kind_metric: str, y: np.ndarray, pred: np.ndarray, classes: np.ndarray | None = None) -> float:
    if kind_metric == "r2":
        return float(r2_score(y, pred))
    if kind_metric == "auroc":
        return float(roc_auc_score(y, pred[:, 1] if pred.ndim == 2 else pred))
    if kind_metric == "balanced_acc":
        return float(balanced_accuracy_score(y, classes[pred.argmax(1)]))
    raise ValueError(kind_metric)


def bootstrap_ci(y, pred, groups, kind_metric: str, classes=None, n: int = 1000, seed: int = 0):
    """95% CI from resampling GROUPS (patients) with replacement. Skips resamples where a metric is undefined."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        try:
            vals.append(score(kind_metric, y[idx], pred[idx], classes))
        except ValueError:  # e.g. only one class in the resample
            continue
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), len(vals)


def evaluate(X, y, groups, kind: str, metric: str, n_boot: int = 1000, seed: int = 0) -> dict:
    """One probe result: point estimate from out-of-fold predictions plus a patient-bootstrap CI."""
    classes = np.unique(y) if kind == "logistic" else None
    yy = y if kind == "ridge" else np.searchsorted(classes, y)
    pred = oof_predict(X, yy, groups, kind)
    est = score(metric, yy, pred, np.arange(len(classes)) if classes is not None else None)
    lo, hi, k = bootstrap_ci(yy, pred, groups, metric, np.arange(len(classes)) if classes is not None else None, n_boot, seed)
    return {"estimate": est, "ci_low": lo, "ci_high": hi, "n_scans": int(len(np.unique(groups))), "n_boot_used": k}
