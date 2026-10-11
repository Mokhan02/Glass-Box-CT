"""Deployable quality flag for the hepatic-tumour class: logistic regression on predicted volume, mean probability and mean entropy inside the predicted mask (whole scan).
Fitted by scripts/fit_quality_flag.py on 75 held-out HCC cases; it flags when the estimated probability of tumour Dice < 0.7 reaches a threshold chosen for >= 90% sensitivity out of fold.
It is a research QC signal: it says nothing about structures it was not validated on (organs fail too rarely in the held-out data to fit one)."""
from __future__ import annotations

import json
import os

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "flag_model.json")


def load_model(path: str = MODEL_PATH) -> dict:
    return json.load(open(path))


def tumour_features(prob: np.ndarray, voxel_mm3: float) -> dict | None:
    """Features from a class-26 probability map (whole scan). None if nothing exceeds 0.5 (the flag cannot assess an empty prediction)."""
    m = prob > 0.5
    if not m.any():
        return None
    q = np.clip(prob[m].astype(np.float64), 1e-6, 1 - 1e-6)
    return {"volume_cm3": float(m.sum() * voxel_mm3 / 1000.0), "mean_probability": float(q.mean()), "mean_entropy": float((-(q * np.log(q) + (1 - q) * np.log(1 - q))).mean())}


def assess(feats: dict | None, model: dict | None = None) -> dict:
    model = model or load_model()
    if feats is None:
        return {"assessed": False, "reason": model["empty_prediction_rule"], "flag": "not assessable"}
    x = np.array([np.log1p(feats["volume_cm3"]), feats["mean_probability"], feats["mean_entropy"]])
    z = (x - np.array(model["mean"])) / np.array(model["scale"]); contrib = z * np.array(model["coef"])
    p = float(1 / (1 + np.exp(-(contrib.sum() + model["intercept"]))))
    names = ["predicted tumour volume", "mean model probability inside the predicted tumour", "mean prediction entropy inside the predicted tumour"]
    order = np.argsort(-np.abs(contrib))
    return {"assessed": True, "failure_probability": p, "flag": "likely wrong" if p >= model["flag_threshold_probability"] else "no flag",
            "drivers": [{"name": names[i], "pushes_toward": "failure" if contrib[i] > 0 else "success", "weight_fraction": float(abs(contrib[i]) / np.abs(contrib).sum())} for i in order],
            "flag_threshold_probability": model["flag_threshold_probability"], "validated_on": model["trained_on"], "caveat": "Validated for the hepatic-tumour class on treated HCC only; the base failure rate there was %.0f%%, so a flag is common." % (100 * model["base_failure_rate_in_training_data"])}
