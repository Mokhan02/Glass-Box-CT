import numpy as np

from glassbox.trust.gating import cv_flags, flagged, run_gating


def _cases(n=120, seed=0, gating_helps=True):
    """Synthetic: low-volume / low-confidence cases are the failures. Default under-measures them badly; recalibration fixes them but over-measures the good cases."""
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        fail = rng.random() < 0.5; vol = rng.normal(2.0 if fail else 5.5, 0.4); mp = rng.normal(0.74 if fail else 0.92, 0.03); ent = rng.normal(0.5 if fail else 0.2, 0.05)
        rd = rng.normal(-0.45, 0.08) if fail else rng.normal(-0.03, 0.05)
        rr = rng.normal(0.0, 0.08) if fail else (rng.normal(0.35 if gating_helps else -0.02, 0.1))
        out.append({"fold": i % 5, "features": [vol, mp, ent], "dice_default": 0.4 if fail else 0.85, "dice_recal": 0.7 if fail else 0.8, "rel_default": rd, "rel_recal": rr})
    return out


def test_the_flag_is_refitted_out_of_fold_and_catches_the_failures():
    cases = _cases(); fl = np.array(cv_flags(cases)); fail = np.array([c["dice_default"] < 0.7 for c in cases])
    assert fl[fail].mean() >= 0.85 and fl[~fail].mean() < 0.4  # sensitivity target of 90% on training folds, mostly quiet on good cases


def test_gating_is_shipped_only_when_it_beats_both_with_intervals_excluding_zero():
    r = run_gating(_cases(gating_helps=True))
    assert r["ship_gated_policy"] and r["policies"]["gated"]["over_20pct"] < r["policies"]["recalibrated"]["over_20pct"] and r["policies"]["gated"]["mean_abs_error"] < r["policies"]["default"]["mean_abs_error"]
    # when recalibration is harmless on good cases, the gated policy cannot beat always-recalibrated on over-estimation, so it is not shipped
    r2 = run_gating(_cases(gating_helps=False)); assert not r2["ship_gated_policy"] and "NOT SHIPPED" in r2["decision"]


def test_an_empty_default_prediction_is_flagged_and_cases_without_targets_are_excluded():
    assert flagged(None, None) is True and flagged(None, [1, 1, 1]) is False
    cases = _cases(); cases[0]["rel_default"] = float("nan"); assert run_gating(cases)["n_cases_with_target"] == len(cases) - 1
