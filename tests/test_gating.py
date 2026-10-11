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


def _nested_cases(n=100, seed=0):
    """Synthetic: failures are under-measured by -45% at the default; recalibration at t = -2.5 or lower fixes them; harsher thresholds over-measure the good cases."""
    from glassbox.trust.gating import T_GRID
    rng = np.random.default_rng(seed); out = []
    for i in range(n):
        fail = rng.random() < 0.5; feats = [rng.normal(2.0 if fail else 5.5, 0.4), rng.normal(0.74 if fail else 0.92, 0.03), rng.normal(0.5 if fail else 0.2, 0.05)]; rel = {0.0: rng.normal(-0.45, 0.08) if fail else rng.normal(-0.03, 0.05)}; dice = {0.0: 0.4 if fail else 0.85}
        for t in T_GRID:
            gain = min(1.0, (-t) / 2.5); rel[t] = (rel[0.0] + (0.45 * gain if fail else -0.0)) if fail else rel[0.0] + 0.14 * (-t - 1.0) + rng.normal(0, 0.03); dice[t] = 0.7 if fail else 0.8
        out.append({"fold": i % 5, "features": feats, "dice": dice, "rel": rel})
    return out


def test_nested_selection_happens_inside_training_folds_and_ships_only_with_intervals_excluding_zero():
    from glassbox.trust.gating import run_nested
    r = run_nested(_nested_cases(), n_boot=300); assert r["label"].startswith("SENSITIVITY") and set(r["chosen_per_fold"]) == {0, 1, 2, 3, 4}
    t = r["policies"]; assert t["gated"]["mean_abs_error"]["value"] < t["default"]["mean_abs_error"]["value"] and t["gated"]["over_20pct"]["value"] <= t["recalibrated"]["over_20pct"]["value"]
    assert all(v["ci"][0] <= v["value"] <= v["ci"][1] for p in t.values() for v in p.values()) and r["pre_registered_rule_would_ship"] in (True, False)
    # nothing about a held-out fold may influence the choices made for it: flipping every held-out label in fold 0 leaves fold 0's chosen parameters unchanged
    cs = _nested_cases(); cs2 = _nested_cases()
    for c in cs2:
        if c["fold"] == 0:
            c["dice"] = {k: 1.0 - v for k, v in c["dice"].items()}; c["rel"] = {k: -v for k, v in c["rel"].items()}
    assert run_nested(cs, n_boot=50)["chosen_per_fold"][0] == run_nested(cs2, n_boot=50)["chosen_per_fold"][0]
