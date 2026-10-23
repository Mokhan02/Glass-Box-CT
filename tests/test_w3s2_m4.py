import numpy as np

from glassbox.seg import w3s2_m4 as M


def _case(shift, seed, n_les=2, bright=False):
    """Synthetic annotated slab: two spherical tumours; the margin is high inside; `shift` lowers the margin of the tumour tissue below 0 (a model that misses it at the default threshold but separates it from liver by a few units)."""
    rng = np.random.default_rng(seed); X, Y, Z = 80, 80, 12; lab = np.ones((X, Y, Z), np.uint8); mg = rng.normal(-8, 0.5, (X, Y, Z)).astype(np.float32); yy, xx = np.mgrid[0:X, 0:Y]
    for k in range(n_les):
        cx, cy = 20 + 40 * k, 40; m = ((xx - cx) ** 2 + (yy - cy) ** 2) < 12 ** 2; lab[m, :] = 2; mg[m, :] = shift + rng.normal(0, 0.4, (int(m.sum()), Z))
    return M.CaseMap(mg, lab, (1.0, 1.0, 2.5))


def test_default_misses_and_shift_recovers_with_nested_threshold_choice():
    cases = [f"c{i:02d}" for i in range(30)]; group = {c: ["dark", "bright"] for c in cases}
    maps = {c: _case(-2.0 if i % 2 else 1.0, i) for i, c in enumerate(cases)}
    # bright lesion (rank 2) is missed at the default threshold in every case: make its margin -2.5 while the dark one is +1
    for i, c in enumerate(cases):
        rng = np.random.default_rng(100 + i); m = maps[c]; m.m[m.glab == m.lesions[1]["id"]] = -2.5 + rng.normal(0, 0.3, int((m.glab == m.lesions[1]["id"]).sum())); m.m[m.glab == m.lesions[0]["id"]] = 1.0
    r = M.m4_run(cases, maps, group, B=200, arm="test")
    assert r["tables"]["default"]["detection_gap"] > 0.9 and r["tables"]["shifted"]["detection_gap"] < 0.1          # the shift fixes the gap
    assert r["tables"]["default"]["median_signed_error"] < -0.3 and abs(r["tables"]["shifted"]["median_signed_error"]) < 0.15
    assert set(r["chosen_thresholds_per_fold"].values()) <= set(M.T_GRID) and all(t <= -2.5 for t in r["chosen_thresholds_per_fold"].values())
    assert r["verdict"] == "FIX GENERALISES" and r["gap_criterion_met"] and r["bias_criterion_met"]


def test_no_effect_when_the_default_is_already_right():
    cases = [f"c{i:02d}" for i in range(30)]; group = {c: ["dark", "bright"] for c in cases}; maps = {c: _case(1.5, i) for i, c in enumerate(cases)}
    r = M.m4_run(cases, maps, group, B=200, arm="test2")
    assert r["verdict"] == "DOES NOT GENERALISE" and abs(r["tables"]["shifted"]["median_signed_error"] - r["tables"]["default"]["median_signed_error"]) < 0.1


def test_over_estimation_cost_is_reported():
    cases = [f"c{i:02d}" for i in range(30)]; group = {c: ["dark", "bright"] for c in cases}; maps = {c: _case(1.5, i) for i, c in enumerate(cases)}
    for c in cases:                                                    # a halo of margin -3 around each tumour: lower thresholds over-estimate
        m = maps[c]; dil = np.zeros_like(m.tum)
        from scipy import ndimage
        dil = ndimage.binary_dilation(m.tum, iterations=4) & ~m.tum; m.m[dil] = -3.0
    r = M.m4_run(cases, maps, group, B=100, arm="test3")
    assert r["tables"]["shifted"]["over_20pct"] >= r["tables"]["default"]["over_20pct"]
