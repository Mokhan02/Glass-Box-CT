import numpy as np

from glassbox.seg import w3s2_stats as S


def test_leace_removes_the_concept_and_random_eraser_is_a_projection():
    rng = np.random.default_rng(0); z = (rng.random(4000) > 0.5).astype(float); X = rng.normal(size=(4000, 12)); X[:, 0] += 2.5 * z; X[:, 3] -= 1.5 * z
    e = S.leace(X, z); Xe = S.apply_eraser(X, e)
    from sklearn.linear_model import LogisticRegression
    pre = LogisticRegression(max_iter=500).fit(X[:2000], z[:2000]).score(X[2000:], z[2000:]); post = LogisticRegression(max_iter=500).fit(Xe[:2000], z[:2000]).score(Xe[2000:], z[2000:])
    assert pre > 0.9 and post < 0.56                                                  # a linear probe falls to chance after rank-1 LEACE
    assert np.var(np.einsum("ni,i->n", Xe, e["a"])) < 1e-10                          # the erased direction is constant after erasure
    assert abs(np.corrcoef(np.einsum("ni,i->n", Xe, rng.normal(size=12)), z)[0, 1]) < 1e-8                      # no linear functional of x' correlates with the concept
    r = S.random_eraser(X.mean(0), 12, 100); Xr = S.apply_eraser(X, r); assert np.allclose((Xr - X.mean(0)) @ r["d"], 0, atol=1e-8) and np.allclose(S.apply_eraser(Xr, r), Xr)      # idempotent projection


def test_boot_stat_is_seeded_by_arm_name_and_covers_the_truth():
    rng = np.random.default_rng(1); cases = np.repeat([f"c{i}" for i in range(40)], 3); v = rng.normal(-0.3, 0.2, 120)
    a = S.boot_stat(cases, v, np.mean, "armX"); assert a == S.boot_stat(cases, v, np.mean, "armX") and a != S.boot_stat(cases, v, np.mean, "armY") and a[1] < a[0] < a[2] and a[2] < 0


def _m2(shift_leace=-0.3, shift_other=0.0, liver=0.0, n=60, probe_upper=0.55, seed=2):
    rng = np.random.default_rng(seed); cs = np.repeat([f"c{i}" for i in range(n)], 2); mk = lambda m: rng.normal(m, 0.05, 2 * n)      # noqa: E731
    return S.m2_verdict({"leace": (cs, mk(shift_leace)), "random": (cs, mk(shift_other)), "hu": (cs, mk(shift_other))}, (np.array([f"c{i}" for i in range(n)]), rng.normal(liver, 0.01, n)), probe_upper)


def test_m2_verdict_categories():
    r = _m2(); assert r["verdict"] == "USED" and r["meets_M2"] and not r["flags"]
    assert _m2(shift_leace=-0.3, liver=-0.1)["flags"] == ["erasure too blunt"] and not _m2(shift_leace=-0.3, liver=-0.1)["meets_M2"]
    r = _m2(shift_leace=0.0); assert r["verdict"] == "NOT NECESSARY" and not r["meets_M2"]
    r = _m2(shift_leace=0.2); assert r["verdict"] == "NOT NECESSARY" and "erasure improves" in r["flags"]
    r = _m2(shift_leace=-0.3, shift_other=-0.29); assert r["verdict"] == "NOT NECESSARY"                  # the controls erase as much: not specific
    r = _m2(probe_upper=0.8); assert r["verdict"] == "USED" and not r["meets_M2"] and any("not testing necessity" in f for f in r["flags"])


def _arm(shift_edit, shift_ctrl, n, direction, name, seed):
    rng = np.random.default_rng(seed); cs = np.array([f"c{i}" for i in range(n)]); e = rng.normal(shift_edit, 0.05, n); c = rng.normal(shift_ctrl, 0.05, n); return S.m3_arm_stats(cs, e, e - c, name, direction)


def test_m3_verdict_rule():
    assert S.m3_verdict(_arm(-0.4, 0.0, 30, -1, "A", 1), _arm(0.4, 0.0, 15, +1, "B", 2)) == "SUPPORTED"
    assert S.m3_verdict(_arm(0.0, 0.0, 30, -1, "A", 3), _arm(0.0, 0.0, 15, +1, "B", 4)) in ("NOT SUPPORTED", "INCONCLUSIVE")
    assert S.m3_verdict(_arm(0.02, 0.0, 30, -1, "A", 5), _arm(-0.02, 0.0, 15, +1, "B", 6)) == "NOT SUPPORTED"          # point estimates in the wrong direction, >= 10 lesions each
    assert S.m3_verdict(_arm(-0.4, 0.0, 30, -1, "A", 7), _arm(0.4, 0.0, 8, +1, "B", 8)) == "INCONCLUSIVE"                # Arm B too small
    assert S.m3_verdict(_arm(-0.4, 0.0, 30, -1, "A", 9), _arm(0.0, 0.0, 15, +1, "B", 10)) == "INCONCLUSIVE"              # only Arm A meets its condition
