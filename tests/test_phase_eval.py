import numpy as np

from glassbox.trust.phase_eval import evaluate, kappa


def _rows(n=200, seed=0):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        art = rng.random() < 0.4; bright = art if rng.random() < 0.8 else not art; miss = (rng.random() < (0.5 if bright else 0.1))
        rows.append({"estimate": "arterial-like" if art else "venous-like", "pred_contrast": 10.0 if bright else -10.0, "ref_contrast": 15.0 if bright else -15.0, "missed": miss})
    return rows


def test_kappa_and_agreement_on_a_planted_relationship():
    r = evaluate(_rows(), n_boot=300); a = r["arterial_vs_ref_contrast_positive"]
    assert 0.7 < a["agreement"] < 0.9 and 0.4 < a["kappa"] < 0.8 and a["agreement_ci"][0] < a["agreement"] < a["agreement_ci"][1]
    assert abs(kappa(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0])) - 1.0) < 1e-9


def test_miss_rate_is_higher_when_the_warning_fires_and_the_interval_excludes_zero():
    m = evaluate(_rows(400), n_boot=500)["miss_rate"]
    assert m["warning_fires"]["rate"] > m["warning_quiet"]["rate"] and m["difference_fires_minus_quiet"]["ci"][0] > 0
