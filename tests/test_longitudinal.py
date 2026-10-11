import numpy as np

from glassbox.trust.longitudinal import CATEGORIES, fit_bias_model, match_lesions, recist_category, simulate


def test_recist_1_1_target_response_rules_including_the_5_mm_floor_and_the_nadir():
    assert recist_category(60, 0) == "CR" and recist_category(60, 42) == "PR" and recist_category(60, 42.1) == "SD" and recist_category(60, 72) == "PD" and recist_category(60, 71.9) == "SD"
    assert recist_category(20, 24) == "SD"            # +20% but only +4 mm: not progression
    assert recist_category(20, 25) == "PD"            # +25% and +5 mm
    assert recist_category(60, 50, nadir=40) == "PD"  # relative to the nadir, not the baseline
    assert recist_category(60, 59, new_lesion=True) == "PD"


def test_matching_follows_lesions_by_position_and_reports_disappeared_and_new():
    a = [{"centroid_mm": [0, 0, 0], "diameter_mm": 30}, {"centroid_mm": [100, 0, 0], "diameter_mm": 15}, {"centroid_mm": [0, 100, 0], "diameter_mm": 12}]
    b = [{"centroid_mm": [102, 3, 0], "diameter_mm": 14}, {"centroid_mm": [2, 1, 0], "diameter_mm": 25}, {"centroid_mm": [200, 200, 0], "diameter_mm": 20}]
    m = match_lesions(a, b); assert m["matched"] == [(0, 1), (1, 0)] and m["disappeared"] == [2] and m["new"] == [2]
    assert match_lesions(a, [{"centroid_mm": [0, 0, 0], "diameter_mm": 200}])["matched"] == []  # an implausible size jump is not the same lesion


def _rows(seed=0, n=300):
    """Planted: detection rises with size, measured = true - 6 mm - 0.1 * true (size-dependent under-measurement); recalibrated has none of that."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        g = float(rng.uniform(10, 100)); br = bool(rng.random() < 0.35); pd = 1 / (1 + np.exp(-(-4 + 1.6 * np.log(g) - 1.0 * br))); dm = g - 6 - 0.1 * g + rng.normal(0, 3) if rng.random() < pd else 0.0
        out.append({"gt": g, "bright": br, "default": max(dm, 0.0), "recal": max(g + rng.normal(0, 3), 0.1)})
    return out


def test_the_bias_model_recovers_the_planted_size_dependent_under_measurement():
    m = fit_bias_model(_rows()); d, r = m["methods"]["default"], m["methods"]["recal"]
    assert abs(d["size_coef"][0] + 6) < 3 and abs(d["size_coef"][1] - 0.9) < 0.08 and d["median_error_mm_detected"] < -4 and abs(r["median_error_mm_detected"]) < 3 and d["detect_coef"][2] < 0  # bright is missed more


def test_simulation_is_labelled_simulated_deterministic_and_shows_calibration_changing_the_category():
    m = fit_bias_model(_rows()); a = simulate(m, [60, 40], -0.25, False, False, n=800); b = simulate(m, [60, 40], -0.25, False, False, n=800)
    assert a == b and a["simulated"] is True and "not real patients" in a["notice"] and a["truth"]["category"] == "SD"
    pr = {k: sum(v["category_probability"][c] for c in CATEGORIES) + v["baseline_not_measurable"] for k, v in a["methods"].items()}; assert all(abs(x - 1) < 1e-9 for x in pr.values())
    # a planted additive bias exaggerates shrinkage: the default is more often wrong than the recalibrated measurement for this true stable disease
    assert a["methods"]["default"]["wrong_probability"] > a["methods"]["recal"]["wrong_probability"] and a["calibration_dependent"]["probability_default_and_recalibrated_differ"] > 0
    assert simulate(m, [60, 40], 0.0, False, False, n=400)["truth"]["category"] == "SD"


def test_api_serves_only_a_real_model_and_labels_every_response_simulated(tmp_path, monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from glassbox.app.server import create_app
    (tmp_path / "index.json").write_text("{}"); model = fit_bias_model(_rows()); model.update({"simulated": True, "fixture": True, "source": "TEST FIXTURE", "presets": []}); mp = tmp_path / "m.json"; mp.write_text(json.dumps(model))
    monkeypatch.setattr("glassbox.trust.longitudinal.MODEL_PATH", str(mp)); monkeypatch.delenv("GLASSBOX_ALLOW_FIXTURE", raising=False); c = TestClient(create_app(str(tmp_path)))
    assert c.get("/api/longitudinal/model").json()["available"] is False and c.post("/api/longitudinal/simulate", json={"baseline_mm": [50], "change": -0.2}).status_code == 503  # fixture refused
    monkeypatch.setenv("GLASSBOX_ALLOW_FIXTURE", "1"); r = c.post("/api/longitudinal/simulate", json={"baseline_mm": [50, 30], "change": -0.2}).json()
    assert r["simulated"] is True and "not real patients" in r["notice"] and r["fixture"] is True and r["pair"]["matching"]["matched"] == [[0, 0], [1, 1]]
    assert c.post("/api/longitudinal/simulate", json={"baseline_mm": [3], "change": 0.1}).status_code == 422 and c.post("/api/longitudinal/simulate", json={"baseline_mm": [50], "change": 9}).status_code == 422
    assert c.post("/api/longitudinal/simulate", json={"baseline_mm": [8, 9], "change": 0.1}).status_code == 422  # no measurable target lesion
