import numpy as np

from glassbox.quality.flag import assess, load_model, tumour_features


def _map(radius, prob):
    p = np.full((40, 40, 40), 0.01, np.float32); zz, yy, xx = np.mgrid[:40, :40, :40]; p[(zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2 <= radius ** 2] = prob; return p


def test_empty_prediction_is_not_assessable_and_says_so():
    a = assess(tumour_features(np.full((10, 10, 10), 0.2, np.float32), 1.0))
    assert a["assessed"] is False and a["flag"] == "not assessable" and "not the same as no tumour" in a["reason"]


def test_small_low_confidence_predictions_are_flagged_and_large_confident_ones_are_not():
    small = assess(tumour_features(_map(2, 0.62), voxel_mm3=3.4)); big = assess(tumour_features(_map(14, 0.98), voxel_mm3=3.4))
    assert small["flag"] == "likely wrong" and big["flag"] == "no flag"
    assert small["failure_probability"] > big["failure_probability"]


def test_drivers_are_the_size_and_confidence_terms_and_weights_sum_to_one():
    a = assess(tumour_features(_map(3, 0.7), 3.4)); assert abs(sum(d["weight_fraction"] for d in a["drivers"]) - 1) < 1e-9
    assert all(("tumour" in d["name"]) for d in a["drivers"]) and a["flag_threshold_probability"] == load_model()["flag_threshold_probability"]


def test_features_use_the_whole_scan_and_real_voxel_volume():
    f = tumour_features(_map(5, 0.9), voxel_mm3=1000.0)  # 1 cm3 voxels
    assert abs(f["volume_cm3"] - (_map(5, 0.9) > 0.5).sum()) < 1e-6
