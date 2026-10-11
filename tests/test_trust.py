from glassbox.agent.tools import qty
from glassbox.trust.verdict import REVIEW, TRUSTED, UNRELIABLE, assess, sentence


def L(i, d, v, mp=0.9):
    return {"id": i, "longest_axial_diameter": qty(d, "mm"), "volume": qty(v, "cm3"), "mean_probability": mp, "recist": "target candidate" if d >= 10 else "non-target"}


def Q(flag="no flag", p=0.1, drivers=None):
    return {"assessed": True, "flag": flag, "failure_probability": p, "flag_threshold_probability": 0.46, "drivers": drivers or []}


def test_quiet_flag_and_dark_tumour_is_trusted_with_a_reason_from_the_quality_audit():
    v = assess({"lesions": [L(1, 40, 20.0)], "tumour_liver_contrast": qty(-20, "HU")}, Q(p=0.05))
    assert v["level"] == TRUSTED and v["lesions"][0]["level"] == TRUSTED and v["reasons"][0]["source"] == "quality_audit" and sentence(v["level"])


def test_a_triggered_flag_makes_the_scan_and_its_lesions_unreliable_and_names_the_drivers():
    d = [{"name": "predicted tumour volume", "pushes_toward": "failure"}, {"name": "mean model probability inside the predicted tumour", "pushes_toward": "failure"}]
    v = assess({"lesions": [L(1, 40, 20.0), L(2, 5, 0.05)]}, Q("likely wrong", 0.99, d))
    assert v["level"] == UNRELIABLE and [x["level"] for x in v["lesions"]] == [UNRELIABLE, UNRELIABLE] and "the size of the predicted tumour" in v["reasons"][0]["text"]


def test_a_bright_tumour_downgrades_a_quiet_flag_to_review_and_says_why():
    v = assess({"lesions": [L(1, 73, 180.0)], "tumour_liver_contrast": qty(25.2, "HU")}, Q(p=0.146))
    assert v["level"] == REVIEW and v["lesions"][0]["level"] == REVIEW and any(r["code"] == "bright_tumour" for r in v["reasons"])


def test_the_phase_estimate_plays_no_part_in_verdicts_and_the_bright_warning_uses_measured_contrast_only():
    import inspect
    assert "phase" not in inspect.signature(assess).parameters  # kappa 0.04 against the reference contrast sign: not a basis for a verdict
    dark = assess({"lesions": [L(1, 40, 20.0)], "tumour_liver_contrast": qty(-5, "HU")}, Q(p=0.05)); assert dark["level"] == TRUSTED and not any("phase" in r["code"] for r in dark["reasons"])
    bright = assess({"lesions": [L(1, 40, 20.0)], "tumour_liver_contrast": qty(5, "HU")}, Q(p=0.05)); assert bright["level"] == REVIEW and any(r["code"] == "bright_tumour" for r in bright["reasons"])


def test_lesion_rules_fragments_small_and_low_confidence():
    v = assess({"lesions": [L(1, 40, 20.0), L(2, 2.0, 0.01), L(3, 6.0, 0.3), L(4, 30, 5.0, mp=0.5)]}, Q(p=0.05))
    assert [x["level"] for x in v["lesions"]] == [TRUSTED, UNRELIABLE, REVIEW, REVIEW]


def test_many_fragments_put_the_scan_in_review_and_no_tumour_is_review_not_trusted():
    frags = [L(i, 1.0, 0.0) for i in range(6)]
    assert assess({"lesions": [L(9, 40, 20.0)] + frags}, Q(p=0.05))["level"] == REVIEW
    v = assess({"lesions": []}, {"assessed": False, "flag": "not assessable", "reason": "x"}); assert v["level"] == REVIEW and v["reasons"][0]["code"] == "no_tumour"


def test_a_flag_close_to_the_threshold_is_review():
    assert assess({"lesions": [L(1, 40, 20.0)]}, Q(p=0.3))["level"] == REVIEW
