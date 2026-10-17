from glassbox.seg.w2_arms import ARMS, SPECIAL_INDEX, ZERO_SHOT_MIN, arm_passes, expected_point_label, lesion_passes


def test_special_classes_need_point_label_3_and_others_1():
    assert expected_point_label(26) == 3 and expected_point_label(133) == 1 and expected_point_label(129) == 1
    assert 26 in SPECIAL_INDEX and 129 not in SPECIAL_INDEX and 133 not in SPECIAL_INDEX and ZERO_SHOT_MIN == 133


def test_arm_table_matches_the_prereg():
    assert (ARMS["kidney_133_1"]["label"], ARMS["kidney_133_1"]["point_label"]) == (133, 1)
    assert (ARMS["hcc_26_3"]["label"], ARMS["hcc_26_3"]["point_label"]) == (26, 3)
    assert (ARMS["hcc_133_1"]["label"], ARMS["hcc_133_1"]["point_label"]) == (133, 1)
    assert (ARMS["hcc_26_1"]["label"], ARMS["hcc_26_1"]["point_label"]) == (26, 1)
    for name in ("kidney_133_1", "hcc_26_3", "hcc_133_1"):                 # every PRIMARY arm sends the label the model expects
        assert ARMS[name]["role"] == "primary" and ARMS[name]["point_label"] == expected_point_label(ARMS[name]["label"])
    assert ARMS["hcc_26_1"]["point_label"] != expected_point_label(26)       # the v1-equivalent arm deliberately sends the wrong one


def _rows(spec):
    return [{"pred_over_lesion_volume": v, "pred_frac_in_lesion": s, "stratum": st} for v, s, st in spec]


def test_lesion_rule():
    assert lesion_passes(2.9, 0.31) and not lesion_passes(3.0, 0.9) and not lesion_passes(1.0, 0.3)


def test_arm_needs_80_percent_overall_and_60_percent_per_stratum():
    good = [(1.5, 0.7, "a")] * 4 + [(1.5, 0.7, "b")] * 4 + [(1.5, 0.7, "c")] * 2
    assert arm_passes(_rows(good))["passes"]
    one_bad = [(9.0, 0.1, "a")] + good[1:]                                  # 9/10 pass overall (>= 80%), stratum a 3/4 = 75% (>= 60%)
    assert arm_passes(_rows(one_bad))["passes"]
    small_bad = [(9.0, 0.1, "c")] * 1 + good[:-1]                           # stratum c 1/2 = 50% < 60%
    r = arm_passes(_rows(small_bad)); assert not r["passes"] and r["pass_fraction"] == 0.9
    many_bad = [(9.0, 0.1, "a")] * 3 + good[3:]                             # 7/10 = 70% < 80%
    assert not arm_passes(_rows(many_bad))["passes"]
