import numpy as np

from glassbox.trust.phase import estimate_phase, tag_hint

Z = (1.5, 1.5, 1.5)


def _vol(aorta_hu, pv_hu, vessel=14):
    ct = np.full((60, 60, 60), 40.0, np.float32); liver = np.zeros(ct.shape, bool); liver[10:50, 10:50, 20:40] = True
    a = np.zeros(ct.shape, bool); a[5:5 + vessel, 5:5 + vessel, 15:45] = True; p = np.zeros(ct.shape, bool); p[30:30 + vessel, 30:30 + vessel, 22:38] = True
    ct[a] = aorta_hu; ct[p] = pv_hu
    return ct, a, p, liver


def test_the_pre_registered_rule_one_case_per_branch():
    for aorta, pv, want in ((45, 45, "unenhanced"), (260, 120, "arterial-like"), (200, 60, "arterial-like"), (150, 120, "venous-like"), (150, 60, "indeterminate"), (199.9, 100, "venous-like"), (99.9, 150, "unenhanced")):
        ct, a, p, l = _vol(aorta, pv); assert estimate_phase(ct, a, p, l, Z)["estimate"] == want, (aorta, pv)


def test_a_vessel_that_is_too_small_after_erosion_is_not_measurable():
    ct, a, p, l = _vol(300, 150, vessel=4)  # 4 voxels wide: erodes to 2 x 2 x 28 = 112 voxels, under the 200-voxel minimum
    out = estimate_phase(ct, a, p, l, Z); assert out["estimate"] == "indeterminate" and out["aorta_hu"] is None


def test_the_aorta_is_limited_to_the_slab_around_the_liver():
    ct, a, p, l = _vol(150, 120); ct[:, :, 0:12] = 40.0; a2 = a.copy(); a2[5:19, 5:19, 0:12] = True; ct[a2 & ~a] = 400.0  # a very bright segment far below the liver must not count
    assert estimate_phase(ct, a2, p, l, Z)["estimate"] == "venous-like"


def test_dicom_tags_are_reported_and_flag_a_disagreement_but_never_override():
    ct, a, p, l = _vol(260, 120)
    out = estimate_phase(ct, a, p, l, Z, {"SeriesDescription": "Portal venous 5.0"}); assert out["estimate"] == "arterial-like" and out["tag_disagrees"] and "check the phase" in out["note"]
    assert not estimate_phase(ct, a, p, l, Z, {"SeriesDescription": "Late arterial"})["tag_disagrees"]
    assert "HU only" in estimate_phase(ct, a, p, l, Z)["note"]
    assert tag_hint({"SeriesDescription": "NON-CONTRAST"})["hint"] == "unenhanced" and tag_hint({"SeriesDescription": "Delayed 3 min"})["hint"] == "delayed" and tag_hint({})["hint"] is None
