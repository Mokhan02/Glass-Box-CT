import json
import os

import nibabel as nib
import numpy as np

from glassbox.agent.backends import PrecomputedBackend
from glassbox.agent.tools import ToolBox
from glassbox.bodycomp.lesions import measure_lesions
from glassbox.pipeline.scan import ORGANS, PROMPTED, build_results
from glassbox.seg import classes as C


def _scan(tmp_path, tumour_radius=6, tumour_prob=0.95, with_l3=True, tumour_hu=70.0):
    shape = (80, 80, 50); zoom = (0.8, 0.8, 3.0); ct = np.full(shape, -50.0, np.float32); probs = {c: np.full(shape, 0.01, np.float32) for c in PROMPTED}
    xx, yy, zz = np.mgrid[:80, :80, :50]
    def blob(cx, cy, cz, rx, ry, rz): return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1
    liver = blob(30, 40, 25, 22, 20, 15); probs[C.LIVER][liver] = 0.97; ct[liver] = 100.0
    t = blob(30, 40, 25, tumour_radius, tumour_radius, 6); probs[C.HEPATIC_TUMOR][t] = tumour_prob; ct[t] = tumour_hu
    if with_l3:
        probs[C.VERTEBRA_L3][blob(60, 40, 25, 8, 8, 5)] = 0.95
        for cls, cx in ((C.LEFT_ILIOPSOAS, 52), (C.RIGHT_ILIOPSOAS, 68), (C.LEFT_AUTOCHTHON, 55), (C.RIGHT_AUTOCHTHON, 65)):
            m = blob(cx, 52 if cls in (C.LEFT_AUTOCHTHON, C.RIGHT_AUTOCHTHON) else 30, 25, 5, 5, 8); probs[cls][m] = 0.95; ct[m] = 45.0
    path = os.path.join(tmp_path, "ct.nii.gz"); nib.save(nib.Nifti1Image(ct, np.diag([*zoom, 1.0])), path); return path, probs


def test_results_files_are_written_and_pass_the_agent_unit_check(tmp_path):
    path, probs = _scan(str(tmp_path)); out = os.path.join(str(tmp_path), "res"); build_results(path, probs, out, "scanX", {"source": "synthetic", "in_distribution": False})
    tb = ToolBox(PrecomputedBackend(out))
    for tool in ("segment_ct", "body_composition", "measure_lesions", "quality_audit"):
        r = tb.call(tool, {"scan_id": "scanX"}); assert r["ok"], (tool, r.get("error"))
    les = tb.evidence[2]["result"]; assert les["n_lesions"] == 1 and 26 in les["prompted_classes"] and les["lesions"][0]["longest_axial_diameter"]["unit"] == "mm"
    assert les["validation_context"]["detected_lesion_median_relative_error"]["value"] == -17
    assert os.path.exists(os.path.join(out, "scanX", "tumour.nii.gz")) and os.path.exists(os.path.join(out, "scanX", "organs.nii.gz"))


def test_body_composition_reports_unavailable_when_l3_is_outside_the_scan(tmp_path):
    path, probs = _scan(str(tmp_path), with_l3=False); r = build_results(path, probs, os.path.join(str(tmp_path), "res"), "s", {})
    assert r["body_composition"]["available"] is False and "field of view" in r["body_composition"]["reason"]


def test_empty_tumour_prediction_says_it_cannot_assess_and_that_absence_is_not_proven(tmp_path):
    path, probs = _scan(str(tmp_path), tumour_prob=0.2); r = build_results(path, probs, os.path.join(str(tmp_path), "res"), "s", {})
    assert r["lesions"]["n_lesions"] == 0 and "not the same as no tumour" in r["lesions"]["note"]
    q = r["quality_audit"]["structures"][0]; assert q["assessed"] is False and q["flag"] == "not assessable"


def test_lesion_diameter_matches_a_known_sphere_and_recist_labels():
    shape = (60, 60, 30); xx, yy, zz = np.mgrid[:60, :60, :30]; m = ((xx - 30) ** 2 + (yy - 30) ** 2 + ((zz - 15) * 3.0 / 1.0) ** 2) <= 10 ** 2  # radius 10 voxels in-plane
    les = measure_lesions(m, None, (1.0, 1.0, 1.0)); assert abs(les[0]["diameter_mm"] - 20.0) < 1.5 and les[0]["recist"] == "target candidate"
    tiny = np.zeros(shape, bool); tiny[10:14, 10:14, 10:12] = True; assert measure_lesions(tiny, None, (1, 1, 1))[0]["recist"].startswith("non-target")


def test_organ_label_map_uses_larger_probability_for_overlaps(tmp_path):
    path, probs = _scan(str(tmp_path)); probs[C.SPLEEN if hasattr(C, "SPLEEN") else 3][30, 40, 25] = 0.99; from glassbox.pipeline.scan import organ_label_map
    lab = organ_label_map(probs); assert lab[30, 40, 25] == list(ORGANS.values()).index(3) + 1


def test_file_voxel_mapping_round_trips_for_a_flipped_axis_volume(tmp_path):
    import nibabel as nib
    from glassbox.pipeline.scan import canonical, to_file_voxel
    arr = np.zeros((10, 12, 8), np.uint8); arr[7, 3, 2] = 1; aff = np.diag([-1.0, 2.0, 3.0, 1.0]); aff[0, 3] = 9  # x axis stored flipped
    c, _, aff_c = canonical(arr, aff); v = tuple(int(i) for i in np.argwhere(c)[0]); assert tuple(to_file_voxel(v, aff_c, aff)) == (7, 3, 2)


def test_result_files_carry_voxels_in_the_file_frame_for_the_viewer(tmp_path):
    path, probs = _scan(str(tmp_path)); r = build_results(path, probs, os.path.join(str(tmp_path), "res"), "s", {})
    v = r["lesions"]["lesions"][0]["voxel_in_file"]; assert abs(v[0] - 30) <= 2 and abs(v[1] - 40) <= 2 and abs(v[2] - 25) <= 2  # the synthetic tumour sits at (30, 40, 25)
    assert len(r["body_composition"]["l3_voxel_in_file"]) == 3


def test_viewer_copies_keep_click_to_locate_on_the_lesion_after_resampling(tmp_path):
    from glassbox.pipeline.viewer import make_viewer_files
    path, probs = _scan(str(tmp_path)); out = os.path.join(str(tmp_path), "res"); r = build_results(path, probs, out, "s", {}); d = os.path.join(out, "s")
    before = r["lesions"]["lesions"][0]["voxel_in_file"]; info = make_viewer_files(path, d, mm=3.0)
    L = json.load(open(os.path.join(d, "lesions.json"))); v = L["lesions"][0]["voxel_in_file"]; assert v != before and L["viewer_grid_mm"] == {"value": 3.0, "unit": "mm"}
    tum = nib.load(os.path.join(d, "tumour.nii.gz")); arr = np.asarray(tum.dataobj); assert arr.shape == tuple(info["viewer_shape"]) and arr[tuple(v)] == 1  # the remapped voxel is inside the tumour
    B = json.load(open(os.path.join(d, "body_composition.json"))); assert np.asarray(nib.load(os.path.join(d, "organs.nii.gz")).dataobj)[tuple(B["l3_voxel_in_file"])] > 0


def test_contrast_is_measured_and_caution_appears_only_for_bright_tumours(tmp_path):
    from glassbox.app.report import template_report
    from glassbox.agent.grounding import check_report
    for hu, bright in ((70.0, False), (140.0, True)):
        d = tmp_path / str(int(hu)); d.mkdir()
        path, probs = _scan(str(d), tumour_hu=hu); out = os.path.join(str(d), "res"); r = build_results(path, probs, out, "s", {"source": "synthetic"})["lesions"]
        c = r["tumour_liver_contrast"]; assert c["unit"] == "HU" and abs(c["value"] - (hu - 100.0)) < 1.0
        assert ("contrast_caution" in r) is bright
        tb = ToolBox(PrecomputedBackend(out))
        for t in ("segment_ct", "body_composition", "measure_lesions", "quality_audit"):
            assert tb.call(t, {"scan_id": "s"})["ok"]
        rep = template_report(tb.evidence); chk = check_report(rep, tb.evidence)
        assert chk["unsupported_numbers"] == [] and chk["missing_units"] == [], chk
        assert ("tends to miss tumours brighter than the surrounding liver" in rep) is bright
        assert "Contrast here is measured on the predicted tumour only" in rep or not bright


def test_no_contrast_is_reported_when_nothing_is_predicted(tmp_path):
    path, probs = _scan(str(tmp_path), tumour_prob=0.2); r = build_results(path, probs, os.path.join(str(tmp_path), "res"), "s", {})["lesions"]
    assert "tumour_liver_contrast" not in r and "contrast_caution" not in r


def test_every_liver_report_carries_the_standing_limitation_even_with_no_lesion(tmp_path):
    from glassbox.agent.grounding import check_report
    from glassbox.app.report import template_report
    for prob in (0.95, 0.2):
        d = tmp_path / str(prob); d.mkdir(); path, probs = _scan(str(d), tumour_prob=prob); out = os.path.join(str(d), "res"); build_results(path, probs, out, "s", {"source": "synthetic"})
        tb = ToolBox(PrecomputedBackend(out))
        for t in ("segment_ct", "body_composition", "measure_lesions", "quality_audit"):
            assert tb.call(t, {"scan_id": "s"})["ok"]
        rep = template_report(tb.evidence)
        assert "known to miss some liver tumours brighter than the surrounding liver (research finding, n=89). A small or absent tumour prediction does not rule out a tumour." in rep
        chk = check_report(rep, tb.evidence); assert chk["unsupported_numbers"] == [] and chk["missing_units"] == [], chk


def test_cached_report_is_rebuilt_when_the_tool_results_change(tmp_path):
    from glassbox.app.report import make_report
    path, probs = _scan(str(tmp_path)); out = os.path.join(str(tmp_path), "res"); build_results(path, probs, out, "s", {"source": "synthetic"})
    first = make_report(out, "s"); assert make_report(out, "s")["inputs_sha"] == first["inputs_sha"]  # unchanged inputs: served from cache
    les = os.path.join(out, "s", "lesions.json"); d = json.load(open(les)); d["standing_limitation"] = "A DIFFERENT NOTE."; json.dump(d, open(les, "w"))
    second = make_report(out, "s"); assert "A DIFFERENT NOTE." in second["report"] and second["inputs_sha"] != first["inputs_sha"]


def test_contrast_text_carries_no_bare_numbers_the_checker_could_misattribute(tmp_path):
    path, probs = _scan(str(tmp_path), tumour_hu=140.0); les = build_results(path, probs, os.path.join(str(tmp_path), "res"), "s", {})["lesions"]
    assert not any(ch.isdigit() for ch in les["contrast_basis"]) and les["liver_reference_margin"] == {"value": 5, "unit": "mm"}


def test_align_to_viewer_puts_masks_on_the_viewer_grid_and_remaps_voxels(tmp_path):
    import nibabel as nib
    from glassbox.pipeline.viewer import align_to_viewer
    d = str(tmp_path); big = np.zeros((60, 60, 40), np.uint8); big[30:40, 20:30, 10:20] = 1  # original 1 mm grid, lesion centre about (34.5, 24.5, 14.5)
    nib.save(nib.Nifti1Image(big, np.diag([1.0, 1.0, 1.0, 1.0])), os.path.join(d, "tumour.nii.gz")); nib.save(nib.Nifti1Image(big, np.diag([1.0, 1.0, 1.0, 1.0])), os.path.join(d, "organs.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros((30, 30, 20), np.int16), np.diag([2.0, 2.0, 2.0, 1.0])), os.path.join(d, "ct.nii.gz"))  # viewer grid: 2 mm
    json.dump({"lesions": [{"id": 1, "voxel_in_file": [35, 25, 15]}]}, open(os.path.join(d, "lesions.json"), "w")); json.dump({"available": False}, open(os.path.join(d, "body_composition.json"), "w"))
    assert align_to_viewer(d) is True and align_to_viewer(d) is False  # second call: nothing to do
    t = nib.load(os.path.join(d, "tumour.nii.gz")); L = json.load(open(os.path.join(d, "lesions.json")))
    from glassbox.agent.tools import bare_floats
    assert t.shape == (30, 30, 20) and L["viewer_grid_mm"] == {"value": 2.0, "unit": "mm"} and bare_floats(L) == []  # the agent's unit check must still accept the file
    v = L["lesions"][0]["voxel_in_file"]; assert v == [18, 12, 8] and np.asarray(t.dataobj)[tuple(v)] == 1  # lands inside the lesion on the new grid


def test_recalibrated_view_is_written_next_to_the_default_and_default_is_unchanged(tmp_path):
    path, probs = _scan(str(tmp_path), tumour_prob=0.30)  # a tumour the default threshold (p > 0.5) misses
    out = os.path.join(str(tmp_path), "res"); r = build_results(path, probs, out, "s", {"source": "synthetic"}, recal_logit=-3.0)
    assert r["lesions"]["n_lesions"] == 0  # default untouched
    rc = json.load(open(os.path.join(out, "s", "recalibrated.json")))
    assert rc["default"]["n_lesions"] == 0 and rc["recalibrated"]["n_lesions"] == 1 and rc["recalibrated"]["sum_of_diameters"]["unit"] == "mm" and rc["threshold_logit"] == -3.0
    assert os.path.exists(os.path.join(out, "s", "tumour_recal.nii.gz"))
    r2 = build_results(path, probs, os.path.join(str(tmp_path), "res2"), "s", {})  # without a recalibration threshold nothing extra is written
    assert not os.path.exists(os.path.join(str(tmp_path), "res2", "s", "recalibrated.json")) and r2["lesions"]["n_lesions"] == 0


def test_outline_layers_are_boundaries_and_the_reference_outline_is_dashed(tmp_path):
    import nibabel as nib
    from glassbox.pipeline.viewer import make_outline_layers
    d = str(tmp_path); aff = np.diag([1.5, 1.5, 1.5, 1.0]); m = np.zeros((40, 40, 6), np.uint8); m[10:30, 10:30, 2:4] = 1  # a square tumour on two slices
    nib.save(nib.Nifti1Image(np.zeros((40, 40, 6), np.int16), aff), os.path.join(d, "ct.nii.gz"))
    for n in ("tumour", "reference_tumour"):
        nib.save(nib.Nifti1Image(m, aff), os.path.join(d, f"{n}.nii.gz"))
    assert make_outline_layers(d) == ["tumour_outline", "reference_outline"]
    solid = np.asarray(nib.load(os.path.join(d, "tumour_outline.nii.gz")).dataobj) > 0; dash = np.asarray(nib.load(os.path.join(d, "reference_outline.nii.gz")).dataobj) > 0
    assert solid[:, :, 2].sum() == 76 and not solid[15:25, 15:25, 2].any() and not solid[:, :, 0].any()  # 20x20 square: 76 boundary pixels, none inside, none off-lesion
    assert 0 < dash[:, :, 2].sum() < solid[:, :, 2].sum() and (dash & ~solid).sum() == 0  # dashed is a proper subset of the boundary


def test_a_baked_model_report_is_kept_by_a_server_without_a_model_but_replaced_when_another_model_is_configured(tmp_path):
    from glassbox.agent.config import LLMConfig
    from glassbox.app.report import make_report
    path, probs = _scan(str(tmp_path)); out = os.path.join(str(tmp_path), "res"); build_results(path, probs, out, "s", {"source": "synthetic"})
    rep = make_report(out, "s"); rp = os.path.join(out, "s", "report.json"); d = json.load(open(rp))
    d.update({"mode": "agent", "report": "BAKED MODEL REPORT", "writer_sha": "some-other-provider"}); json.dump(d, open(rp, "w"))  # as if a model had written it elsewhere
    assert make_report(out, "s")["report"] == "BAKED MODEL REPORT"  # no model configured here: keep serving it
    fake = (object(), LLMConfig("https://other.test/v1", "k", "m", "f"))  # a model that is not the writer: the cache is not used (the call below must try the model and fall back)
    assert make_report(out, "s", llm=fake)["report"] != "BAKED MODEL REPORT"
