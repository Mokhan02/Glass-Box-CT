import json
import os

import nibabel as nib
import numpy as np

from glassbox.agent.backends import PrecomputedBackend
from glassbox.agent.tools import ToolBox
from glassbox.bodycomp.lesions import measure_lesions
from glassbox.pipeline.scan import ORGANS, PROMPTED, build_results
from glassbox.seg import classes as C


def _scan(tmp_path, tumour_radius=6, tumour_prob=0.95, with_l3=True):
    shape = (80, 80, 50); zoom = (0.8, 0.8, 3.0); ct = np.full(shape, -50.0, np.float32); probs = {c: np.full(shape, 0.01, np.float32) for c in PROMPTED}
    xx, yy, zz = np.mgrid[:80, :80, :50]
    def blob(cx, cy, cz, rx, ry, rz): return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2 <= 1
    liver = blob(30, 40, 25, 22, 20, 15); probs[C.LIVER][liver] = 0.97; ct[liver] = 100.0
    t = blob(30, 40, 25, tumour_radius, tumour_radius, 6); probs[C.HEPATIC_TUMOR][t] = tumour_prob; ct[t] = 70.0
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
