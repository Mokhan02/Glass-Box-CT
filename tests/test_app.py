import json
import os

from fastapi.testclient import TestClient

from glassbox.app.server import create_app
from glassbox.pipeline.scan import build_results
from tests.test_pipeline import _scan


def _demo(tmp_path):
    d = os.path.join(str(tmp_path), "demo"); path, probs = _scan(str(tmp_path)); build_results(path, probs, d, "scan_a", {"source": "synthetic", "in_distribution": False})
    os.replace(path, os.path.join(d, "scan_a", "ct.nii.gz"))
    json.dump({"scans": [{"id": "scan_a", "title": "Synthetic"}]}, open(os.path.join(d, "index.json"), "w")); return d


def test_api_lists_scans_serves_only_allowlisted_files_and_rejects_bad_ids(tmp_path):
    c = TestClient(create_app(_demo(tmp_path)))
    assert c.get("/api/health").json()["live_segmentation"] is False and c.get("/api/scans").json()["scans"][0]["id"] == "scan_a"
    assert c.get("/api/scans/scan_a/files/ct.nii.gz").status_code == 200 and c.get("/api/scans/scan_a/files/report.json").status_code == 404
    assert c.get("/api/scans/..%2F..%2Fetc/summary").status_code in (400, 404) and c.get("/api/scans/nope/summary").status_code == 404
    assert c.get("/").status_code == 200 and "Research use only" in c.get("/").text


def test_template_report_passes_the_grounding_and_unit_check_and_states_the_known_bias(tmp_path):
    c = TestClient(create_app(_demo(tmp_path))); r = c.post("/api/scans/scan_a/report").json()
    assert r["mode"] == "template" and r["ok"], (r["unsupported_numbers"], r["missing_units"])
    assert r["report"].endswith("Research use only, not for clinical decisions.") and "Known bias" in r["report"] and "never overestimated" in r["report"]
    assert c.post("/api/scans/scan_a/report").json()["report"] == r["report"]  # cached, reproducible


def test_live_segmentation_is_disabled_by_default_and_uploads_are_validated(tmp_path):
    c = TestClient(create_app(_demo(tmp_path))); assert c.post("/api/segment", files={"file": ("x.nii.gz", b"abc")}).status_code == 503
    called = []; c2 = TestClient(create_app(_demo(tmp_path), live=lambda p, s: called.append(s)))
    assert c2.post("/api/segment", files={"file": ("x.txt", b"abc")}).status_code == 400
    assert c2.post("/api/segment", files={"file": ("x.nii.gz", b"abc")}).status_code == 200 and called
