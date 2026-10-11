"""Backends that produce tool results. PrecomputedBackend serves JSON written by the pipeline (the demo's cached-results
fallback, so the app works even when the GPU endpoint is cold). A live backend that calls the segmentation endpoint over HTTP
implements the same five methods."""
from __future__ import annotations

import json
import os


class PrecomputedBackend:
    FILES = {"segment_ct": "segmentation.json", "body_composition": "body_composition.json",
             "measure_lesions": "lesions.json", "quality_audit": "quality_audit.json"}

    def __init__(self, root: str, allow_fixture: bool = False):
        r = os.path.abspath(root).lower()
        if not allow_fixture and any(w in r for w in ("scratch", "fixture", "/tmp/")):
            raise PermissionError(f"refusing to load results from {root!r}: it looks like scratch or fixture data, not pipeline output")
        self.root, self.allow_fixture = root, allow_fixture

    def _load(self, path: str) -> dict:
        d = json.load(open(path))
        if isinstance(d, dict) and d.get("fixture") and not self.allow_fixture:
            raise PermissionError(f"{path} is marked fixture=true (made-up test data) and cannot be served")
        return d

    def _scan(self, scan_id: str, key: str) -> dict:
        if os.sep in scan_id or scan_id.startswith("."):  # scan ids are folder names, never paths
            raise ValueError("invalid scan_id")
        path = os.path.join(self.root, scan_id, self.FILES[key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"no {key} result for scan {scan_id!r}")
        return self._load(path)

    def segment_ct(self, scan_id): return self._scan(scan_id, "segment_ct")
    def body_composition(self, scan_id): return self._scan(scan_id, "body_composition")
    def measure_lesions(self, scan_id): return self._scan(scan_id, "measure_lesions")
    def quality_audit(self, scan_id): return self._scan(scan_id, "quality_audit")
