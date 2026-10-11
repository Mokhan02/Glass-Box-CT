"""Smoke-test a running Glass-Box CT server (local container, Nebius endpoint, fallback host). Needs only the standard library.
Usage: python scripts/smoke_deployed.py https://your-url [--token TOKEN]   (--token only if the host puts a Bearer token in front of it)"""
import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("--token", default=""); a = ap.parse_args(); base = a.url.rstrip("/")
hdr = {"Authorization": f"Bearer {a.token}"} if a.token else {}


def get(path, data=None, method=None):
    req = urllib.request.Request(base + path, data=data, method=method, headers={**hdr, **({"Content-Type": "application/json"} if data else {})}); t = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=60); return r.status, r.read(), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t


fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else "")); (None if ok else fails.append(name))

s, b, _ = get("/api/health"); h = json.loads(b) if s == 200 else {}; check("health", s == 200 and h.get("ok"), f"language_model={h.get('language_model')} provider={h.get('provider')} model={h.get('model')}")
s, b, _ = get("/api/scans"); scans = json.loads(b)["scans"] if s == 200 else []; check("five demo scans listed", len(scans) == 5, ", ".join(x["id"] for x in scans))
for p in ("/", "/evidence", "/why", "/follow-up", "/about", "/scan/demo_2"):
    s, b, _ = get(p); check(f"page {p}", s == 200 and b"<div id=\"root\">" in b)
if scans:
    sid = scans[1]["id"]
    for f in ("ct.nii.gz", "tumour.nii.gz", "thumb.png"):
        s, b, _ = get(f"/api/scans/{sid}/files/{f}"); check(f"file {f}", s == 200 and len(b) > 1000, f"{len(b)} bytes")
    s, b, dt = get(f"/api/scans/{sid}/report", method="POST"); r = json.loads(b) if s == 200 else {}
    check("report (cached, instant)", s == 200 and r.get("ok") and r.get("cached") and dt < 2.0, f"mode={r.get('mode')} {dt:.2f}s")
    s, b, _ = get(f"/api/scans/{sid}/ask", data=json.dumps({"question": "Should we treat this with chemotherapy?"}).encode(), method="POST")
    check("ask: clinical question refused (or 503 when no model)", (s == 200 and b"refused" in b and b'"refused": true' in b) or s == 503, f"status {s}")
if scans:
    check("every scan carries a Trusted / Review / Unreliable verdict", all(x.get("verdict") in ("Trusted", "Review", "Unreliable") for x in scans), ", ".join(str(x.get("verdict")) for x in scans))
    s, b, _ = get(f"/api/scans/{sid}/summary"); sm = json.loads(b) if s == 200 else {}; check("summary has verdict, recalibrated, reference layers and NO phase in the main payload", bool(sm.get("trust")) and "phase" not in sm and bool(sm.get("recalibrated")) and "reference_outline" in sm.get("layers", []))
    s, b, _ = get(f"/api/scans/{sid}/audit.json"); a_ = json.loads(b) if s == 200 else {}; check("audit.json downloads and has the integrity block", s == 200 and "sha256" in a_.get("integrity", {}) and len(a_.get("model", {}).get("checkpoint_sha256", "")) == 64)
    s, b, _ = get(f"/api/scans/{sid}/audit.pdf"); check("audit.pdf downloads", s == 200 and b[:5] == b"%PDF-" and len(b) > 3000, f"{len(b)} bytes")
s, b, _ = get("/api/longitudinal/model"); lm = json.loads(b) if s == 200 else {}; check("two-timepoint model is available, real (not a fixture) and labelled simulated", s == 200 and lm.get("available") and not lm.get("fixture") and lm.get("simulated"))
s, b, _ = get("/api/longitudinal/simulate", data=json.dumps({"baseline_mm": [60, 30], "change": -0.2, "persistence": 0.5}).encode(), method="POST"); sim = json.loads(b) if s == 200 else {}; check("simulation runs and every response is stamped simulated", s == 200 and sim.get("simulated") is True and "not real patients" in sim.get("notice", ""))
s, _, _ = get("/api/scans/..%2f..%2fetc/files/ct.nii.gz"); check("path traversal rejected", s in (400, 404))
s, _, _ = get("/api/scans/demo_2/files/report.json"); check("internal files not downloadable", s == 404)
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}"); sys.exit(1 if fails else 0)
