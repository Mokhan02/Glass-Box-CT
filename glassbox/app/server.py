"""Glass-Box CT web server. Serves the viewer UI, sample-scan files and reports; optional live segmentation when a GPU model is loaded.
Research use only. Never upload patient data."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from glassbox.agent.ask import MAX_QUESTION, ask_events
from glassbox.agent.backends import PrecomputedBackend
from glassbox.agent.tools import ToolBox
from glassbox.app.report import make_report
from glassbox.trust.audit import audit_pdf, build_audit
from glassbox.trust.longitudinal import SIMULATED, load_model, simulate
from glassbox.trust.verdict import assess

SCAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
VOLUMES = ("ct", "organs", "tumour", "tumour_outline", "tumour_recal", "tumour_recal_outline", "reference_outline")  # what the viewer may draw; a scan lists the ones it has
FILES = {f"{v}.nii.gz" for v in VOLUMES} | {"labels.json", "thumb.png"}
SUMMARY_PARTS = ("segmentation", "body_composition", "lesions", "quality_audit", "recalibrated", "reference", "phase")
DIST = os.environ.get("GLASSBOX_FRONTEND_DIST") or os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
MAX_UPLOAD_MB = 300
ASK_PER_MINUTE, ASK_PER_HOUR = 6, 120
BANNER = "Research use only. Not for clinical decisions. Do not upload patient data."


class Question(BaseModel):
    question: str


class SimRequest(BaseModel):
    baseline_mm: list[float]
    change: float
    bright_baseline: bool = False
    bright_followup: bool = False


def create_app(demo_dir: str, live=None, llm=None) -> FastAPI:
    """live: optional callable (ct_path, scan_id) -> None that writes result files under demo_dir/<scan_id>/ (GPU). llm: optional (client, cfg) for the agent."""
    app = FastAPI(title="Glass-Box CT", description=BANNER); demo_dir = os.path.abspath(demo_dir)

    def scan_dir(scan_id: str) -> str:
        if not SCAN_ID.match(scan_id):
            raise HTTPException(400, "invalid scan id")
        d = os.path.join(demo_dir, scan_id)
        if not os.path.isdir(d):
            raise HTTPException(404, "unknown scan")
        return d

    @app.get("/api/health")
    def health():
        h = {"ok": True, "live_segmentation": live is not None, "language_model": llm is not None, "notice": BANNER}
        if llm is not None:  # which provider and model write reports (never the key)
            h.update({"provider": llm[1].provider_host, "model": llm[1].model, "fast_model": llm[1].fast_model})
        return h

    def trust(d: str) -> dict | None:
        """The Trusted / Review / Unreliable verdict, computed from the stored tool results (deterministic; shared with the audit record)."""
        try:
            les = json.load(open(os.path.join(d, "lesions.json"))); qa = json.load(open(os.path.join(d, "quality_audit.json")))["structures"][0]
            ph = os.path.join(d, "phase.json")
            return assess(les, qa, json.load(open(ph)) if os.path.exists(ph) else None)
        except (OSError, KeyError, IndexError, ValueError):
            return None

    @app.get("/api/scans")
    def scans():
        p = os.path.join(demo_dir, "index.json")
        data = json.load(open(p)) if os.path.exists(p) else {"scans": []}
        for sc in data["scans"]:
            t = trust(os.path.join(demo_dir, sc["id"])); sc["verdict"] = t["level"] if t else None
        return data

    @app.get("/api/scans/{scan_id}/summary")
    def summary(scan_id: str):
        d = scan_dir(scan_id)
        out = {n: json.load(open(os.path.join(d, f"{n}.json"))) for n in SUMMARY_PARTS if os.path.exists(os.path.join(d, f"{n}.json"))}
        out["layers"] = [v for v in VOLUMES if os.path.exists(os.path.join(d, f"{v}.nii.gz"))]
        out["trust"] = trust(d)
        return out

    @app.post("/api/scans/{scan_id}/report")
    def report(scan_id: str, refresh: bool = False):
        scan_dir(scan_id)
        return make_report(demo_dir, scan_id, llm=llm, refresh=refresh)

    asked: list[float] = []; lock = threading.Lock()  # a small global limit protects the language-model key from a public page

    @app.post("/api/scans/{scan_id}/ask")
    def ask(scan_id: str, body: Question):
        scan_dir(scan_id)
        if llm is None:
            return JSONResponse({"error": "Asking questions needs a language model, and none is configured on this server."}, status_code=503)
        if len(body.question) > MAX_QUESTION * 2:
            raise HTTPException(413, "question too long")
        with lock:
            now = time.time(); asked[:] = [t for t in asked if now - t < 3600]
            if len([t for t in asked if now - t < 60]) >= ASK_PER_MINUTE or len(asked) >= ASK_PER_HOUR:
                return JSONResponse({"error": "Too many questions just now. Please wait a moment."}, status_code=429)
            asked.append(now)
        client, cfg = llm; tb = ToolBox(PrecomputedBackend(demo_dir))

        def stream():
            try:
                for ev in ask_events(scan_id, body.question, client, cfg, tb):
                    yield json.dumps(ev) + "\n"
            except Exception as e:  # provider error mid-question: say so, never a stack trace
                yield json.dumps({"type": "answer", "answer": "The language model is unavailable right now, so I can't answer. The report and measurements above still work.", "ok": False, "refused": False, "error": type(e).__name__}) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    def long_model() -> dict | None:
        m = load_model()
        return m if m and (not m.get("fixture") or os.environ.get("GLASSBOX_ALLOW_FIXTURE") == "1") else None  # a test-fixture model is never served unless explicitly allowed

    @app.get("/api/longitudinal/model")
    def longitudinal_model():
        m = long_model()
        if m is None:
            return {"available": False, "simulated": True, "notice": SIMULATED}
        return {"available": True, "simulated": True, "notice": SIMULATED, "fixture": bool(m.get("fixture")), "source": m.get("source"), "n_lesions": m["n_lesions"], "presets": m.get("presets", []),
                "median_error_mm_detected": {k: v["median_error_mm_detected"] for k, v in m["methods"].items()}}

    @app.post("/api/longitudinal/simulate")
    def longitudinal_simulate(body: SimRequest):
        m = long_model()
        if m is None:
            return JSONResponse({"error": "The two-timepoint bias model is not installed on this server."}, status_code=503)
        if not (1 <= len(body.baseline_mm) <= 3) or any(not 5 <= d <= 250 for d in body.baseline_mm) or not -0.95 <= body.change <= 1.5:
            raise HTTPException(422, "baseline_mm: 1 to 3 sizes of 5 to 250 mm; change: -0.95 to 1.5")
        try:
            r = simulate(m, body.baseline_mm, body.change, body.bright_baseline, body.bright_followup, n=1500)
        except ValueError as e:
            raise HTTPException(422, str(e))
        r["fixture"] = bool(m.get("fixture")); return r

    @app.get("/api/scans/{scan_id}/audit.json")
    def audit_json(scan_id: str):
        scan_dir(scan_id)
        return Response(json.dumps(build_audit(demo_dir, scan_id, llm=llm), indent=1), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="glassbox-audit-{scan_id}.json"'})

    @app.get("/api/scans/{scan_id}/audit.pdf")
    def audit_pdf_file(scan_id: str):
        scan_dir(scan_id)
        return Response(audit_pdf(build_audit(demo_dir, scan_id, llm=llm)), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="glassbox-audit-{scan_id}.pdf"'})

    @app.get("/api/scans/{scan_id}/files/{name}")
    def files(scan_id: str, name: str):
        if name not in FILES:
            raise HTTPException(404, "no such file")
        p = os.path.join(scan_dir(scan_id), name)
        if not os.path.exists(p):
            raise HTTPException(404, "no such file")
        return FileResponse(p)

    @app.post("/api/segment")
    async def segment(file: UploadFile = File(...)):
        if live is None:
            return JSONResponse({"error": "Live segmentation is not enabled on this server. Use a sample scan.", "notice": BANNER}, status_code=503)
        if not (file.filename or "").lower().endswith((".nii", ".nii.gz")):
            raise HTTPException(400, "upload a NIfTI file (.nii or .nii.gz)")
        scan_id = "upload_" + os.urandom(4).hex(); d = os.path.join(demo_dir, scan_id); os.makedirs(d)
        size = 0
        with open(os.path.join(d, "ct.nii.gz"), "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * (1 << 20):
                    raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
                f.write(chunk)
        live(os.path.join(d, "ct.nii.gz"), scan_id)
        return {"scan_id": scan_id, "notice": BANNER}

    static = os.path.join(os.path.dirname(__file__), "static"); dist = os.path.abspath(DIST); built = os.path.exists(os.path.join(dist, "index.html"))
    if built:  # the React app (frontend/dist): hashed assets, and index.html for every client-side route
        app.mount("/assets", StaticFiles(directory=os.path.join(dist, "assets")), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "not found")
            f = os.path.normpath(os.path.join(dist, path))
            if path and f.startswith(dist + os.sep) and os.path.isfile(f):
                return FileResponse(f)
            return FileResponse(os.path.join(dist, "index.html"))
    else:  # no frontend build on this machine: serve the original single-file page
        app.mount("/static", StaticFiles(directory=static), name="static")

        @app.get("/")
        def index():
            return FileResponse(os.path.join(static, "index.html"))

    return app
