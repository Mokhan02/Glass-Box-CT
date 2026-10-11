"""Glass-Box CT web server. Serves the viewer UI, sample-scan files and reports; optional live segmentation when a GPU model is loaded.
Research use only. Never upload patient data."""
from __future__ import annotations

import json
import os
import re
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from glassbox.app.report import make_report

SCAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
VOLUMES = ("ct", "organs", "tumour", "tumour_outline", "tumour_recal", "tumour_recal_outline", "reference_outline")  # what the viewer may draw; a scan lists the ones it has
FILES = {f"{v}.nii.gz" for v in VOLUMES} | {"labels.json", "thumb.png"}
SUMMARY_PARTS = ("segmentation", "body_composition", "lesions", "quality_audit", "recalibrated", "reference")
DIST = os.environ.get("GLASSBOX_FRONTEND_DIST") or os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
MAX_UPLOAD_MB = 300
BANNER = "Research use only. Not for clinical decisions. Do not upload patient data."


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

    @app.get("/api/scans")
    def scans():
        p = os.path.join(demo_dir, "index.json")
        return json.load(open(p)) if os.path.exists(p) else {"scans": []}

    @app.get("/api/scans/{scan_id}/summary")
    def summary(scan_id: str):
        d = scan_dir(scan_id)
        out = {n: json.load(open(os.path.join(d, f"{n}.json"))) for n in SUMMARY_PARTS if os.path.exists(os.path.join(d, f"{n}.json"))}
        out["layers"] = [v for v in VOLUMES if os.path.exists(os.path.join(d, f"{v}.nii.gz"))]
        return out

    @app.post("/api/scans/{scan_id}/report")
    def report(scan_id: str, refresh: bool = False):
        scan_dir(scan_id)
        return make_report(demo_dir, scan_id, llm=llm, refresh=refresh)

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
