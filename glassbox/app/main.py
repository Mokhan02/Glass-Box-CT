"""Server entry point: `uvicorn --factory glassbox.app.main:build_app --host 0.0.0.0 --port 8000`.
Environment: GLASSBOX_DEMO_DIR (sample scans and results; default ./demo), GLASSBOX_LIVE=1 and GLASSBOX_WEIGHTS=/models/model.pt to enable live segmentation on a GPU,
and NEBIUS_API_KEY + LLM_MODEL + LLM_FAST_MODEL (Nebius Token Factory) to have the agent write reports (development may use LLM_BASE_URL / LLM_API_KEY for another provider;
without a model the deterministic template is used). GLASSBOX_ENV=production allows Token Factory only and refuses to start otherwise."""
from __future__ import annotations

import os
import shutil

from glassbox.agent.config import ConfigError, from_env, is_production, make_client
import logging
import threading

from glassbox.app.report import warm_reports
from glassbox.app.server import create_app

log = logging.getLogger("uvicorn.error")


def startup_line(llm, demo: str, live: bool) -> str:
    """The one line that says which provider and models write reports on this server (never the key)."""
    who = f"language model provider={llm[1].provider_host} model={llm[1].model} fast_model={llm[1].fast_model}" if llm else "language model=none (reports use the deterministic template; Ask is disabled)"
    return f"Glass-Box CT: {who}; demo_dir={demo}; live_segmentation={'on' if live else 'off'}"


def build_app():
    demo = os.environ.get("GLASSBOX_DEMO_DIR", "demo"); os.makedirs(demo, exist_ok=True); live = None
    if os.environ.get("GLASSBOX_LIVE") == "1":
        import torch

        from glassbox.pipeline.scan import build_results, infer_probs
        from glassbox.pipeline.viewer import make_viewer_files
        from glassbox.seg.infer import load_network

        device = torch.device("cuda"); net = load_network(os.environ["GLASSBOX_WEIGHTS"], device)

        def live(ct_path: str, scan_id: str) -> None:
            d = os.path.join(demo, scan_id); original = os.path.join(d, "original.nii.gz"); shutil.move(ct_path, original)
            probs = infer_probs(original, net, device)
            build_results(original, probs, demo, scan_id, {"source": "user upload", "in_distribution": "unknown", "model": "NV-Segment-CT vista3d132"})
            make_viewer_files(original, d)  # browser-sized copies; measurements were made on the original grid
    try:
        cfg = from_env(); llm = (make_client(cfg), cfg)
    except ConfigError:
        if is_production():  # no silent degradation: the runtime must call Token Factory, so a bad provider setting stops the server with the reason
            raise
        llm = None
    log.info(startup_line(llm, demo, live is not None))
    app = create_app(demo, live=live, llm=llm)
    if os.environ.get("GLASSBOX_WARM_REPORTS", "1") != "0":  # build any report that is not cached yet in the background; cached ones load instantly
        threading.Thread(target=warm_reports, args=(demo, llm, log.info), daemon=True).start()
    return app
