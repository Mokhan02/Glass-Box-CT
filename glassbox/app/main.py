"""Server entry point: `uvicorn --factory glassbox.app.main:build_app --host 0.0.0.0 --port 8000`.
Environment: GLASSBOX_DEMO_DIR (sample scans and results; default ./demo), GLASSBOX_LIVE=1 and GLASSBOX_WEIGHTS=/models/model.pt to enable live segmentation on a GPU,
and LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (optional LLM_FAST_MODEL) to have the agent write reports (otherwise the deterministic template is used)."""
from __future__ import annotations

import os
import shutil

from glassbox.agent.config import ConfigError, from_env, make_client
from glassbox.app.server import create_app


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
        llm = None
    return create_app(demo, live=live, llm=llm)
