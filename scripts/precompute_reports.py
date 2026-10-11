"""Build the cached report of every demo scan with whatever language model .env configures (or the template if none), so a deployed server shows them instantly.
Run this after setting the final provider (see docs/TOKEN_FACTORY.md) and BEFORE building the Docker image: the cache key includes provider, models and prompt.
Usage: python scripts/precompute_reports.py [data/demo]"""
import os
import sys

sys.path.insert(0, ".")
from glassbox.agent.config import ConfigError, from_env, make_client  # noqa: E402
from glassbox.app.report import warm_reports  # noqa: E402

demo = sys.argv[1] if len(sys.argv) > 1 else "data/demo"
try:
    cfg = from_env(); llm = (make_client(cfg), cfg); print(f"language model: {cfg.provider_host} {cfg.model} (fast: {cfg.fast_model})")
except ConfigError as e:
    llm = None; print(f"no language model configured ({e}); building template reports")
warm_reports(demo, llm)
