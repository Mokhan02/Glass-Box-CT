"""Run the agent once against a results folder and print the report, tool calls and grounding check.
Usage: python scripts/agent_smoke.py <results_dir> <scan_id> [--allow-fixture]   (reads LLM_* from .env)
--allow-fixture is required for made-up test data; such output is stamped and must never be published."""
import json
import sys
import time

sys.path.insert(0, ".")
from glassbox.agent.agent import run_agent  # noqa: E402
from glassbox.agent.backends import PrecomputedBackend  # noqa: E402
from glassbox.agent.config import from_env, make_client  # noqa: E402
from glassbox.agent.tools import ToolBox  # noqa: E402

cfg = from_env()
print(cfg)
t = time.time()
out = run_agent(sys.argv[2], make_client(cfg), cfg, ToolBox(PrecomputedBackend(sys.argv[1], allow_fixture='--allow-fixture' in sys.argv)))
print(f"\n--- {time.time() - t:.0f}s | tools called: {[e['tool'] + ('' if e['ok'] else ' (ERROR)') for e in out['evidence']]}")
print("--- report:\n" + out["report"])
print("\n--- unsupported numbers:", out["unsupported_numbers"] or "none", "| missing units:", out["missing_units"] or "none", "| provider:", out["provider_host"])
