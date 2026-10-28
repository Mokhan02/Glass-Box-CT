"""Live smoke test, one real Token Factory call per role, in production mode (GLASSBOX_ENV=production is forced).
Runs the agent (fast model: tool loop; reasoning model: report writer) and one Ask question on a demo scan, and prints the
model, host and latency of every chat call. Never prints the key. Usage: python scripts/live_smoke_roles.py <demo_dir> <scan_id>"""
import os
import sys
import time

os.environ["GLASSBOX_ENV"] = "production"
sys.path.insert(0, ".")
from glassbox.agent.agent import run_agent  # noqa: E402
from glassbox.agent.ask import ask_events  # noqa: E402
from glassbox.agent.backends import PrecomputedBackend  # noqa: E402
from glassbox.agent.config import from_env, make_client  # noqa: E402
from glassbox.agent.tools import ToolBox  # noqa: E402

demo, scan = sys.argv[1], sys.argv[2]
cfg = from_env(); print(cfg)
client = make_client(cfg); log = []
real = client.chat.completions.create


def spy(**kw):
    t = time.time(); r = real(**kw); m = r.choices[0].message
    log.append((kw["model"], "tools" if kw.get("tools") else "text", "tool_calls" if getattr(m, "tool_calls", None) else "message", time.time() - t)); return r


client.chat.completions.create = spy
base = str(client.base_url).rstrip("/")
print("client base_url:", base)
assert base.endswith("api.tokenfactory.nebius.com/v1"), "not Token Factory"
out = run_agent(scan, client, cfg, ToolBox(PrecomputedBackend(demo)))
print(f"agent: attempts={out['attempts']} ok={out['ok']} unsupported={out['unsupported_numbers']} missing_units={out['missing_units']} provider={out['provider_host']}")
n = len(log)
ans = [e for e in ask_events(scan, "How big is the main tumour and how far can I trust that measurement?", client, cfg, ToolBox(PrecomputedBackend(demo))) if e.get("type") == "answer"]
print("ask answer ok:", ans[0].get("ok"), "refused:", ans[0].get("refused"), "|", ans[0]["answer"][:200].replace("\n", " "))
print("\nrole / model / call kind / result / seconds")
for i, (m, k, r, s) in enumerate(log):
    print(f"{'agent' if i < n else 'ask  '} {m} {k} -> {r} {s:.1f}s")
