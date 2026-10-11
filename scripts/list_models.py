"""List the models your configured provider offers, and optionally test that one can call tools (the agent's fast model must).
Reads LLM_BASE_URL and LLM_API_KEY from .env / the environment; never prints the key.
Usage: python scripts/list_models.py [--grep nemotron] [--test MODEL_ID]"""
import argparse
import json
import sys
import urllib.request

sys.path.insert(0, ".")
from glassbox.agent.config import load_env  # noqa: E402
import os  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--grep", default=""); ap.add_argument("--test", default=""); a = ap.parse_args()
load_env(); base = os.environ.get("LLM_BASE_URL", "").rstrip("/"); key = os.environ.get("LLM_API_KEY", "")
if not base or not key:
    sys.exit("set LLM_BASE_URL and LLM_API_KEY in .env first")
host = base.split("//")[-1].split("/")[0]


def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


if a.test:
    tools = [{"type": "function", "function": {"name": "get_value", "description": "Return a value for a scan.", "parameters": {"type": "object", "properties": {"scan_id": {"type": "string"}}, "required": ["scan_id"]}}}]
    r = call("/chat/completions", {"model": a.test, "messages": [{"role": "user", "content": "Call the get_value tool for scan_id 'x'. Do not answer in text."}], "tools": tools, "tool_choice": "auto", "max_tokens": 200})
    tc = (r["choices"][0]["message"].get("tool_calls") or [])
    print(f"{host} {a.test}: tool calling {'WORKS' if tc else 'did NOT produce a tool call'}")
    sys.exit(0 if tc else 1)
ids = sorted(m["id"] for m in call("/models")["data"]); shown = [i for i in ids if a.grep.lower() in i.lower()]
print(f"{host}: {len(ids)} models, {len(shown)} match {a.grep!r}"); print(*shown, sep="\n")
