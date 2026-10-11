"""Screenshot every screen of the React app at desktop (1440) and phone (390) width, record console/network errors. Dev-only (Playwright).
Usage: python scripts/ui_shots.py [--out screenshots/redesign] [--demo data/demo]"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="screenshots/redesign"); ap.add_argument("--demo", default="data/demo"); ap.add_argument("--port", type=int, default=8011); ap.add_argument("--scans", nargs="*", default=["demo_2", "demo_4"]); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); root = os.path.abspath(".")
env = {**os.environ, "GLASSBOX_DEMO_DIR": os.path.abspath(a.demo), "PYTHONPATH": root}
for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_FAST_MODEL"):
    env.pop(k, None)
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "glassbox.app.main:build_app", "--host", "127.0.0.1", "--port", str(a.port)], cwd=tempfile.mkdtemp(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"
for _ in range(60):
    try:
        urllib.request.urlopen(base + "/api/health", timeout=1); break
    except Exception:
        time.sleep(0.5)
errors = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        for tag, vp in (("1440", {"width": 1440, "height": 900}), ("390", {"width": 390, "height": 844})):
            ctx = b.new_context(viewport=vp, device_scale_factor=1, is_mobile=(tag == "390")); pg = ctx.new_page(); cur = {"page": ""}
            pg.on("console", lambda m: errors.append((cur["page"], m.type, m.text[:200])) if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda e: errors.append((cur["page"], "pageerror", str(e)[:200])))
            pg.on("response", lambda r: errors.append((cur["page"], f"http{r.status}", r.url)) if r.status >= 400 else None)
            for name, path, wait in [("landing", "/", "ul li img"), *[(f"scan_{s}", f"/scan/{s}", "text=Quality flag") for s in a.scans], ("why", "/why", "text=From brightness to a missed tumour"), ("about", "/about", "text=A research instrument")]:
                cur["page"] = f"{tag}/{name}"; pg.goto(base + path); pg.wait_for_selector(wait, timeout=40000)
                if name.startswith("scan_"):
                    pg.wait_for_function("() => document.querySelector('canvas') && document.querySelector('[role=status]') === null", timeout=60000); time.sleep(3.0)
                if name == "why":
                    for y in range(0, 6000, 700):
                        pg.evaluate(f"window.scrollTo(0,{y})"); time.sleep(0.25)
                    pg.evaluate("window.scrollTo(0,0)"); time.sleep(0.8)
                else:
                    time.sleep(1.2)
                pg.screenshot(path=f"{a.out}/{tag}_{name}.png", full_page=True); print("shot", tag, name)
            ctx.close()
        b.close()
finally:
    srv.terminate()
print("ERRORS:", *(errors or ["none"]), sep="\n  ")
