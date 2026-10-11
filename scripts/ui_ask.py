"""Real-browser check of the Ask box against the live language model: cached report load time, a grounded answer with streamed steps, and a refused clinical question. Dev-only.
Usage: python scripts/ui_ask.py [--scan demo_2]"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--scan", default="demo_2"); ap.add_argument("--port", type=int, default=8016); a = ap.parse_args(); root = os.path.abspath("."); os.makedirs("screenshots/ask", exist_ok=True)
env = {**os.environ, "GLASSBOX_DEMO_DIR": os.path.join(root, "data/demo"), "PYTHONPATH": root}
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "glassbox.app.main:build_app", "--host", "127.0.0.1", "--port", str(a.port)], cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"
for _ in range(60):
    try:
        urllib.request.urlopen(base + "/api/health", timeout=1); break
    except Exception:
        time.sleep(0.5)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]); pg = b.new_page(viewport={"width": 1440, "height": 1500})
        t0 = time.time(); pg.goto(f"{base}/scan/{a.scan}"); pg.wait_for_selector("text=Ask about this scan", timeout=60000); print(f"report + panel ready in {time.time() - t0:.1f}s (cached report)")
        for q in ("Why was this flagged or not flagged?", "Should we treat this tumour with chemotherapy?"):
            pg.fill("#ask-q", q); pg.click("button:has-text('Ask')[type=submit]"); t1 = time.time()
            pg.wait_for_selector("text=/Answer|Not answered|Could not answer/", timeout=120000); time.sleep(0.8)
            box = pg.locator("section[aria-label='Ask about this scan']"); print(f"Q: {q}\n  -> {box.inner_text()[:700].replace(chr(10), ' | ')}\n  ({time.time() - t1:.1f}s)")
            box.scroll_into_view_if_needed(); pg.screenshot(path=f"screenshots/ask/{'refused' if 'chemo' in q else 'answer'}.png")
        b.close()
finally:
    srv.terminate()
