"""Screenshot every demo scan at 1440px (top of page and scrolled, to show the sticky viewer) and measure how the CT sits in the viewer.
Fit/centre check: NiiVue's own tile rectangle for the slice must cover >= 85% of the canvas's shorter side and be centred within 3% of the canvas.
Usage: python scripts/ui_demos.py --step step1 [--demo data/demo] [--scans demo_1 ...]"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--step", required=True); ap.add_argument("--demo", default="data/demo"); ap.add_argument("--port", type=int, default=8014); ap.add_argument("--scans", nargs="*", default=["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]); ap.add_argument("--live", action="store_true", help="run from the repo root so .env is read and the language-model agent writes the reports"); ap.add_argument("--w", type=int, default=1440); ap.add_argument("--h", type=int, default=900); a = ap.parse_args()
out = f"screenshots/revision/{a.step}"; os.makedirs(out, exist_ok=True); root = os.path.abspath(".")
env = {**os.environ, "GLASSBOX_DEMO_DIR": os.path.abspath(a.demo), "PYTHONPATH": root}
for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_FAST_MODEL"):
    if not a.live:
        env.pop(k, None)
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "glassbox.app.main:build_app", "--host", "127.0.0.1", "--port", str(a.port)], cwd=root if a.live else tempfile.mkdtemp(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
base = f"http://127.0.0.1:{a.port}"
for _ in range(60):
    try:
        urllib.request.urlopen(base + "/api/health", timeout=1); break
    except Exception:
        time.sleep(0.5)
problems, rows = [], []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": a.w, "height": a.h}); errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:200])); pg.on("console", lambda m: errs.append(m.text[:200]) if m.type == "error" else None)
        for sid in a.scans:
            errs.clear(); pg.goto(f"{base}/scan/{sid}")
            pg.wait_for_function("() => window.__nv && window.__nv.volumes.length >= 3 && document.querySelector('section[aria-label=Key-numbers]') !== null || (window.__nv && window.__nv.volumes.length >= 3 && Array.from(document.querySelectorAll('section')).some(s => s.getAttribute('aria-label') === 'Key numbers'))", timeout=300000); time.sleep(3.5)
            m = pg.evaluate("""() => { const nv = window.__nv, c = nv.canvas, t = nv.screenSlices.filter(s => s.leftTopWidthHeight && s.leftTopWidthHeight[2] > 0)[0].leftTopWidthHeight, r = c.getBoundingClientRect(), k = c.width / r.width;
                return {canvas: [Math.round(c.width / k), Math.round(c.height / k)], tile: t.map(v => Math.round(v / k)), vols: nv.volumes.map(v => [v.name, +v.opacity.toFixed(2)])} }""")
            cw, ch = m["canvas"]; l, t, w, h = m["tile"]; cover = max(w / min(cw, ch), h / min(cw, ch)); cx, cy = abs((l + w / 2) - cw / 2) / cw, abs((t + h / 2) - ch / 2) / ch
            row = {"scan": sid, "canvas": m["canvas"], "tile": m["tile"], "coverage_of_short_side": round(cover, 2), "centre_offset": [round(cx, 3), round(cy, 3)], "layers": m["vols"], "errors": list(errs)}; rows.append(row)
            if cover < 0.85 or cx > 0.03 or cy > 0.03: problems.append(f"{sid}: CT not fitted/centred (tile {m['tile']} in canvas {m['canvas']})")
            if errs: problems.append(f"{sid}: {errs[:2]}")
            pg.screenshot(path=f"{out}/{sid}_top.png"); pg.evaluate("window.scrollTo(0, 900)"); time.sleep(0.8); pg.screenshot(path=f"{out}/{sid}_scrolled.png"); pg.evaluate("window.scrollTo(0, 0)")
            print("shot", sid, row["coverage_of_short_side"], row["centre_offset"], flush=True)
        b.close()
finally:
    srv.terminate()
json.dump(rows, open(f"{out}/metrics.json", "w"), indent=1); print("PROBLEMS:", *(problems or ["none"]), sep="\n  ")
