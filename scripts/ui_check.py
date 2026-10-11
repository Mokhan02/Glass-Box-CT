"""Real-browser check of the web app (Playwright + Chromium). Starts the server on data/demo, opens every demo scan, asserts what a user would see, records console/page/network errors, screenshots.
Dev-only (pip install playwright; python -m playwright install chromium). Usage: python scripts/ui_check.py [--shots screenshots/phase1] [--port 8010]"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--shots", default="screenshots/phase1"); ap.add_argument("--port", type=int, default=8010); ap.add_argument("--demo", default="data/demo"); a = ap.parse_args()
root = os.path.abspath("."); os.makedirs(a.shots, exist_ok=True)
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
problems, report = [], []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        pg = b.new_page(viewport={"width": 1440, "height": 1200}); errs = []
        pg.on("console", lambda m: errs.append(("console." + m.type, m.text)) if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(("pageerror", str(e))))
        pg.on("requestfailed", lambda r: errs.append(("requestfailed", r.url + " " + str(r.failure))))
        pg.on("response", lambda r: errs.append(("http" + str(r.status), r.url)) if r.status >= 400 else None)
        pg.goto(base + "/"); pg.wait_for_selector("#scanSel option", state="attached", timeout=15000)
        ids = pg.eval_on_selector_all("#scanSel option", "els => els.map(e => e.value)")
        print("scans in selector:", ids)
        for i, sid in enumerate(ids):
            errs.clear(); pg.select_option("#scanSel", sid)
            pg.wait_for_function("() => document.querySelector('#report p') && !document.querySelector('#report').textContent.startsWith('Writing')", timeout=30000)
            try:
                pg.wait_for_function("() => typeof nv !== 'undefined' && nv && nv.volumes.length === 3", timeout=60000); vols = pg.evaluate("() => nv.volumes.length")
            except Exception:
                vols = pg.evaluate("() => (typeof nv !== 'undefined' && nv) ? nv.volumes.length : -1")
            time.sleep(2.5)
            canvas = f"{a.shots}/canvas_{sid}.png"; pg.locator("#gl").screenshot(path=canvas); arr = np.asarray(Image.open(canvas).convert("L")); nonblack = float((arr > 12).mean())
            pg.screenshot(path=f"{a.shots}/page_{sid}.png", full_page=True)
            text = pg.inner_text("#report"); flag = pg.inner_text("#flag"); les = pg.inner_text("#lesions"); evid = pg.inner_text("#evidence"); status = pg.inner_text("#status")
            cols = np.flatnonzero((arr > 12).any(axis=0)); rows_ = np.flatnonzero((arr > 12).any(axis=1))
            extent = (round((cols.max() - cols.min() + 1) / arr.shape[1], 2), round((rows_.max() - rows_.min() + 1) / arr.shape[0], 2)) if len(cols) else (0, 0)
            ls_ = json.load(urllib.request.urlopen(f"{base}/api/scans/{sid}/summary"))["lesions"]["lesions"]
            cross = pg.evaluate("() => Array.from(nv.scene.crosshairPos)"); want = pg.evaluate("(v) => Array.from(nv.vox2frac(v, 0))", ls_[0]["voxel_in_file"]) if ls_ else None
            row = {"scan": sid, "image_extent_wh": extent, "crosshair_on_largest_lesion": bool(want and max(abs(x - y) for x, y in zip(cross, want)) < 0.02), "volumes_loaded": vols, "canvas_nonblack_fraction": round(nonblack, 3), "flag": flag.split("\n")[0][:80], "has_limitation": "known to miss some liver tumours" in text,
                   "has_contrast_line": "contrast" in text.lower(), "has_caution": "Caution (research finding)" in text, "numbers_checked": "every number matches a tool result" in evid, "lesion_rows": les.count("\n"), "status": status, "errors": [e for e in errs]}
            report.append(row)
            if vols != 3: problems.append(f"{sid}: only {vols} volumes loaded")
            if extent[0] < 0.5 or extent[1] < 0.5: problems.append(f"{sid}: image fills only {extent} of the canvas")
            if ls_ and not row["crosshair_on_largest_lesion"]: problems.append(f"{sid}: view did not start on the largest lesion")
            if nonblack < 0.02: problems.append(f"{sid}: canvas looks blank ({nonblack:.3f} non-black)")
            if not row["has_limitation"]: problems.append(f"{sid}: standing limitation missing")
            if not row["numbers_checked"]: problems.append(f"{sid}: number check not shown as passing")
            if status.strip(): problems.append(f"{sid}: status line shows '{status}'")
            for kind, msg in errs:
                if "favicon" in msg: continue
                problems.append(f"{sid}: {kind}: {msg[:160]}")
        b.close()
finally:
    srv.terminate()
json.dump(report, open(os.path.join(a.shots, "ui_check.json"), "w"), indent=1)
for r in report:
    print({k: v for k, v in r.items() if k != "errors"})
print("PROBLEMS:", *(problems or ["none"]), sep="\n  ")
sys.exit(1 if problems else 0)
